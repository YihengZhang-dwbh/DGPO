from __future__ import annotations

from functools import partial
from typing import Literal

import dataclasses
import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
import optax
from jax import Array
from jax import numpy as jnp

from flow_policy.networks import MlpWeights
from . import math_utils, networks, rollouts


@jdc.pytree_dataclass
class IgmmConfig:
    # ==========================================
    # 👑 流匹配核心参数 (The Flow)
    # ==========================================
    flow_steps: jdc.Static[int] = 10
    training_flow_steps: jdc.Static[int] = 2
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = "u_but_supervise_as_eps"
    timestep_embed_dim: jdc.Static[int] = 8
    discretize_t_for_training: jdc.Static[bool] = True
    policy_mlp_output_scale: float = 0.25

    # ==========================================
    # 👑 全局退火方差调度 (DDPM-Style Annealing)
    # ==========================================
    num_epsilon_samples: jdc.Static[int] = 16
    clipping_epsilon: float = 0.2

    # 🚀 降维打击：网络不再预测方差，使用全局统一的随时间衰减的超参数
    initial_std: float = 0.5  # 训练初期的超大探索方差
    final_std: float = 0.05  # 训练末期的精准打击方差

    action_clip: jdc.Static[Literal["hard", "margin", "tanh", "fold", "scale_clip"]] = "margin"
    clip_margin: float = 10.0

    # ==========================================
    # 经典的 On-Policy / V-Net 参数
    # ==========================================
    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    learning_rate: float = 3e-4
    normalize_observations: jdc.Static[bool] = True
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 30
    num_minibatches: jdc.Static[int] = 32
    num_timesteps: jdc.Static[int] = 180_000_000
    num_updates_per_batch: jdc.Static[int] = 16
    reward_scaling: float = 10.0
    unroll_length: jdc.Static[int] = 30

    gae_lambda: float = 0.95
    normalize_advantage: jdc.Static[bool] = True
    value_loss_coeff: float = 0.25

    def __post_init__(self) -> None:
        assert self.timestep_embed_dim % 2 == 0

    @property
    def iterations_per_env(self) -> int:
        return (self.num_minibatches * self.batch_size * self.unroll_length) // self.num_envs

    @property
    def total_training_steps(self) -> int:
        outer_iters = self.num_timesteps // (self.iterations_per_env * self.num_envs)
        return outer_iters * self.num_updates_per_batch * self.num_minibatches


@jdc.pytree_dataclass
class IgmmParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class IgmmActionInfo:
    raw_action: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))

    eps_batch: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    log_p_old: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    gae_vs: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    gae_advantages: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))


@jdc.pytree_dataclass
class FlowSchedule:
    t_current: Array
    t_next: Array


IgmmTransition = rollouts.TransitionStruct[IgmmActionInfo]


@jdc.pytree_dataclass
class IgmmState:
    env: jdc.Static[mjp.MjxEnv]
    config: IgmmConfig
    params: IgmmParams
    obs_stats: math_utils.RunningStats

    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState

    prng: Array
    steps: Array

    @staticmethod
    @jdc.jit
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: IgmmConfig) -> IgmmState:
        obs_size = env.observation_size
        action_size = env.action_size

        # 🚀 降维打击：去掉方差，ODE 状态回归最纯粹的动作维度！
        theta_dim = action_size

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(
            prng0,
            (obs_size + theta_dim + config.timestep_embed_dim, 32, 32, 32, 32, theta_dim),
        )
        critic_net = networks.mlp_init(prng1, (obs_size, 256, 256, 256, 256, 256, 1))

        network_params = IgmmParams(actor_net, critic_net)

        # 保留全局梯度裁剪作为最后一道防线
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam()
        )

        return IgmmState(
            env=env, config=config, params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt=opt, opt_state=opt.init(network_params),
            prng=prng2, steps=jnp.zeros((), dtype=jnp.int32),
        )

    def _apply_clip(self, x: Array) -> Array:
        cfg = self.config
        if cfg.action_clip == "hard":
            return jnp.clip(x, -1.0, 1.0)
        elif cfg.action_clip == "margin":
            return jnp.clip(x, -cfg.clip_margin, cfg.clip_margin)
        elif cfg.action_clip == "tanh":
            return cfg.clip_margin * jnp.tanh(x / cfg.clip_margin)
        elif cfg.action_clip == "fold":
            T = 4.0 * cfg.clip_margin
            t = cfg.clip_margin
            return jnp.abs(x - T * jnp.floor((x + 3 * t) / T) + t) - t
        return x

    def get_schedule(self, num_steps: int) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, num_steps + 1)
        return FlowSchedule(t_current=full_t_path[:-1], t_next=full_t_path[1:])

    def embed_timestep(self, t: Array) -> Array:
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def get_current_std(self) -> Array:
        """🚀 带有总体衰减包络的余弦波动探索 (Cyclical Cosine Annealing)"""
        progress = jnp.clip(self.steps / self.config.total_training_steps, 0.0, 1.0)

        # 设定整个训练过程波动的周期数 (比如 3 次大波谷大波峰)
        num_cycles = 3.0

        # 当前处于哪个周期的什么位置 (0.0 到 1.0 之间循环)
        cycle_progress = (progress * num_cycles) % 1.0

        # 经典的余弦退火波形：在每个周期内从 1.0 平滑降到 0.0
        cosine_wave = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))

        # 加一个线性的衰减包络：让每次波峰都比上一次矮，防止大后期还在瞎探索
        envelope = 1.0 - progress

        # 最终的 sigma：基础下限 + 波动幅度 * 余弦波 * 衰减包络
        return self.config.final_std + (self.config.initial_std - self.config.final_std) * cosine_wave * envelope

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, IgmmActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / (
                    self.obs_stats.std + 1e-8) if self.config.normalize_observations else obs
        (*batch_dims, obs_dim) = obs.shape
        theta_dim = self.env.action_size

        def euler_step(carry: Array, inputs: tuple[FlowSchedule, Array]) -> tuple[Array, Array]:
            x_t = carry
            schedule_t, _ = inputs
            dt = schedule_t.t_next - schedule_t.t_current

            velocity = networks.flow_mlp_fwd(
                self.params.policy, obs_norm, x_t,
                jnp.broadcast_to(
                    self.embed_timestep(schedule_t.t_current[None]),
                    (*batch_dims, self.config.timestep_embed_dim),
                ),
            ) * self.config.policy_mlp_output_scale

            return x_t + dt * velocity, x_t

        prng_sample, prng_action = jax.random.split(prng, 2)
        eps_1 = jax.random.normal(prng_sample, (*batch_dims, theta_dim))

        sch = self.get_schedule(self.config.flow_steps)
        # 🚀 ODE 直接输出均值 mu，没有任何冗余的切片！
        mu_1, _ = jax.lax.scan(
            euler_step, init=eps_1,
            xs=(sch, jnp.zeros((self.config.flow_steps, 1))),
        )

        current_std = self.get_current_std()
        raw_action = mu_1 if deterministic else mu_1 + current_std * jax.random.normal(prng_action, mu_1.shape)
        action_clipped = self._apply_clip(raw_action)

        return action_clipped, IgmmActionInfo(raw_action=raw_action)

    @jdc.jit
    def training_step(self, transitions: IgmmTransition) -> tuple[IgmmState, dict[str, Array]]:
        config, state = self.config, self

        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)

        obs_norm = (transitions.obs - state.obs_stats.mean) / (
                    state.obs_stats.std + 1e-8) if config.normalize_observations else transitions.obs
        value_pred = networks.value_mlp_fwd(state.params.value, obs_norm)
        bootstrap_obs_norm = (transitions.next_obs[-1:, :, :] - state.obs_stats.mean) / (state.obs_stats.std + 1e-8)
        bootstrap_value = networks.value_mlp_fwd(state.params.value, bootstrap_obs_norm)

        gae_vs, gae_advantages = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation, discount=transitions.discount * config.discounting,
                rewards=transitions.reward * config.reward_scaling, values=value_pred,
                bootstrap_value=bootstrap_value, gae_lambda=config.gae_lambda,
            )
        )

        global_adv_mean = jnp.mean(gae_advantages)
        if config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

        # ==========================================
        # 提取旧策略概率 p(a|s; mu_old, current_std)
        # ==========================================
        prng_eps = jax.random.fold_in(state.prng, state.steps)
        M = config.num_epsilon_samples
        theta_dim = state.env.action_size

        eps_batch = jax.random.normal(prng_eps, (*transitions.reward.shape, M, theta_dim))

        def forward_ode_step_old(carry, t_tup):
            x = carry
            t_curr, t_next = t_tup
            t_embed = state.embed_timestep(jnp.broadcast_to(t_curr, (*x.shape[:-1], 1)))
            obs_b = jnp.broadcast_to(obs_norm[..., None, :], (*x.shape[:-1], obs_norm.shape[-1]))
            v = networks.flow_mlp_fwd(jax.lax.stop_gradient(state.params.policy), obs_b, x,
                                      t_embed) * config.policy_mlp_output_scale
            return x + (t_next - t_curr) * v, None

        train_sch = state.get_schedule(config.training_flow_steps)
        mu_old, _ = jax.lax.scan(forward_ode_step_old, eps_batch, (train_sch.t_current, train_sch.t_next))
        mu_old = jax.lax.stop_gradient(mu_old)

        current_std = state.get_current_std()
        a_b = transitions.action[..., None, :]

        # 🚀 极致干净的对数概率
        log_p_old = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(current_std) - ((a_b - mu_old) ** 2) / (
                    2 * current_std ** 2 + 1e-8)
        log_p_old = jnp.sum(log_p_old, axis=-1)

        new_action_info = jdc.replace(
            transitions.action_info,
            eps_batch=eps_batch,
            log_p_old=log_p_old,
            gae_vs=gae_vs,
            gae_advantages=gae_advantages
        )
        cached_transitions = jdc.replace(transitions, action_info=new_action_info)

        def step_batch(carry_state: IgmmState, _):
            step_prng = jax.random.fold_in(carry_state.prng, carry_state.steps)
            new_state, metrics = jax.lax.scan(
                partial(IgmmState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)),
                init=carry_state,
                xs=cached_transitions.prepare_minibatches(step_prng, config.num_minibatches, config.batch_size),
            )
            return new_state, metrics

        state, metrics = jax.lax.scan(step_batch, init=state, length=config.num_updates_per_batch)
        metrics["advantages_mean"] = global_adv_mean
        return state, metrics

    def _step_minibatch(self, transitions: IgmmTransition, prng: Array) -> tuple[IgmmState, dict[str, Array]]:
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: IgmmState._compute_policy_loss(jdc.replace(self, params=params), transitions, prng),
            has_aux=True,
        )(self.params)

        param_update, new_opt_state = self.opt.update(grads, self.opt_state)
        param_update = jax.tree.map(lambda x: -self.config.learning_rate * x, param_update)
        with jdc.copy_and_mutate(self) as state:
            state.params = jax.tree.map(jnp.add, self.params, param_update)
            state.opt_state = new_opt_state
            state.steps = state.steps + 1
        return state, metrics

    def _compute_policy_loss(self, transitions: IgmmTransition, prng: Array) -> tuple[Array, dict[str, Array]]:
        obs_norm = (transitions.obs - self.obs_stats.mean) / (
                    self.obs_stats.std + 1e-8) if self.config.normalize_observations else transitions.obs
        value_pred = networks.value_mlp_fwd(self.params.value, obs_norm)

        eps_batch = transitions.action_info.eps_batch
        log_p_old = transitions.action_info.log_p_old
        gae_vs = transitions.action_info.gae_vs
        A = transitions.action_info.gae_advantages

        current_std = self.get_current_std()

        def forward_ode_step_new(carry, t_tup):
            x = carry
            t_curr, t_next = t_tup
            t_embed = self.embed_timestep(jnp.broadcast_to(t_curr, (*x.shape[:-1], 1)))
            obs_b = jnp.broadcast_to(obs_norm[..., None, :], (*x.shape[:-1], obs_norm.shape[-1]))
            # 前向计算包含梯度
            v = networks.flow_mlp_fwd(self.params.policy, obs_b, x, t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * v, None

        train_sch = self.get_schedule(self.config.training_flow_steps)
        # 🚀 预测的纯粹就是 mu_pred，没有任何切片和多余维度
        mu_pred, _ = jax.lax.scan(forward_ode_step_new, eps_batch, (train_sch.t_current, train_sch.t_next))

        a_b = transitions.action[..., None, :]

        # 计算新策略下的概率
        log_p_pred = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(current_std) - ((a_b - mu_pred) ** 2) / (
                    2 * current_std ** 2 + 1e-8)
        log_p_pred = jnp.sum(log_p_pred, axis=-1)

        # 稳健截断 ratio，因为已经去掉了指数爆炸的根源，这里实际上起双保险作用
        log_ratio = jnp.clip(log_p_pred - log_p_old, -10.0, 10.0)
        ratio = jnp.exp(log_ratio)

        surr1 = ratio * A[..., None]
        surr2 = jnp.clip(ratio, 1.0 - self.config.clipping_epsilon, 1.0 + self.config.clipping_epsilon) * A[..., None]

        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        v_loss = jnp.mean(v_error ** 2) * self.config.value_loss_coeff

        return policy_loss + v_loss, {
            "policy_loss": policy_loss,
            "v_loss": v_loss,
            "ratio_mean": jnp.mean(ratio),
            "current_sigma": current_std
        }