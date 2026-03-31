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
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = "u_but_supervise_as_eps"
    timestep_embed_dim: jdc.Static[int] = 8
    discretize_t_for_training: jdc.Static[bool] = True
    policy_mlp_output_scale: float = 0.25

    # ==========================================
    # 👑 IGMM 独家范式超参 (Wasserstein Geodesic Reprojection)
    # ==========================================
    # 是否使用多噪声构建稠密流场。如果为 False，则回归最严格的 1v1 On-Policy 映射
    use_dense_flow: jdc.Static[bool] = True
    num_epsilon_samples: jdc.Static[int] = 48

    # 相当于 PPO 的概率缩放步长 (eta)
    target_k_scaling: float = 0.1
    clipping_epsilon: float = 0.2

    # 方差边界，防止数值溢出或完全确定性
    min_log_std: float = -5.0
    max_log_std: float = 2.0

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


@jdc.pytree_dataclass
class IgmmParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class IgmmActionInfo:
    # 环境交互数据
    raw_action: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    eps_1: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))

    # 全局目标缓存容器 (用于内层加速)
    eps_batch: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    theta_2: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    gae_vs: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))


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
        assert isinstance(obs_size, int)

        theta_dim = action_size * 2

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(
            prng0,
            (obs_size + theta_dim + config.timestep_embed_dim, 32, 32, 32, 32, theta_dim),
        )
        critic_net = networks.mlp_init(prng1, (obs_size, 256, 256, 256, 256, 256, 1))

        network_params = IgmmParams(actor_net, critic_net)
        opt = optax.scale_by_adam()

        return IgmmState(
            env=env, config=config, params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt=opt, opt_state=opt.init(network_params),  # type: ignore
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

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(t_current=full_t_path[:-1], t_next=full_t_path[1:])

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, IgmmActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else obs
        (*batch_dims, obs_dim) = obs.shape
        theta_dim = self.env.action_size * 2

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

        # 环境交互：仅采样唯一的本源噪声 eps_1
        eps_1 = jax.random.normal(prng_sample, (*batch_dims, theta_dim))

        theta_1, _ = jax.lax.scan(
            euler_step, init=eps_1,
            xs=(self.get_schedule(), jnp.zeros((self.config.flow_steps, 1))),
        )

        mu_1, log_std_1 = jnp.split(theta_1, 2, axis=-1)
        log_std_1 = jnp.clip(log_std_1, self.config.min_log_std, self.config.max_log_std)
        sigma_1 = jnp.exp(log_std_1)

        raw_action = mu_1 if deterministic else mu_1 + sigma_1 * jax.random.normal(prng_action, mu_1.shape)
        action_clipped = self._apply_clip(raw_action)

        # 返回执行的动作，并妥善缓存本源 eps_1
        return action_clipped, IgmmActionInfo(raw_action=raw_action, eps_1=eps_1)

    @jdc.jit
    def training_step(self, transitions: IgmmTransition) -> tuple[IgmmState, dict[str, Array]]:
        config, state = self.config, self

        # --- 观测特征归一化 ---
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)

        obs_norm = (
                               transitions.obs - state.obs_stats.mean) / state.obs_stats.std if config.normalize_observations else transitions.obs
        value_pred = networks.value_mlp_fwd(state.params.value, obs_norm)
        bootstrap_obs_norm = (transitions.next_obs[-1:, :, :] - state.obs_stats.mean) / state.obs_stats.std
        bootstrap_value = networks.value_mlp_fwd(state.params.value, bootstrap_obs_norm)

        # ==========================================
        # 🚀 计算 GAE 优势
        # ==========================================
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
        # 👑 模式切换：构建流场的起点
        # ==========================================
        prng_eps = jax.random.fold_in(state.prng, state.steps)
        act_dim = state.env.action_size
        theta_dim = act_dim * 2

        if config.use_dense_flow:
            # 模式 B: 撒 M 个全新随机噪声，构建稠密流场
            M = config.num_epsilon_samples
            eps_batch = jax.random.normal(prng_eps, (*transitions.reward.shape, M, theta_dim))
        else:
            # 模式 A: 纯正 On-Policy，提取创造真实动作的唯一噪声
            M = 1
            eps_batch = transitions.action_info.eps_1[..., None, :]

        # 统一进行一次全局静态 ODE 积分，找到靶点起点 theta_1
        def forward_ode_step(carry, t_tup):
            x = carry
            t_curr, t_next = t_tup
            t_embed = state.embed_timestep(jnp.broadcast_to(t_curr, (*x.shape[:-1], 1)))
            obs_b = jnp.broadcast_to(obs_norm[..., None, :], (*x.shape[:-1], obs_norm.shape[-1]))
            # 停止梯度！寻找投影目标时绝不让梯度穿过 ODE
            v = networks.flow_mlp_fwd(jax.lax.stop_gradient(state.params.policy), obs_b, x,
                                      t_embed) * config.policy_mlp_output_scale
            return x + (t_next - t_curr) * v, None

        sch = state.get_schedule()
        theta_1, _ = jax.lax.scan(forward_ode_step, eps_batch, (sch.t_current, sch.t_next))
        theta_1 = jax.lax.stop_gradient(theta_1)

        mu_1, log_std_1 = jnp.split(theta_1, 2, axis=-1)

        # ==========================================
        # 👑 绝对闭式解：最优传输连线投影 (Wasserstein Geodesic)
        # ==========================================
        # 1. 计算对数概率增量 Delta L
        ratio_target = 1.0 + config.target_k_scaling * gae_advantages[..., None, None]
        ratio_target_clipped = jnp.clip(ratio_target, 1.0 - config.clipping_epsilon, 1.0 + config.clipping_epsilon)
        ratio_target_clipped = jnp.maximum(ratio_target_clipped, 1e-4)  # 防止 log(0) 爆炸
        delta_L = jnp.log(ratio_target_clipped)

        # 2. 完美的几何消元反解：连线步长 rho_inv = 1 / rho = exp(-Delta L)
        rho_inv = jnp.exp(-delta_L)

        # 唯一的真动作 a_true 扩展形状以匹配 M 维
        a_b = transitions.action[..., None, :]

        # 3. 极简闭式推导：参数沿着连线无冲突同步移动
        log_std_2 = jnp.clip(log_std_1 - delta_L, config.min_log_std, config.max_log_std)
        mu_2 = mu_1 + (1.0 - rho_inv) * (a_b - mu_1)

        # 4. 组装终极目标 theta_2
        theta_2 = jax.lax.stop_gradient(jnp.concatenate([mu_2, log_std_2], axis=-1))

        # ==========================================
        # 🚀 目标缓存注入
        # ==========================================
        new_action_info = jdc.replace(
            transitions.action_info, eps_batch=eps_batch, theta_2=theta_2, gae_vs=gae_vs
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
        # ==========================================
        # 🚀 极其轻盈的内层回归
        # ==========================================
        obs_norm = (
                               transitions.obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else transitions.obs
        value_pred = networks.value_mlp_fwd(self.params.value, obs_norm)

        # 提取目标缓存
        eps_batch = transitions.action_info.eps_batch
        theta_2 = transitions.action_info.theta_2
        gae_vs = transitions.action_info.gae_vs

        M = eps_batch.shape[-2]
        batch_shape = eps_batch.shape[:-2]

        if self.config.discretize_t_for_training:
            t_idx = jax.random.randint(prng, (*batch_shape, M, 1), 0, self.config.flow_steps)
            t = self.get_schedule().t_current[t_idx]
        else:
            t = jax.random.uniform(prng, (*batch_shape, M, 1))

        # 流场监督：起点按比例滑向终点
        x_t = t * eps_batch + (1.0 - t) * theta_2
        obs_b_fit = jnp.broadcast_to(obs_norm[..., None, :], (*x_t.shape[:-1], obs_norm.shape[-1]))

        # 单次前向通过
        vel_pred = networks.flow_mlp_fwd(
            self.params.policy, obs_b_fit, x_t, self.embed_timestep(t)
        ) * self.config.policy_mlp_output_scale

        if self.config.output_mode == "u":
            flow_loss = jnp.mean((vel_pred - (eps_batch - theta_2)) ** 2, axis=-1)
        else:
            x0_pred = x_t - t * vel_pred
            x1_pred = x0_pred + vel_pred
            flow_loss = jnp.mean((eps_batch - x1_pred) ** 2, axis=-1)

        policy_loss = jnp.mean(flow_loss)

        # Value Loss
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        v_loss = jnp.mean(v_error ** 2) * self.config.value_loss_coeff

        return policy_loss + v_loss, {"policy_loss": policy_loss, "v_loss": v_loss}