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
    # 👑 IGMM 独家范式超参 (Infinite GMM & Target Reprojection)
    # ==========================================
    # 每次更新时，用来探路、构建稠密流场的噪声条数
    num_epsilon_samples: jdc.Static[int] = 48
    # 相当于 PPO 的概率缩放步长 (eta)，决定了优势转化为位移的剧烈程度
    target_k_scaling: float = 0.1
    clipping_epsilon: float = 0.2

    # 方差边界，防止网络学成完全的确定性或者崩盘
    min_log_std: float = -5.0
    max_log_std: float = 2.0

    action_clip: jdc.Static[Literal["hard", "margin", "tanh", "fold", "scale_clip"]] = "margin"
    clip_margin: float = 10

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
    # 一切投影推导都在 loss 函数里基于 a_true On-the-fly 完成，
    # 不再需要缓存庞大的噪声池，内存开销直降到底。
    raw_action: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))


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

        # 👑 策略网络输出升维：预测 mu 和 log_sigma
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

        # 1. 采样单一基准噪声
        eps_1 = jax.random.normal(prng_sample, (*batch_dims, theta_dim))

        # 2. ODE 求解输出参数
        theta_1, _ = jax.lax.scan(
            euler_step, init=eps_1,
            xs=(self.get_schedule(), jnp.zeros((self.config.flow_steps, 1))),
        )

        mu_1, log_std_1 = jnp.split(theta_1, 2, axis=-1)
        log_std_1 = jnp.clip(log_std_1, self.config.min_log_std, self.config.max_log_std)
        sigma_1 = jnp.exp(log_std_1)

        # 3. 基于策略内蕴方差的真实探索
        raw_action = mu_1 if deterministic else mu_1 + sigma_1 * jax.random.normal(prng_action, mu_1.shape)
        action_clipped = self._apply_clip(raw_action)

        return action_clipped, IgmmActionInfo(raw_action=raw_action)

    @jdc.jit
    def training_step(self, transitions: IgmmTransition) -> tuple[IgmmState, dict[str, Array]]:
        config, state = self.config, self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)

        def step_batch(carry_state: IgmmState, _):
            step_prng = jax.random.fold_in(carry_state.prng, carry_state.steps)
            new_state, metrics = jax.lax.scan(
                partial(IgmmState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)),
                init=carry_state,
                xs=transitions.prepare_minibatches(step_prng, config.num_minibatches, config.batch_size),
            )
            return new_state, metrics

        state, metrics = jax.lax.scan(step_batch, init=state, length=config.num_updates_per_batch)
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
        prng_eps, prng_mask, prng_t = jax.random.split(prng, 3)
        (timesteps, batch_dim) = transitions.reward.shape
        act_dim = self.env.action_size
        theta_dim = act_dim * 2
        M = self.config.num_epsilon_samples

        obs_norm = (
                               transitions.obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else transitions.obs
        value_pred = networks.value_mlp_fwd(self.params.value, obs_norm)

        bootstrap_obs_norm = (transitions.next_obs[-1:, :, :] - self.obs_stats.mean) / self.obs_stats.std
        bootstrap_value = networks.value_mlp_fwd(self.params.value, bootstrap_obs_norm)

        gae_vs, gae_advantages = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * self.config.discounting,
                rewards=transitions.reward * self.config.reward_scaling,
                values=value_pred,
                bootstrap_value=bootstrap_value,
                gae_lambda=self.config.gae_lambda,
            )
        )

        metrics = {"advantages_mean": jnp.mean(gae_advantages)}
        if self.config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

        # A: (T, B), a_true: (T, B, act_dim)
        A = gae_advantages
        a_true = transitions.action

        # ==========================================
        # 👑 1. 批量映射 (The Flow to The Param Space)
        # ==========================================
        eps_batch = jax.random.normal(prng_eps, (timesteps, batch_dim, M, theta_dim))

        def forward_ode_step(carry, t_tup):
            x = carry
            t_curr, t_next = t_tup
            t_embed = self.embed_timestep(jnp.broadcast_to(t_curr, (*x.shape[:-1], 1)))
            obs_b = jnp.broadcast_to(obs_norm[..., None, :], (*x.shape[:-1], obs_norm.shape[-1]))
            # 停止梯度，绝不让梯度穿过 ODE
            v = networks.flow_mlp_fwd(jax.lax.stop_gradient(self.params.policy), obs_b, x,
                                      t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * v, None

        sch = self.get_schedule()
        theta_1, _ = jax.lax.scan(forward_ode_step, eps_batch, (sch.t_current, sch.t_next))
        theta_1 = jax.lax.stop_gradient(theta_1)

        mu_1, log_std_1 = jnp.split(theta_1, 2, axis=-1)
        sigma_1 = jnp.exp(log_std_1)

        # ==========================================
        # 👑 2. 万剑归宗的目标投影 (Target Reprojection)
        # ==========================================
        eta = self.config.target_k_scaling
        clip_eps = self.config.clipping_epsilon

        ratio_target = 1.0 + eta * A[..., None, None]
        ratio_target_clipped = jnp.where(
            A[..., None, None] > 0,
            jnp.minimum(ratio_target, 1.0 + clip_eps),
            jnp.maximum(ratio_target, 1.0 - clip_eps)
        )
        ratio_target_clipped = jnp.maximum(ratio_target_clipped, 1e-4)
        delta_L = jnp.log(ratio_target_clipped)

        # 为广播准备形状: (T, B, 1, act_dim)
        a_b = a_true[..., None, :]

        # --- 路线 A: 移动均值 mu (极限天花板闭式解) ---
        L_max_mu = -0.5 * jnp.log(2 * jnp.pi) - log_std_1
        L_1 = L_max_mu - ((a_b - mu_1) ** 2) / (2 * sigma_1 ** 2 + 1e-8)
        L_star = jnp.minimum(L_1 + delta_L, L_max_mu)

        sign_dir = jnp.where(a_b >= mu_1, 1.0, -1.0)
        mu_2 = a_b - sign_dir * jnp.sqrt(jnp.maximum(0.0, 2 * sigma_1 ** 2 * (L_max_mu - L_star)))

        # --- 路线 B: 收缩/放大方差 sigma (对数方差一阶安全游走) ---
        grad_y = ((a_b - mu_1) ** 2) / (sigma_1 ** 2 + 1e-8) - 1.0
        grad_y_clipped = jnp.clip(grad_y, -10.0, 10.0)

        log_std_2_raw = log_std_1 + delta_L * grad_y_clipped * 0.5
        log_std_2 = jnp.clip(log_std_2_raw, self.config.min_log_std, self.config.max_log_std)

        # ==========================================
        # 👑 3. 完美轮换交替法则 (Alternating Optimization)
        # ==========================================
        is_mu_turn = jax.random.bernoulli(prng_mask, 0.5, shape=(timesteps, batch_dim, M, 1))

        mu_target = jnp.where(is_mu_turn, mu_2, mu_1)
        log_std_target = jnp.where(is_mu_turn, log_std_1, log_std_2)
        theta_2 = jax.lax.stop_gradient(jnp.concatenate([mu_target, log_std_target], axis=-1))

        # ==========================================
        # 👑 4. 纯净流形回归 (Dense Flow Regression)
        # ==========================================
        if self.config.discretize_t_for_training:
            t_idx = jax.random.randint(prng_t, (timesteps, batch_dim, M, 1), 0, self.config.flow_steps)
            t = sch.t_current[t_idx]
        else:
            t = jax.random.uniform(prng_t, (timesteps, batch_dim, M, 1))

        x_t = t * eps_batch + (1.0 - t) * theta_2
        obs_b_fit = jnp.broadcast_to(obs_norm[..., None, :], (*x_t.shape[:-1], obs_norm.shape[-1]))

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

        # --- Value Loss ---
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        v_loss = jnp.mean(v_error ** 2) * self.config.value_loss_coeff
        total_loss = policy_loss + v_loss

        # --- Metrics ---
        metrics["policy_loss"] = policy_loss
        metrics["v_loss"] = v_loss
        metrics["target_reproj_mu_diff"] = jnp.mean(jnp.abs(mu_target - mu_1))
        metrics["target_reproj_sigma_diff"] = jnp.mean(jnp.abs(jnp.exp(log_std_target) - sigma_1))

        return total_loss, metrics