from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
import optax
from jax import Array
from jax import numpy as jnp

from flow_policy.networks import MlpWeights
from . import math_utils, networks, rollouts


@jdc.pytree_dataclass
class DGPOFMConfig:
    # --- 全新 Q-Guided 生成控制核心 ---
    resampling_alpha: float = 0.1
    num_generated_actions: jdc.Static[int] = 7  # 每个状态额外生成 7 个动作 (凑成 8 个的簇)

    # 控制损失权重
    w_v_loss: float = 1.0
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4

    # Flow parameters.
    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = "u_but_supervise_as_eps"
    timestep_embed_dim: jdc.Static[int] = 8
    n_samples_per_action: jdc.Static[int] = 8
    average_losses_before_exp: jdc.Static[bool] = False
    discretize_t_for_training: jdc.Static[bool] = True
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25

    loss_mode: jdc.Static[Literal["dgpo", "denoising_mdp"]] = "dgpo"
    final_steps_only: jdc.Static[bool] = False
    sde_sigma: float = 0.0
    clipping_epsilon: float = 0.05

    # PPO Base Config
    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    normalize_observations: jdc.Static[bool] = True
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 30
    num_minibatches: jdc.Static[int] = 32
    num_timesteps: jdc.Static[int] = 60_000_000
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
class DGPOFMParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class DGPOFMActionInfo:
    pass


@jdc.pytree_dataclass
class FlowSchedule:
    t_current: Array
    t_next: Array


DGPOFMTransition = rollouts.TransitionStruct[DGPOFMActionInfo]


@jdc.pytree_dataclass
class DGPOFMState:
    env: jdc.Static[mjp.MjxEnv]
    config: DGPOFMConfig
    params: DGPOFMParams
    obs_stats: math_utils.RunningStats

    opt_policy: jdc.Static[optax.GradientTransformation]
    opt_value: jdc.Static[optax.GradientTransformation]
    opt_state_policy: optax.OptState
    opt_state_value: optax.OptState

    prng: Array
    steps: Array

    @staticmethod
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: DGPOFMConfig) -> DGPOFMState:
        obs_size = env.observation_size
        action_size = env.action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(
            prng0,
            (obs_size + action_size + config.timestep_embed_dim, 32, 32, 32, 32, action_size),
        )

        # 👑 核心进化 1：Critic 升级为 Q(s, a) 网络
        critic_net = networks.mlp_init(
            prng1,
            (obs_size + action_size, 256, 256, 256, 256, 256, 1)
        )

        network_params = DGPOFMParams(actor_net, critic_net)
        opt_policy = optax.adam(config.learning_rate_p)
        opt_value = optax.adam(config.learning_rate_v)

        return DGPOFMState(
            env=env,
            config=config,
            params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt_policy=opt_policy,
            opt_value=opt_value,
            opt_state_policy=opt_policy.init(network_params.policy),
            opt_state_value=opt_value.init(network_params.value),
            prng=prng2,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def _step_minibatch(self, transitions: DGPOFMTransition, prng: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        prng_targets, prng_policy = jax.random.split(prng, 2)
        obs_norm = (
                               transitions.obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else transitions.obs

        # 1. 依然先获取打分和 Targets
        pool_actions, pool_weights, target_qs, metrics = self._compute_targets(transitions, obs_norm, prng_targets)

        # ==========================================
        # 👑 2. Q 网络“小灶”先行循环 (先训 Q)
        # ==========================================
        # 这里把 length 改为 4 或 8 (这就是 Q-Network 的 UTD ratio)
        # 意味着在一个 Minibatch 里，Q 网络会拿着同样的经验反复自我修正 4 次
        q_update_steps = 4

        def value_inner_step(carry, _):
            v_params, v_opt_state = carry

            def v_loss_fn(v_p):
                return self._compute_value_loss(
                    v_p, obs_norm, transitions.action, transitions.truncation, target_qs, pool_actions
                )

            v_loss_val, v_grads = jax.value_and_grad(v_loss_fn)(v_params)
            v_updates, next_v_opt_state = self.opt_value.update(v_grads, v_opt_state, v_params)
            next_v_params = optax.apply_updates(v_params, v_updates)
            return (next_v_params, next_v_opt_state), v_loss_val

        (new_value_params, new_opt_state_value), extra_v_losses = jax.lax.scan(
            value_inner_step, (self.params.value, self.opt_state_value), None, length=q_update_steps
        )
        metrics["v_loss"] = extra_v_losses[-1]  # 记录最后一次收敛的 loss

        # ==========================================
        # 👑 3. 速度场 (Policy) 滞后更新
        # ==========================================
        # 注意：如果你想用刚刚训好的、最新鲜的 new_value_params 重新打分，可以在这里再调一次 Q 网络。
        # 但通常为了计算效率，使用步骤 1 里的旧权重 (Delayed Update) 效果就足够好了，甚至更稳定。
        def policy_loss_fn(p_params):
            return self._compute_policy_loss(p_params, obs_norm, pool_actions, pool_weights, prng_policy)

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_opt_state_policy = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)
        new_policy_params = optax.apply_updates(self.params.policy, p_updates)
        metrics.update(p_metrics)

        # 保存更新
        new_params = DGPOFMParams(policy=new_policy_params, value=new_value_params)
        with jdc.copy_and_mutate(self) as state:
            state.params = new_params
            state.opt_state_policy = new_opt_state_policy
            state.opt_state_value = new_opt_state_value
            state.steps = state.steps + 1

        return state, metrics

    def _compute_targets(self, transitions: DGPOFMTransition, obs_norm: Array, prng: Array) -> tuple[
        Array, Array, Array, dict[str, Array]]:
        metrics = dict[str, Array]()
        T, B, obs_dim = obs_norm.shape
        act_dim = self.env.action_size
        N = T * B

        prng_boot, prng_gen, prng_eval = jax.random.split(prng, 3)

        # =========================================================
        # 1. 估算 SARSA(λ) Targets (完全真实的物理环境反馈)
        # =========================================================
        # 当前步骤的 Q(s, a)
        concat_inputs = jnp.concatenate([obs_norm, transitions.action], axis=-1)
        q_pred, h_s = networks.value_mlp_fwd_with_features(self.params.value, concat_inputs)

        # 🚨 检查点：请确保这里就是光秃秃的 q_pred，千万不要有 [..., 0] 或 .squeeze(-1)
        q_pred = jax.lax.stop_gradient(q_pred)

        # Bootstrap 步骤的 Q(s', a') -> 需要生成一个 a'
        bootstrap_obs = transitions.next_obs[-1:, :, :]
        if self.config.normalize_observations:
            bootstrap_obs = (bootstrap_obs - self.obs_stats.mean) / self.obs_stats.std

        # 快速为 next_obs 生成一个动作
        def boot_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            # 简化逻辑：先变成 (1, 1) 的输入，得到 (1, 8) 的输出
            t_embed_raw = self.embed_timestep(jnp.array([t_curr])[..., None])  # (1, 8)
            # 广播到 (1, B, embed_dim)
            t_embed = jnp.broadcast_to(t_embed_raw[:, None, :], (1, B, self.config.timestep_embed_dim))

            vel = networks.flow_mlp_fwd(self.params.policy, bootstrap_obs, x,
                                        t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        boot_noise = jax.random.normal(prng_boot, (1, B, act_dim))
        schedule = self.get_schedule()
        bootstrap_act, _ = jax.lax.scan(boot_step_fn, boot_noise, (schedule.t_current, schedule.t_next))

        bootstrap_concat = jnp.concatenate([bootstrap_obs, bootstrap_act], axis=-1)
        bootstrap_q, _ = networks.value_mlp_fwd_with_features(self.params.value, bootstrap_concat)

        # 🚨 检查点：这里也必须是光秃秃的
        bootstrap_q = jax.lax.stop_gradient(bootstrap_q)

        # 借用 GAE 的数学框架，计算 SARSA(λ) 返回值作为 Target Q
        gae_qs, _ = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * self.config.discounting,
                rewards=transitions.reward * self.config.reward_scaling,
                values=q_pred,
                bootstrap_value=bootstrap_q,
                gae_lambda=self.config.gae_lambda,
            )
        )

        # =========================================================
        # 2. 核心：自生成 K 个动作与“原配动作”组成候选池
        # =========================================================
        K = self.config.num_generated_actions
        flat_obs = obs_norm.reshape((N, obs_dim))
        flat_acts_real = transitions.action.reshape((N, 1, act_dim))

        # 并行批量生成 K 个动作
        x_t = jax.random.normal(prng_gen, (N, K, act_dim))
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K, obs_dim))

        def gen_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            # 同样简化：得到 (1, 8) 的基础嵌入
            t_embed_raw = self.embed_timestep(jnp.array([t_curr])[..., None])  # (1, 8)
            # 广播到 (N, K, embed_dim)
            t_embed = jnp.broadcast_to(t_embed_raw[:, None, :], (N, K, self.config.timestep_embed_dim))

            # 必须 stop_gradient
            p_params = jax.lax.stop_gradient(self.params.policy)
            vel = networks.flow_mlp_fwd(p_params, obs_b, x, t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        generated_acts, _ = jax.lax.scan(gen_step_fn, x_t, (schedule.t_current, schedule.t_next))

        # 拼成 K+1 大小的动作池
        pool_actions = jnp.concatenate([flat_acts_real, generated_acts], axis=1)  # (N, K+1, act_dim)

        # =========================================================
        # 3. 裁判打分：用 Q 网络评估这 K+1 个动作
        # =========================================================
        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K + 1, obs_dim))
        concat_pool = jnp.concatenate([obs_pool_b, pool_actions], axis=-1)

        # 原报错代码：q_pool = jax.lax.stop_gradient(q_pool[..., 0])
        # 👑 终极修复：
        q_pool, _ = networks.value_mlp_fwd_with_features(self.params.value, concat_pool)
        q_pool = jax.lax.stop_gradient(q_pool)  # 此时它天然就是完美的 (N, K+1)

        # =========================================================
        # 4. Softmax 加权 (Q-Guided Weighting)
        # =========================================================
        local_scale = jnp.max(jnp.abs(q_pool - jnp.mean(q_pool, axis=-1, keepdims=True)), axis=-1)
        dynamic_alpha = self.config.resampling_alpha * (local_scale + 1e-6)

        logits = (q_pool - jnp.max(q_pool, axis=-1, keepdims=True)) / dynamic_alpha[:, None]
        pool_probs = jax.nn.softmax(logits, axis=-1)  # (N, K+1)

        # --- 监控指标 ---
        metrics["q_guided/q_real_mean"] = jnp.mean(q_pool[:, 0])
        metrics["q_guided/q_generated_mean"] = jnp.mean(q_pool[:, 1:])
        metrics["q_guided/prob_real_mean"] = jnp.mean(pool_probs[:, 0])
        metrics["q_guided/dynamic_alpha_mean"] = jnp.mean(dynamic_alpha)

        return jax.lax.stop_gradient(pool_actions), jax.lax.stop_gradient(pool_probs), gae_qs, metrics

    # ==========================================
    # 5. AW-Flow: 局部优势加权速度场
    # ==========================================
    def _compute_policy_loss(self, policy_params, obs_norm, actions_pool, weights_pool, prng):
        N, K_plus_1, act_dim = actions_pool.shape
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))

        prng_eps, prng_t = jax.random.split(prng, 2)
        # 👑 你的神级修复：为每个候选动作分配【独立】的噪声和时间戳！
        # 这样网络就能把不同的动作映射到不同的高斯噪声球面上，保留双峰分布！
        eps = jax.random.normal(prng_eps, (N, K_plus_1, act_dim))
        t_idx = jax.random.randint(prng_t, (N, K_plus_1, 1), 0, self.config.flow_steps)
        t = self.get_schedule().t_current[t_idx]

        # ODE 轨迹起点 (完全独立的轨迹)
        x_t = t * eps + (1.0 - t) * actions_pool
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))

        t_embed = self.embed_timestep(t)  # (N, K+1, t_dim)

        vel_pred = networks.flow_mlp_fwd(
            policy_params, obs_b, x_t, t_embed
        ) * self.config.policy_mlp_output_scale

        if self.config.output_mode == "u_but_supervise_as_eps":
            x1_pred = (x_t - t * vel_pred) + vel_pred
            error_sq = jnp.sum((eps - x1_pred) ** 2, axis=-1)
        else:
            error_sq = jnp.sum((vel_pred - (eps - actions_pool)) ** 2, axis=-1)

        # 依然用 Softmax 权重加权，但现在它是在对分布进行重塑，而不是对动作进行求均值
        policy_loss = jnp.mean(jnp.sum(weights_pool * error_sq, axis=-1))

        return policy_loss, {"policy_loss": policy_loss}

    # ==========================================
    # 6. Q 网络训练 (拟合真实回报)
    # ==========================================
    # 修改 _compute_value_loss，我们需要传入生成的动作 pool_actions 来惩罚它们
    def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs, pool_actions):
        # 1. 计算真实动作的 Q 值，拟合 Target (保持不变)
        concat_inputs = jnp.concatenate([obs_norm, actions], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(value_params, concat_inputs)

        v_error = (target_qs - q_pred) * (1 - truncation)
        mse_loss = jnp.mean(v_error ** 2)

        # 👑 2. 加入 CQL 保守惩罚项 (打破死亡螺旋的核心)
        # 把生成的假动作丢进去算 Q 值
        # 👑 2. 加入 CQL 保守惩罚项 (打破死亡螺旋的核心)
        N, K_plus_1, act_dim = pool_actions.shape

        # 👇 关键修复：先把 (T, B, obs_dim) 展平为 (N, obs_dim)
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))

        # 然后再增加维度并广播
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, obs_norm.shape[-1]))
        concat_pool = jnp.concatenate([obs_b, pool_actions], axis=-1)

        q_pool_fake, _ = networks.value_mlp_fwd_with_features(value_params, concat_pool)

        # 惩罚项：如果 Q 网络给生成的动作打分太高，就产生巨大的 Loss 惩罚它！
        # 排除掉第0个(真实的动作)，只惩罚后面 7 个生成的动作
        cql_penalty = jnp.mean(q_pool_fake[:, 1:])

        # 综合 Loss：拟合真实回报 + 压制假动作的幻觉分数 (cql_weight 比如设为 1.0 或 5.0)
        cql_weight = 0.1
        total_v_loss = (mse_loss + cql_weight * cql_penalty) * self.config.value_loss_coeff * self.config.w_v_loss

        return total_v_loss

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(t_current=full_t_path[:-1], t_next=full_t_path[1:])

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.config.timestep_embed_dim)
        return out

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, DGPOFMActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else obs
        (*batch_dims, obs_dim) = obs.shape
        assert obs_dim == self.env.observation_size

        def euler_step(carry: Array, inputs: tuple[FlowSchedule, Array]) -> tuple[Array, Array]:
            x_t = carry
            schedule_t, noise = inputs
            dt = schedule_t.t_next - schedule_t.t_current
            velocity = networks.flow_mlp_fwd(
                self.params.policy, obs_norm, x_t,
                jnp.broadcast_to(self.embed_timestep(schedule_t.t_current[None]),
                                 (*batch_dims, self.config.timestep_embed_dim))
            ) * self.config.policy_mlp_output_scale
            return x_t + dt * velocity + self.config.sde_sigma * noise, x_t

        prng_sample, prng_loss, prng_feather, prng_noise = jax.random.split(prng, num=4)
        noise_path = jax.random.normal(prng_noise, (self.config.flow_steps, *batch_dims, self.env.action_size))
        x0, _ = jax.lax.scan(euler_step, jax.random.normal(prng_sample, (*batch_dims, self.env.action_size)),
                             (self.get_schedule(), noise_path))

        if not deterministic:
            x0 = x0 + jax.random.normal(prng_feather, (*batch_dims, self.env.action_size)) * self.config.feather_std
        return x0, DGPOFMActionInfo()

    @jdc.jit
    def training_step(self, transitions: DGPOFMTransition) -> tuple[DGPOFMState, dict[str, Array]]:
        config = self.config
        state = self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)
        del self

        def step_batch(state: DGPOFMState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                partial(DGPOFMState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)),
                init=state,
                xs=transitions.prepare_minibatches(step_prng, config.num_minibatches, config.batch_size),
            )
            return state, metrics

        state, metrics = jax.lax.scan(step_batch, init=state, length=config.num_updates_per_batch)
        return state, metrics