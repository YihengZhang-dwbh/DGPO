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
    resampling_alpha_k: float = 0.1
    resampling_alpha_min: float = 0.3
    use_dynamic_alpha: jdc.Static[bool] = False
    num_generated_actions_min: jdc.Static[int] = 1
    num_generated_actions_max: jdc.Static[int] = 8
    num_epsilon_samples: jdc.Static[int] = 8

    # 👑 新增：纯重采样开关 (True 为根据概率抛骰子硬采样，False 为加权)
    use_hard_resampling: jdc.Static[bool] = True

    # 控制损失权重
    w_v_loss: float = 1.0
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4
    loop_v: jdc.Static[int] = 1

    # 👑 CQL 与 Hinge 控制
    use_hinge_cql: jdc.Static[bool] = True
    # 扩充 Literal 类型，加入 exponential 和 inverse
    cql_decay_mode: jdc.Static[Literal["none", "linear", "cosine", "exponential", "inverse", "auto"]] = "auto"
    cql_init_weight: float = 0.1
    cql_final_weight: float = 0.0001
    cql_decay_ratio: float = 0.5

    # 👑 新增：Auto-CQL 超参
    # 👑 Auto-CQL / PI 控制器超参
    cql_target_margin: float = 10  # 容忍底线
    cql_alpha_lr: float = 3e-4  # 也就是 Ki (积分系数/基础学习率)，负责长期稳态
    cql_alpha_kp: float = 0.05  # 👑 新增：Kp (比例系数)，负责瞬间镇压坏动作的力度#
    # 👑 新增：是否开启权重上下限的安全锁
    cql_clip_alpha: jdc.Static[bool] = False

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
    num_timesteps: jdc.Static[int] = 180000000
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

    # 👑 新增：系统自己学习的动态权重 (存对数是为了保证权重永远为正)
    log_cql_weight: Array

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
            # 初始化对数权重
            log_cql_weight=jnp.log(jnp.array(config.cql_init_weight, dtype=jnp.float32)),
        )

    def _step_minibatch(self, transitions: DGPOFMTransition, prng: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        # 👑 计算当前进度下的动态 K
        # 假设在 decay_ratio (例如 50% 进度) 时达到最大 K
        total_steps = self.config.num_timesteps // (self.config.num_envs * self.config.unroll_length)
        progress = jnp.minimum(1.0, self.steps / (total_steps * self.config.cql_decay_ratio))

        # 线性增加 K
        current_K = (self.config.num_generated_actions_min +
                     progress * (self.config.num_generated_actions_max - self.config.num_generated_actions_min)).astype(
            jnp.int32)

        # 将 current_K 传给 _compute_targets
        pool_actions, pool_weights, target_qs, metrics = self._compute_targets(transitions, obs_norm, prng_targets,
                                                                               current_K)
        # ... 后续逻辑 ...
        # ==========================================
        # 👑 2. Q 网络小灶 & Auto-CQL 动态调温
        # ==========================================
        q_update_steps = self.config.loop_v

        # 循环不仅传递参数，还传递当前的 log_cql_weight
        def value_inner_step(carry, _):
            v_params, v_opt_state, current_log_alpha = carry

            # 还原出真正的权重 (永远为正)
            current_alpha = jax.lax.stop_gradient(jnp.exp(current_log_alpha))

            def v_loss_fn(v_p):
                # 传入 current_alpha
                total_loss, penalty = self._compute_value_loss(
                    v_p, obs_norm, transitions.action, transitions.truncation, target_qs, pool_actions, current_alpha
                )
                return total_loss, penalty

            # 拿到 Q 网络的梯度和当前的 penalty
            (v_loss_val, current_penalty), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(v_params)

            # 更新 Q 网络
            v_updates, next_v_opt_state = self.opt_value.update(v_grads, v_opt_state, v_params)
            next_v_params = optax.apply_updates(v_params, v_updates)

            # 👑 魔法发生：更新动态权重 (Lagrange Update)
            if self.config.cql_decay_mode == "auto":
                # 公式：如果 penalty 大于 target_margin，差值为正，log_alpha 变大 (加强惩罚)
                # 反之，log_alpha 变小 (松绑)
                alpha_grad = current_penalty - self.config.cql_target_margin
                next_log_alpha = current_log_alpha + self.config.cql_alpha_lr * alpha_grad

                # 限制上下限，防止爆炸或跌穿
                next_log_alpha = jnp.clip(
                    next_log_alpha,
                    a_min=jnp.log(self.config.cql_final_weight),
                    a_max=jnp.log(self.config.cql_init_weight)
                )
            else:
                # 如果没开 auto，就保持不动 (你可以把前面写的各种衰减 schedule 放这里，为了纯粹，我们这里只演示 auto)
                next_log_alpha = current_log_alpha

            aux_metrics = {
                "v_loss/total": v_loss_val,
                "v_loss/cql_penalty": current_penalty,
                "v_loss/current_cql_weight": jnp.exp(next_log_alpha)
            }
            return (next_v_params, next_v_opt_state, next_log_alpha), aux_metrics

        # 执行 scan 循环
        (new_value_params, new_opt_state_value, new_log_alpha), extra_v_metrics = jax.lax.scan(
            value_inner_step,
            (self.params.value, self.opt_state_value, self.log_cql_weight),
            None,
            length=q_update_steps
        )

        # jax.lax.scan 会把字典里的标量堆叠成数组，我们取最后一次循环的值 [-1]
        for k, v in extra_v_metrics.items():
            metrics[k] = v[-1]

        # 👑 提取出刚刚算完的、最新鲜的 V-Loss
        final_v_loss = extra_v_metrics["v_loss/total"][-1]

        # ==========================================
        # 👑 3. 带着 V-Loss 重新打分 (Fresh Weighting)
        # ==========================================
        # 使用最新的 Q 网络和 V-Loss，计算出带有自适应避震器的 pool_weights
        fresh_pool_weights, q_metrics = self._compute_fresh_weights(
            new_value_params, obs_norm, pool_actions, final_v_loss
        )
        # 把监控指标加进去 (这会覆盖掉 step 1 里旧的打分 metrics)
        metrics.update(q_metrics)

        # ==========================================
        # 👑 4. 速度场 (Policy) 滞后更新
        # ==========================================
        def policy_loss_fn(p_params):
            # 这里的 fresh_pool_weights 已经是防抖处理过的了！
            return self._compute_policy_loss(p_params, obs_norm, pool_actions, fresh_pool_weights, prng_policy)

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
            state.log_cql_weight = new_log_alpha  # 👑 更新动态变量
            state.steps = state.steps + 1

        return state, metrics

    def _compute_fresh_weights(self, value_params, obs_norm, pool_actions, final_v_loss) -> tuple[
        Array, dict[str, Array]]:
        """使用最新更新的 Q 网络和 V-Loss 来计算动作的 Softmax 权重"""
        N, K_plus_1, act_dim = pool_actions.shape
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))

        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))
        concat_pool = jnp.concatenate([obs_pool_b, pool_actions], axis=-1)

        # 用刚刚更新完的 Q 网络打分
        q_pool, _ = networks.value_mlp_fwd_with_features(value_params, concat_pool)
        q_pool = jax.lax.stop_gradient(q_pool)

        # ==========================================
        # 👑 Loss-Guided Temperature (基于 V-Loss 的避震器)
        # ==========================================
        if self.config.use_dynamic_alpha:
            # final_v_loss 是个标量 (比如 65 或者 892)
            # 因为 v_loss 主要是 MSE，我们开个根号 (RMSE) 把它拉回到和 Q 值同一线性维度
            rmse = jnp.abs(final_v_loss + 1e-8)

            # 用 RMSE 作为缩放因子。
            # 如果 RMSE=30 (裁判极度迷茫)，alpha = 0.1 * 31 = 3.1，拉平分布保命
            # 如果 RMSE=1 (裁判极其自信)，alpha = 0.1 * 2 = 0.2，锋利选择好动作
            alpha = jnp.maximum(self.config.resampling_alpha_k * rmse, self.config.resampling_alpha_min)
        else:
            alpha = self.config.resampling_alpha_min

        # Softmax 归一化
        logits = (q_pool - jnp.max(q_pool, axis=-1, keepdims=True)) / alpha
        pool_probs = jax.nn.softmax(logits, axis=-1)

        metrics = {
            "q_guided/q_real_mean": jnp.mean(q_pool[:, 0]),
            "q_guided/q_generated_mean": jnp.mean(q_pool[:, 1:]),
            "q_guided/prob_real_mean": jnp.mean(pool_probs[:, 0]),
            "q_guided/alpha_mean": jnp.mean(alpha)  # 👑 监控它随着 V-Loss 的起伏！
        }
        return jax.lax.stop_gradient(pool_probs), metrics

    def _compute_targets(self, transitions, obs_norm, prng, current_K):
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
        K = current_K
        x_t = jax.random.normal(prng_gen, (N, K, act_dim))
        # ... 生成 pool_actions (N, K+1, act_dim) ...

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
        if self.config.use_dynamic_alpha and False:
            # 原有的动态缩放逻辑 (根据极差自适应)
            local_scale = jnp.max(jnp.abs(q_pool - jnp.mean(q_pool, axis=-1, keepdims=True)), axis=-1)
            alpha = self.config.resampling_alpha_min * (local_scale + 1e-6)
            alpha = alpha[:, None]  # (N, 1) 以便广播
        else:
            # 👑 新的静态逻辑：直接使用全局固定的温度值
            alpha = self.config.resampling_alpha_min

        # 👑 全局大乱斗打分 (Softmax 覆盖所有 K+1 个动作)
        logits = (q_pool - jnp.max(q_pool, axis=-1, keepdims=True)) / alpha
        pool_probs = jax.nn.softmax(logits, axis=-1)  # (N, K+1)


        # --- 监控指标 ---
        metrics["q_guided/q_real_mean"] = jnp.mean(q_pool[:, 0])
        metrics["q_guided/q_generated_mean"] = jnp.mean(q_pool[:, 1:])
        metrics["q_guided/prob_real_mean"] = jnp.mean(pool_probs[:, 0])
        metrics["q_guided/alpha_mean"] = jnp.mean(alpha)  # 统一指标名称

        return jax.lax.stop_gradient(pool_actions), jax.lax.stop_gradient(pool_probs), gae_qs, metrics

    # ==========================================
    # 5. AW-Flow: 局部优势加权速度场
    # ==========================================
    def _compute_policy_loss(self, policy_params, obs_norm, actions_pool, weights_pool, prng):
        N, K_plus_1, act_dim = actions_pool.shape
        M = self.config.num_epsilon_samples # 固定为 8
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))

        if self.config.use_hard_resampling:
            prng_idx, prng_eps, prng_t = jax.random.split(prng, 3)

            # 👑 1. 大乱斗采样：为 M 个噪声分别从 K+1 个动作中选出赢家
            # weights_pool: (N, K+1) -> sampled_indices: (N, M)
            logits = jnp.log(weights_pool + 1e-8)
            sampled_indices = jax.random.categorical(prng_idx, logits[:, None, :], axis=-1)

            # 👑 2. 批量提取选中的动作 (N, M, act_dim)
            # 使用 take_along_axis 确保支持动态的 K+1 维度
            sampled_actions = jnp.take_along_axis(
                actions_pool[:, None, :, :],
                sampled_indices[:, :, None, None],
                axis=2
            ).squeeze(2)

            # 3. 带着这 M 个动作去配对 M 个噪声
            eps = jax.random.normal(prng_eps, (N, M, act_dim))
            t_idx = jax.random.randint(prng_t, (N, M, 1), 0, self.config.flow_steps)
            t = self.get_schedule().t_current[t_idx]

            # ... 后续 ODE 训练逻辑保持不变 ...

            # 3. 把选中的动作和观测，广播复制 M 份以对齐张量
            a_target = jnp.broadcast_to(sampled_actions[:, None, :], (N, M, act_dim))
            obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, M, flat_obs.shape[-1]))

            # 4. 构造 M 条截然不同的 ODE 轨迹
            x_t = t * eps + (1.0 - t) * a_target
            t_embed = self.embed_timestep(t)  # (N, M, t_dim)

            # 5. 前向传播计算速度场
            vel_pred = networks.flow_mlp_fwd(policy_params, obs_b, x_t, t_embed) * self.config.policy_mlp_output_scale

            if self.config.output_mode == "u_but_supervise_as_eps":
                x1_pred = (x_t - t * vel_pred) + vel_pred
                error_sq = jnp.sum((eps - x1_pred) ** 2, axis=-1)
            else:
                error_sq = jnp.sum((vel_pred - (eps - a_target)) ** 2, axis=-1)

            # 6. 对这 M 个不同时空点上的 Error 求平均，得到极低方差的平滑梯度
            policy_loss = jnp.mean(error_sq)

        else:
            # ==========================================
            # 路线 B：软加权 (Soft Weighting - 你之前的版本)
            # ==========================================
            prng_eps, prng_t = jax.random.split(prng, 2)
            eps = jax.random.normal(prng_eps, (N, K_plus_1, act_dim))
            t_idx = jax.random.randint(prng_t, (N, K_plus_1, 1), 0, self.config.flow_steps)
            t = self.get_schedule().t_current[t_idx]

            x_t = t * eps + (1.0 - t) * actions_pool
            obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))
            t_embed = self.embed_timestep(t)

            vel_pred = networks.flow_mlp_fwd(policy_params, obs_b, x_t, t_embed) * self.config.policy_mlp_output_scale

            if self.config.output_mode == "u_but_supervise_as_eps":
                x1_pred = (x_t - t * vel_pred) + vel_pred
                error_sq = jnp.sum((eps - x1_pred) ** 2, axis=-1)
            else:
                error_sq = jnp.sum((vel_pred - (eps - actions_pool)) ** 2, axis=-1)

            # 加权求和
            policy_loss = jnp.mean(jnp.sum(weights_pool * error_sq, axis=-1))

        return policy_loss, {"policy_loss": policy_loss}

    # ==========================================
    # 6. Q 网络训练 (拟合真实回报)
    # ==========================================
    def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs, pool_actions,
                            current_cql_weight):
        # 1. 计算 MSE Loss
        concat_inputs = jnp.concatenate([obs_norm, actions], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(value_params, concat_inputs)
        v_error = (target_qs - q_pred) * (1 - truncation)
        mse_loss = jnp.mean(v_error ** 2)

        # 2. 计算假动作的 Q 值
        N, K_plus_1, act_dim = pool_actions.shape
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, obs_norm.shape[-1]))
        concat_pool = jnp.concatenate([obs_b, pool_actions], axis=-1)
        q_pool_fake, _ = networks.value_mlp_fwd_with_features(value_params, concat_pool)

        # 3. 计算 Penalty
        # 👑 核心防爆锁：切断惩罚项对真实动作 Q 值的反向传播！
        q_real_sg = jax.lax.stop_gradient(q_pool_fake[:, 0:1])
        q_fake = q_pool_fake[:, 1:]

        if self.config.use_hinge_cql:
            # Hinge 惩罚：只把高于 real_q 的 fake_q 往下压，绝不把 real_q 往上拔！
            cql_penalty = jnp.mean(jax.nn.relu(q_fake - q_real_sg))
        else:
            cql_penalty = jnp.mean(q_fake - q_real_sg)

        # 4. 综合 Loss (接收外面传进来的 current_cql_weight)
        total_v_loss = (
                                   mse_loss + current_cql_weight * cql_penalty) * self.config.value_loss_coeff * self.config.w_v_loss

        return total_v_loss, cql_penalty

    # # 修改 _compute_value_loss，我们需要传入生成的动作 pool_actions 来惩罚它们
    # def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs, pool_actions):
    #     # 1. 计算真实动作的 Q 值，拟合 Target
    #     concat_inputs = jnp.concatenate([obs_norm, actions], axis=-1)
    #     q_pred, _ = networks.value_mlp_fwd_with_features(value_params, concat_inputs)
    #
    #     v_error = (target_qs - q_pred) * (1 - truncation)
    #     mse_loss = jnp.mean(v_error ** 2)
    #
    #     # 2. 计算假动作的 Q 值
    #     N, K_plus_1, act_dim = pool_actions.shape
    #     flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))
    #     obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, obs_norm.shape[-1]))
    #     concat_pool = jnp.concatenate([obs_b, pool_actions], axis=-1)
    #
    #     q_pool_fake, _ = networks.value_mlp_fwd_with_features(value_params, concat_pool)
    #
    #     # 3. 静态分支：决定是否使用 Hinge
    #     if self.config.use_hinge_cql:
    #         cql_penalty = jnp.mean(jax.nn.relu(q_pool_fake[:, 1:] - q_pool_fake[:, 0:1]))
    #     else:
    #         cql_penalty = jnp.mean(q_pool_fake[:, 1:])
    #
    #     # 👑 4. 静态分支：决定调度器模式 (五大模式火力全开)
    #     init_w = self.config.cql_init_weight
    #     final_w = self.config.cql_final_weight
    #
    #     # 算出进度 (0.0 到 1.0)
    #     steps_per_iter = self.config.iterations_per_env * self.config.num_envs
    #     total_iterations = self.config.num_timesteps // steps_per_iter
    #     total_updates = total_iterations * self.config.num_minibatches * self.config.num_updates_per_batch
    #     decay_updates = total_updates * self.config.cql_decay_ratio
    #
    #     progress = jnp.minimum(1.0, self.steps / decay_updates)
    #
    #     if self.config.cql_decay_mode == "none":
    #         current_cql_weight = init_w
    #
    #     elif self.config.cql_decay_mode == "linear":
    #         current_cql_weight = init_w - progress * (init_w - final_w)
    #
    #     elif self.config.cql_decay_mode == "cosine":
    #         cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    #         current_cql_weight = final_w + (init_w - final_w) * cosine_decay
    #
    #
    #     elif self.config.cql_decay_mode == "exponential":
    #
    #         # 👑 换用最稳定、最标准的自然指数衰减写法: W_init * e^(-k * progress)
    #         # 计算衰减系数 k (由于 final_w < init_w，这里的 decay_rate 必然是正数)
    #         decay_rate = -jnp.log(jnp.maximum(final_w, 1e-8) / jnp.maximum(init_w, 1e-8))
    #         # 使用 jnp.exp 确保严格的负指数递减
    #
    #         current_cql_weight = init_w * jnp.exp(-decay_rate * progress)
    #
    #     elif self.config.cql_decay_mode == "inverse":
    #         # 反比例衰减: W_init / (1 + c * progress)
    #         # 通过代数推导，当 progress=1 时到达 final_w 的常数 c 为:
    #         c = (init_w / jnp.maximum(final_w, 1e-8)) - 1.0
    #         current_cql_weight = init_w / (1.0 + c * progress)
    #
    #     else:
    #         raise ValueError(f"未知的衰减模式: {self.config.cql_decay_mode}")
    #
    #     # 终极安全锁：由于浮点精度问题，以防万一越界，钳制在上下限之间
    #     current_cql_weight = jnp.clip(current_cql_weight, final_w, init_w)
    #
    #     # 5. 综合 Loss (后续不变)
    #     total_v_loss = (
    #                                mse_loss + current_cql_weight * cql_penalty) * self.config.value_loss_coeff * self.config.w_v_loss
    #
    #     # 👑 把当前的 weight 和 penalty 传出去，方便监控
    #     aux_metrics = {
    #         "v_loss/total": total_v_loss,
    #         "v_loss/mse": mse_loss,
    #         "v_loss/cql_penalty": cql_penalty,
    #         "v_loss/current_cql_weight": current_cql_weight
    #     }
    #
    #     return total_v_loss, aux_metrics

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