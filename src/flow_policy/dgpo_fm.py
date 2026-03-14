from __future__ import annotations

from functools import partial
from typing import Literal, assert_never

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
    # --- 重采样控制核心 ---
    resampling_alpha: float = 0.1
    # [新增]: 添加了 "cluster" 模式
    resampling_mode: jdc.Static[Literal["knn", "radius", "both", "cluster"]] = "knn"
    fixed_radius: float = 0.5
    resampling_topk: jdc.Static[int] = 32
    # [新增]: 聚类中心数量
    num_clusters: jdc.Static[int] = 64
    # [新增]: 语义特征融合权重 (0.0 表示纯物理状态聚类)
    semantic_weight: float = 0.0
    L2_combined_regularized: jdc.Static[bool] = True

    # [新增]: 岛屿拓扑模式 (forward: 价值层造岛+物理层压缩; reverse: 物理层造岛+语义层压缩)
    island_mode: jdc.Static[Literal["forward", "reverse"]] = "forward"
    num_value_buckets: jdc.Static[int] = 8

    # 控制损失权重
    w_v_loss: float = 1.
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4

    # 关键修改：JAX 内部的分支逻辑通常要求 M 的大小或采样开关在编译期确定
    use_subsampling: jdc.Static[bool] = True
    subsampling_m: jdc.Static[int] = 512
    # --------------------

    # Flow parameters.
    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = (
        "u_but_supervise_as_eps"
    )
    timestep_embed_dim: jdc.Static[int] = 8
    """"Must be divisible by 2."""
    n_samples_per_action: jdc.Static[int] = 8
    average_losses_before_exp: jdc.Static[bool] = False

    discretize_t_for_training: jdc.Static[bool] = True
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25

    loss_mode: jdc.Static[Literal["dgpo", "denoising_mdp"]] = "dgpo"
    final_steps_only: jdc.Static[bool] = False

    # Fixed noise level for sampling via denoising MDP. This is used for
    # DDPO-style policy updates.
    sde_sigma: float = 0.0

    clipping_epsilon: float = 0.05

    # Based on Brax PPO config:
    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    # learning_rate: float = 3e-4
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
        """Number of iterations (=policy forward passes) per environment at the
        start of each training step."""
        return (
                self.num_minibatches * self.batch_size * self.unroll_length
        ) // self.num_envs


@jdc.pytree_dataclass
class DGPOFMParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class DGPOFMActionInfo:
    pass  # 零负担，不需要存任何多余的 Loss 和噪声


@jdc.pytree_dataclass
class FlowSchedule:
    t_current: Array  # (*, flow_steps) - timesteps at the start of each step
    t_next: Array  # (*, flow_steps) - timesteps at the end of each step


DGPOFMTransition = rollouts.TransitionStruct[DGPOFMActionInfo]


@jdc.pytree_dataclass
class DGPOFMState:
    """PPO agent state."""

    env: jdc.Static[mjp.MjxEnv]
    config: DGPOFMConfig
    params: DGPOFMParams
    obs_stats: math_utils.RunningStats

    # --- 核心修改 1：拆分优化器和状态 ---
    opt_policy: jdc.Static[optax.GradientTransformation]
    opt_value: jdc.Static[optax.GradientTransformation]

    opt_state_policy: optax.OptState
    opt_state_value: optax.OptState
    # --------------------------------

    prng: Array
    steps: Array

    @staticmethod
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: DGPOFMConfig) -> DGPOFMState:
        obs_size = env.observation_size
        action_size = env.action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(
            # Policy takes both observation and action as input. We'll just concatenate them!
            prng0,
            (
                obs_size + action_size + config.timestep_embed_dim,
                32,
                32,
                32,
                32,
                action_size,
            ),
        )
        critic_net = networks.mlp_init(prng1, (obs_size, 256, 256, 256, 256, 256, 1))

        network_params = DGPOFMParams(actor_net, critic_net)

        # --- 核心修改 2：独立初始化两个 Adam 优化器 ---
        # Policy 保持基础学习率
        opt_policy = optax.adam(config.learning_rate_p)
        # Value 采用大步长 (比如 2.5 倍)
        opt_value = optax.adam(config.learning_rate_v)

        return DGPOFMState(
            env=env,
            config=config,
            params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            # 分别存入 State
            opt_policy=opt_policy,
            opt_value=opt_value,
            opt_state_policy=opt_policy.init(network_params.policy),
            opt_state_value=opt_value.init(network_params.value),
            prng=prng2,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    # ==========================================
    # 1. 核心调度：_step_minibatch
    # ==========================================
    def _step_minibatch(
            self, transitions: DGPOFMTransition, prng: Array
    ) -> tuple[DGPOFMState, dict[str, Array]]:

        assert transitions.reward.shape == (self.config.unroll_length, self.config.batch_size)
        prng_targets, prng_policy = jax.random.split(prng, 2)

        # 1. 预计算归一化观测值
        obs_norm = (
                               transitions.obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else transitions.obs

        # 1. 预计算归一化观测值
        obs_norm = (
                               transitions.obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else transitions.obs

        # 2. 获取权重 (把接收变量从 a_hat 改为 weights)
        weights, target_vs, metrics = self._compute_targets(transitions, obs_norm, prng_targets)

        # 提取当前 Batch 的原始动作 (N, action_size)
        N = self.config.unroll_length * self.config.batch_size
        flat_actions = transitions.action.reshape((N, self.env.action_size))

        # 3. Policy 训练 1 次 (传入真实的 flat_actions 和 weights)
        def policy_loss_fn(p_params):
            return self._compute_policy_loss(p_params, obs_norm, flat_actions, weights, prng_policy)

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_opt_state_policy = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)
        new_policy_params = optax.apply_updates(self.params.policy, p_updates)
        metrics.update(p_metrics)

        # 4. Value 训练 N 次 (使用已经拆分出的 _compute_value_loss)
        def value_inner_step(carry, _):
            v_params, v_opt_state = carry

            def v_loss_fn(v_p):
                # 传入 obs_norm, truncation 和 target_vs
                return self._compute_value_loss(v_p, obs_norm, transitions.truncation, target_vs)

            v_loss_val, v_grads = jax.value_and_grad(v_loss_fn)(v_params)
            v_updates, next_v_opt_state = self.opt_value.update(v_grads, v_opt_state, v_params)
            next_v_params = optax.apply_updates(v_params, v_updates)

            return (next_v_params, next_v_opt_state), v_loss_val

        # 执行小灶循环
        (new_value_params, new_opt_state_value), extra_v_losses = jax.lax.scan(
            value_inner_step,
            (self.params.value, self.opt_state_value),
            None,
            length=1  # 根据你的观察，4步通常能显著压低 v_loss
        )
        metrics["v_loss"] = extra_v_losses[-1]

        # 5. 组装更新
        new_params = DGPOFMParams(policy=new_policy_params, value=new_value_params)
        with jdc.copy_and_mutate(self) as state:
            state.params = new_params
            state.opt_state_policy = new_opt_state_policy
            state.opt_state_value = new_opt_state_value
            state.steps = state.steps + 1

        return state, metrics

    # ==========================================
    # 2. 目标生成：_compute_targets (不参与求导)
    # ==========================================
    def _compute_targets(self, transitions: DGPOFMTransition, obs_norm: Array, prng: Array) -> tuple[Array, Array, dict[str, Array]]:
        metrics = dict[str, Array]()
        (timesteps, batch_dim) = transitions.reward.shape
        N = timesteps * batch_dim

        # === 提取并阻断梯度，防止聚类操作破坏 Critic 的主线学习 ===
        value_pred, h_s = networks.value_mlp_fwd_with_features(self.params.value, obs_norm)
        h_s = jax.lax.stop_gradient(h_s)

        if self.config.normalize_observations:
            bootstrap_obs_norm = (transitions.next_obs[-1:, :, :] - self.obs_stats.mean) / self.obs_stats.std
        else:
            bootstrap_obs_norm = transitions.next_obs[-1:, :, :]

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

        metrics["advantages_mean"] = jnp.mean(gae_advantages)
        metrics["advantages_std"] = jnp.std(gae_advantages)
        metrics["advantages_max"] = jnp.max(gae_advantages)
        metrics["advantages_min"] = jnp.min(gae_advantages)

        if self.config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

        # --- 核心：重采样逻辑必须放在 if 外面 ---
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        flat_actions = transitions.action.reshape((N, self.env.action_size))
        flat_adv = gae_advantages.reshape((N,))
        prng_resample = prng

        # === 修复后的融合逻辑：保留幅度信息 ===
        flat_hs = h_s.reshape((N, -1))
        # 1. 先拼接出原始融合向量 (保留原始幅度)
        raw_combined = jnp.concatenate([flat_obs, flat_hs], axis=-1)

        # 2. 【核心】：执行跨样本的同维度归一化
        # 计算当前 Batch 的均值和标准差
        mean_feat = jnp.mean(raw_combined, axis=0, keepdims=True)
        std_feat = jnp.std(raw_combined, axis=0, keepdims=True) + 1e-8

        # 标准化：(x - mu) / sigma
        # 现在每一个维度的期望值都是 0，方差都是 1
        standardized_features = (raw_combined - mean_feat) / std_feat

        # 3. 最后再应用 semantic_weight 进行“块级别”的权重分配
        # 此时权重分配是极其精准的，因为两边的基准面（方差）已经完全平齐了
        obs_dim = flat_obs.shape[-1]
        combined_features = jnp.concatenate([
            standardized_features[:, :obs_dim] * ((1.0 - self.config.semantic_weight) ** 0.5),
            standardized_features[:, obs_dim:] * (self.config.semantic_weight ** 0.5)
        ], axis=-1)

        if self.config.resampling_mode == "cluster":
            # [K-Means 聚类逻辑...]
            C = self.config.num_clusters
            prng_cluster, prng_resample = jax.random.split(prng_resample)
            initial_indices = jax.random.choice(prng_cluster, N, shape=(C,), replace=False)
            centers = jax.lax.stop_gradient(combined_features[initial_indices])

            # 提前计算特征的平方和，避免在循环内重复计算
            sq_norms_features = jnp.sum(combined_features ** 2, axis=-1)

            labels = jnp.zeros((N,), dtype=jnp.int32)
            for _ in range(3):
                # --- 高速距离计算 (利用矩阵乘法代替广播减法) ---
                sq_norms_centers = jnp.sum(centers ** 2, axis=-1)
                dist_to_centers = jnp.maximum(
                    0.0,
                    sq_norms_features[:, None] + sq_norms_centers[None, :] - 2 * jnp.matmul(combined_features,
                                                                                            centers.T)
                )

                labels = jnp.argmin(dist_to_centers, axis=-1)
                one_hot = jax.nn.one_hot(labels, C)
                centers = jnp.matmul(one_hot.T, combined_features) / (jnp.sum(one_hot, axis=0)[:, None] + 1e-8)
                centers = jax.lax.stop_gradient(centers)

            # --- 1. 离群点检测 (同样使用高速算法) ---
            sq_norms_centers = jnp.sum(centers ** 2, axis=-1)
            dist_to_centers = jnp.maximum(
                0.0,
                sq_norms_features[:, None] + sq_norms_centers[None, :] - 2 * jnp.matmul(combined_features, centers.T)
            )
            min_dists_sq = jnp.min(dist_to_centers, axis=-1)

            # 由于加入了 h_s，距离绝对值会变大，这里的阈值可能需要按比例放大 （不用了，已L2范数约束为1）
            is_outlier = min_dists_sq > (self.config.fixed_radius ** 2)

            one_hot_labels = jax.nn.one_hot(labels, C)
            valid_one_hot = one_hot_labels * (~is_outlier[:, None])

            # =========================================================
            # 4. 簇内 Softmax 概率加权 (Local Advantage-Weighted)
            # =========================================================
            masked_adv_cluster = jnp.where(valid_one_hot, flat_adv[:, None], -jnp.inf)
            local_adv_pool = jnp.where(valid_one_hot, flat_adv[:, None], 0.0)

            # 计算簇内动态温度
            local_scale_c = jax.lax.stop_gradient(jnp.max(jnp.abs(local_adv_pool), axis=0))
            dynamic_alpha_c = self.config.resampling_alpha * (local_scale_c + 1e-6)

            # 减去最大值防止指数爆炸
            max_adv_c = jnp.max(masked_adv_cluster, axis=0, keepdims=True)
            logits_c = jnp.where(valid_one_hot, (masked_adv_cluster - max_adv_c) / dynamic_alpha_c[None, :], -1e9)

            # --- 核心：手动安全 Softmax 计算簇内概率 ---
            # (用 -1e9 代替 -inf 防止空簇导致的 NaN 错误)
            exp_logits = jnp.exp(logits_c)
            sum_exp = jnp.sum(exp_logits, axis=0, keepdims=True) + 1e-8
            probs_c = exp_logits / sum_exp  # [N, C] 矩阵，每列和为 1

            # # --- 缩放对齐：概率乘以簇大小，保证全局梯度尺度不塌缩 ---
            cluster_sizes = jnp.sum(valid_one_hot, axis=0)
            # scaled_weights_c = probs_c * cluster_sizes[None, :]
            #
            # # 提取每个样本在所属簇中的软权重
            # sample_weights = jnp.sum(scaled_weights_c * valid_one_hot, axis=-1)
            #
            # # 对于离群点，赋予中立权重 1.0 (等效于普通的行为克隆)
            # final_weights = jnp.where(is_outlier, 1.0, sample_weights)

            # 1. 直接取概率，不乘 cluster_sizes
            # 此时每个簇对 Loss 的总贡献都是 1.0，不再看人头
            sample_weights = jnp.sum(probs_c * valid_one_hot, axis=-1)

            # 2. 必须做全局归一化，保证整个 Batch 的平均权重为 1.0
            # 否则 Loss 会变小 N/C 倍（比如小 500 倍），导致训练根本不动
            final_weights = sample_weights / (jnp.mean(sample_weights) + 1e-8)

            # 3. 离群点处理（保持不变）
            final_weights = jnp.where(is_outlier, 1.0, final_weights)

            # --- 监控指标 ---
            metrics["dgpo/avg_neighbor_count"] = jnp.mean(cluster_sizes)
            metrics["dgpo/dynamic_alpha_mean"] = jnp.mean(dynamic_alpha_c)
            metrics["dgpo/outlier_ratio"] = jnp.mean(is_outlier)
            metrics["dgpo/est_clusters"] = jnp.array(C, dtype=jnp.float32)
            metrics["dgpo/weight_max"] = jnp.max(final_weights)  # 监控一下最大权重倍率
            # 注意这里不再返回 a_hat，而是返回我们算好的连续权重！
            return jax.lax.stop_gradient(final_weights), gae_vs, metrics
        else:
            raise ValueError("Mode must be cluster")


    # ==========================================
    # 3. 损失函数定义 (纯净计算)
    # ==========================================
    # 参数列表增加 actions 和 weights
    def _compute_policy_loss(self, policy_params, obs_norm, actions, weights, prng):
        (timesteps, batch_dim, obs_dim) = obs_norm.shape
        N = timesteps * batch_dim
        flat_obs = obs_norm.reshape((N, obs_dim))

        prng_eps, prng_t = jax.random.split(prng, 2)
        eps = jax.random.normal(prng_eps, (N, self.env.action_size))
        t_idx = jax.random.randint(prng_t, (N, 1), 0, self.config.flow_steps)
        t = self.get_schedule().t_current[t_idx]

        # ODE 轨迹起点是真实的 actions
        x_t = t * eps + (1.0 - t) * actions
        velocity_pred = networks.flow_mlp_fwd(
            policy_params, flat_obs, x_t, self.embed_timestep(t)
        ) * self.config.policy_mlp_output_scale

        # 分别计算每个样本的 MSE (注意使用 axis=-1 求和保留 N 维度)
        if self.config.output_mode == "u_but_supervise_as_eps":
            x1_pred = (x_t - t * velocity_pred) + velocity_pred
            error_sq = jnp.sum((eps - x1_pred) ** 2, axis=-1)
        else:
            error_sq = jnp.sum((velocity_pred - (eps - actions)) ** 2, axis=-1)

        # 👑 终极奥义：局部优势加权回归！
        policy_loss = jnp.mean(weights * error_sq)

        return policy_loss, {"policy_loss": policy_loss}


    def _compute_value_loss(self, value_params, obs_norm, truncation, target_vs):
        v_pred, _ = networks.value_mlp_fwd_with_features(value_params, obs_norm)
        v_error = (target_vs - v_pred) * (1 - truncation)
        # 统一使用 value_loss_coeff * w_v_loss (如果你想要双重加权)
        return jnp.mean(v_error ** 2) * self.config.value_loss_coeff * self.config.w_v_loss

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        t_current = full_t_path[:-1]
        return FlowSchedule(
            t_current=t_current,
            t_next=full_t_path[1:],
        )


    def embed_timestep(self, t: Array) -> Array:
        """Embed (*, 1) timestep into (*, timestep_embed_dim)."""
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.config.timestep_embed_dim)
        return out

    def sample_action(
            self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, DGPOFMActionInfo]:
        """Sample an action from the policy given an observation."""
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = obs

        (*batch_dims, obs_dim) = obs.shape
        assert obs_dim == self.env.observation_size

        def euler_step(
                carry: Array, inputs: tuple[FlowSchedule, Array]
        ) -> tuple[Array, Array]:
            x_t = carry
            assert x_t.shape == (*batch_dims, self.env.action_size)
            schedule_t, noise = inputs
            assert schedule_t.t_current.shape == ()
            assert schedule_t.t_next.shape == ()
            assert noise.shape == x_t.shape

            dt = schedule_t.t_next - schedule_t.t_current

            velocity = (
                    networks.flow_mlp_fwd(
                        self.params.policy,
                        obs_norm,
                        x_t,
                        jnp.broadcast_to(
                            self.embed_timestep(schedule_t.t_current[None]),
                            (*batch_dims, self.config.timestep_embed_dim),
                        ),
                    )
                    * self.config.policy_mlp_output_scale
            )

            x_t_next = x_t + dt * velocity + self.config.sde_sigma * noise
            assert x_t_next.shape == x_t.shape
            return x_t_next, x_t

        prng_sample, prng_loss, prng_feather, prng_noise = jax.random.split(prng, num=4)

        noise_path = jax.random.normal(
            prng_noise,
            (self.config.flow_steps, *batch_dims, self.env.action_size),
        )
        x0, x_t_path = jax.lax.scan(
            euler_step,
            init=jax.random.normal(prng_sample, (*batch_dims, self.env.action_size)),
            xs=(self.get_schedule(), noise_path),
        )

        if not deterministic:
            perturb = (
                    jax.random.normal(prng_feather, (*batch_dims, self.env.action_size))
                    * self.config.feather_std
            )
            x0 = x0 + perturb

        # 极致精简：直接返回，没有任何多余的判断！
        return x0, DGPOFMActionInfo()

    @jdc.jit
    def training_step(
            self, transitions: DGPOFMTransition
    ) -> tuple[DGPOFMState, dict[str, Array]]:
        # We're use a (T, B) shape convention, corresponding to a "scan of the
        # vmap" and not a "vmap of the scan".
        config = self.config
        assert transitions.reward.shape == (config.iterations_per_env, config.num_envs)

        # Update observation statistics.
        state = self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)
        del self

        def step_batch(state: DGPOFMState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                partial(
                    DGPOFMState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)
                ),
                init=state,
                xs=transitions.prepare_minibatches(
                    step_prng, config.num_minibatches, config.batch_size
                ),
            )
            return state, metrics

        # Do N updates over the full batch of transitions.
        state, metrics = jax.lax.scan(
            step_batch,
            init=state,
            length=config.num_updates_per_batch,
        )

        return state, metrics