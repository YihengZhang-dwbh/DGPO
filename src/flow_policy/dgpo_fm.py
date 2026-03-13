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
    # [新增]: 价值分层聚类参数
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

        # 2. 预计算固定目标 (a_hat 和 target_vs)
        # 注意：这里我们把 obs_norm 传进去
        a_hat, target_vs, metrics = self._compute_targets(transitions, obs_norm, prng_targets)

        # 3. Policy 训练 1 次
        def policy_loss_fn(p_params):
            return self._compute_policy_loss(p_params, obs_norm, a_hat, prng_policy)

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
        # --- 原有代码 ---
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        flat_actions = transitions.action.reshape((N, self.env.action_size))
        flat_adv = gae_advantages.reshape((N,))
        prng_resample = prng

        # =========================================================
        # --- 新增插入：1. 提取价值 Rank 并构造扭曲空间向量 ---
        # =========================================================
        K_v = self.config.num_value_buckets
        prng_v, prng_resample = jax.random.split(prng_resample)

        flat_vs = gae_vs.reshape((N, 1))
        v_centers = jax.lax.stop_gradient(flat_vs[jax.random.choice(prng_v, N, shape=(K_v,), replace=False)])

        for _ in range(5):
            v_dist = jnp.sum((flat_vs[:, None, :] - v_centers[None, :, :]) ** 2, axis=-1)
            v_labels = jnp.argmin(v_dist, axis=-1)
            v_one_hot = jax.nn.one_hot(v_labels, K_v)
            v_centers = jnp.matmul(v_one_hot.T, flat_vs) / (jnp.sum(v_one_hot, axis=0)[:, None] + 1e-8)

        # 获取严格的价值阶级 Rank (0 到 K_v - 1)
        sorted_v_indices = jnp.argsort(v_centers.squeeze())
        ranks = jnp.argsort(sorted_v_indices)
        point_ranks = ranks[v_labels].astype(jnp.float32)[:, None]

        # 空间扭曲压缩：最大物理距离压缩到 0.5 以内 (平方 < 0.25)
        max_norm = jax.lax.stop_gradient(jnp.max(jnp.linalg.norm(flat_obs, axis=-1, keepdims=True))) + 1e-8
        # s_T = (flat_obs / max_norm) * 0.25
        # # 你的神级向量：[压缩物理状态, 价值阶层]
        # fused_vectors = jnp.concatenate([s_T, point_ranks], axis=-1)

        # 你的神级向量：[压缩物理状态, 价值阶层]
        fused_vectors = jnp.concatenate([flat_obs, point_ranks*4*max_norm], axis=-1)

        # === 极简融合逻辑 ===
        flat_hs = h_s.reshape((N, -1))
        flat_hs_norm = flat_hs / (jnp.linalg.norm(flat_hs, axis=-1, keepdims=True) + 1e-8)

        # 直接乘上 config 里的权重！
        combined_features = jnp.concatenate([
            flat_obs,
            flat_hs_norm * self.config.semantic_weight
        ], axis=-1)

        # ==========================================================
        # === 新增：对融合后的向量进行全局 L2 归一化 (投影到超球面) ===
        # ==========================================================
        if self.config.L2_combined_regularized:
            combined_features = combined_features / (jnp.linalg.norm(combined_features, axis=-1, keepdims=True) + 1e-8)

        if self.config.resampling_mode == "cluster":
            K_v = self.config.num_value_buckets
            C = self.config.num_clusters

            prng_v, prng_p, prng_resample = jax.random.split(prng_resample, 3)

            # =========================================================
            # 3. 全局自适应 K-Means (天然等效两次聚类)
            # =========================================================
            centers = jax.lax.stop_gradient(fused_vectors[jax.random.choice(prng_p, N, shape=(C,), replace=False)])
            sq_norms_fused = jnp.sum(fused_vectors ** 2, axis=-1)

            labels = jnp.zeros((N,), dtype=jnp.int32)
            for _ in range(10):
                sq_norms_centers = jnp.sum(centers ** 2, axis=-1)
                dist = jnp.maximum(0.0,
                                   sq_norms_fused[:, None] + sq_norms_centers[None, :] - 2 * jnp.matmul(fused_vectors,
                                                                                                        centers.T))

                labels = jnp.argmin(dist, axis=-1)
                one_hot = jax.nn.one_hot(labels, C)
                centers = jnp.matmul(one_hot.T, fused_vectors) / (jnp.sum(one_hot, axis=0)[:, None] + 1e-8)
                centers = jax.lax.stop_gradient(centers)

            # =========================================================
            # 4. 离群点检测 (极其关键：必须换回原汁原味的未压缩物理空间！)
            # =========================================================
            # 我们只需要知道每个中心被分配到了哪些状态，用这些状态算出原始的物理中心
            one_hot_labels = jax.nn.one_hot(labels, C)
            # 还原出纯物理的聚类中心
            physical_centers = jnp.matmul(one_hot_labels.T, flat_obs) / (
                        jnp.sum(one_hot_labels, axis=0)[:, None] + 1e-8)
            physical_centers = jax.lax.stop_gradient(physical_centers)

            # 在原尺寸下计算物理距离，判定离群点，完美保留你的 fixed_radius 物理意义！
            sq_norms_obs = jnp.sum(flat_obs ** 2, axis=-1)
            sq_norms_phys_centers = jnp.sum(physical_centers ** 2, axis=-1)
            phys_dist = jnp.maximum(0.0,
                                    sq_norms_obs[:, None] + sq_norms_phys_centers[None, :] - 2 * jnp.matmul(flat_obs,
                                                                                                            physical_centers.T))

            # --- 在 _compute_targets 内部 ---

            # 1. 获取观测维度
            obs_dim = flat_obs.shape[-1]

            # 2. 计算动态物理半径：2.0 * sqrt(dim)
            # 使用 jnp.sqrt 保证 JAX 追踪，* 2.0 是你的硬核约束
            dynamic_radius = 2.0 * jnp.sqrt(obs_dim)

            # 3. 执行离群点检测
            # (假设你还在使用 cluster 模式)
            sq_norms_obs = jnp.sum(flat_obs ** 2, axis=-1)
            sq_norms_phys_centers = jnp.sum(physical_centers ** 2, axis=-1)
            phys_dist_sq = jnp.maximum(0.0,
                                       sq_norms_obs[:, None] + sq_norms_phys_centers[None, :] - 2 * jnp.matmul(flat_obs,
                                                                                                               physical_centers.T))

            min_dists_sq = jnp.min(phys_dist_sq, axis=-1)

            # 判定：平方距离 > 半径的平方
            is_outlier = min_dists_sq > (dynamic_radius ** 2)

            # --- 后续逻辑保持不变 ---
            valid_one_hot = one_hot_labels * (~is_outlier[:, None])

            # =========================================================
            # 5. 簇内 Gumbel-Max 选优 (保持原样)
            # =========================================================
            masked_adv = jnp.where(valid_one_hot, flat_adv[:, None], -jnp.inf)
            local_adv_pool = jnp.where(valid_one_hot, flat_adv[:, None], 0.0)

            local_scale_c = jax.lax.stop_gradient(jnp.max(jnp.abs(local_adv_pool), axis=0))
            dynamic_alpha_c = self.config.resampling_alpha * (local_scale_c + 1e-6)

            max_adv_c = jnp.max(masked_adv, axis=0, keepdims=True)
            logits_c = jnp.where(valid_one_hot, (masked_adv - max_adv_c) / dynamic_alpha_c[None, :], -jnp.inf)

            gumbel_noise_c = jax.random.gumbel(prng_resample, shape=(N, C))
            sampled_idx_per_cluster = jnp.argmax(logits_c + gumbel_noise_c, axis=0)

            cluster_sampled_actions = flat_actions[sampled_idx_per_cluster]
            a_hat_normal = cluster_sampled_actions[labels]

            a_hat = jnp.where(is_outlier[:, None], flat_actions, a_hat_normal)

            # 监控指标
            cluster_sizes = jnp.sum(valid_one_hot, axis=0)
            metrics["dgpo/avg_neighbor_count"] = jnp.mean(cluster_sizes)
            metrics["dgpo/dynamic_alpha_mean"] = jnp.mean(dynamic_alpha_c)
            metrics["dgpo/outlier_ratio"] = jnp.mean(is_outlier)
            metrics["dgpo/est_clusters"] = jnp.array(C, dtype=jnp.float32)
            metrics["dgpo/local_scale_max"] = jnp.max(local_scale_c)
            # --- 替换原有 else: 分支下的距离计算部分 ---
        else:
            if self.config.use_subsampling:
                M = self.config.subsampling_m
                prng_pool, prng_resample = jax.random.split(prng_resample)
                # actual_m = jnp.minimum(M, N)
                # ✅ 修改后：使用原生 Python 的 min 函数，保持 actual_m 为静态整数！
                actual_m = min(M, N)
                candidate_indices = jax.random.choice(prng_pool, N, shape=(actual_m,), replace=False)

                # 1. 计算融合空间距离 (用于确定拓扑邻域)
                dist_input_fused = jax.lax.stop_gradient(fused_vectors)
                dist_cand_fused = jax.lax.stop_gradient(fused_vectors[candidate_indices])
                sq_all_fused = jnp.sum(dist_input_fused ** 2, axis=-1)
                sq_cand_fused = jnp.sum(dist_cand_fused ** 2, axis=-1)
                dist_sq_fused = jnp.maximum(0.0, sq_all_fused[:, None] + sq_cand_fused[None, :] - 2 * jnp.matmul(
                    dist_input_fused, dist_cand_fused.T))

                # 2. 计算纯物理距离 (用于判断离群点/Radius 阈值)
                dist_input_phys = jax.lax.stop_gradient(flat_obs)
                dist_cand_phys = jax.lax.stop_gradient(flat_obs[candidate_indices])
                sq_all_phys = jnp.sum(dist_input_phys ** 2, axis=-1)
                sq_cand_phys = jnp.sum(dist_cand_phys ** 2, axis=-1)
                dist_sq_phys = jnp.maximum(0.0, sq_all_phys[:, None] + sq_cand_phys[None, :] - 2 * jnp.matmul(
                    dist_input_phys, dist_cand_phys.T))

                cand_adv = flat_adv[candidate_indices]
                gumbel_shape = (N, actual_m)
                pool_indices_ref = candidate_indices
                scaling_factor = N / actual_m
            else:
                # 不用子采样的情况同理，算两遍
                dist_input_fused = jax.lax.stop_gradient(fused_vectors)
                dist_sq_fused = jnp.maximum(0.0, jnp.sum(dist_input_fused ** 2, axis=-1)[:, None] +
                                            jnp.sum(dist_input_fused ** 2, axis=-1)[None, :] - 2 * jnp.matmul(
                    dist_input_fused, dist_input_fused.T))

                dist_input_phys = jax.lax.stop_gradient(flat_obs)
                dist_sq_phys = jnp.maximum(0.0, jnp.sum(dist_input_phys ** 2, axis=-1)[:, None] +
                                           jnp.sum(dist_input_phys ** 2, axis=-1)[None, :] - 2 * jnp.matmul(
                    dist_input_phys, dist_input_phys.T))

                cand_adv = flat_adv
                gumbel_shape = (N, N)
                pool_indices_ref = jnp.arange(N)
                scaling_factor = 1.0

            # 提取出两种距离矩阵
            dist_matrix_fused = jnp.sqrt(dist_sq_fused + 1e-8)
            dist_matrix_phys = jnp.sqrt(dist_sq_phys + 1e-8)

            # =========================================================
            # 👑 利用数学构造，生成“绝对阶级隔离墙”
            # =========================================================
            # # 只要距离 < 1.0，在数学上绝对保证它们属于同一个 Value Rank！
            # same_class_mask = dist_matrix_fused < 1.0
            # 既然放大系数是 4 * max_norm，那么只要距离小于这个系数，就绝对是同一阶级
            same_class_mask = dist_matrix_fused < (2 * max_norm)

            # =========================================================
            # 2. 模式切换 (所有模式都必须服从阶级隔离)
            # =========================================================
            mode = self.config.resampling_mode
            if mode == "knn":
                # 在同阶级内，按真实的物理距离找 Top 5%
                # 注意：如果同阶级的人数不够 5%，超出的部分会被 same_class_mask 无情斩断，绝不跨阶级拉人！
                deltas = jnp.quantile(dist_matrix_phys, q=0.05, axis=-1)
                mask_knn = dist_matrix_phys < deltas[:, None]
                mask = mask_knn & same_class_mask

            elif mode == "radius":
                # 完美的双重认证：必须是同一个价值阶级 AND 真实物理距离达标
                mask_radius = dist_matrix_phys < self.config.fixed_radius
                mask = mask_radius & same_class_mask

            elif mode == "both":
                mask_radius = dist_matrix_phys < self.config.fixed_radius
                K_val = self.config.resampling_topk
                _, topk_indices = jax.lax.top_k(-dist_matrix_phys, k=K_val)
                mask_topk = jnp.zeros_like(mask_radius, dtype=jnp.bool_).at[
                    jnp.arange(N)[:, None], topk_indices
                ].set(True)
                mask = mask_radius & mask_topk & same_class_mask

            else:
                mask = (dist_matrix_phys < self.config.fixed_radius) & same_class_mask

            # 1. 找出同阶级内，候选池里离你物理最近的那个人的距离
            # (用 jnp.where 把不同阶级的人距离设为无穷大，这样 min 就只会挑同类)
            min_phys_dist_in_class = jnp.min(
                jnp.where(same_class_mask, dist_matrix_phys, jnp.inf),
                axis=-1, keepdims=True
            )

            # 2. 只有当这个距离不是无穷大（即候选池里确实有同类）时，才把它作为保底
            safe_in_class_mask = (dist_matrix_phys == min_phys_dist_in_class) & same_class_mask

            # 3. 最终的 mask
            mask = mask | safe_in_class_mask

            # --- 计算每个状态邻域内的动态 Alpha ---
            local_adv_pool = jnp.where(mask, cand_adv[None, :], 0.0)
            local_scale = jax.lax.stop_gradient(jnp.max(jnp.abs(local_adv_pool), axis=-1))
            dynamic_alpha = self.config.resampling_alpha * (local_scale + 1e-6)

            # --- 监控指标计算 ---
            neighbor_counts = jnp.sum(mask, axis=-1)
            avg_neighbors = jnp.mean(neighbor_counts)
            metrics["dgpo/avg_neighbor_count"] = avg_neighbors
            metrics["dgpo/isolated_ratio"] = jnp.mean(neighbor_counts <= 1.0)
            metrics["dgpo/est_clusters"] = N / (avg_neighbors * scaling_factor + 1e-8)

            # --- 核心重采样计算 ---
            masked_adv = jnp.where(mask, cand_adv[None, :], -jnp.inf)
            max_adv = jnp.max(masked_adv, axis=-1, keepdims=True)

            logits = jnp.where(mask, (masked_adv - max_adv) / dynamic_alpha[:, None], -jnp.inf)

            gumbel_noise = jax.random.gumbel(prng_resample, shape=gumbel_shape)
            sampled_rel_indices = jnp.argmax(logits + gumbel_noise, axis=-1)

            a_hat = flat_actions[pool_indices_ref[sampled_rel_indices]]

            # --- 指标监控 ---
            metrics["dgpo/dynamic_alpha_mean"] = jnp.mean(dynamic_alpha)
            metrics["dgpo/local_scale_max"] = jnp.max(local_scale)

        return jax.lax.stop_gradient(a_hat), gae_vs, metrics

    # ==========================================
    # 3. 损失函数定义 (纯净计算)
    # ==========================================
    def _compute_policy_loss(self, policy_params, obs_norm, a_hat, prng):
        (timesteps, batch_dim, obs_dim) = obs_norm.shape
        N = timesteps * batch_dim
        flat_obs = obs_norm.reshape((N, obs_dim))

        prng_eps, prng_t = jax.random.split(prng, 2)
        eps = jax.random.normal(prng_eps, (N, self.env.action_size))
        t_idx = jax.random.randint(prng_t, (N, 1), 0, self.config.flow_steps)
        t = self.get_schedule().t_current[t_idx]

        x_t = t * eps + (1.0 - t) * a_hat
        velocity_pred = networks.flow_mlp_fwd(
            policy_params, flat_obs, x_t, self.embed_timestep(t)
        ) * self.config.policy_mlp_output_scale

        # 兼容你原来的 output_mode 逻辑
        if self.config.output_mode == "u_but_supervise_as_eps":
            x1_pred = (x_t - t * velocity_pred) + velocity_pred
            policy_loss = jnp.mean((eps - x1_pred) ** 2)
        else:
            policy_loss = jnp.mean((velocity_pred - (eps - a_hat)) ** 2)

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
