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

    # 控制损失权重
    w_v_loss: float = 1.
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4

    # 关键修改：JAX 内部的分支逻辑通常要求 M 的大小或采样开关在编译期确定
    use_subsampling: jdc.Static[bool] = False
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

    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState

    prng: Array
    steps: Array

    @staticmethod
    @jdc.jit
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

        # We'll manage learning rate ourselves!
        opt = optax.scale_by_adam()

        return DGPOFMState(
            env=env,
            config=config,
            params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt=opt,
            opt_state=opt.init(network_params),
            prng=prng2,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

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

    def _step_minibatch(
            self, transitions: DGPOFMTransition, prng: Array
    ) -> tuple[DGPOFMState, dict[str, Array]]:
        """One training step over a minibatch of transitions."""

        assert transitions.reward.shape == (
            self.config.unroll_length,
            self.config.batch_size,
        )
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: DGPOFMState._compute_dgpofm_loss(
                jdc.replace(self, params=params),
                transitions,
                prng,
            ),
            has_aux=True,
        )(self.params)
        assert isinstance(grads, DGPOFMParams)
        assert isinstance(loss, Array)
        assert isinstance(metrics, dict)

        # 1. 正常的 Adam 梯度变换 (矩估计归一化)
        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)

        # 2. 手动应用非对称学习率更新
        policy_lr = self.config.learning_rate_p
        value_lr = self.config.learning_rate_v  # * 2.5

        # 使用 jdc.replace 重新组装更新项
        # 我们对 policy 和 value 的 pytree 分别乘上负的学习率
        new_updates = jdc.replace(
            updates,
            policy=jax.tree.map(lambda x: -policy_lr * x, updates.policy),
            value=jax.tree.map(lambda x: -value_lr * x, updates.value)
        )

        # 1. 获取第一轮联合更新后的参数 (Policy 在这里完成了本轮的唯一一次更新)
        new_params = optax.apply_updates(self.params, new_updates)

        # === 增量修改 2：实装 Critic 专属的局部 Adam 微调 ===
        target_vs = metrics["_target_vs"]
        del metrics["_target_vs"]  # 阅后即焚，不污染外层数据结构

        # 核心：单独定义一个纯净的局部 Adam 优化器，不干扰全局动量
        extra_v_opt = optax.adam(self.config.learning_rate_v)
        extra_v_opt_state = extra_v_opt.init(new_params.value)

        def extra_value_update(carry, _):
            curr_v_params, curr_v_opt_state = carry

            def v_loss_fn(v_params):
                # 重构归一化特征
                if self.config.normalize_observations:
                    obs_norm = (transitions.obs - self.obs_stats.mean) / self.obs_stats.std
                else:
                    obs_norm = transitions.obs

                v_pred, _ = networks.value_mlp_fwd_with_features(v_params, obs_norm)
                v_error = (target_vs - v_pred) * (1 - transitions.truncation)
                return jnp.mean(v_error ** 2)

            # 获取 Value 梯度
            v_loss_val, v_grads = jax.value_and_grad(v_loss_fn)(curr_v_params)

            # 使用局部 Adam 步进
            v_updates, next_v_opt_state = extra_v_opt.update(v_grads, curr_v_opt_state, curr_v_params)
            # 这里不用乘负学习率了，因为 optax.adam 内部已经自带了下降逻辑
            next_v_params = optax.apply_updates(curr_v_params, v_updates)

            return (next_v_params, next_v_opt_state), v_loss_val

        # 强制 Critic 额外用 Adam 猛练 4 次！
        (final_v_params, _), extra_v_losses = jax.lax.scan(
            extra_value_update,
            (new_params.value, extra_v_opt_state),
            None,
            length=4
        )

        # 把小灶练出来的 Critic 拼回主参数里
        final_params = jdc.replace(new_params, value=final_v_params)

        # 记录小灶结束后的最终残差
        metrics["v_loss_after_extra"] = extra_v_losses[-1] * self.config.value_loss_coeff
        # =======================================================

        with jdc.copy_and_mutate(self) as state:
            state.params = final_params
            state.opt_state = new_opt_state  # 全局优化器状态只接受了第一次的更新，完美保护
            state.steps = state.steps + 1
        return state, metrics

    def _compute_dgpofm_loss(
            self, transitions: DGPOFMTransition, prng: Array
    ) -> tuple[Array, dict[str, Array]]:

        (timesteps, batch_dim) = transitions.reward.shape
        assert transitions.obs.shape == (
            timesteps,
            batch_dim,
            self.env.observation_size,
        )
        assert transitions.action.shape == (
            timesteps,
            batch_dim,
            self.env.action_size,
        )

        metrics = dict[str, Array]()

        if self.config.normalize_observations:
            obs_norm = (transitions.obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = transitions.obs

        prng_resample, prng_eps, prng_t = jax.random.split(prng, 3)
        (timesteps, batch_dim) = transitions.reward.shape
        N = timesteps * batch_dim

        # 1. 获取 h_s (替换掉原来的 value_pred = networks.value_mlp_fwd(...))
        value_pred, h_s = networks.value_mlp_fwd_with_features(self.params.value, obs_norm)

        assert value_pred.shape == (timesteps, batch_dim)

        bootstrap_obs_norm = (
                                     transitions.next_obs[-1:, :, :] - self.obs_stats.mean
                             ) / self.obs_stats.std
        bootstrap_value = networks.value_mlp_fwd(self.params.value, bootstrap_obs_norm)
        assert bootstrap_value.shape == (1, batch_dim)

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

        # Log advantage statistics before normalization
        metrics["advantages_mean"] = jnp.mean(gae_advantages)
        metrics["advantages_std"] = jnp.std(gae_advantages)
        metrics["advantages_min"] = jnp.min(gae_advantages)
        metrics["advantages_max"] = jnp.max(gae_advantages)

        if self.config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (
                    gae_advantages.std() + 1e-8
            )

        # --- 1. 展平张量 (直接使用归一化后的 obs) ---
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        flat_actions = transitions.action.reshape((N, self.env.action_size))
        flat_adv = gae_advantages.reshape((N,))

        # =========================================================================
        # [核心增量修改]：使用 if/else 完全隔离 Cluster 模式与原有模式
        # =========================================================================
        if self.config.resampling_mode == "cluster":
            # --- 全新：基于状态空间的高效聚类重采样 (O(N*C) 复杂度) ---
            C = self.config.num_clusters
            prng_cluster, prng_resample = jax.random.split(prng_resample)

            # 1. 初始化聚类中心 (随机从 flat_obs 中抽 C 个)
            initial_indices = jax.random.choice(prng_cluster, N, shape=(C,), replace=False)
            centers = jax.lax.stop_gradient(flat_obs[initial_indices])

            # 2. 手动展开 3 次 K-Means 迭代 (避免 scan 导致的历史堆叠)
            labels = jnp.zeros((N,), dtype=jnp.int32)
            for _ in range(3):
                # 计算所有点到中心的距离 (N, C)
                dist_to_centers = jnp.sum((flat_obs[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
                labels = jnp.argmin(dist_to_centers, axis=-1)  # (N,)

                # 更新聚类中心
                one_hot = jax.nn.one_hot(labels, C)  # (N, C)
                centers = jnp.matmul(one_hot.T, flat_obs) / (jnp.sum(one_hot, axis=0)[:, None] + 1e-8)
                centers = jax.lax.stop_gradient(centers)

            # 此时 labels 的形状是完美的 (N,)
            one_hot_labels = jax.nn.one_hot(labels, C)  # (N, C)

            # 3. 簇内动态温度与 Gumbel-Max 采样
            # ... (接下来的代码完全保持原样) ...
            # 掩码优势矩阵：点i如果不属于簇c，则其优势为 -inf
            masked_adv_cluster = jnp.where(one_hot_labels, flat_adv[:, None], -jnp.inf)

            # 计算每个簇的局部动态 Alpha (C,)
            local_adv_pool = jnp.where(one_hot_labels, flat_adv[:, None], 0.0)
            local_scale_c = jax.lax.stop_gradient(jnp.max(jnp.abs(local_adv_pool), axis=0))
            dynamic_alpha_c = self.config.resampling_alpha * (local_scale_c + 1e-6)

            # 簇内 Logits 计算 (N, C)
            max_adv_c = jnp.max(masked_adv_cluster, axis=0, keepdims=True)  # (1, C)
            logits_c = jnp.where(one_hot_labels, (masked_adv_cluster - max_adv_c) / dynamic_alpha_c[None, :], -jnp.inf)

            # 为每个簇独立进行 1 次 Gumbel-Max 采样，找出最佳候选索引
            gumbel_noise_c = jax.random.gumbel(prng_resample, shape=(N, C))
            sampled_idx_per_cluster = jnp.argmax(logits_c + gumbel_noise_c, axis=0)  # (C,)

            # 提取每个簇的"行为原语"，并分配给该簇的所有状态成员
            cluster_sampled_actions = flat_actions[sampled_idx_per_cluster]  # (C, action_dim)
            a_hat = cluster_sampled_actions[labels]  # (N, action_dim)

            # 4. 指标监控对齐 (适配 Cluster 模式)
            cluster_sizes = jnp.sum(one_hot_labels, axis=0)
            metrics["dgpo/avg_neighbor_count"] = jnp.mean(cluster_sizes)  # 这里等同于簇的平均大小
            metrics["dgpo/isolated_ratio"] = jnp.mean(cluster_sizes <= 1.0)
            metrics["dgpo/est_clusters"] = jnp.array(C, dtype=jnp.float32)
            metrics["dgpo/dynamic_alpha_mean"] = jnp.mean(dynamic_alpha_c)
            metrics["dgpo/local_scale_max"] = jnp.max(local_scale_c)

        else:
            # =========================================================================
            # [原有逻辑区]：如果不是 cluster，则原封不动走 KNN / Radius / Both 的老路
            # =========================================================================
            if self.config.use_subsampling:
                M = self.config.subsampling_m
                prng_pool, prng_resample = jax.random.split(prng_resample)
                actual_m = jnp.minimum(M, N)
                candidate_indices = jax.random.choice(prng_pool, N, shape=(actual_m,), replace=False)

                dist_input_all = jax.lax.stop_gradient(flat_obs)
                dist_input_cand = jax.lax.stop_gradient(flat_obs[candidate_indices])

                sq_norms_all = jnp.sum(dist_input_all ** 2, axis=-1)
                sq_norms_cand = jnp.sum(dist_input_cand ** 2, axis=-1)
                dist_sq = jnp.maximum(0.0,
                                      sq_norms_all[:, None] + sq_norms_cand[None, :] - 2 * jnp.matmul(dist_input_all,
                                                                                                      dist_input_cand.T))

                cand_adv = flat_adv[candidate_indices]
                gumbel_shape = (N, actual_m)
                pool_indices_ref = candidate_indices
                scaling_factor = N / actual_m
            else:
                dist_input = jax.lax.stop_gradient(flat_obs)
                sq_norms = jnp.sum(dist_input ** 2, axis=-1)
                dist_sq = jnp.maximum(0.0,
                                      sq_norms[:, None] + sq_norms[None, :] - 2 * jnp.matmul(dist_input, dist_input.T))

                cand_adv = flat_adv
                gumbel_shape = (N, N)
                pool_indices_ref = jnp.arange(N)
                scaling_factor = 1.0

            # --- 邻域划分算法选择 ---
            # 1. 计算距离
            dist_matrix = jnp.sqrt(dist_sq + 1e-8)

            # 2. 模式切换
            mode = self.config.resampling_mode
            if mode == "knn":
                deltas = jnp.quantile(dist_matrix, q=0.05, axis=-1)
                mask = dist_matrix < deltas[:, None]
            elif mode == "radius":
                mask = dist_matrix < self.config.fixed_radius
            elif mode == "both":
                mask_radius = dist_matrix < self.config.fixed_radius
                K_val = self.config.resampling_topk
                _, topk_indices = jax.lax.top_k(-dist_matrix, k=K_val)
                mask_topk = jnp.zeros_like(mask_radius, dtype=jnp.bool_).at[
                    jnp.arange(N)[:, None], topk_indices
                ].set(True)
                mask = mask_radius & mask_topk
            else:
                mask = dist_matrix < self.config.fixed_radius  # Default

            # 3. 兜底
            mask = mask | (dist_matrix == jnp.min(dist_matrix, axis=-1, keepdims=True))

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

        # =========================================================================
        # 重采样块结束，继续后续的流匹配 (不受任何影响)
        # =========================================================================

        # 7. 纯净流匹配 ODE 更新
        eps = jax.random.normal(prng_eps, (N, self.env.action_size))
        if self.config.discretize_t_for_training:
            t_idx = jax.random.randint(prng_t, (N, 1), 0, self.config.flow_steps)
            t = self.get_schedule().t_current[t_idx]
        else:
            t = jax.random.uniform(prng_t, (N, 1))

        x_t = t * eps + (1.0 - t) * a_hat
        velocity_pred = networks.flow_mlp_fwd(
            self.params.policy, flat_obs, x_t, self.embed_timestep(t)
        ) * self.config.policy_mlp_output_scale

        if self.config.output_mode == "u_but_supervise_as_eps":
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
            policy_loss = jnp.mean((eps - x1_pred) ** 2)
        else:
            policy_loss = jnp.mean((velocity_pred - (eps - a_hat)) ** 2)

        # 8. 整合 metrics 返回
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        v_loss = jnp.mean(v_error ** 2) * self.config.value_loss_coeff
        total_loss = policy_loss + v_loss * self.config.w_v_loss

        metrics["policy_loss"] = policy_loss
        metrics["v_loss"] = v_loss
        metrics["advantages_mean"] = jnp.mean(gae_advantages)
        # === 增量修改 1：将真实回报目标暴露出去 ===
        metrics["_target_vs"] = jax.lax.stop_gradient(gae_vs)
        return total_loss, metrics