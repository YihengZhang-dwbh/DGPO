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
    # Closed
    resampling_alpha: float = 0.1  # 新增：控制重采样的“温度”
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
    learning_rate: float = 3e-4
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
            opt_state=opt.init(network_params),  # type: ignore
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
    ) -> tuple[Array, DGPOFMActionInfo]:  # <-- 修复了这里的类型提示
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

        param_update, new_opt_state = self.opt.update(grads, self.opt_state)  # type: ignore
        param_update = jax.tree.map(
            lambda x: -self.config.learning_rate * x, param_update
        )
        with jdc.copy_and_mutate(self) as state:
            state.params = jax.tree.map(jnp.add, self.params, param_update)
            state.opt_state = new_opt_state
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

        # 2. 展平张量
        flat_h_s = h_s.reshape((N, -1))
        flat_actions = transitions.action.reshape((N, self.env.action_size))
        flat_adv = gae_advantages.reshape((N,))
        flat_obs = obs_norm.reshape((N, self.env.observation_size))

        # ========================== 极速子采样逻辑开始 ==========================
        # 3. 随机选择一个固定的候选池 M (例如 1024)，显著降低计算量
        M = 1024
        key_sample, prng_resample = jax.random.split(prng_resample)
        # 随机选出 M 个“参考邻居”
        candidate_indices = jax.random.choice(key_sample, N, shape=(M,), replace=False)

        h_s_candidates = flat_h_s[candidate_indices]  # (M, D)
        actions_candidates = flat_actions[candidate_indices]  # (M, A)
        adv_candidates = flat_adv[candidate_indices]  # (M,)

        # 4. 只计算 N x M 的距离矩阵 (不再是 N x N)
        # 矩阵大小从 9亿 缩小到 3千万，显存压力瞬间消失
        def compute_dist_row(single_h):
            # 计算单个状态对 M 个候选者的距离
            return jnp.sum((single_h - h_s_candidates) ** 2, axis=-1)

        # 使用 vmap 批量计算 N 个状态对应的 M 个距离
        dist_matrix_sub = jax.vmap(compute_dist_row)(flat_h_s)  # (N, M)

        # 5. 在 M 个候选者中找最近的 k 个
        k_neighbors = 8  # 因为候选池小了，k 也可以适当减小提升速度
        neg_dists, top_k_sub_indices = jax.lax.top_k(-dist_matrix_sub, k_neighbors)

        # 获取这些邻居在候选池中的 Advantage
        # neighbor_advs 形状为 (N, k_neighbors)
        neighbor_advs = adv_candidates[top_k_sub_indices]

        # 6. Gumbel-Max 重采样 (在 k 个最近邻中选优势最大的)
        max_adv = jnp.max(neighbor_advs, axis=-1, keepdims=True)
        logits = (neighbor_advs - max_adv) / self.config.resampling_alpha

        gumbel_noise = jax.random.gumbel(prng_resample, shape=(N, k_neighbors))
        resample_choice = jnp.argmax(logits + gumbel_noise, axis=-1)  # (N,)

        # 映射回动作空间
        # 从 top_k_sub_indices 中选出被选中的那个参考点的索引
        final_indices_in_candidates = jnp.take_along_axis(
            top_k_sub_indices, resample_choice[:, None], axis=1
        ).squeeze()

        # 最终的目标动作 a_hat
        a_hat = actions_candidates[final_indices_in_candidates]
        # ========================== 极速子采样逻辑结束 ==========================

        # --- 恢复全量监控指标 ---
        # 在子采样模式下，我们通过距离分布来模拟原本的指标
        neighbor_counts = jnp.sum(dist_matrix_sub <= individual_deltas[:, None], axis=-1)

        # 1. 平均邻居数 (在 K-NN 下这通常等于 k_neighbors)
        avg_neighbor_count = jnp.mean(neighbor_counts)

        # 2. 推估类数量 (考虑采样率 M/N)
        # 这里的逻辑是：如果 M 个样本里能找到 k 个邻居，那么全量 N 样本里理论上有 k*(N/M) 个
        est_clusters = N / (avg_neighbor_count * (N / M) + 1e-8)

        # 3. 孤立样本比：在 M 个候选人里连 1 个够近的邻居都很难找
        # 我们定义距离大于某个阈值或邻居数极少为孤立
        isolated_ratio = jnp.mean(neighbor_counts <= 1.0)

        # 4. 拥挤样本比：在子采样池里，邻居占比超过了 10% (说明局部性失效)
        crowded_ratio = jnp.mean(neighbor_counts > (M * 0.1))
        # ----------------------

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

        # 8. 整合 metrics 返回 (替换原本 FPO 的复杂 ratio metrics)
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        v_loss = jnp.mean(v_error ** 2) * self.config.value_loss_coeff
        total_loss = policy_loss + v_loss

        metrics["policy_loss"] = policy_loss
        metrics["v_loss"] = v_loss
        metrics["advantages_mean"] = jnp.mean(gae_advantages)

        # 重点：在这里添加新指标，它们会被打印到 txt 和控制台
        metrics["dgpo/est_clusters"] = est_clusters
        metrics["dgpo/isolated_ratio"] = isolated_ratio
        metrics["dgpo/crowded_ratio"] = crowded_ratio
        metrics["dgpo/avg_neighbors"] = avg_neighbor_count


        return total_loss, metrics
