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

    use_hard_resampling: jdc.Static[bool] = True

    w_v_loss: float = 1.0
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4
    loop_v: jdc.Static[int] = 1

    use_hinge_cql: jdc.Static[bool] = True
    cql_decay_mode: jdc.Static[Literal["none", "linear", "cosine", "exponential", "inverse", "auto"]] = "auto"
    cql_init_weight: float = 0.1
    cql_final_weight: float = 0.0001
    cql_decay_ratio: float = 0.5

    cql_target_margin: float = 10
    cql_alpha_lr: float = 3e-4
    cql_alpha_kp: float = 0.05
    cql_clip_alpha: jdc.Static[bool] = False

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
    log_cql_weight: Array
    prng: Array
    steps: Array

    @staticmethod
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: DGPOFMConfig) -> DGPOFMState:
        obs_size = env.observation_size
        action_size = env.action_size
        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(prng0,
                                      (obs_size + action_size + config.timestep_embed_dim, 32, 32, 32, 32, action_size))
        critic_net = networks.mlp_init(prng1, (obs_size + action_size, 256, 256, 256, 256, 256, 1))
        network_params = DGPOFMParams(actor_net, critic_net)
        opt_policy = optax.adam(config.learning_rate_p)
        opt_value = optax.adam(config.learning_rate_v)
        return DGPOFMState(
            env=env, config=config, params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt_policy=opt_policy, opt_value=opt_value,
            opt_state_policy=opt_policy.init(network_params.policy),
            opt_state_value=opt_value.init(network_params.value),
            prng=prng2, steps=jnp.zeros((), dtype=jnp.int32),
            log_cql_weight=jnp.log(jnp.array(config.cql_init_weight, dtype=jnp.float32)),
        )

    def _step_minibatch(self, transitions: DGPOFMTransition, prng: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        prng_targets, prng_policy = jax.random.split(prng, 2)
        obs_norm = (transitions.obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else transitions.obs

        # 👑 修复：计算进度时，总步数应该是静态的
        steps_per_iter = self.config.iterations_per_env * self.config.num_envs
        # 确保这里参与计算的都是静态常数
        total_iterations = self.config.num_timesteps // steps_per_iter
        decay_steps = total_iterations * self.config.cql_decay_ratio
        progress = jnp.minimum(1.0, self.steps / jnp.maximum(1.0, decay_steps))

        current_K = (self.config.num_generated_actions_min +
                     progress * (self.config.num_generated_actions_max - self.config.num_generated_actions_min)).astype(jnp.int32)

        # 1. 传给 compute_targets
        pool_actions, pool_weights, target_qs, metrics = self._compute_targets(transitions, obs_norm, prng_targets, current_K)

        q_update_steps = self.config.loop_v

        def value_inner_step(carry, _):
            v_params, v_opt_state, current_log_alpha = carry
            current_alpha = jax.lax.stop_gradient(jnp.exp(current_log_alpha))

            def v_loss_fn(v_p):
                total_loss, penalty = self._compute_value_loss(
                    v_p, obs_norm, transitions.action, transitions.truncation, target_qs, pool_actions, current_alpha,
                    current_K
                )
                return total_loss, penalty

            (v_loss_val, current_penalty), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(v_params)
            v_updates, next_v_opt_state = self.opt_value.update(v_grads, v_opt_state, v_params)
            next_v_params = optax.apply_updates(v_params, v_updates)

            if self.config.cql_decay_mode == "auto":
                alpha_grad = current_penalty - self.config.cql_target_margin
                next_log_alpha = current_log_alpha + self.config.cql_alpha_lr * alpha_grad
                next_log_alpha = jnp.clip(next_log_alpha, a_min=jnp.log(self.config.cql_final_weight),
                                          a_max=jnp.log(self.config.cql_init_weight))
            else:
                next_log_alpha = current_log_alpha

            return (next_v_params, next_v_opt_state, next_log_alpha), {"v_loss/total": v_loss_val,
                                                                       "v_loss/cql_penalty": current_penalty,
                                                                       "v_loss/current_cql_weight": jnp.exp(
                                                                           next_log_alpha)}

        (new_value_params, new_opt_state_value, new_log_alpha), extra_v_metrics = jax.lax.scan(value_inner_step,
                                                                                               (self.params.value,
                                                                                                self.opt_state_value,
                                                                                                self.log_cql_weight),
                                                                                               None,
                                                                                               length=q_update_steps)
        for k, v in extra_v_metrics.items(): metrics[k] = v[-1]

        final_v_loss = extra_v_metrics["v_loss/total"][-1]
        fresh_pool_weights, q_metrics = self._compute_fresh_weights(new_value_params, obs_norm, pool_actions,
                                                                    final_v_loss, current_K)
        metrics.update(q_metrics)

        def policy_loss_fn(p_params):
            return self._compute_policy_loss(p_params, obs_norm, pool_actions, fresh_pool_weights, prng_policy)

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_opt_state_policy = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)
        new_policy_params = optax.apply_updates(self.params.policy, p_updates)
        metrics.update(p_metrics)

        new_params = DGPOFMParams(policy=new_policy_params, value=new_value_params)
        with jdc.copy_and_mutate(self) as state:
            state.params, state.opt_state_policy, state.opt_state_value, state.log_cql_weight = new_params, new_opt_state_policy, new_opt_state_value, new_log_alpha
            state.steps += 1
        return state, metrics

    def _compute_fresh_weights(self, value_params, obs_norm, pool_actions, final_v_loss, current_K) -> tuple[
        Array, dict[str, Array]]:
        N, K_plus_1, act_dim = pool_actions.shape
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))
        q_pool, _ = networks.value_mlp_fwd_with_features(value_params,
                                                         jnp.concatenate([obs_pool_b, pool_actions], axis=-1))
        q_pool = jax.lax.stop_gradient(q_pool)

        if self.config.use_dynamic_alpha:
            rmse = jnp.abs(final_v_loss + 1e-8)
            alpha = jnp.maximum(self.config.resampling_alpha_k * rmse, self.config.resampling_alpha_min)
        else:
            alpha = self.config.resampling_alpha_min

        # 👑 Mask 逻辑
        indices = jnp.arange(K_plus_1)
        mask = indices <= current_K
        q_pool_masked = jnp.where(mask[None, :], q_pool, -1e10)

        logits = (q_pool_masked - jnp.max(q_pool_masked, axis=-1, keepdims=True)) / alpha
        pool_probs = jax.nn.softmax(logits, axis=-1)

        return jax.lax.stop_gradient(pool_probs), {"q_guided/q_real_mean": jnp.mean(q_pool[:, 0]),
                                                   "q_guided/prob_real_mean": jnp.mean(pool_probs[:, 0]),
                                                   "q_guided/alpha_mean": jnp.mean(alpha)}

    def _compute_targets(self, transitions, obs_norm, prng, current_K):
        metrics = dict[str, Array]()
        T, B, _ = obs_norm.shape
        obs_dim = self.env.observation_size
        act_dim = self.env.action_size
        N = T * B

        prng_boot, prng_gen, _ = jax.random.split(prng, 3)

        # --- 1. GAE 计算 (保持不变) ---
        concat_inputs = jnp.concatenate([obs_norm, transitions.action], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(self.params.value, concat_inputs)
        q_pred = jax.lax.stop_gradient(q_pred)

        bootstrap_obs = transitions.next_obs[-1:, :, :]
        if self.config.normalize_observations:
            bootstrap_obs = (bootstrap_obs - self.obs_stats.mean) / self.obs_stats.std

        def boot_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            t_embed = jnp.broadcast_to(self.embed_timestep(jnp.array([t_curr])[..., None])[:, None, :],
                                       (1, B, self.config.timestep_embed_dim))
            vel = networks.flow_mlp_fwd(self.params.policy, bootstrap_obs, x,
                                        t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        schedule = self.get_schedule()
        bootstrap_act, _ = jax.lax.scan(boot_step_fn, jax.random.normal(prng_boot, (1, B, act_dim)),
                                        (schedule.t_current, schedule.t_next))
        bootstrap_q, _ = networks.value_mlp_fwd_with_features(self.params.value,
                                                              jnp.concatenate([bootstrap_obs, bootstrap_act], axis=-1))
        bootstrap_q = jax.lax.stop_gradient(bootstrap_q)

        gae_qs, _ = jax.lax.stop_gradient(rollouts.compute_gae(
            truncation=transitions.truncation,
            discount=transitions.discount * self.config.discounting,
            rewards=transitions.reward * self.config.reward_scaling,
            values=q_pred,
            bootstrap_value=bootstrap_q,
            gae_lambda=self.config.gae_lambda))

        # =========================================================
        # 👑 2. 物理级动作生成 (K_max 静态锁定)
        # =========================================================
        K_max = self.config.num_generated_actions_max
        flat_obs = obs_norm.reshape((N, obs_dim))
        flat_acts_real = transitions.action.reshape((N, 1, act_dim))

        # 定义 gen_step_fn (修复 NameError)
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_max, obs_dim))

        def gen_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            t_embed = jnp.broadcast_to(self.embed_timestep(jnp.array([t_curr])[..., None])[:, None, :],
                                       (N, K_max, self.config.timestep_embed_dim))
            p_params = jax.lax.stop_gradient(self.params.policy)
            vel = networks.flow_mlp_fwd(p_params, obs_b, x, t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        # 生成全量 K_max 动作
        x_t_gen = jax.random.normal(prng_gen, (N, K_max, act_dim))
        generated_acts_all, _ = jax.lax.scan(gen_step_fn, x_t_gen, (schedule.t_current, schedule.t_next))

        # 👑 物理切片：抹除尚未解锁的动作
        # 这一步极其关键，它保证了 Q 网络在 Iteration 0 只看到 1 个假动作
        gen_mask = jnp.arange(K_max) < current_K
        generated_acts = generated_acts_all * gen_mask[None, :, None]
        pool_actions = jnp.concatenate([flat_acts_real, generated_acts], axis=1)  # (N, K_max+1, act_dim)

        # --- 3. 裁判打分 ---
        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_max + 1, obs_dim))
        q_pool, _ = networks.value_mlp_fwd_with_features(self.params.value,
                                                         jnp.concatenate([obs_pool_b, pool_actions], axis=-1))
        q_pool = jax.lax.stop_gradient(q_pool)

        # 👑 逻辑屏蔽：除了 real 和 current_K 内的假动作，其余全部禁言
        full_mask = jnp.arange(K_max + 1) <= current_K
        q_pool_masked = jnp.where(full_mask[None, :], q_pool, -1e10)

        # Softmax 重采样概率
        pool_probs = jax.nn.softmax(
            (q_pool_masked - jnp.max(q_pool_masked, axis=-1, keepdims=True)) / self.config.resampling_alpha_min,
            axis=-1)

        return jax.lax.stop_gradient(pool_actions), jax.lax.stop_gradient(pool_probs), gae_qs, {
            "q_guided/q_real_mean": jnp.mean(q_pool[:, 0])}

    def _compute_policy_loss(self, policy_params, obs_norm, actions_pool, weights_pool, prng):
        N, K_plus_1, act_dim = actions_pool.shape
        M = self.config.num_epsilon_samples
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))
        prng_idx, prng_eps, prng_t = jax.random.split(prng, 3)

        # 👑 M 独立重采样逻辑 (不依赖动态 K，直接用 weights_pool)
        sampled_indices = jax.random.categorical(prng_idx, jnp.log(weights_pool + 1e-8)[:, None, :], axis=-1)
        sampled_actions = jnp.take_along_axis(actions_pool[:, None, :, :], sampled_indices[:, :, None, None],
                                              axis=2).squeeze(2)

        eps = jax.random.normal(prng_eps, (N, M, act_dim))
        t = self.get_schedule().t_current[jax.random.randint(prng_t, (N, M, 1), 0, self.config.flow_steps)]
        x_t = t * eps + (1.0 - t) * sampled_actions
        t_embed = self.embed_timestep(t)
        vel_pred = networks.flow_mlp_fwd(policy_params,
                                         jnp.broadcast_to(flat_obs[:, None, :], (N, M, flat_obs.shape[-1])), x_t,
                                         t_embed) * self.config.policy_mlp_output_scale

        error_sq = jnp.sum(
            (eps - ((x_t - t * vel_pred) + vel_pred)) ** 2 if self.config.output_mode == "u_but_supervise_as_eps" else (
                                                                                                                                   vel_pred - (
                                                                                                                                       eps - sampled_actions)) ** 2,
            axis=-1)
        return jnp.mean(error_sq), {"policy_loss": jnp.mean(error_sq)}

    def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs, pool_actions,
                            current_cql_weight, current_K):
        # 👑 修复：这里同样要把动态的 _ 换成静态的 observation_size
        obs_dim = self.env.observation_size
        q_pred, _ = networks.value_mlp_fwd_with_features(value_params, jnp.concatenate([obs_norm, actions], axis=-1))
        mse_loss = jnp.mean(((target_qs - q_pred) * (1 - truncation)) ** 2)

        N, K_plus_1, _ = pool_actions.shape
        # 👑 修复：reshape 使用静态 obs_dim
        flat_obs = obs_norm.reshape((N, obs_dim))
        q_pool_fake, _ = networks.value_mlp_fwd_with_features(value_params, jnp.concatenate(
            [jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, obs_dim)), pool_actions], axis=-1))

        # 👑 核心修复：只惩罚“第一个”假动作（或者前 current_K 个）
        # 我们不再用 mean，因为 mean 会随 K 变化改变梯度量级
        penalty_all = jax.nn.relu(q_pool_fake[:, 1:] - q_real_sg)  # (N, K_max)

        # 关键：我们只取前 current_K 个动作的惩罚，并且只除以 current_K
        active_mask = jnp.arange(self.config.num_generated_actions_max) < current_K
        cql_penalty = jnp.sum(penalty_all * active_mask[None, :]) / jnp.maximum(1.0, jnp.sum(active_mask))

        total_v_loss = (
                                   mse_loss + current_cql_weight * cql_penalty) * self.config.value_loss_coeff * self.config.w_v_loss
        return total_v_loss, cql_penalty

    def get_schedule(self) -> FlowSchedule:
        t = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(t_current=t[:-1], t_next=t[1:])

    def embed_timestep(self, t: Array) -> Array:
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, DGPOFMActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else obs

        def euler_step(carry, inputs):
            x_t, (sch, noise) = carry, inputs
            vel = networks.flow_mlp_fwd(self.params.policy, obs_norm, x_t,
                                        jnp.broadcast_to(self.embed_timestep(sch.t_current[None]), (*obs.shape[:-1],
                                                                                                    self.config.timestep_embed_dim))) * self.config.policy_mlp_output_scale
            return x_t + (sch.t_next - sch.t_current) * vel + self.config.sde_sigma * noise, x_t

        x, _ = jax.lax.scan(euler_step, jax.random.normal(prng, (*obs.shape[:-1], self.env.action_size)),
                            (self.get_schedule(), jax.random.normal(jax.random.split(prng)[1],
                                                                    (self.config.flow_steps, *obs.shape[:-1],
                                                                     self.env.action_size))))
        return x, DGPOFMActionInfo()

    @jdc.jit
    def training_step(self, transitions: DGPOFMTransition) -> tuple[DGPOFMState, dict[str, Array]]:
        if self.config.normalize_observations:
            with jdc.copy_and_mutate(self) as state: state.obs_stats = state.obs_stats.update(transitions.obs)

        def step_batch(state, _):
            state, metrics = jax.lax.scan(partial(DGPOFMState._step_minibatch, prng=jax.random.fold_in(state.prng, 0)),
                                          init=state,
                                          xs=transitions.prepare_minibatches(state.prng, self.config.num_minibatches,
                                                                             self.config.batch_size))
            return state, metrics

        state, metrics = jax.lax.scan(step_batch, init=self, length=self.config.num_updates_per_batch)
        return state, metrics