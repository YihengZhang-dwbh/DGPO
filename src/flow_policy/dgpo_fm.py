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
class DGPOFMConfig:
    # ==========================================
    # 1. Flow Matching Base Parameters
    # ==========================================
    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = "u_but_supervise_as_eps"
    timestep_embed_dim: jdc.Static[int] = 8
    policy_mlp_output_scale: float = 0.25

    # Number of noise paths (eps) to fit per REAL action
    num_epsilon_samples: jdc.Static[int] = 8

    # ==========================================
    # 2. Dynamic Lipschitz Repulsive Flow & Routing
    # ==========================================
    # 👑 Three choices for the Z partition constant
    z_estimation_mode: jdc.Static[Literal["local_max", "global_ema", "fixed"]] = "local_max"
    z_temperature: float = 1.0  # T in the exp(A/T) formula
    z_fixed_max_adv: float = 3.0  # Used if mode is 'fixed'
    z_ema_rate: float = 0.99  # Used if mode is 'global_ema'

    # Pure Lipschitz bounded safe distance
    lipschitz_k: float = 0.1
    max_repel_radius: float = 2.0

    # ==========================================
    # 3. Action Boundary Safety
    # ==========================================
    action_clip: jdc.Static[Literal["hard", "margin", "tanh", "fold", "scale_clip"]] = "margin"
    clip_margin: float = 1.1

    # ==========================================
    # 4. RL Infrastructure & EMA Trust Region
    # ==========================================
    batch_size: jdc.Static[int] = 1024
    num_minibatches: jdc.Static[int] = 32
    num_updates_per_batch: jdc.Static[int] = 16
    unroll_length: jdc.Static[int] = 30
    num_envs: jdc.Static[int] = 2048

    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4
    loop_v: jdc.Static[int] = 1

    discounting: float = 0.995
    gae_lambda: float = 0.95
    reward_scaling: float = 10.0
    value_loss_coeff: float = 0.25
    w_v_loss: float = 1.0

    normalize_observations: jdc.Static[bool] = True
    normalize_advantage: jdc.Static[bool] = True
    sde_sigma: float = 0.0
    feather_std: float = 0.0

    beta_r: float = 0.9
    beta_v: float = 0.9
    tolerance_r: float = -10
    tolerance_v: float = 10

    episode_length: int = 1000
    num_timesteps: jdc.Static[int] = 180000000
    num_evals: jdc.Static[int] = 30

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
    target_vs: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    advantages: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))


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

    ema_reward: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    ema_reward_sq: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    ema_v_loss: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    ema_v_loss_sq: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))

    # Track global maximum advantage for Rejection Sampling Z-estimation
    ema_max_adv: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))

    @staticmethod
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: DGPOFMConfig) -> DGPOFMState:
        obs_size = env.observation_size
        action_size = env.action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)

        # Actor takes obs, action, and time embed
        actor_net = networks.mlp_init(prng0,
                                      (obs_size + action_size + config.timestep_embed_dim, 32, 32, 32, 32, action_size))

        # 👑 CRITICAL CHANGE: Critic is now a pure V-Network! It ONLY takes observations!
        critic_net = networks.mlp_init(prng1, (obs_size, 256, 256, 256, 256, 256, 1))

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

    def _compute_value_loss(self, value_params, obs_norm, truncation, target_vs):
        # Pure V-Network forward
        v_pred, _ = networks.value_mlp_fwd_with_features(value_params, obs_norm)
        v_pred = v_pred.reshape(target_vs.shape)
        v_error = (target_vs - v_pred) * (1 - truncation)
        mse_loss = jnp.mean(v_error ** 2)
        return mse_loss * self.config.value_loss_coeff * self.config.w_v_loss

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(t_current=full_t_path[:-1], t_next=full_t_path[1:])

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        return out

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, DGPOFMActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else obs
        (*batch_dims, obs_dim) = obs.shape

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

        prng_sample, prng_noise = jax.random.split(prng, num=2)
        noise_path = jax.random.normal(prng_noise, (self.config.flow_steps, *batch_dims, self.env.action_size))
        x0, _ = jax.lax.scan(euler_step, jax.random.normal(prng_sample, (*batch_dims, self.env.action_size)),
                             (self.get_schedule(), noise_path))

        x_raw = x0
        if not deterministic:
            prng_feather = jax.random.fold_in(prng, 0)
            noise = jax.random.normal(prng_feather, x_raw.shape)
            x_raw = x_raw + noise * self.config.feather_std

        x_final = self._apply_clip(x_raw)
        return x_final, DGPOFMActionInfo()

    def _update_critic_only(self, transitions: DGPOFMTransition, prng: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        cfg = self.config
        obs_dim = self.env.observation_size
        N = transitions.obs.size // obs_dim
        obs_flat = ((transitions.obs - self.obs_stats.mean) / self.obs_stats.std
                    if cfg.normalize_observations else transitions.obs).reshape((N, obs_dim))
        target_vs = transitions.action_info.target_vs.reshape((N, 1))

        def value_inner_step(carry, _):
            v_p, v_opt = carry

            def v_loss_fn(p):
                total = self._compute_value_loss(p, obs_flat, transitions.truncation.reshape((N, 1)), target_vs)
                return total, total

            (v_loss_val, _), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(v_p)
            v_updates, next_v_opt = self.opt_value.update(v_grads, v_opt, v_p)
            return (optax.apply_updates(v_p, v_updates), next_v_opt), {"v_loss/total": v_loss_val}

        (new_v_params, new_v_opt_state), extra_v_metrics = jax.lax.scan(
            value_inner_step, (self.params.value, self.opt_state_value), None, length=cfg.loop_v
        )
        new_state = jdc.replace(self, params=jdc.replace(self.params, value=new_v_params),
                                opt_state_value=new_v_opt_state)
        return new_state, extra_v_metrics

    def _update_actor_only(self, transitions: DGPOFMTransition, prng: Array, global_v_loss: Array,
                           global_max_adv: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        cfg, sch = self.config, self.get_schedule()
        prng_pol, next_prng = jax.random.split(prng, 2)

        obs_dim = self.env.observation_size
        act_dim = self.env.action_size
        N = transitions.obs.size // obs_dim
        obs_flat = ((transitions.obs - self.obs_stats.mean) / self.obs_stats.std
                    if cfg.normalize_observations else transitions.obs).reshape((N, obs_dim))

        mb_mean_reward = jnp.mean(transitions.reward)
        final_v_loss = global_v_loss

        # 👑 Pure Real Actions and Real GAE Advantages
        a_real_flat = transitions.action.reshape((N, act_dim))
        adv_flat = transitions.action_info.advantages.reshape((N, 1))

        def policy_loss_fn(p_params):
            M = cfg.num_epsilon_samples
            p_eps, p_t, p_trust, p_route = jax.random.split(prng_pol, 4)

            a_target = jnp.broadcast_to(a_real_flat[:, None, :], (N, M, act_dim))
            a_adv = jnp.broadcast_to(adv_flat, (N, M))  # <--- 修复完毕，完美二维广播

            if cfg.action_clip in ["hard", "margin", "fold"]:
                a_target = self._apply_clip(a_target)

            # --- Z-Score EMA Trust Region ---
            t_outer = (self.steps // (cfg.num_updates_per_batch * cfg.num_minibatches)) + 1.0
            bc_v = 1.0 - jnp.power(cfg.beta_v, t_outer)
            bc_r = 1.0 - jnp.power(cfg.beta_r, t_outer)

            hat_r, hat_r_sq = self.ema_reward / bc_r, self.ema_reward_sq / bc_r
            hat_v, hat_v_sq = self.ema_v_loss / bc_v, self.ema_v_loss_sq / bc_v

            r_std = jnp.sqrt(jnp.maximum(hat_r_sq - jnp.square(hat_r), 0.0)) + 1e-5
            v_std = jnp.sqrt(jnp.maximum(hat_v_sq - jnp.square(hat_v), 0.0)) + 1e-5

            r_z = (mb_mean_reward - hat_r) / r_std
            v_z = (final_v_loss - hat_v) / v_std

            r_trust = jnp.clip(jnp.exp(-jnp.maximum(-r_z + cfg.tolerance_r, 0.0) / 0.5), 0.01, 1.0)
            v_trust = jnp.clip(jnp.exp(-jnp.maximum(v_z - cfg.tolerance_v, 0.0) / 0.5), 0.01, 1.0)

            # Only real actions now, so we always use r_trust as combined trust proxy
            trust_mask = (jax.random.uniform(p_trust, (N, M)) < r_trust).astype(jnp.float32)

            # --- 👑 Rejection Sampling Z-Estimation & Routing ---
            if cfg.z_estimation_mode == "local_max":
                Z_adv = jnp.max(a_adv)
            elif cfg.z_estimation_mode == "global_ema":
                Z_adv = global_max_adv
            elif cfg.z_estimation_mode == "fixed":
                Z_adv = cfg.z_fixed_max_adv
            else:
                Z_adv = jnp.max(a_adv)

            # Compute Target Probability (p*) guaranteed to be un-biased and bounded
            p_star = jnp.exp((a_adv - Z_adv) / cfg.z_temperature)
            p_star = jnp.clip(p_star, 0.0, 1.0)

            is_attract = (jax.random.uniform(p_route, (N, M)) < p_star).astype(jnp.float32)
            is_repel = 1.0 - is_attract

            # --- Flow Matching & Hinge Loss ---
            eps = jax.random.normal(p_eps, (N, M, act_dim))
            t = sch.t_current[jax.random.randint(p_t, (N, M, 1), 0, cfg.flow_steps)]

            x_t = t * eps + (1.0 - t) * a_target

            t_embed = self.embed_timestep(t)
            obs_p = jnp.broadcast_to(obs_flat[:, None, :], (N, M, obs_dim))
            vel = networks.flow_mlp_fwd(p_params, obs_p, x_t, t_embed) * cfg.policy_mlp_output_scale

            if cfg.output_mode == "u_but_supervise_as_eps":
                err = jnp.sum((eps - ((x_t - t * vel) + vel)) ** 2, axis=-1)
            else:
                err = jnp.sum((vel - (eps - a_target)) ** 2, axis=-1)

            dist = jnp.sqrt(err + 1e-8)
            loss_attract = err

            # Dynamic Lipschitz Repulsion (allows R -> 0 asymptotically)
            dynamic_radius = cfg.lipschitz_k * jnp.abs(a_adv)
            R_safe = jnp.minimum(dynamic_radius, cfg.max_repel_radius)
            loss_repel = jnp.maximum(0.0, R_safe - dist) ** 2

            err_combined = is_attract * loss_attract + is_repel * loss_repel
            loss = jnp.mean(err_combined * trust_mask)

            return loss, {
                "policy_loss": loss,
                "q_guided/real_trust_prob": r_trust,
                "q_guided/fake_trust_prob": v_trust,
                "q_guided/attract_ratio": jnp.mean(is_attract),
                "q_guided/mean_R_safe": jnp.mean(R_safe),
                "q_guided/Z_adv_anchor": Z_adv,
                "q_guided/p_star_mean": jnp.mean(p_star),
            }

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_p_opt = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)

        new_state = jdc.replace(
            self,
            params=jdc.replace(self.params, policy=optax.apply_updates(self.params.policy, p_updates)),
            opt_state_policy=new_p_opt,
            steps=self.steps + 1,
            prng=next_prng
        )

        return new_state, p_metrics

    @jdc.jit
    def training_step(self, transitions: DGPOFMTransition) -> tuple[DGPOFMState, dict[str, Array]]:
        config, state = self.config, self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)

        obs_norm = (
                               transitions.obs - state.obs_stats.mean) / state.obs_stats.std if config.normalize_observations else transitions.obs

        # 👑 Pure V-Network Forward (No actions required!)
        v_pred, _ = networks.value_mlp_fwd_with_features(state.params.value, obs_norm)
        v_pred = jax.lax.stop_gradient(v_pred)

        bootstrap_obs = transitions.next_obs[-1:, :, :]
        if config.normalize_observations:
            bootstrap_obs = (bootstrap_obs - state.obs_stats.mean) / state.obs_stats.std

        # 👑 Massive Compute Saver: No ODE required to bootstrap V!
        bootstrap_v, _ = networks.value_mlp_fwd_with_features(state.params.value, bootstrap_obs)

        target_vs, advs = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * config.discounting,
                rewards=transitions.reward * config.reward_scaling,
                values=v_pred,
                bootstrap_value=jax.lax.stop_gradient(bootstrap_v),
                gae_lambda=config.gae_lambda,
            )
        )

        # Normalize advantages properly across the global batch
        if config.normalize_advantage:
            advs = (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-8)

        new_action_info = jdc.replace(
            transitions.action_info,
            target_vs=target_vs,
            advantages=advs
        )
        new_transitions = jdc.replace(transitions, action_info=new_action_info)

        # Update EMA for global max advantage Z-Estimation
        batch_max_adv = jnp.max(advs)
        new_ema_max_adv = jnp.where(
            state.steps == 0,
            batch_max_adv,
            config.z_ema_rate * state.ema_max_adv + (1.0 - config.z_ema_rate) * batch_max_adv
        )

        def critic_epoch_step(carry_state, _):
            minibatches = new_transitions.prepare_minibatches(
                jax.random.fold_in(carry_state.prng, carry_state.steps), config.num_minibatches, config.batch_size
            )

            def minibatch_scan_fn(ms, mb):
                return ms._update_critic_only(mb, jax.random.fold_in(ms.prng, ms.steps + 1))

            return jax.lax.scan(minibatch_scan_fn, init=carry_state, xs=minibatches)

        state_after_v, all_v_metrics = jax.lax.scan(critic_epoch_step, init=state, length=config.num_updates_per_batch)
        current_global_v_loss = jnp.mean(all_v_metrics["v_loss/total"])

        batch_reward = jnp.mean(transitions.reward)
        new_state = jdc.replace(
            state_after_v,
            ema_max_adv=new_ema_max_adv,  # 👑 Record global Z anchor
            ema_reward=config.beta_r * state_after_v.ema_reward + (1.0 - config.beta_r) * batch_reward,
            ema_reward_sq=config.beta_r * state_after_v.ema_reward_sq + (1.0 - config.beta_r) * jnp.square(
                batch_reward),
            ema_v_loss=config.beta_v * state_after_v.ema_v_loss + (1.0 - config.beta_v) * current_global_v_loss,
            ema_v_loss_sq=config.beta_v * state_after_v.ema_v_loss_sq + (1.0 - config.beta_v) * jnp.square(
                current_global_v_loss)
        )

        def actor_epoch_step(carry_state, _):
            minibatches = new_transitions.prepare_minibatches(
                jax.random.fold_in(carry_state.prng, carry_state.steps), config.num_minibatches, config.batch_size
            )

            def minibatch_scan_fn(ms, mb):
                return ms._update_actor_only(mb, jax.random.fold_in(ms.prng, ms.steps + 2), current_global_v_loss,
                                             new_state.ema_max_adv)

            return jax.lax.scan(minibatch_scan_fn, init=carry_state, xs=minibatches)

        final_state, all_p_metrics = jax.lax.scan(actor_epoch_step, init=new_state, length=config.num_updates_per_batch)

        return final_state, {**all_v_metrics, **all_p_metrics}