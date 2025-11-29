from __future__ import annotations

import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
import optax
from jax import Array
from jax import numpy as jnp

from flow_policy.networks import MlpWeights

from . import math_utils, networks, rollouts


@jdc.pytree_dataclass
class EncoderConfig:
    # Based on Brax PPO config, but outputs z instead of actions:
    action_repeat: jdc.Static[int]
    batch_size: jdc.Static[int]
    discounting: float
    entropy_cost: float
    episode_length: int
    learning_rate: float
    normalize_observations: jdc.Static[bool]
    num_envs: jdc.Static[int]
    num_evals: jdc.Static[int]
    num_minibatches: jdc.Static[int]
    num_timesteps: jdc.Static[int]
    num_updates_per_batch: jdc.Static[int]
    reward_scaling: float
    unroll_length: jdc.Static[int]

    # New parameter for latent dimension
    z_dim: jdc.Static[int] = 6

    gae_lambda: float = 0.95
    normalize_advantage: jdc.Static[bool] = True
    clipping_epsilon: float = 0.15  # Balanced: stable but allows reasonable exploration

    value_loss_coeff: float = 0.25

    # Z regularization to prevent explosion
    z_regularization: float = 0.0

    # Gradient clipping (FPO uses 0.5)
    max_grad_norm: jdc.Static[float] = 0.5

    # Entropy constraint strategy (optional, same as V-MPO_z)
    use_tanh_jacobian_for_z: jdc.Static[bool] = False  # Default: no tanh correction (original behavior)

    def __post_init__(self):
        assert self.action_repeat == 1  # "action repeat is dumb" - Kevin (?)

    @property
    def iterations_per_env(self) -> int:
        """Number of iterations (=policy forward passes) per environment at the
        start of each training step."""
        return (
            self.num_minibatches * self.batch_size * self.unroll_length
        ) // self.num_envs


@jdc.pytree_dataclass
class ActorCriticParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class EncoderActionInfo:
    log_prob: Array
    mean: Array
    std: Array


PpoZTransition = rollouts.TransitionStruct[EncoderActionInfo]


@jdc.pytree_dataclass
class EncoderState:
    """PPO-Z agent state that outputs latent z instead of actions."""

    env: jdc.Static[mjp.MjxEnv]
    config: EncoderConfig
    params: ActorCriticParams
    obs_stats: math_utils.RunningStats
    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState
    prng: Array
    steps: Array

    @staticmethod
    @jdc.jit
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: EncoderConfig) -> EncoderState:
        obs_size = env.observation_size
        z_dim = config.z_dim  # Use z_dim instead of action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        # Actor outputs z distribution (mean and std for z_dim dimensions)
        actor_net = networks.mlp_init(
            prng0, (obs_size, 32, 32, 32, 32, z_dim * 2)
        )
        critic_net = networks.mlp_init(prng1, (obs_size, 256, 256, 256, 256, 256, 1))

        network_params = ActorCriticParams(actor_net, critic_net)

        # We'll manage learning rate ourselves!
        opt = optax.scale_by_adam()
        return EncoderState(
            env=env,
            config=config,
            params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt=opt,
            opt_state=opt.init(network_params),  # type: ignore
            prng=prng2,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def sample_z(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, EncoderActionInfo]:
        """Sample a latent z from the policy given an observation."""
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = obs
        z_dist = networks.gaussian_policy_fwd(self.params.policy, obs_norm)
        if deterministic:
            # Use deterministic z during evaluation.
            return z_dist.loc, EncoderActionInfo(
                log_prob=jnp.zeros_like(z_dist.loc[..., 0]),
                mean=z_dist.loc,
                std=z_dist.scale,
            )
        z = z_dist.sample(prng)
        log_probs = jnp.sum(z_dist.log_prob(z), axis=-1)
        return z, EncoderActionInfo(
            log_prob=log_probs,
            mean=z_dist.loc,
            std=z_dist.scale,
        )

    @jdc.jit
    def training_step(
        self, transitions: PpoZTransition
    ) -> tuple[EncoderState, dict[str, Array]]:
        # We're use a (T, B) shape convention, corresponding to a "scan of the
        # vmap" and not a "vmap of the scan".
        config = self.config
        assert transitions.reward.shape == (config.iterations_per_env, config.num_envs)
        # Update observation statistics and store them in the state.
        if self.config.normalize_observations:
            with jdc.copy_and_mutate(self) as state:
                state.obs_stats = self.obs_stats.update(transitions.obs)
        else:
            state = self
        del self

        def step_batch(state: EncoderState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                lambda state, minibatch: state._step_minibatch(
                    minibatch,
                    # state.steps will increment after each minibatch.
                    entropy_prng=jax.random.fold_in(step_prng, state.steps),
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
        self,
        transitions: PpoZTransition,
        entropy_prng: Array,
    ) -> tuple[EncoderState, dict[str, Array]]:
        """One training step over a minibatch of transitions."""

        assert transitions.reward.shape == (
            self.config.unroll_length,
            self.config.batch_size,
        )
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: EncoderState._compute_ppo_loss(
                jdc.replace(self, params=params),
                transitions,
                entropy_prng,
            ),
            has_aux=True,
        )(self.params)
        assert isinstance(grads, ActorCriticParams)
        assert isinstance(loss, Array)
        assert isinstance(metrics, dict)

        # Track detailed gradient metrics
        grad_norm_before = optax.global_norm(grads)
        metrics["grad_norm_before_clip"] = grad_norm_before
        metrics["policy_grad_norm"] = optax.global_norm(grads.policy)
        metrics["value_grad_norm"] = optax.global_norm(grads.value)

        # Track per-layer gradient metrics for the first and last layers
        metrics["policy_first_layer_grad_norm"] = jnp.linalg.norm(grads.policy[0][0])
        metrics["policy_last_layer_grad_norm"] = jnp.linalg.norm(grads.policy[-1][0])
        metrics["value_first_layer_grad_norm"] = jnp.linalg.norm(grads.value[0][0])
        metrics["value_last_layer_grad_norm"] = jnp.linalg.norm(grads.value[-1][0])

        # Apply gradient clipping (like FPO)
        if self.config.max_grad_norm > 0.0:
            scale = jnp.minimum(1.0, self.config.max_grad_norm / (grad_norm_before + 1e-8))
            grads = jax.tree.map(lambda g: g * scale, grads)
            metrics["grad_norm"] = optax.global_norm(grads)
            metrics["grad_clipped"] = grad_norm_before > self.config.max_grad_norm
        else:
            metrics["grad_norm"] = grad_norm_before
            metrics["grad_clipped"] = jnp.array(False)

        param_update, new_opt_state = self.opt.update(grads, self.opt_state)  # type: ignore
        param_update = jax.tree.map(
            lambda x: -self.config.learning_rate * x, param_update
        )
        with jdc.copy_and_mutate(self) as state:
            state.params = jax.tree.map(jnp.add, self.params, param_update)
            state.opt_state = new_opt_state
            state.steps = state.steps + 1
        return state, metrics

    def _compute_ppo_loss(
        self, transitions: PpoZTransition, entropy_prng: Array
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
            self.config.z_dim,  # transitions.action stores z, not actions
        )

        metrics = dict[str, Array]()

        if self.config.normalize_observations:
            obs_norm = (transitions.obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = transitions.obs
        value_pred = networks.value_mlp_fwd(self.params.value, obs_norm)
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

        # Now we're computing log probs for z, not actions
        z_dist = networks.gaussian_policy_fwd(self.params.policy, obs_norm)
        new_log_probs = jnp.sum(z_dist.log_prob(transitions.action), axis=-1)
        old_log_probs = transitions.action_info.log_prob
        assert new_log_probs.shape == old_log_probs.shape

        # Log z distribution statistics
        metrics["z_mean"] = jnp.mean(z_dist.loc)
        metrics["z_std"] = jnp.mean(z_dist.scale)
        metrics["z_min"] = jnp.min(transitions.action)  # transitions.action contains z
        metrics["z_max"] = jnp.max(transitions.action)
        metrics["log_prob_diff"] = jnp.mean(new_log_probs - old_log_probs)
        metrics["log_prob_ratio"] = jnp.mean(jnp.exp(new_log_probs - old_log_probs))

        rho_s = jnp.exp(new_log_probs - old_log_probs)
        surrogate_loss1 = rho_s * gae_advantages
        surrogate_loss2 = (
            jnp.clip(
                rho_s,
                1 - self.config.clipping_epsilon,
                1 + self.config.clipping_epsilon,
            )
            * gae_advantages
        )

        # Log clipping statistics
        metrics["clipped_ratio_mean"] = jnp.mean(
            jnp.abs(rho_s - 1.0) > self.config.clipping_epsilon
        )
        metrics["policy_ratio_mean"] = jnp.mean(rho_s)
        metrics["policy_ratio_min"] = jnp.min(rho_s)
        metrics["policy_ratio_max"] = jnp.max(rho_s)

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
        metrics["policy_loss"] = policy_loss
        metrics["surrogate_loss1_mean"] = jnp.mean(surrogate_loss1)
        metrics["surrogate_loss2_mean"] = jnp.mean(surrogate_loss2)

        # Don't supervise value function on truncated timesteps.
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)

        # Value function statistics
        metrics["value_mean"] = jnp.mean(value_pred)
        metrics["value_std"] = jnp.std(value_pred)
        metrics["value_min"] = jnp.min(value_pred)
        metrics["value_max"] = jnp.max(value_pred)
        metrics["value_target_mean"] = jnp.mean(gae_vs)
        metrics["value_error_mean"] = jnp.mean(v_error)
        metrics["value_error_std"] = jnp.std(v_error)

        v_loss = jnp.mean(v_error**2) * self.config.value_loss_coeff
        metrics["v_loss"] = v_loss

        # Entropy calculation with optional tanh_jacobian (same as V-MPO_z)
        if self.config.use_tanh_jacobian_for_z:
            # Optional: Add tanh_jacobian for stronger constraint (like V-MPO_z)
            entropy = jnp.mean(
                jnp.sum(
                    z_dist.entropy()
                    + math_utils.tanh_log_det_jacobian(z_dist.sample(entropy_prng)),
                    axis=-1,
                )
            )
        else:
            # Default: No tanh correction (original behavior)
            entropy = jnp.mean(jnp.sum(z_dist.entropy(), axis=-1))

        metrics["entropy"] = entropy
        entropy_loss = -self.config.entropy_cost * entropy
        metrics["entropy_loss"] = entropy_loss

        # Compute the total loss that will be used for optimization
        total_loss = policy_loss + v_loss + entropy_loss

        # Add z regularization to prevent explosion
        # CRITICAL: Regularize z_dist parameters (loc, scale), NOT sampled z values!
        # Sampled values are constants and have no gradient w.r.t. policy params
        z_reg_loss = self.config.z_regularization * (
            jnp.mean(jnp.square(z_dist.loc)) +  # Penalize mean
            jnp.mean(jnp.square(z_dist.scale))  # Penalize std
        )
        total_loss = total_loss + z_reg_loss
        metrics["z_reg_loss"] = z_reg_loss
        metrics["z_l2_norm"] = jnp.sqrt(jnp.mean(jnp.square(transitions.action)))

        metrics["total_loss"] = total_loss

        return total_loss, metrics
