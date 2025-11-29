"""Flow Matching policy - Direct copy from FPO implementation."""

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp
from typing import NamedTuple

from flow_policy.networks import MlpWeights
from . import math_utils, networks


class FlowSchedule(NamedTuple):
    """Flow schedule from FPO."""
    t_current: Array
    t_next: Array


@jdc.pytree_dataclass
class DecoderFMConfig:
    """Configuration for Flow Matching - matching FPO."""

    # Flow parameters
    flow_steps: jdc.Static[int] = 10
    timestep_embed_dim: jdc.Static[int] = 8

    # Network architecture - increased for supervised learning
    hidden_dims: jdc.Static[tuple[int, ...]] = (64, 64, 64, 64)
    policy_output_scale: float = 1.0  # Changed from 0.25 for supervised learning

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: jdc.Static[int] = 8192
    num_epochs: jdc.Static[int] = 50
    n_samples_per_action: jdc.Static[int] = 8  # FPO default

    # Data parameters
    normalize_observations: jdc.Static[bool] = True

    # SDE parameters from FPO (usually 0)
    sde_sigma: float = 0.0
    feather_std: float = 0.0


@jdc.pytree_dataclass
class DecoderFMState:
    """Flow Matching model state - based on FPO."""

    config: DecoderFMConfig
    params: MlpWeights  # Flow network parameters
    obs_stats: math_utils.RunningStats  # Observation statistics
    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState
    prng: Array
    steps: Array

    @staticmethod
    def init(
        prng: Array,
        obs_dim: int,
        action_dim: int,
        config: DecoderFMConfig
    ) -> DecoderFMState:
        """Initialize FM state - matching FPO."""

        prng0, prng1 = jax.random.split(prng)

        # Network takes: [obs, action, timestep_embed] as input - LIKE FPO!
        input_dim = obs_dim + action_dim + config.timestep_embed_dim

        # Build network architecture - small like FPO
        layer_dims = (input_dim,) + config.hidden_dims + (action_dim,)
        flow_net = networks.mlp_init(prng0, layer_dims)

        # Initialize optimizer
        opt = optax.adam(config.learning_rate)

        return DecoderFMState(
            config=config,
            params=flow_net,
            obs_stats=math_utils.RunningStats.init((obs_dim,)),
            opt=opt,
            opt_state=opt.init(flow_net),
            prng=prng1,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def get_schedule(self) -> FlowSchedule:
        """Get flow schedule - copied from FPO."""
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        t_current = full_t_path[:-1]
        return FlowSchedule(
            t_current=t_current,
            t_next=full_t_path[1:],
        )

    def embed_timestep(self, t: Array) -> Array:
        """Embed timestep - copied from FPO.

        Args:
            t: Timestep, shape (*, 1)

        Returns:
            Embedded timestep, shape (*, timestep_embed_dim)
        """
        assert t.shape[-1] == 1, f"Expected (..., 1), got {t.shape}"
        freqs = jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * (2 ** freqs[None, :])
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def flow_forward(
        self,
        obs_norm: Array,
        x_t: Array,
        t_embed: Array,
    ) -> Array:
        """Forward pass through flow network - like FPO.

        Args:
            obs_norm: Normalized observations, shape (*, obs_dim)
            x_t: Current state, shape (*, action_dim)
            t_embed: Embedded timestep, shape (*, timestep_embed_dim)

        Returns:
            Velocity field, shape (*, action_dim)
        """
        # Use networks.flow_mlp_fwd like FPO
        velocity = networks.flow_mlp_fwd(
            self.params,
            obs_norm,
            x_t,
            t_embed,
        )
        return velocity * self.config.policy_output_scale

    def sample_action(
        self,
        obs: Array,
        prng: Array,
        deterministic: bool = False,
    ) -> Array:
        """Sample action - copied from FPO's sample_action."""

        # Normalize observations if needed
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        else:
            obs_norm = obs

        # Handle batch dimensions
        single_obs = obs.ndim == 1
        if single_obs:
            obs_norm = obs_norm[None, :]

        (*batch_dims, obs_dim) = obs_norm.shape
        action_dim = self.params[-1][0].shape[-1]

        # Define euler step - copied from FPO
        def euler_step(
            carry: Array, inputs: tuple[FlowSchedule, Array]
        ) -> tuple[Array, Array]:
            x_t = carry
            assert x_t.shape == (*batch_dims, action_dim)
            schedule_t, noise = inputs
            assert schedule_t.t_current.shape == ()
            assert schedule_t.t_next.shape == ()
            assert noise.shape == x_t.shape

            # Compute dt as the difference between current and next timestep
            dt = schedule_t.t_next - schedule_t.t_current

            # Get velocity from flow model
            velocity = self.flow_forward(
                obs_norm,
                x_t,
                jnp.broadcast_to(
                    self.embed_timestep(schedule_t.t_current[None]),
                    (*batch_dims, self.config.timestep_embed_dim),
                ),
            )

            # SDE step with optional noise
            x_t_next = x_t + dt * velocity + self.config.sde_sigma * noise
            assert x_t_next.shape == x_t.shape
            return x_t_next, x_t

        # Split random keys
        prng_sample, prng_noise, prng_feather = jax.random.split(prng, num=3)

        # Generate noise path
        noise_path = jax.random.normal(
            prng_noise,
            (self.config.flow_steps, *batch_dims, action_dim),
        )

        # Run integration using scan
        x0, x_t_path = jax.lax.scan(
            euler_step,
            init=jax.random.normal(prng_sample, (*batch_dims, action_dim)),
            xs=(self.get_schedule(), noise_path),
        )

        # Add perturbation if not deterministic
        if not deterministic:
            perturb = (
                jax.random.normal(prng_feather, (*batch_dims, action_dim))
                * self.config.feather_std
            )
            x0 = x0 + perturb

        # No clipping - FPO doesn't clip actions
        action = x0

        if single_obs:
            action = action.squeeze(0)

        return action

    def sample_action_from_z(
        self, obs: Array, z: Array, prng: Array, deterministic: bool = True
    ) -> Array:
        """Sample action starting from given z instead of N(0,I).

        Args:
            obs: Observation
            z: Starting latent variable from PPO_z
            prng: Random key for noise generation
            deterministic: Whether to add feather noise

        Returns:
            Action sampled from the flow model
        """
        # Handle single observation
        single_obs = obs.ndim == 1

        # Normalize observation
        obs_norm = (obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        if single_obs:
            obs_norm = obs_norm[None, :]
            z = z[None, :]

        (*batch_dims, obs_dim) = obs_norm.shape
        action_dim = self.params[-1][0].shape[-1]

        # Define euler step - same as in sample_action
        def euler_step(
            carry: Array, inputs: tuple[FlowSchedule, Array]
        ) -> tuple[Array, Array]:
            x_t = carry
            assert x_t.shape == (*batch_dims, action_dim)
            schedule_t, noise = inputs
            assert schedule_t.t_current.shape == ()
            assert schedule_t.t_next.shape == ()
            assert noise.shape == x_t.shape

            # Compute dt as the difference between current and next timestep
            dt = schedule_t.t_next - schedule_t.t_current

            # Get velocity from flow model
            velocity = self.flow_forward(
                obs_norm,
                x_t,
                jnp.broadcast_to(
                    self.embed_timestep(schedule_t.t_current[None]),
                    (*batch_dims, self.config.timestep_embed_dim),
                ),
            )

            # SDE step with optional noise
            x_t_next = x_t + dt * velocity + self.config.sde_sigma * noise
            assert x_t_next.shape == x_t.shape
            return x_t_next, x_t

        # Split random keys for noise generation
        prng_noise, prng_feather = jax.random.split(prng, num=2)

        # Generate noise path
        noise_path = jax.random.normal(
            prng_noise,
            (self.config.flow_steps, *batch_dims, action_dim),
        )

        # Run integration using scan, starting from z instead of N(0,I)
        x0, x_t_path = jax.lax.scan(
            euler_step,
            init=z,  # Use z as starting point instead of sampling from N(0,I)
            xs=(self.get_schedule(), noise_path),
        )

        # Add perturbation if not deterministic
        if not deterministic:
            perturb = (
                jax.random.normal(prng_feather, (*batch_dims, action_dim))
                * self.config.feather_std
            )
            x0 = x0 + perturb

        # No clipping - FPO doesn't clip actions
        action = x0

        if single_obs:
            action = action.squeeze(0)

        return action

    def compute_cfm_loss(
        self,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
    ) -> Array:
        """Compute CFM loss - copied from FPO._compute_cfm_loss.

        Args:
            obs_norm: Normalized observations, shape (batch, obs_dim)
            action: Target actions, shape (batch, action_dim)
            eps: Noise samples, shape (batch, n_samples, action_dim)
            t: Time samples, shape (batch, n_samples, 1)

        Returns:
            Loss values, shape (batch, n_samples)
        """
        (*batch_dims, action_dim) = action.shape
        samples_dim = self.config.n_samples_per_action

        assert eps.shape == (*batch_dims, samples_dim, action_dim)
        assert t.shape == (*batch_dims, samples_dim, 1)

        # Interpolate between action and noise
        x_t = t * eps + (1.0 - t) * action[..., None, :]

        # Get network prediction
        obs_dim = obs_norm.shape[-1]
        sample_shape = (*batch_dims, samples_dim)

        velocity_pred = self.flow_forward(
            jnp.broadcast_to(obs_norm[..., None, :], (*sample_shape, obs_dim)),
            x_t,
            self.embed_timestep(t),
        )

        # Target velocity
        velocity_gt = eps - action[..., None, :]  # u = x1 - x0

        # MSE loss
        loss = jnp.mean((velocity_pred - velocity_gt) ** 2, axis=-1)

        assert loss.shape == (*batch_dims, samples_dim)
        return loss

    def train_step(
        self,
        batch_obs: Array,
        batch_actions: Array,
    ) -> tuple[DecoderFMState, dict[str, Array]]:
        """Training step - based on FPO's training logic."""

        batch_size = batch_obs.shape[0]
        action_dim = batch_actions.shape[1]

        # Normalize observations
        if self.config.normalize_observations:
            obs_norm = (batch_obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        else:
            obs_norm = batch_obs

        # Sample random noise and timesteps
        prng_eps, prng_t, self_prng = jax.random.split(self.prng, 3)

        # Sample eps and t for each action (like FPO)
        eps = jax.random.normal(
            prng_eps,
            (batch_size, self.config.n_samples_per_action, action_dim)
        )
        t = jax.random.uniform(
            prng_t,
            (batch_size, self.config.n_samples_per_action, 1)
        )

        def loss_fn(params):
            # Create state with new params
            state_with_params = jdc.replace(self, params=params)

            # Compute CFM loss
            cfm_loss = state_with_params.compute_cfm_loss(
                obs_norm,
                batch_actions,
                eps,
                t
            )

            # Average over samples and batch
            loss = jnp.mean(cfm_loss)

            metrics = {
                "loss": loss,
                "velocity_mean": 0.0,  # Placeholder
                "velocity_std": 0.0,   # Placeholder
            }

            return loss, metrics

        # Compute gradients and update
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.params
        )

        updates, new_opt_state = self.opt.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.params, updates)

        # Update state
        with jdc.copy_and_mutate(self) as new_state:
            new_state.params = new_params
            new_state.opt_state = new_opt_state
            new_state.prng = self_prng
            new_state.steps = self.steps + 1

        return new_state, metrics