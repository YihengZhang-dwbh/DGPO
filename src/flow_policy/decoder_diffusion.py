"""Diffusion policy using DDPM/DDIM - Decoder replacement for Flow Matching."""

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp
from typing import NamedTuple

from flow_policy.networks import MlpWeights
from . import math_utils, networks


class DiffusionSchedule(NamedTuple):
    """Diffusion schedule (similar to FlowSchedule)."""
    t_current: Array  # Current timestep
    t_next: Array     # Next timestep
    alpha_t: Array    # alpha_t for current timestep
    alpha_next: Array # alpha_t for next timestep


@jdc.pytree_dataclass
class DecoderDiffusionConfig:
    """Configuration for Diffusion - matching FM interface."""

    # Diffusion parameters
    diffusion_steps: jdc.Static[int] = 10  # Same as FM's flow_steps
    timestep_embed_dim: jdc.Static[int] = 8

    # Network architecture - 4 layers, width 64 (matching paper)
    hidden_dims: jdc.Static[tuple[int, ...]] = (64, 64, 64, 64)
    policy_output_scale: float = 1.0  # Changed from 0.25 for supervised learning

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: jdc.Static[int] = 8192
    num_epochs: jdc.Static[int] = 50
    n_samples_per_action: jdc.Static[int] = 8

    # Data parameters
    normalize_observations: jdc.Static[bool] = True

    # Diffusion-specific parameters
    beta_schedule: jdc.Static[str] = "cosine"  # linear or cosine (cosine recommended for fewer steps)
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # SDE parameter (same as FM)
    sde_sigma: float = 0.0  # 0.0 for deterministic sampling (DDIM)
    feather_std: float = 0.0


@jdc.pytree_dataclass
class DecoderDiffusionState:
    """Diffusion model state - matching FMState interface."""

    config: DecoderDiffusionConfig
    params: MlpWeights  # Denoising network parameters
    obs_stats: math_utils.RunningStats
    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState
    prng: Array
    steps: Array

    @staticmethod
    def init(
        prng: Array,
        obs_dim: int,
        action_dim: int,
        config: DecoderDiffusionConfig
    ) -> DecoderDiffusionState:
        """Initialize Diffusion state - same as FMState.init."""

        prng0, prng1 = jax.random.split(prng)

        # Network takes: [obs, action, timestep_embed] as input - SAME AS FM!
        input_dim = obs_dim + action_dim + config.timestep_embed_dim

        # Build network architecture - same as FM
        layer_dims = (input_dim,) + config.hidden_dims + (action_dim,)
        denoise_net = networks.mlp_init(prng0, layer_dims)

        # Initialize optimizer
        opt = optax.adam(config.learning_rate)

        return DecoderDiffusionState(
            config=config,
            params=denoise_net,
            obs_stats=math_utils.RunningStats.init((obs_dim,)),
            opt=opt,
            opt_state=opt.init(denoise_net),
            prng=prng1,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def get_beta_schedule(self) -> tuple[Array, Array, Array]:
        """Compute beta schedule and derived quantities.

        Returns:
            betas: Beta values, shape (diffusion_steps,)
            alphas: Alpha values (1 - beta), shape (diffusion_steps,)
            alphas_cumprod: Cumulative product of alphas, shape (diffusion_steps,)
        """
        steps = self.config.diffusion_steps

        if self.config.beta_schedule == "linear":
            # Linear schedule
            betas = jnp.linspace(self.config.beta_start, self.config.beta_end, steps)
            alphas = 1.0 - betas
            alphas_cumprod = jnp.cumprod(alphas)
        elif self.config.beta_schedule == "cosine":
            # Cosine schedule (from Improved DDPM paper)
            # Directly define alphas_cumprod trajectory
            s = 0.008
            t = jnp.linspace(0, steps, steps + 1) / steps
            alphas_cumprod_full = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
            alphas_cumprod_full = alphas_cumprod_full / alphas_cumprod_full[0]

            # Take only the timesteps we need [0, 1, ..., T-1]
            alphas_cumprod = alphas_cumprod_full[:-1]

            # Derive betas from alphas_cumprod
            # alpha_t = alpha_cumprod[t] / alpha_cumprod[t-1]
            # For t=0, alpha_cumprod[-1] is treated as 1.0
            alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), alphas_cumprod[:-1]])
            alphas = alphas_cumprod / alphas_cumprod_prev
            betas = 1.0 - alphas
            betas = jnp.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.config.beta_schedule}")

        return betas, alphas, alphas_cumprod

    def get_schedule(self) -> DiffusionSchedule:
        """Get diffusion schedule - matching FM's get_schedule interface.

        For diffusion_steps=T, DDIM sampling requires T-1 denoising steps:
        x_{T-1} → x_{T-2} → ... → x_1 → x_0

        Returns schedule with (diffusion_steps - 1) elements.
        """
        # Get alphas_cumprod
        _, _, alphas_cumprod = self.get_beta_schedule()

        # Timesteps for denoising: T-1 → T-2 → ... → 1 → 0 (0-indexed)
        timesteps = jnp.arange(self.config.diffusion_steps - 1, -1, -1)  # [T-1, T-2, ..., 1, 0]
        t_current = timesteps[:-1]  # [T-1, T-2, ..., 1], length: T-1
        t_next = timesteps[1:]      # [T-2, T-3, ..., 0], length: T-1

        # Alpha values at current timesteps
        alpha_t = alphas_cumprod[t_current]  # Length: T-1

        # Alpha values at next timesteps
        # For t=0, alpha should be 1.0 (clean data, no noise)
        alpha_next_from_schedule = alphas_cumprod[t_next]
        alpha_next = jnp.where(t_next == 0, 1.0, alpha_next_from_schedule)  # Length: T-1

        return DiffusionSchedule(
            t_current=t_current.astype(jnp.float32),
            t_next=t_next.astype(jnp.float32),
            alpha_t=alpha_t,
            alpha_next=alpha_next,
        )

    def embed_timestep(self, t: Array) -> Array:
        """Embed timestep - same as FM.

        Args:
            t: Timestep, shape (*, 1)

        Returns:
            Embedded timestep, shape (*, timestep_embed_dim)
        """
        assert t.shape[-1] == 1, f"Expected (..., 1), got {t.shape}"
        # Normalize to [0, 1]
        t_normalized = t / self.config.diffusion_steps
        freqs = jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t_normalized * (2 ** freqs[None, :])
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def denoise_forward(
        self,
        obs_norm: Array,
        x_t: Array,
        t_embed: Array,
    ) -> Array:
        """Forward pass through denoising network - matching FM's flow_forward.

        Args:
            obs_norm: Normalized observations, shape (*, obs_dim)
            x_t: Noisy state, shape (*, action_dim)
            t_embed: Embedded timestep, shape (*, timestep_embed_dim)

        Returns:
            Predicted noise, shape (*, action_dim)
        """
        # Use networks.flow_mlp_fwd - same as FM
        noise_pred = networks.flow_mlp_fwd(
            self.params,
            obs_norm,
            x_t,
            t_embed,
        )
        return noise_pred * self.config.policy_output_scale

    def sample_action(
        self,
        obs: Array,
        prng: Array,
        deterministic: bool = False,
    ) -> Array:
        """Sample action - matching FM's interface, using DDIM sampling."""

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

        # DDIM sampling step (deterministic when sde_sigma=0)
        def ddim_step(
            carry: Array, inputs: tuple[DiffusionSchedule, Array]
        ) -> tuple[Array, Array]:
            x_t = carry
            schedule_t, noise = inputs

            # Get noise prediction
            t_embed = jnp.broadcast_to(
                self.embed_timestep(schedule_t.t_current[None, None]),
                (*batch_dims, self.config.timestep_embed_dim),
            )
            noise_pred = self.denoise_forward(obs_norm, x_t, t_embed)

            # DDIM update: x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1 - alpha_{t-1}) * noise_pred
            # Where x_0_pred = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
            sqrt_alpha_t = jnp.sqrt(schedule_t.alpha_t)
            sqrt_one_minus_alpha_t = jnp.sqrt(1 - schedule_t.alpha_t)
            sqrt_alpha_next = jnp.sqrt(schedule_t.alpha_next)
            sqrt_one_minus_alpha_next = jnp.sqrt(1 - schedule_t.alpha_next)

            # Predict x_0
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

            # Compute x_{t-1}
            x_t_next = sqrt_alpha_next * x_0_pred + sqrt_one_minus_alpha_next * noise_pred

            # Add stochastic noise (DDPM-style, automatically 0 when sde_sigma=0)
            x_t_next = x_t_next + self.config.sde_sigma * noise

            return x_t_next, x_t

        # Split random keys
        prng_sample, prng_noise, prng_feather = jax.random.split(prng, num=3)

        # Generate noise path
        noise_path = jax.random.normal(
            prng_noise,
            (self.config.diffusion_steps - 1, *batch_dims, action_dim),
        )

        # Start from N(0,I) - matching FM's interface
        x_T = jax.random.normal(prng_sample, (*batch_dims, action_dim))

        # Run denoising using scan
        x_0, x_t_path = jax.lax.scan(
            ddim_step,
            init=x_T,
            xs=(self.get_schedule(), noise_path),
        )

        # Add perturbation (feather noise, automatically 0 when feather_std=0 or deterministic=True)
        # Note: deterministic is a Python bool (static), so this if is safe
        if not deterministic:
            perturb = (
                jax.random.normal(prng_feather, (*batch_dims, action_dim))
                * self.config.feather_std
            )
            x_0 = x_0 + perturb

        # No clipping - same as FM
        action = x_0

        if single_obs:
            action = action.squeeze(0)

        return action

    def sample_action_from_z(
        self, obs: Array, z: Array, prng: Array, deterministic: bool = True
    ) -> Array:
        """Sample action starting from z - MATCHING FM's KEY METHOD.

        Args:
            obs: Observation
            z: Starting latent variable from PPO_z
            prng: Random key
            deterministic: Whether to add feather noise

        Returns:
            Action sampled from the diffusion model
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

        # DDIM sampling step - same as sample_action
        def ddim_step(
            carry: Array, inputs: tuple[DiffusionSchedule, Array]
        ) -> tuple[Array, Array]:
            x_t = carry
            schedule_t, noise = inputs

            # Get noise prediction
            t_embed = jnp.broadcast_to(
                self.embed_timestep(schedule_t.t_current[None, None]),
                (*batch_dims, self.config.timestep_embed_dim),
            )
            noise_pred = self.denoise_forward(obs_norm, x_t, t_embed)

            # DDIM update
            sqrt_alpha_t = jnp.sqrt(schedule_t.alpha_t)
            sqrt_one_minus_alpha_t = jnp.sqrt(1 - schedule_t.alpha_t)
            sqrt_alpha_next = jnp.sqrt(schedule_t.alpha_next)
            sqrt_one_minus_alpha_next = jnp.sqrt(1 - schedule_t.alpha_next)

            # Predict x_0
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

            # Compute x_{t-1}
            x_t_next = sqrt_alpha_next * x_0_pred + sqrt_one_minus_alpha_next * noise_pred

            # Add stochastic noise (automatically 0 when sde_sigma=0)
            x_t_next = x_t_next + self.config.sde_sigma * noise

            return x_t_next, x_t

        # Split random keys
        prng_noise, prng_feather = jax.random.split(prng, num=2)

        # Generate noise path
        noise_path = jax.random.normal(
            prng_noise,
            (self.config.diffusion_steps - 1, *batch_dims, action_dim),
        )

        # KEY: Start from z instead of N(0,I) - SAME AS FM!
        x_0, x_t_path = jax.lax.scan(
            ddim_step,
            init=z,  # Use z as starting point
            xs=(self.get_schedule(), noise_path),
        )

        # Add perturbation (feather noise, automatically 0 when feather_std=0 or deterministic=True)
        # Note: deterministic is a Python bool (static), so this if is safe
        if not deterministic:
            perturb = (
                jax.random.normal(prng_feather, (*batch_dims, action_dim))
                * self.config.feather_std
            )
            x_0 = x_0 + perturb

        # No clipping - same as FM
        action = x_0

        if single_obs:
            action = action.squeeze(0)

        return action

    def compute_ddpm_loss(
        self,
        obs_norm: Array,
        action: Array,
        noise: Array,
        t: Array,
    ) -> Array:
        """Compute DDPM denoising loss - replacing FM's compute_cfm_loss.

        Args:
            obs_norm: Normalized observations, shape (batch, obs_dim)
            action: Target actions (x_0), shape (batch, action_dim)
            noise: Noise samples, shape (batch, n_samples, action_dim)
            t: Timesteps (integers in [0, diffusion_steps]), shape (batch, n_samples, 1)

        Returns:
            Loss values, shape (batch, n_samples)
        """
        (*batch_dims, action_dim) = action.shape
        samples_dim = self.config.n_samples_per_action

        assert noise.shape == (*batch_dims, samples_dim, action_dim)
        assert t.shape == (*batch_dims, samples_dim, 1)

        # Get alpha_t for the sampled timesteps
        _, _, alphas_cumprod = self.get_beta_schedule()

        # t is in [0, diffusion_steps], clamp to valid range
        t_int = jnp.clip(t.astype(jnp.int32), 0, self.config.diffusion_steps - 1)
        alpha_t = alphas_cumprod[t_int[..., 0]]  # shape: (batch, n_samples)

        # Compute x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        sqrt_alpha_t = jnp.sqrt(alpha_t)[..., None]  # (batch, n_samples, 1)
        sqrt_one_minus_alpha_t = jnp.sqrt(1 - alpha_t)[..., None]

        x_t = sqrt_alpha_t * action[..., None, :] + sqrt_one_minus_alpha_t * noise

        # Get network prediction
        obs_dim = obs_norm.shape[-1]
        sample_shape = (*batch_dims, samples_dim)

        noise_pred = self.denoise_forward(
            jnp.broadcast_to(obs_norm[..., None, :], (*sample_shape, obs_dim)),
            x_t,
            self.embed_timestep(t),
        )

        # MSE loss: predict noise
        loss = jnp.mean((noise_pred - noise) ** 2, axis=-1)

        assert loss.shape == (*batch_dims, samples_dim)
        return loss

    def train_step(
        self,
        batch_obs: Array,
        batch_actions: Array,
    ) -> tuple[DecoderDiffusionState, dict[str, Array]]:
        """Training step - matching FMState.train_step interface."""

        batch_size = batch_obs.shape[0]
        action_dim = batch_actions.shape[1]

        # Normalize observations
        if self.config.normalize_observations:
            obs_norm = (batch_obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        else:
            obs_norm = batch_obs

        # Sample random noise and timesteps
        prng_noise, prng_t, self_prng = jax.random.split(self.prng, 3)

        # Sample noise and timesteps for each action
        noise = jax.random.normal(
            prng_noise,
            (batch_size, self.config.n_samples_per_action, action_dim)
        )
        # Timesteps in [0, diffusion_steps]
        t = jax.random.randint(
            prng_t,
            (batch_size, self.config.n_samples_per_action, 1),
            0,
            self.config.diffusion_steps
        ).astype(jnp.float32)

        def loss_fn(params):
            # Create state with new params
            state_with_params = jdc.replace(self, params=params)

            # Compute DDPM loss
            ddpm_loss = state_with_params.compute_ddpm_loss(
                obs_norm,
                batch_actions,
                noise,
                t
            )

            # Average over samples and batch
            loss = jnp.mean(ddpm_loss)

            metrics = {
                "loss": loss,
                "noise_pred_mean": 0.0,  # Placeholder
                "noise_pred_std": 0.0,   # Placeholder
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
