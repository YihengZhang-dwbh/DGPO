"""Create an identity Diffusion model for first iteration.

This creates a Diffusion checkpoint where z = a (identity mapping).
The denoising network is initialized to zero, so the denoising process doesn't change the input.
"""

import datetime
import pickle
import sys
from pathlib import Path
from typing import Annotated

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import tyro
from jax import numpy as jnp
from mujoco_playground import dm_control_suite, locomotion, registry

from flow_policy.decoder_diffusion import DecoderDiffusionConfig, DecoderDiffusionState


def main(
    env_name: Annotated[
        str,
        tyro.conf.arg(
            constructor=tyro.extras.literal_type_from_choices(
                dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS
            )
        ),
    ] = "CheetahRun",
    output_dir: str = "diffusion_models",
    seed: int = 42,
) -> None:
    """Create identity Diffusion checkpoint for first iteration.

    Args:
        env_name: Environment name (to determine obs_dim and action_dim)
        output_dir: Directory to save the checkpoint
        seed: Random seed for initialization
    """

    # Load environment to get dimensions
    env_config = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_config)

    obs_dim = env.observation_size
    action_dim = env.action_size

    # Create Diffusion config - matching playground version parameters
    config = DecoderDiffusionConfig(
        diffusion_steps=10,
        timestep_embed_dim=8,
        hidden_dims=(128, 128, 128, 128),  # Playground version (was 64 in fpo-main)
        policy_output_scale=1.0,
        learning_rate=3e-4,
        batch_size=8192,  # Playground version (was 2048 in fpo-main)
        num_epochs=1,  # Not used for identity
        n_samples_per_action=8,
        normalize_observations=True,
        beta_schedule="linear",  # Linear for identity (small scaling ~1.05x), cosine for training
        beta_start=0.0001,
        beta_end=0.02,
        sde_sigma=0.0,
        feather_std=0.0,
    )

    # Initialize Diffusion state
    prng = jax.random.PRNGKey(seed)
    diffusion_state = DecoderDiffusionState.init(prng, obs_dim, action_dim, config)

    # Zero out all network parameters to create identity mapping
    # When noise_pred = 0, DDIM denoising becomes identity: x_{t-1} = x_t

    def zero_params(params):
        """Recursively zero all parameters."""
        if isinstance(params, tuple):
            return tuple(zero_params(p) for p in params)
        elif isinstance(params, list):
            return [zero_params(p) for p in params]
        else:
            # It's a JAX array
            return jnp.zeros_like(params)

    zeroed_params = zero_params(diffusion_state.params)

    # Update Diffusion state with zeroed parameters
    import jax_dataclasses as jdc
    with jdc.copy_and_mutate(diffusion_state) as diffusion_state:
        diffusion_state.params = zeroed_params

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_path / f"diffusion_identity_{env_name}_{timestamp}.pkl"

    checkpoint = {
        "params": diffusion_state.params,
        "obs_stats": diffusion_state.obs_stats,
        "config": config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_name": env_name,
        "is_identity": True,  # Mark this as identity checkpoint
        "epoch": 0,
        "train_loss": 0.0,
        "val_loss": 0.0,
    }

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Init decoder (Diffusion): {checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(main)
