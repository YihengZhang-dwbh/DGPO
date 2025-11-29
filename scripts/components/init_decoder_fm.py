"""Create an identity Flow Matching model for first iteration.

This creates an FM checkpoint where z = a (identity mapping).
The velocity field is initialized to zero, so flow matching doesn't change the input.
"""

import datetime
import pickle
from pathlib import Path
from typing import Annotated

import jax
import tyro
from jax import numpy as jnp
from mujoco_playground import dm_control_suite, locomotion, registry

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState


def main(
    env_name: Annotated[
        str,
        tyro.conf.arg(
            constructor=tyro.extras.literal_type_from_choices(
                dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS
            )
        ),
    ] = "CheetahRun",
    output_dir: str = "fm_models",
    seed: int = 42,
) -> None:
    """Create identity FM checkpoint for first iteration.

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

    # Create FM config
    # Use minimal network since we want identity mapping
    config = DecoderFMConfig(
        flow_steps=10,
        timestep_embed_dim=8,
        hidden_dims=(64, 64, 64, 64),  # Standard architecture
        policy_output_scale=1.0,
        learning_rate=3e-4,
        batch_size=2048,
        num_epochs=1,  # Not used for identity
        n_samples_per_action=8,
        normalize_observations=True,
        sde_sigma=0.0,
        feather_std=0.0,
    )

    # Initialize FM state
    prng = jax.random.PRNGKey(seed)
    fm_state = DecoderFMState.init(prng, obs_dim, action_dim, config)

    # Zero out all network parameters to create identity mapping
    # When velocity = 0, x_t stays constant during integration
    # So if we start from z, we end at z (identity mapping)

    def zero_params(params):
        """Recursively zero all parameters."""
        if isinstance(params, tuple):
            return tuple(zero_params(p) for p in params)
        elif isinstance(params, list):
            return [zero_params(p) for p in params]
        else:
            # It's a JAX array
            return jnp.zeros_like(params)

    zeroed_params = zero_params(fm_state.params)

    # Update FM state with zeroed parameters
    import jax_dataclasses as jdc
    with jdc.copy_and_mutate(fm_state) as fm_state:
        fm_state.params = zeroed_params

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_path / f"fm_identity_{env_name}_{timestamp}.pkl"

    checkpoint = {
        "params": fm_state.params,
        "obs_stats": fm_state.obs_stats,
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

    print(f"Init decoder (FM): {checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(main)
