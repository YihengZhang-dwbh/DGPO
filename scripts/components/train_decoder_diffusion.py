"""Train Diffusion model to learn PPO action distribution."""

import datetime
import pickle
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import jax_dataclasses as jdc
import numpy as np
import tyro
from jax import numpy as jnp
from tqdm import tqdm

from flow_policy.decoder_diffusion import DecoderDiffusionConfig, DecoderDiffusionState


def train_diffusion(
    data_path: str = "data/ppo_training_data_WalkerWalk_20250928_212057.pkl",
    num_epochs: int = 80,  # Match playground version
    batch_size: int = 8192,  # Match playground version
    learning_rate: float = 3e-4,
    validation_split: float = 0.1,
    max_samples: int | None = 10000000,  # Match playground version (10M)
    episode_length: int = 1000,  # Episode length for reward-based filtering
    reward_percentile: float = 0.0,  # Keep episodes above this percentile (0-1)
    min_episode_reward: float | None = None,  # Minimum episode reward threshold
    hybrid_sampling: bool = False,  # Enable hybrid sampling strategy
    high_quality_ratio: float = 0.8,  # Ratio of high-quality samples in hybrid mode
    high_quality_percentile: float = 0.5,  # Percentile threshold for high-quality episodes
    output_dir: str = "diffusion_models",
    seed: int = 42,
    hidden_size: int = 128,  # Match playground version
    num_layers: int = 4,  # Number of hidden layers
) -> None:
    """Train Diffusion model on collected PPO data.

    Args:
        data_path: Path to PPO data pickle file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        max_samples: Maximum number of samples to use (None for all)
        episode_length: Length of each episode for grouping
        reward_percentile: Keep episodes with reward >= this percentile (0-1)
        min_episode_reward: If set, keep episodes with total reward >= this value
        hybrid_sampling: Enable hybrid sampling (mix high-quality + random coverage)
        high_quality_ratio: In hybrid mode, ratio of high-quality samples (0-1)
        high_quality_percentile: In hybrid mode, percentile for high-quality (0-1)
        output_dir: Directory to save models
        seed: Random seed
        hidden_size: Size of hidden layers (default: 64)
        num_layers: Number of hidden layers (default: 4)
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]

    # Episode-based filtering using rewards
    if "rewards" in data and (reward_percentile > 0 or min_episode_reward is not None or hybrid_sampling):
        rewards = data["rewards"]

        # Group data by episodes
        n_episodes = len(states) // episode_length
        if len(states) % episode_length != 0:
            trim_to = n_episodes * episode_length
            states = states[:trim_to]
            actions = actions[:trim_to]
            rewards = rewards[:trim_to]

        episodes = []
        for i in range(n_episodes):
            start = i * episode_length
            end = start + episode_length
            episode_total_reward = rewards[start:end].sum()
            episodes.append({
                'idx': i,
                'start': start,
                'end': end,
                'total_reward': float(episode_total_reward)
            })

        episode_rewards = np.array([ep['total_reward'] for ep in episodes])

        # Apply filtering or hybrid sampling
        if hybrid_sampling:
            hq_threshold = np.percentile(episode_rewards, high_quality_percentile * 100)
            high_quality_episodes = [ep for ep in episodes if ep['total_reward'] >= hq_threshold]

            n_high_quality = int(len(episodes) * high_quality_ratio)
            n_coverage = len(episodes) - n_high_quality

            if len(high_quality_episodes) >= n_high_quality:
                sampled_hq = np.random.choice(len(high_quality_episodes), n_high_quality, replace=False)
                selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]
            else:
                sampled_hq = np.random.choice(len(high_quality_episodes), n_high_quality, replace=True)
                selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]

            sampled_coverage = np.random.choice(len(episodes), n_coverage, replace=False)
            selected_coverage_episodes = [episodes[i] for i in sampled_coverage]

            keep_episodes = selected_hq_episodes + selected_coverage_episodes

        elif min_episode_reward is not None:
            keep_episodes = [ep for ep in episodes if ep['total_reward'] >= min_episode_reward]
        else:
            threshold = np.percentile(episode_rewards, reward_percentile * 100)
            keep_episodes = [ep for ep in episodes if ep['total_reward'] >= threshold]

        # Rebuild data from kept episodes
        keep_indices = []
        for ep in keep_episodes:
            keep_indices.extend(range(ep['start'], ep['end']))

        states = states[keep_indices]
        actions = actions[keep_indices]
        rewards = rewards[keep_indices]
    else:
        if "rewards" in data:
            rewards = data["rewards"]

    # Optionally subsample data for faster training
    if max_samples is not None and len(states) > max_samples:
        sample_indices = np.random.choice(len(states), max_samples, replace=False)
        states = states[sample_indices]
        actions = actions[sample_indices]
        if "rewards" in data:
            rewards = rewards[sample_indices]

    # Split data
    n_samples = len(states)
    n_train = int(n_samples * (1 - validation_split))
    indices = np.random.permutation(n_samples)

    train_states = states[indices[:n_train]]
    train_actions = actions[indices[:n_train]]
    val_states = states[indices[n_train:]]
    val_actions = actions[indices[n_train:]]

    # Initialize Diffusion model
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]

    # Build hidden dims from parameters
    hidden_dims = tuple([hidden_size] * num_layers)

    config = DecoderDiffusionConfig(
        diffusion_steps=10,
        timestep_embed_dim=8,  # FPO uses 8
        hidden_dims=hidden_dims,  # Configurable network size
        policy_output_scale=1.0,  # Changed to 1.0 for supervised learning
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        n_samples_per_action=8,  # FPO's actual default
        normalize_observations=True,
        beta_schedule="cosine",  # Cosine schedule for better noise coverage with 10 steps
        beta_start=0.0001,
        beta_end=0.02,
        sde_sigma=0.0,
        feather_std=0.0,
    )

    prng = jax.random.PRNGKey(seed)
    diffusion_state = DecoderDiffusionState.init(prng, obs_dim, action_dim, config)

    # Update statistics
    with jdc.copy_and_mutate(diffusion_state) as diffusion_state:
        diffusion_state.obs_stats = diffusion_state.obs_stats.update(jnp.array(train_states))

    # Training loop
    n_batches = n_train // batch_size
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 20  # Early stopping patience

    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        epoch_metrics = []

        # Shuffle training data
        perm = np.random.permutation(n_train)

        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = perm[start_idx:end_idx]

            batch_obs = jnp.array(train_states[batch_indices])
            batch_actions = jnp.array(train_actions[batch_indices])

            # Training step
            diffusion_state, metrics = diffusion_state.train_step(batch_obs, batch_actions)

            epoch_losses.append(float(metrics["loss"]))
            epoch_metrics.append(metrics)

        # Compute epoch statistics
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        val_batch_losses = []
        n_val_batches = min(50, len(val_states) // batch_size)  # Increased validation coverage

        for i in range(n_val_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_obs = jnp.array(val_states[start_idx:end_idx])
            batch_actions = jnp.array(val_actions[start_idx:end_idx])

            # Use DDPM loss for validation (same as training)
            # Normalize observations
            if diffusion_state.config.normalize_observations:
                obs_norm = (batch_obs - diffusion_state.obs_stats.mean) / (diffusion_state.obs_stats.std + 1e-8)
            else:
                obs_norm = batch_obs

            # Sample noise and timesteps
            prng_val_noise, prng_val_t, prng = jax.random.split(diffusion_state.prng, 3)
            val_noise = jax.random.normal(
                prng_val_noise,
                (batch_size, diffusion_state.config.n_samples_per_action, action_dim)
            )
            val_t = jax.random.randint(
                prng_val_t,
                (batch_size, diffusion_state.config.n_samples_per_action, 1),
                0,
                diffusion_state.config.diffusion_steps
            ).astype(jnp.float32)

            # Compute DDPM loss
            ddpm_loss = diffusion_state.compute_ddpm_loss(
                obs_norm,
                batch_actions,
                val_noise,
                val_t
            )
            val_loss = jnp.mean(ddpm_loss)
            val_batch_losses.append(float(val_loss))

        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Save checkpoint
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = output_path / f"diffusion_model_best_{timestamp}.pkl"

            checkpoint = {
                "params": diffusion_state.params,  # Only save parameters
                "obs_stats": diffusion_state.obs_stats,
                "config": config,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Save final model
    final_file = output_path / f"diffusion_model_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    final_checkpoint = {
        "params": diffusion_state.params,  # Only save parameters
        "obs_stats": diffusion_state.obs_stats,
        "config": config,
        "epoch": num_epochs,
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_history": train_losses,
        "val_history": val_losses,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }

    with open(final_file, "wb") as f:
        pickle.dump(final_checkpoint, f)

    print(f"Decoder (Diffusion) done: loss={best_val_loss:.4f}, output={checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(train_diffusion)
