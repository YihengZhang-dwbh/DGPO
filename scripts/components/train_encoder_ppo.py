"""Train encoder (PPO in latent space) with generative decoder."""

import datetime
import sys
import time
from typing import Annotated, Literal
from pathlib import Path
import pickle

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import jax_dataclasses as jdc
import numpy as onp
import tyro
from jax import numpy as jnp
from mujoco_playground import dm_control_suite, locomotion, registry
from mujoco_playground.config import dm_control_suite_params
from tqdm import tqdm

from flow_policy import encoder_ppo


def main(
    env_name: Annotated[
        str,
        tyro.conf.arg(
            constructor=tyro.extras.literal_type_from_choices(
                dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS
            )
        ),
    ] = "WalkerWalk",
    decoder_type: Literal["fm", "diffusion"] = "fm",
    decoder_model_path: str | None = None,
    exp_name: str = "encoder",
    learning_rate: float | None = None,
    clipping_epsilon: float | None = None,
    num_timesteps: int | None = None,
    z_dim: int | None = None,
    seed: int = 42,
    apply_tanh_in_rollout: bool = True,
    z_regularization: float = 0.0,
    max_grad_norm: float = 0.5,
    use_tanh_jacobian_for_z: bool = False,

    # Early stopping parameters
    eval_frequency: int = 1000000,
    min_steps: int = 0,
    improvement_threshold: float | None = 15.0,
    improvement_ratio_threshold: float | None = None,
    improvement_window: int = 5,
    reward_drop_threshold: float | None = 100.0,
    reward_drop_ratio: float | None = None,
    early_stopping: bool = False,
) -> None:
    """Train encoder with generative decoder (FM or Diffusion)."""

    # Dynamic imports based on decoder type
    if decoder_type == "fm":
        from flow_policy.decoder_fm import DecoderFMState as DecoderState
        from flow_policy.agent import EncoderFMAgent as Agent
        from flow_policy.rollout_encoder import (
            BatchedRolloutStateEncoderFM as BatchedRolloutState,
            eval_policy_encoder_fm as eval_policy
        )
        decoder_glob_pattern = "fm_models/fm_model_best_*.pkl"
        decoder_fallback = "fm_models_fixed_val/fm_model_best_20250929_142112.pkl"
    else:  # diffusion
        from flow_policy.decoder_diffusion import DecoderDiffusionState as DecoderState
        from flow_policy.agent import EncoderDiffusionAgent as Agent
        from flow_policy.rollout_encoder import (
            BatchedRolloutStateEncoderDiffusion as BatchedRolloutState,
            eval_policy_encoder_diffusion as eval_policy
        )
        decoder_glob_pattern = "diffusion_models/diffusion_model_best_*.pkl"
        decoder_fallback = None

    # Load environment config
    env_config = registry.get_default_config(env_name)
    ppo_params = dm_control_suite_params.brax_ppo_config(env_name)

    if learning_rate is not None:
        ppo_params.learning_rate = learning_rate
    if clipping_epsilon is not None:
        ppo_params.clipping_epsilon = clipping_epsilon
    if num_timesteps is not None:
        ppo_params.num_timesteps = num_timesteps

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"encoder_{decoder_type}_{env_name}_{exp_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load decoder model
    if decoder_model_path is None:
        import glob
        decoder_models = glob.glob(decoder_glob_pattern)
        if decoder_models:
            decoder_model_path = sorted(decoder_models)[-1]
        elif decoder_fallback:
            decoder_model_path = decoder_fallback
        else:
            raise ValueError(f"No {decoder_type} model found. Please specify --decoder_model_path")

    with open(decoder_model_path, "rb") as f:
        decoder_checkpoint = pickle.load(f)

    # Initialize environment
    env = registry.load(env_name, config=env_config)

    if z_dim is None:
        z_dim = env.action_size

    # Create encoder config
    ppo_params['z_dim'] = z_dim
    ppo_params['z_regularization'] = z_regularization
    ppo_params['max_grad_norm'] = max_grad_norm
    ppo_params['use_tanh_jacobian_for_z'] = use_tanh_jacobian_for_z
    config = encoder_ppo.EncoderConfig(**ppo_params)

    # Initialize encoder state
    encoder_state = encoder_ppo.EncoderState.init(
        prng=jax.random.key(seed),
        env=env,
        config=config
    )

    # Create decoder state from checkpoint
    decoder_prng = jax.random.PRNGKey(seed + 1000)
    decoder_state = DecoderState.init(
        decoder_prng,
        decoder_checkpoint['obs_dim'],
        decoder_checkpoint['action_dim'],
        decoder_checkpoint['config']
    )

    # Load decoder params and stats
    with jdc.copy_and_mutate(decoder_state) as decoder_state:
        decoder_state.params = decoder_checkpoint["params"]
        decoder_state.obs_stats = decoder_checkpoint["obs_stats"]

    # Create combined agent
    if decoder_type == "fm":
        agent = Agent(
            ppo_z_state=encoder_state,
            fm_state=decoder_state,
        )
    else:
        agent = Agent(
            ppo_z_state=encoder_state,
            diffusion_state=decoder_state,
        )

    # Initialize rollout state
    rollout_state = BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed + 1),
        num_envs=config.num_envs,
    )

    # Save configuration
    config_file = results_dir / "config.txt"
    with open(config_file, "w") as f:
        f.write(f"Algorithm: Encoder + {decoder_type.upper()}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"z_dim: {z_dim}\n")
        f.write(f"Decoder model: {decoder_model_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\nEncoder Parameters:\n")
        for key, value in vars(config).items():
            f.write(f"  {key}: {value}\n")

    # Create metrics file
    train_file = results_dir / "train_metrics.txt"
    with open(train_file, "w") as f:
        f.write(f"Training Metrics\n")
        f.write(f"{'='*60}\n")

    # Training loop
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)

    # Early stopping setup
    steps_per_iter = config.iterations_per_env * config.num_envs
    if early_stopping:
        if improvement_window < 2:
            raise ValueError("improvement_window must be at least 2 when early_stopping is enabled.")

        eval_step_interval = max(1, eval_frequency // steps_per_iter)
        eval_iters = set(range(0, outer_iters, eval_step_interval))
        eval_iters.add(max(outer_iters - 1, 0))

        recent_rewards: list[tuple[int, float]] = []
        min_steps_iters = max(0, min_steps // steps_per_iter)
    else:
        eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))
        recent_rewards = []
        min_steps_iters = 0

    times = [time.time()]
    best_reward = -float('inf')
    stop_training = False
    last_iteration = -1

    for i in tqdm(range(outer_iters)):
        last_iteration = i
        # Evaluation
        if i in eval_iters:
            eval_outputs = eval_policy(
                agent,
                prng=jax.random.fold_in(agent.ppo_z_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
                apply_tanh_in_rollout=apply_tanh_in_rollout,
            )

            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}
            current_reward = float(s_np['reward_mean'])
            reward_std = float(s_np['reward_std'])

            print(f"Eval metrics at step {i}:")
            print(
                f"  Reward: mean={s_np['reward_mean']:.2f}, min={s_np['reward_min']:.2f}, "
                f"max={s_np['reward_max']:.2f}, std={s_np['reward_std']:.2f}"
            )
            print(
                f"  Steps:  mean={s_np['steps_mean']:.1f}, min={s_np['steps_min']:.1f}, "
                f"max={s_np['steps_max']:.1f}, std={s_np['steps_std']:.1f}"
            )

            eval_outputs.log_to_file(results_dir, step=i)

            # Save best model
            if current_reward >= best_reward - 1e-6:
                best_reward = current_reward

                if early_stopping:
                    recent_rewards.clear()
                    recent_rewards.append((i, current_reward))

                # Use legacy key names for compatibility with collect_data scripts
                checkpoint = {
                    "ppo_z_params": agent.ppo_z_state.params,
                    "ppo_z_obs_stats": agent.ppo_z_state.obs_stats,
                    "config": config,
                    "env_name": env_name,
                    "decoder_type": decoder_type,
                    "iteration": i,
                    "reward": current_reward,
                    "z_dim": z_dim,
                }
                # Add decoder params with appropriate key names
                if decoder_type == "fm":
                    checkpoint["fm_params"] = decoder_state.params
                    checkpoint["fm_obs_stats"] = decoder_state.obs_stats
                else:
                    checkpoint["diffusion_params"] = decoder_state.params
                    checkpoint["diffusion_obs_stats"] = decoder_state.obs_stats
                best_checkpoint_file = results_dir / "best_checkpoint.pkl"
                with open(best_checkpoint_file, "wb") as f:
                    pickle.dump(checkpoint, f)
            elif early_stopping:
                recent_rewards.append((i, current_reward))
                if len(recent_rewards) > improvement_window:
                    recent_rewards.pop(0)

            # Early stopping check
            if early_stopping and i >= min_steps_iters and best_reward > -float("inf"):
                stop_reason = None
                reward_delta = best_reward - current_reward

                if reward_drop_threshold is not None and reward_delta >= reward_drop_threshold:
                    stop_reason = (
                        f"Reward dropped by {reward_delta:.2f} (>= {reward_drop_threshold}) "
                        f"from best {best_reward:.2f}"
                    )
                elif reward_drop_ratio is not None and best_reward != 0:
                    best_abs = max(abs(best_reward), 1e-6)
                    drop_ratio = reward_delta / best_abs
                    if drop_ratio >= reward_drop_ratio:
                        stop_reason = (
                            f"Reward dropped by {drop_ratio*100:.2f}% (>= {reward_drop_ratio*100:.2f}%) "
                            f"from best {best_reward:.2f}"
                        )

                if stop_reason is None and len(recent_rewards) >= improvement_window:
                    window_improvement = recent_rewards[-1][1] - recent_rewards[0][1]

                    if improvement_threshold is not None and window_improvement <= improvement_threshold:
                        stop_reason = (
                            f"Reward improvement {window_improvement:.2f} over "
                            f"{improvement_window} evals <= threshold {improvement_threshold:.2f}"
                        )
                    elif improvement_ratio_threshold is not None and best_reward != 0:
                        best_abs = max(abs(best_reward), 1e-6)
                        window_ratio = window_improvement / best_abs
                        if window_ratio <= improvement_ratio_threshold:
                            stop_reason = (
                                f"Relative improvement {window_ratio*100:.2f}% over "
                                f"{improvement_window} evals <= "
                                f"{improvement_ratio_threshold*100:.2f}% threshold"
                            )

                if stop_reason is not None:
                    print(f"Early stop at step {i * steps_per_iter}")
                    stop_training = True
        if stop_training:
            times.append(time.time())
            break

        # Training step
        rollout_state, transitions = rollout_state.rollout(
            agent,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
            apply_tanh_in_rollout=apply_tanh_in_rollout,
        )

        agent, metrics = agent.training_step(transitions)

        # Z distribution statistics for logging
        z_values = transitions.action
        z_mean = float(onp.mean(z_values))
        z_std = float(onp.std(z_values))
        z_min = float(onp.min(z_values))
        z_max = float(onp.max(z_values))
        z_abs_max = float(onp.max(onp.abs(z_values)))
        mean_reward = float(onp.mean(transitions.reward))

        with open(train_file, "a") as f:
            f.write(f"\nIteration {i}:\n")
            f.write(f"  mean_reward: {mean_reward:.4f}\n")
            f.write(f"  z_mean: {z_mean:.6f}\n")
            f.write(f"  z_std: {z_std:.6f}\n")
            f.write(f"  z_min: {z_min:.6f}\n")
            f.write(f"  z_max: {z_max:.6f}\n")
            f.write(f"  z_abs_max: {z_abs_max:.6f}\n")

            for k, v in metrics.items():
                f.write(f"  {k}: {float(onp.mean(v)):.6f}\n")

        times.append(time.time())

    # Final summary
    print("First train step time:", times[1] - times[0])
    print("~Train time:", times[-1] - times[1])
    print(f"\nResults saved to: {results_dir}")

    # Save final checkpoint (use legacy key names for compatibility)
    final_checkpoint = {
        "ppo_z_params": agent.ppo_z_state.params,
        "ppo_z_obs_stats": agent.ppo_z_state.obs_stats,
        "config": config,
        "env_name": env_name,
        "decoder_type": decoder_type,
        "final_iteration": last_iteration + 1,
        "best_reward": best_reward,
        "z_dim": z_dim,
    }
    if decoder_type == "fm":
        final_checkpoint["fm_params"] = decoder_state.params
        final_checkpoint["fm_obs_stats"] = decoder_state.obs_stats
    else:
        final_checkpoint["diffusion_params"] = decoder_state.params
        final_checkpoint["diffusion_obs_stats"] = decoder_state.obs_stats
    final_checkpoint_file = results_dir / "final_checkpoint.pkl"
    with open(final_checkpoint_file, "wb") as f:
        pickle.dump(final_checkpoint, f)


if __name__ == "__main__":
    tyro.cli(main)
