import datetime
import time
from typing import Annotated
from pathlib import Path

import jax
import jax_dataclasses as jdc
import numpy as onp
import tyro
from jax import numpy as jnp
from mujoco_playground import dm_control_suite, locomotion, registry
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flow_policy import fpo, rollouts


def main(
    # TODO: these envs are just what i'm testing with initially
    env_name: Annotated[
        str,
        tyro.conf.arg(
            constructor=tyro.extras.literal_type_from_choices(
                dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS
            )
        ),
    ],
    config: fpo.FpoConfig,
    exp_name: str = "",
    seed: int = 0,
) -> None:
    """Main function to train FPO on a specified environment."""

    # Load environment config and PPO parameters.
    env_config = registry.get_default_config(env_name)

    # 如果是DPPO模式，应用DPPO特定参数（论文Table中的最优值）
    if config.loss_mode == "denoising_mdp":
        config = jdc.replace(
            config,
            clipping_epsilon=0.2,  # DPPO论文最优值
            sde_sigma=0.05,        # DPPO论文最优值
        )

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_name = "dppo" if config.loss_mode == "denoising_mdp" else "fpo"
    results_dir = Path("results") / f"{algo_name}_{env_name}_{exp_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config to file
    config_file = results_dir / "config.txt"
    with open(config_file, "w") as f:
        f.write(f"Algorithm: {algo_name.upper()}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\nFPO Parameters:\n")
        config_dict = jdc.asdict(config)
        for key, value in config_dict.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nEnvironment Config:\n")
        f.write(f"{env_config}\n")

    # Create train metrics file
    train_file = results_dir / "train_metrics.txt"
    with open(train_file, "w") as f:
        f.write(f"{algo_name.upper()} Training Metrics - {env_name}\n")
        f.write(f"{'='*60}\n")

    # Initialize.
    env = registry.load(env_name, config=env_config)
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=env, config=config)
    rollout_state = rollouts.BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed),
        num_envs=config.num_envs,
    )

    # Perform rollout.
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    # Eval every 6M steps (matching GoRL iter0), also eval at step 0
    steps_per_iter = config.iterations_per_env * config.num_envs
    eval_interval = 6_000_000 // steps_per_iter
    eval_iters = {0} | set(range(eval_interval - 1, outer_iters, eval_interval))

    times = [time.time()]
    for i in tqdm(range(outer_iters)):
        # Evaluation. Note: this might be better done *after* the training step.
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )

            # Convert to numpy for printing.
            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            # Print summary.
            print(f"Eval metrics at step {i}:")
            print(
                f"  Reward: mean={s_np['reward_mean']:.2f}, min={s_np['reward_min']:.2f}, max={s_np['reward_max']:.2f}, std={s_np['reward_std']:.2f}"
            )
            print(
                f"  Steps:  mean={s_np['steps_mean']:.1f}, min={s_np['steps_min']:.1f}, max={s_np['steps_max']:.1f}, std={s_np['steps_std']:.1f}"
            )

            # Log to file using the new API.
            eval_outputs.log_to_file(results_dir, step=i)

        # Training step.
        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        # Train metric logging to file.
        mean_reward = float(onp.mean(transitions.reward))
        mean_steps = float(transitions.discount.size / jnp.sum(transitions.discount == 0.0))

        # Write to train metrics file
        with open(train_file, "a") as f:
            f.write(f"\nIteration: {i}\n")
            f.write(f"  mean_reward: {mean_reward:.4f}\n")
            f.write(f"  mean_steps: {mean_steps:.2f}\n")

            # Add all training metrics
            for k, v in metrics.items():
                f.write(f"  {k}: {float(onp.mean(v)):.6f}\n")

        times.append(time.time())

    print("First train step time:", times[1] - times[0])
    print("~Train time:", times[-1] - times[1])

    # Save summary
    summary_file = results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"{algo_name.upper()} Training Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Total iterations: {outer_iters}\n")
        f.write(f"Total timesteps: {config.num_timesteps}\n")
        f.write(f"First train step time: {times[1] - times[0]:.2f} seconds\n")
        f.write(f"Total train time: {times[-1] - times[1]:.2f} seconds\n")
        f.write(f"\nResults saved to: {results_dir}\n")

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.FlagConversionOff,))
