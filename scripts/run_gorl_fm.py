"""Complete training pipeline for GoRL (FM decoder).

This script automates the full training loop:
Stage 0: Init decoder → Encoder update → Collect data → Decoder update
Stage 1+: Encoder update → Collect data → Decoder update → Repeat
"""

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import tyro
from mujoco_playground import registry


def run_command(cmd: str, description: str) -> int:
    """Run a shell command and handle errors."""
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        sys.exit(result.returncode)

    return result.returncode


def main(
    env_name: Annotated[str, tyro.conf.arg(help="Environment name (e.g., CheetahRun)")] = "CheetahRun",
    num_stages: Annotated[int, tyro.conf.arg(help="Number of training stages")] = 4,
    encoder_num_timesteps: Annotated[int, tyro.conf.arg(help="Default encoder training timesteps per stage")] = 100000000,
    encoder_timesteps_per_stage: Annotated[str | None, tyro.conf.arg(help="Comma-separated timesteps for each stage (e.g., '60000000,60000000,30000000,30000000')")] = "60000000,60000000,30000000,30000000",
    seed: int = 1,

    # Optional parameters (with defaults from result.md)
    fm_batch_size: int = 8192,
    fm_num_epochs: int = 50,
    fm_learning_rate: float = 0.0003,
    fm_max_samples: int = 10000000,
    fm_hidden_size: int = 64,
    fm_num_layers: int = 4,

    data_collection_iterations: int = 20,
    fm_eval_episodes: int = 20,

    z_regularization: float | None = None,
    max_grad_norm: float = 0.5,

    # FM data filtering/sampling
    fm_hybrid_sampling: bool = False,  # Disabled: use full random sampling
    fm_high_quality_ratio: float = 0.8,
    fm_high_quality_percentile: float = 0.5,

) -> None:
    """Run complete GoRL training pipeline (FM decoder).

    Args:
        env_name: Environment to train on
        num_stages: Total number of stages (0, 1, 2, ...)
        encoder_num_timesteps: Default training steps for all stages
        encoder_timesteps_per_stage: Comma-separated timesteps per stage
        seed: Random seed
        gpu_id: CUDA device ID
    """

    # Create unique run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"gorl_fm_{env_name}_seed{seed}_{timestamp}"

    # Parse encoder timesteps per stage
    if encoder_timesteps_per_stage is not None:
        timesteps_list = [int(x.strip()) for x in encoder_timesteps_per_stage.split(",")]
        if len(timesteps_list) < num_stages:
            timesteps_list.extend([encoder_num_timesteps] * (num_stages - len(timesteps_list)))
        timesteps_list = timesteps_list[:num_stages]
    else:
        timesteps_list = [encoder_num_timesteps] * num_stages

    # Create run directory
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGoRL (FM) - {env_name}")
    print(f"Stages: {num_stages}, Timesteps: {timesteps_list}")

    # Auto-detect action_dim (z_dim) from environment
    env_config = registry.get_default_config(env_name)
    temp_env = registry.load(env_name, config=env_config)
    z_dim = temp_env.action_size
    del temp_env

    # Save pipeline config
    config_file = run_dir / "pipeline_config.txt"
    with open(config_file, "w") as f:
        f.write(f"GoRL (FM) Configuration\n")
        f.write(f"{'='*60}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n")
        f.write(f"Timesteps per stage: {timesteps_list}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"z_dim: {z_dim}\n")

    # Create global continuous metrics file
    global_metrics_file = run_dir / "global_eval_metrics.txt"
    with open(global_metrics_file, "w") as f:
        f.write(f"GoRL (FM) Evaluation Metrics\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n\n")

    # Track checkpoints and cumulative steps across stages
    fm_checkpoint = None
    encoder_checkpoint = None
    cumulative_step = 0

    for stage in range(num_stages):
        print(f"\n=== Stage {stage}/{num_stages-1} ===")

        stage_dir = run_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # STEP 1: Init decoder (only for stage 0)
        # =====================================================================
        if stage == 0:
            cmd = (
                f"python scripts/components/init_decoder_fm.py "
                f"--env_name {env_name} "
                f"--output_dir {stage_dir} "
                f"--seed {seed}"
            )
            run_command(cmd, f"Stage {stage}: Init decoder")

            fm_files = list(stage_dir.glob(f"fm_identity_{env_name}_*.pkl"))
            if not fm_files:
                print(f"ERROR: Identity decoder not found in {stage_dir}")
                sys.exit(1)
            fm_checkpoint = sorted(fm_files)[-1]

        # =====================================================================
        # STEP 2: Encoder update
        # =====================================================================
        if fm_checkpoint is None:
            print(f"ERROR: No decoder checkpoint available for stage {stage}")
            sys.exit(1)

        encoder_exp_name = f"pipeline_{run_id}_stage{stage}"
        stage_timesteps = timesteps_list[stage]

        # Adaptive parameters
        if stage == 0:
            stage_max_grad_norm = max_grad_norm
            stage_clipping_epsilon = 0.15
            if z_regularization is not None:
                stage_z_regularization = z_regularization
            else:
                stage_z_regularization = 0.0 if env_name == "BallInCup" else 0.0005
        else:
            stage_max_grad_norm = 1.0
            stage_clipping_epsilon = 0.3
            stage_z_regularization = z_regularization if z_regularization is not None else 0.001

        cmd = " ".join([
            f"python scripts/components/train_encoder_ppo.py",
            f"--env_name {env_name}",
            f"--decoder_type fm",
            f"--decoder_model_path {fm_checkpoint}",
            f"--z_dim {z_dim}",
            f"--seed {seed}",
            f"--num_timesteps {stage_timesteps}",
            f"--z_regularization {stage_z_regularization}",
            f"--max_grad_norm {stage_max_grad_norm}",
            f"--clipping_epsilon {stage_clipping_epsilon}",
            f"--exp_name {encoder_exp_name}",
        ])

        run_command(cmd, f"Stage {stage}: Encoder update")

        # Find encoder checkpoint
        encoder_pattern = f"encoder_fm_{env_name}_{encoder_exp_name}_*"
        encoder_results = list(Path("results").glob(encoder_pattern))

        if not encoder_results:
            print(f"ERROR: Encoder results not found")
            sys.exit(1)

        encoder_result_dir = sorted(encoder_results)[-1]
        encoder_checkpoint = encoder_result_dir / "best_checkpoint.pkl"

        if not encoder_checkpoint.exists():
            print(f"ERROR: Encoder checkpoint not found: {encoder_checkpoint}")
            sys.exit(1)

        # Merge evaluation metrics to global file
        eval_metrics_file = encoder_result_dir / "eval_metrics.txt"
        if eval_metrics_file.exists():
            with open(eval_metrics_file, "r") as f_in:
                lines = f_in.readlines()

            max_local_step = 0
            with open(global_metrics_file, "a") as f_out:
                f_out.write(f"\nSTAGE {stage}\n")
                for line in lines:
                    if line.startswith("Step:"):
                        local_step = int(line.split(":")[1].strip())
                        global_step = cumulative_step + local_step
                        max_local_step = max(max_local_step, local_step)
                        f_out.write(f"Step: {global_step}\n")
                    else:
                        f_out.write(line)

            cumulative_step += (max_local_step + 1)

        # =====================================================================
        # STEP 3: Collect data
        # =====================================================================
        cmd = (
            f"python scripts/components/collect_data_fm.py "
            f"--env_name {env_name} "
            f"--ppo_z_checkpoint_path {encoder_checkpoint} "
            f"--fm_model_path {fm_checkpoint} "
            f"--num_iterations {data_collection_iterations} "
            f"--output_dir {stage_dir} "
            f"--seed {seed}"
        )
        run_command(cmd, f"Stage {stage}: Collect data")

        data_files = list(stage_dir.glob(f"ppo_z_fm_data_{env_name}_*.pkl"))
        if not data_files:
            print(f"ERROR: Data file not found in {stage_dir}")
            sys.exit(1)
        data_file = sorted(data_files)[-1]

        # =====================================================================
        # STEP 4: Decoder update
        # =====================================================================
        fm_cmd_parts = [
            f"python scripts/components/train_decoder_fm.py",
            f"--data_path {data_file}",
            f"--batch_size {fm_batch_size}",
            f"--num_epochs {fm_num_epochs}",
            f"--learning_rate {fm_learning_rate}",
            f"--max_samples {fm_max_samples}",
            f"--hidden_size {fm_hidden_size}",
            f"--num_layers {fm_num_layers}",
            f"--output_dir {stage_dir}",
            f"--seed {seed}",
        ]

        if fm_hybrid_sampling:
            fm_cmd_parts.extend([
                f"--hybrid_sampling",
                f"--high_quality_ratio {fm_high_quality_ratio}",
                f"--high_quality_percentile {fm_high_quality_percentile}",
            ])

        cmd = " ".join(fm_cmd_parts)
        run_command(cmd, f"Stage {stage}: Decoder update")

        fm_best_files = list(stage_dir.glob(f"fm_model_best_*.pkl"))
        if not fm_best_files:
            print(f"ERROR: Decoder checkpoint not found in {stage_dir}")
            sys.exit(1)
        fm_checkpoint = sorted(fm_best_files)[-1]

        # Save stage summary
        summary_file = stage_dir / "stage_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Stage {stage} Summary\n")
            f.write(f"Decoder: {fm_checkpoint}\n")
            f.write(f"Encoder: {encoder_checkpoint}\n")
            f.write(f"Data: {data_file}\n")

    # =========================================================================
    # Pipeline Complete
    # =========================================================================
    print(f"\nGoRL (FM) complete - {num_stages} stages")
    print(f"Output: {run_dir}")

    # Save final summary
    final_summary = run_dir / "final_summary.txt"
    with open(final_summary, "w") as f:
        f.write(f"GoRL (FM) Summary\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n")
        f.write(f"Final decoder: {fm_checkpoint}\n")
        f.write(f"Final encoder: {encoder_checkpoint}\n")


if __name__ == "__main__":
    tyro.cli(main)
