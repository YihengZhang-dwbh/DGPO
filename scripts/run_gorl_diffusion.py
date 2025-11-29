"""Complete training pipeline for GoRL (Diffusion decoder).

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

    # Optional parameters (aligned with playground version)
    diffusion_batch_size: int = 8192,
    diffusion_num_epochs: int = 50,
    diffusion_learning_rate: float = 0.0003,
    diffusion_max_samples: int = 10000000,
    diffusion_hidden_size: int = 64,
    diffusion_num_layers: int = 4,

    data_collection_iterations: int = 20,
    diffusion_eval_episodes: int = 20,

    z_regularization: float | None = None,
    max_grad_norm: float = 0.5,

    # Diffusion data filtering/sampling
    diffusion_hybrid_sampling: bool = False,
    diffusion_high_quality_ratio: float = 0.8,
    diffusion_high_quality_percentile: float = 0.5,

) -> None:
    """Run complete GoRL training pipeline (Diffusion decoder)."""

    # Create unique run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"gorl_diffusion_{env_name}_seed{seed}_{timestamp}"

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

    print(f"\nGoRL (Diffusion) - {env_name}")
    print(f"Stages: {num_stages}, Timesteps: {timesteps_list}")

    # Auto-detect action_dim (z_dim) from environment
    env_config = registry.get_default_config(env_name)
    temp_env = registry.load(env_name, config=env_config)
    z_dim = temp_env.action_size
    del temp_env

    # Save pipeline config
    config_file = run_dir / "pipeline_config.txt"
    with open(config_file, "w") as f:
        f.write(f"GoRL (Diffusion) Configuration\n")
        f.write(f"{'='*60}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n")
        f.write(f"Timesteps per stage: {timesteps_list}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"z_dim: {z_dim}\n")

    # Create global continuous metrics file
    global_metrics_file = run_dir / "global_eval_metrics.txt"
    with open(global_metrics_file, "w") as f:
        f.write(f"GoRL (Diffusion) Evaluation Metrics\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n\n")

    # Track checkpoints and cumulative steps across stages
    diffusion_checkpoint = None
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
                f"python scripts/components/init_decoder_diffusion.py "
                f"--env_name {env_name} "
                f"--output_dir {stage_dir} "
                f"--seed {seed}"
            )
            run_command(cmd, f"Stage {stage}: Init decoder")

            diffusion_files = list(stage_dir.glob(f"diffusion_identity_{env_name}_*.pkl"))
            if not diffusion_files:
                print(f"ERROR: Identity decoder not found in {stage_dir}")
                sys.exit(1)
            diffusion_checkpoint = sorted(diffusion_files)[-1]

        # =====================================================================
        # STEP 2: Encoder update
        # =====================================================================
        if diffusion_checkpoint is None:
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
            f"--decoder_type diffusion",
            f"--decoder_model_path {diffusion_checkpoint}",
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
        encoder_pattern = f"encoder_diffusion_{env_name}_{encoder_exp_name}_*"
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
            f"python scripts/components/collect_data_diffusion.py "
            f"--env_name {env_name} "
            f"--ppo_z_checkpoint_path {encoder_checkpoint} "
            f"--diffusion_model_path {diffusion_checkpoint} "
            f"--num_iterations {data_collection_iterations} "
            f"--output_dir {stage_dir} "
            f"--seed {seed}"
        )
        run_command(cmd, f"Stage {stage}: Collect data")

        data_files = list(stage_dir.glob(f"ppo_z_diffusion_data_{env_name}_*.pkl"))
        if not data_files:
            print(f"ERROR: Data file not found in {stage_dir}")
            sys.exit(1)
        data_file = sorted(data_files)[-1]

        # =====================================================================
        # STEP 4: Decoder update
        # =====================================================================
        diffusion_cmd_parts = [
            f"python scripts/components/train_decoder_diffusion.py",
            f"--data_path {data_file}",
            f"--batch_size {diffusion_batch_size}",
            f"--num_epochs {diffusion_num_epochs}",
            f"--learning_rate {diffusion_learning_rate}",
            f"--max_samples {diffusion_max_samples}",
            f"--hidden_size {diffusion_hidden_size}",
            f"--num_layers {diffusion_num_layers}",
            f"--output_dir {stage_dir}",
            f"--seed {seed}",
        ]

        if diffusion_hybrid_sampling:
            diffusion_cmd_parts.extend([
                f"--hybrid_sampling",
                f"--high_quality_ratio {diffusion_high_quality_ratio}",
                f"--high_quality_percentile {diffusion_high_quality_percentile}",
            ])

        cmd = " ".join(diffusion_cmd_parts)
        run_command(cmd, f"Stage {stage}: Decoder update")

        diffusion_best_files = list(stage_dir.glob(f"diffusion_model_best_*.pkl"))
        if not diffusion_best_files:
            print(f"ERROR: Decoder checkpoint not found in {stage_dir}")
            sys.exit(1)
        diffusion_checkpoint = sorted(diffusion_best_files)[-1]

        # Save stage summary
        summary_file = stage_dir / "stage_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Stage {stage} Summary\n")
            f.write(f"Decoder: {diffusion_checkpoint}\n")
            f.write(f"Encoder: {encoder_checkpoint}\n")
            f.write(f"Data: {data_file}\n")

    # =========================================================================
    # Pipeline Complete
    # =========================================================================
    print(f"\nGoRL (Diffusion) complete - {num_stages} stages")
    print(f"Output: {run_dir}")

    # Save final summary
    final_summary = run_dir / "final_summary.txt"
    with open(final_summary, "w") as f:
        f.write(f"GoRL (Diffusion) Summary\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Stages: {num_stages}\n")
        f.write(f"Final decoder: {diffusion_checkpoint}\n")
        f.write(f"Final encoder: {encoder_checkpoint}\n")


if __name__ == "__main__":
    tyro.cli(main)
