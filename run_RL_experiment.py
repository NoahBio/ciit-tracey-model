"""RL experiment runner for therapy agent training.

Orchestrates complete training and evaluation pipeline with:
- Config loading and validation
- Directory structure setup
- Training execution
- Policy evaluation
- Final report generation

Usage:
    python run_RL_experiment.py --config configs/default_experiment.yaml
    python run_RL_experiment.py --config configs/default_experiment.yaml --eval-only
    python run_RL_experiment.py --config configs/default_experiment.yaml --no-eval
    python run_RL_experiment.py --config configs/default_experiment.yaml --resume-from models/.../checkpoint.pth
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from src.training.config import TrainingConfig, load_config, save_config
from src.training.train_ppo import train
from src.evaluation.evaluate_policy import (
    evaluate_policy, compute_metrics,
    plot_results, save_results, print_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete RL training experiment with evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (required)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    # Training control
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training (useful for re-running evaluation only)"
    )

    # Evaluation control
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing policy (skip training)"
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip final evaluation after training"
    )

    parser.add_argument(
        "--eval-policy",
        type=str,
        default=None,
        help="Path to policy checkpoint for evaluation (defaults to best policy from training)"
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override number of evaluation episodes (defaults to config value)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device for training/evaluation (auto=cuda if available)"
    )

    # Output directory override
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (defaults to models/{experiment_name})"
    )

    return parser.parse_args()


def generate_report(
    config: TrainingConfig,
    metadata: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Generate markdown experiment report.

    Parameters
    ----------
    config : TrainingConfig
        Experiment configuration
    metadata : dict
        Experiment metadata and results
    output_dir : Path
        Directory to save report

    Returns
    -------
    Path
        Path to generated report
    """

    report_path = output_dir / "experiment_report.md"

    with open(report_path, 'w') as f:
        # Header
        f.write(f"# Experiment Report: {config.experiment_name}\n\n")
        f.write(f"**Generated**: {metadata['timestamp']}\n\n")
        f.write("---\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write("### Environment Parameters\n")
        f.write(f"- **Mechanism**: {config.mechanism}\n")
        f.write(f"- **Patterns**: {', '.join(config.patterns)}\n")
        f.write(f"- **Threshold**: {config.threshold}\n")
        f.write(f"- **Max Sessions**: {config.max_sessions}\n")
        f.write(f"- **Entropy**: {config.entropy}\n")
        f.write(f"- **Bond Alpha**: {config.bond_alpha}\n")
        f.write(f"- **Bond Offset**: {config.bond_offset}\n")
        f.write(f"- **History Weight**: {config.history_weight}\n")
        f.write(f"- **Enable Perception**: {config.enable_perception}\n")
        f.write(f"- **Baseline Accuracy**: {config.baseline_accuracy}\n")
        f.write("\n")

        f.write("### RL Parameters\n")
        f.write(f"- **Total Timesteps**: {config.total_timesteps:,}\n")
        f.write(f"- **Parallel Envs**: {config.n_envs}\n")
        f.write(f"- **Learning Rate**: {config.learning_rate}\n")
        f.write(f"- **Batch Size**: {config.batch_size}\n")
        f.write(f"- **Epochs per Update**: {config.n_epochs}\n")
        f.write(f"- **Gamma**: {config.gamma}\n")
        f.write(f"- **GAE Lambda**: {config.gae_lambda}\n")
        f.write(f"- **Clip Range**: {config.clip_range}\n")
        f.write(f"- **Entropy Coef**: {config.ent_coef}\n")
        f.write(f"- **Hidden Size**: {config.hidden_size}\n")
        f.write(f"- **Seed**: {config.seed}\n")
        f.write("\n")

        # Training Results
        f.write("## Training\n\n")
        if metadata['training_completed']:
            training_time = metadata.get('training_time_seconds')
            if training_time:
                hours = training_time / 3600
                f.write(f"- **Status**: Completed\n")
                f.write(f"- **Duration**: {timedelta(seconds=int(training_time))} ({hours:.2f} hours)\n")
                f.write(f"- **Device**: {metadata['device']}\n")
            else:
                f.write(f"- **Status**: Completed (interrupted)\n")
        else:
            f.write(f"- **Status**: Skipped\n")
        f.write("\n")

        # Evaluation Results
        f.write("## Evaluation\n\n")
        if metadata['evaluation_completed'] and 'eval_metrics' in metadata:
            metrics = metadata['eval_metrics']
            overall = metrics.get('overall', {})

            f.write("### Overall Performance\n")
            f.write(f"- **Success Rate**: {overall.get('success_rate', 0):.1%}\n")
            f.write(f"- **Dropout Rate**: {overall.get('dropout_rate', 0):.1%}\n")
            f.write(f"- **Mean Reward**: {overall.get('mean_reward', 0):.2f} ± {overall.get('std_reward', 0):.2f}\n")
            f.write(f"- **Mean Episode Length**: {overall.get('mean_length', 0):.1f} ± {overall.get('std_length', 0):.1f}\n")
            f.write(f"- **Mean Final RS**: {overall.get('mean_final_rs', 0):.3f} ± {overall.get('std_final_rs', 0):.3f}\n")

            if overall.get('mean_sessions_to_success') is not None:
                f.write(f"- **Sessions to Success**: {overall['mean_sessions_to_success']:.1f} ± {overall.get('std_sessions_to_success', 0):.1f}\n")
            f.write("\n")

            # Per-pattern results
            if 'by_pattern' in metrics:
                f.write("### Per-Pattern Results\n\n")
                f.write("| Pattern | Success Rate | Dropout Rate | Mean Reward | Mean Length |\n")
                f.write("|---------|--------------|--------------|-------------|-------------|\n")

                for pattern, pm in metrics['by_pattern'].items():
                    f.write(f"| {pattern} | {pm['success_rate']:.1%} | {pm['dropout_rate']:.1%} | ")
                    f.write(f"{pm['mean_reward']:.1f} | {pm['mean_length']:.1f} |\n")
                f.write("\n")

            # Action distribution
            if 'action_distribution' in metrics:
                f.write("### Action Distribution\n\n")
                action_names = ['D', 'WD', 'W', 'WS', 'S', 'CS', 'C', 'CD']
                for name, prob in zip(action_names, metrics['action_distribution']):
                    f.write(f"- **{name}**: {prob:.1%}\n")
                f.write("\n")
        else:
            f.write("- **Status**: Skipped or incomplete\n\n")

        # Files
        f.write("## Output Files\n\n")
        f.write(f"- **Models**: `{metadata['output_dir']}/`\n")
        f.write(f"- **Results**: `{metadata['results_dir']}/`\n")
        f.write(f"- **Config**: `{metadata['config_path']}`\n")
        f.write("\n")

        # Footer
        f.write("---\n\n")
        f.write("*Report generated by run_RL_experiment.py*\n")

    return report_path


def run_experiment(
    config_path: Path,
    resume_from: Optional[Path] = None,
    skip_training: bool = False,
    skip_eval: bool = False,
    eval_policy_path: Optional[Path] = None,
    eval_episodes: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run complete RL experiment: training + evaluation.

    Parameters
    ----------
    config_path : Path
        Path to YAML config file
    resume_from : Path, optional
        Checkpoint to resume training from
    skip_training : bool
        Skip training phase
    skip_eval : bool
        Skip evaluation phase
    eval_policy_path : Path, optional
        Specific policy to evaluate (defaults to best from training)
    eval_episodes : int, optional
        Override number of evaluation episodes
    device : str, optional
        Device for computation (cpu/cuda/auto)
    output_dir : Path, optional
        Override output directory

    Returns
    -------
    dict
        Experiment results and metadata
    """

    # 1. LOAD CONFIG
    print("=" * 80)
    print("LOADING CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {config_path}")

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    print(f"Experiment: {config.experiment_name}")
    print(f"Mechanism: {config.mechanism}")
    print(f"Patterns: {config.patterns}")
    print()

    # 2. SETUP DIRECTORIES
    print("=" * 80)
    print("SETTING UP DIRECTORIES")
    print("=" * 80)

    if output_dir is None:
        output_dir = Path("models") / config.experiment_name
    output_dir = Path(output_dir)

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results") / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model output: {output_dir}")
    print(f"Results output: {results_dir}")
    print()

    # 3. SAVE CONFIG COPY
    config_copy_path = output_dir / "experiment_config.yaml"
    save_config(config, config_copy_path)
    print(f"Saved config copy to: {config_copy_path}")
    print()

    # 4. DEVICE SETUP
    if device == "auto" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()

    # 5. TRAINING PHASE
    policy = None
    training_time = None

    if not skip_training:
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Total timesteps: {config.total_timesteps:,}")
        print(f"Parallel environments: {config.n_envs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Expected training time: ~16 hours (Ryzen 5 5600)")
        print()

        start_time = time.time()

        try:
            policy = train(
                config=config,
                output_dir=output_dir,
                resume_from=resume_from
            )

            training_time = time.time() - start_time
            print()
            print("=" * 80)
            print("TRAINING COMPLETED")
            print("=" * 80)
            print(f"Training time: {timedelta(seconds=int(training_time))}")
            print()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            training_time = time.time() - start_time
            print(f"Partial training time: {timedelta(seconds=int(training_time))}")

            # Ask if user wants to continue with evaluation
            if not skip_eval:
                response = input("\nContinue with evaluation? (y/n): ").strip().lower()
                if response != 'y':
                    skip_eval = True

        except Exception as e:
            print(f"\n\nERROR during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("=" * 80)
        print("SKIPPING TRAINING (as requested)")
        print("=" * 80)
        print()

    # 6. EVALUATION PHASE
    eval_results = None
    eval_metrics = None

    if not skip_eval:
        print("=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)

        # Determine which policy to evaluate
        if eval_policy_path is None:
            # Use best policy from training
            eval_policy_path = output_dir / "policy_best.pth"

            # Fallback to final policy if best doesn't exist
            if not eval_policy_path.exists():
                eval_policy_path = output_dir / "policy_final.pth"

            if not eval_policy_path.exists():
                print(f"ERROR: No trained policy found at {output_dir}")
                print("Please provide --eval-policy path or train a model first")
                sys.exit(1)

        print(f"Evaluating policy: {eval_policy_path}")
        print()

        # Determine number of episodes
        n_episodes = eval_episodes if eval_episodes is not None else config.eval_episodes

        try:
            # Run evaluation
            eval_results = evaluate_policy(
                policy_path=eval_policy_path,
                config=config,
                n_episodes=n_episodes,
                device=device,
                patterns=config.patterns
            )

            # Compute metrics
            eval_metrics = compute_metrics(eval_results)

            # Print summary
            print_summary(eval_metrics)

            # Save results
            save_results(eval_results, eval_metrics, results_dir)

            # Generate plots
            print("\nGenerating visualizations...")
            plot_results(eval_results, eval_metrics, results_dir)

            print()
            print("=" * 80)
            print("EVALUATION COMPLETED")
            print("=" * 80)
            print(f"Results saved to: {results_dir}")
            print()

        except Exception as e:
            print(f"\n\nERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Don't exit - still generate report with available data
    else:
        print("=" * 80)
        print("SKIPPING EVALUATION (as requested)")
        print("=" * 80)
        print()

    # 7. GENERATE FINAL REPORT
    print("=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80)

    experiment_metadata = {
        "experiment_name": config.experiment_name,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
        "device": device,
        "training_completed": not skip_training,
        "training_time_seconds": training_time,
        "evaluation_completed": not skip_eval,
        "timestamp": datetime.now().isoformat(),
    }

    if eval_metrics:
        experiment_metadata["eval_metrics"] = eval_metrics

    report_path = generate_report(
        config=config,
        metadata=experiment_metadata,
        output_dir=results_dir
    )

    print(f"Report saved to: {report_path}")
    print()

    return experiment_metadata


def main():
    """Main entry point."""
    args = parse_args()

    # Parse paths
    config_path = Path(args.config)
    resume_from = Path(args.resume_from) if args.resume_from else None
    eval_policy_path = Path(args.eval_policy) if args.eval_policy else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Validate config exists
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Handle eval-only mode
    skip_training = args.eval_only or args.no_train
    skip_eval = args.no_eval

    # Validate conflicting options
    if args.eval_only and args.no_eval:
        print("ERROR: Cannot use --eval-only and --no-eval together")
        sys.exit(1)

    if args.no_train and args.no_eval:
        print("ERROR: Nothing to do (both training and evaluation disabled)")
        sys.exit(1)

    # Run experiment
    print()
    print("=" * 80)
    print(f"RL EXPERIMENT RUNNER")
    print("=" * 80)
    print()

    try:
        results = run_experiment(
            config_path=config_path,
            resume_from=resume_from,
            skip_training=skip_training,
            skip_eval=skip_eval,
            eval_policy_path=eval_policy_path,
            eval_episodes=args.eval_episodes,
            device=args.device,
            output_dir=output_dir
        )

        print()
        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Experiment: {results['experiment_name']}")
        print(f"Results: {results['results_dir']}")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
