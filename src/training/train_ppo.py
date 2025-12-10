"""PPO training script for therapy RL agents using Tianshou."""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger

from src.environment import TherapyEnv
from src.training.config import TrainingConfig, load_config, save_config
from src.training.networks import make_therapy_networks


def make_env(config: TrainingConfig, seed: int) -> gymnasium.Env:
    """
    Create a single TherapyEnv instance.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    seed : int
        Random seed for this environment

    Returns
    -------
    gymnasium.Env
        Configured environment instance (potentially wrapped with OmniscientObservationWrapper)
    """
    env = TherapyEnv(**config.get_env_kwargs(), random_state=seed)

    # Apply omniscient wrapper if requested
    if hasattr(config, 'use_omniscient_wrapper') and config.use_omniscient_wrapper:
        from src.environment.omniscient_wrapper import OmniscientObservationWrapper
        env = OmniscientObservationWrapper(env)

    return env


def make_vectorized_envs(config: TrainingConfig) -> DummyVectorEnv:
    """
    Create vectorized training environments.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration

    Returns
    -------
    DummyVectorEnv
        Vectorized environments for parallel training
    """
    return DummyVectorEnv([
        lambda i=i: make_env(config, config.seed + i)
        for i in range(config.n_envs)
    ])


def make_eval_envs(config: TrainingConfig, n_envs: int = 10) -> DummyVectorEnv:
    """
    Create vectorized evaluation environments.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    n_envs : int, default=10
        Number of parallel evaluation environments

    Returns
    -------
    DummyVectorEnv
        Vectorized environments for evaluation
    """
    return DummyVectorEnv([
        lambda i=i: make_env(config, config.seed + 10000 + i)
        for i in range(n_envs)
    ])


def train(
    config: TrainingConfig,
    output_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None
) -> PPOPolicy:
    """
    Train a PPO agent on the therapy environment.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    output_dir : Path, optional
        Directory to save models and logs. Defaults to models/{experiment_name}
    resume_from : Path, optional
        Path to checkpoint to resume training from

    Returns
    -------
    PPOPolicy
        Trained policy  
    """
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup output directory
    if output_dir is None:
        output_dir = Path("models") / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, output_dir / "config.yaml")
    print(f"Configuration saved to {output_dir / 'config.yaml'}")

    # Create environments
    print(f"Creating {config.n_envs} training environments...")
    train_envs = make_vectorized_envs(config)

    print(f"Creating evaluation environments...")
    eval_envs = make_eval_envs(config, n_envs=min(10, config.n_envs))

    # Get observation and action space
    env = train_envs.workers[0].env  # type: ignore[attr-defined]
    obs_space = env.observation_space
    act_space = env.action_space

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create networks (omniscient or standard)
    if hasattr(config, 'use_omniscient_wrapper') and config.use_omniscient_wrapper:
        print("Creating OMNISCIENT actor and critic networks...")
        from src.training.omniscient_networks import make_omniscient_networks
        actor, critic = make_omniscient_networks(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=(config.hidden_size, config.hidden_size),
            device=device
        )
    else:
        print("Creating standard actor and critic networks...")
        actor, critic = make_therapy_networks(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=(config.hidden_size, config.hidden_size),
            device=device
        )

    # Create optimizers
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate
    )

    # Create PPO policy
    print("Initializing PPO policy...")
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=config.gamma,
        gae_lambda=config.gae_lambda,
        max_grad_norm=config.max_grad_norm,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        reward_normalization=True,
        action_bound_method=None, # Overrides default "clip" for discrete actions
        action_scaling=False,
        eps_clip=config.clip_range,
        value_clip=True,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=False,
        action_space=act_space
    )

    # Load checkpoint if resuming
    checkpoint_metadata = None
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        # Handle both checkpoint formats:
        # 1. Full checkpoint dict with 'model', 'optim', metadata (checkpoint_*.pth)
        # 2. Direct state dict (policy_final.pth, policy_best.pth)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Full checkpoint with metadata
            policy.load_state_dict(checkpoint['model'])
            optim.load_state_dict(checkpoint['optim'])

            # Store metadata for later tensorboard logger initialization
            checkpoint_metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'env_step': checkpoint.get('env_step', 0),
                'gradient_step': checkpoint.get('gradient_step', 0)
            }
            print(f"Resuming from: epoch={checkpoint_metadata['epoch']}, "
                  f"step={checkpoint_metadata['env_step']}, "
                  f"gradient_step={checkpoint_metadata['gradient_step']}")
        else:
            # Direct state dict (no metadata)
            policy.load_state_dict(checkpoint)
            print("Warning: Loading direct state dict without metadata. "
                  "Training will restart from step 0. "
                  "Use checkpoint_*.pth files for proper resumption.")

    # Create collectors
    print("Setting up data collectors...")
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(config.batch_size * config.n_envs, config.n_envs)
    )

    eval_collector = Collector(policy, eval_envs)

    # Create CSV logger for cross-platform compatibility
    csv_path = log_dir / f"{config.experiment_name}_progress.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'epoch', 'step', 'train_reward_mean', 'train_reward_std',
        'train_length_mean', 'test_reward_mean', 'test_reward_std',
        'test_length_mean', 'duration'
    ])
    csv_writer.writeheader()
    print(f"Logging training progress to {csv_path}")

    # Setup tensorboard logger for training metrics and resumption
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))
    tb_logger = TensorboardLogger(writer, update_interval=100)

    # Note: TensorboardLogger.restore_data() has a bug - it reads .step instead of .value
    # So we'll manually restore trainer state instead of using resume_from_log

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        """Save model checkpoint.

        Note: env_step is the CUMULATIVE total from trainer.env_step,
        which should include restored checkpoint steps after resumption.
        """
        if env_step % config.save_freq == 0 or env_step >= config.total_timesteps:
            ckpt_path = output_dir / f"checkpoint_{env_step}.pth"
            print(f"Saving checkpoint: epoch={epoch}, env_step={env_step}, gradient_step={gradient_step}")
            torch.save({
                'model': policy.state_dict(),
                'optim': optim.state_dict(),
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
            }, ckpt_path)

            print(f"Checkpoint saved to {ckpt_path} (epoch={epoch}, step={env_step})")
            return str(ckpt_path)
        return ""

    def log_metrics_fn(epoch: int, env_step: int, info: dict) -> None:
        """Log metrics to CSV."""
        csv_writer.writerow({
            'epoch': epoch,
            'step': env_step,
            'train_reward_mean': info.get('train_reward', 0),
            'train_reward_std': info.get('train_reward_std', 0),
            'train_length_mean': info.get('train_length', 0),
            'test_reward_mean': info.get('test_reward', 0),
            'test_reward_std': info.get('test_reward_std', 0),
            'test_length_mean': info.get('test_length', 0),
            'duration': info.get('duration', 0)
        })
        csv_file.flush()

    # Train
    print("\n" + "=" * 60)
    print(f"Starting training: {config.experiment_name}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num epochs: {config.n_epochs}")
    print("=" * 60 + "\n")

    try:
        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=eval_collector,
            max_epoch=int(config.total_timesteps // (config.batch_size * config.n_envs)),
            step_per_epoch=config.batch_size * config.n_envs,
            repeat_per_collect=config.n_epochs,
            episode_per_test=config.eval_episodes,
            batch_size=config.batch_size,
            step_per_collect=config.batch_size * config.n_envs,
            save_best_fn=lambda policy: torch.save(
                policy.state_dict(),
                output_dir / "policy_best.pth"
            ),
            save_checkpoint_fn=save_checkpoint_fn,
            logger=tb_logger,
            test_in_train=False,
            resume_from_log=False,  # Disabled - has bug, we restore manually instead
            verbose=True,
        )

        # Manually restore trainer state if resuming (TensorboardLogger.restore_data is buggy)
        if checkpoint_metadata is not None:
            print(f"\nRestoring trainer state from checkpoint...")
            print(f"  Checkpoint metadata: epoch={checkpoint_metadata['epoch']}, "
                  f"env_step={checkpoint_metadata['env_step']}, "
                  f"gradient_step={checkpoint_metadata['gradient_step']}")

            trainer.start_epoch = checkpoint_metadata['epoch']
            trainer.epoch = checkpoint_metadata['epoch']
            trainer.env_step = checkpoint_metadata['env_step']
            trainer._gradient_step = checkpoint_metadata['gradient_step']

            print(f"  Trainer state after restoration: epoch={trainer.epoch}, "
                  f"env_step={trainer.env_step}, gradient_step={trainer._gradient_step}")
            print(f"  Training will continue from epoch {trainer.epoch+1} to epoch {trainer.max_epoch}\n")

        # Run full training
        print(f"Starting training loop...")
        result = trainer.run()

        # Log final metrics
        log_metrics_fn(
            getattr(result, 'epoch', 0),  # type: ignore[arg-type]
            getattr(result, 'env_step', 0),  # type: ignore[arg-type]
            {
                'train_reward': getattr(result, 'train_reward', 0),
                'train_reward_std': getattr(result, 'train_reward_std', 0),
                'train_length': getattr(result, 'train_length', 0),
                'test_reward': getattr(result, 'test_reward', 0),
                'test_reward_std': getattr(result, 'test_reward_std', 0),
                'test_length': getattr(result, 'test_length', 0),
                'duration': getattr(result, 'duration', 0)
            }
        )

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best test reward: {getattr(result, 'best_reward', 0):.2f}")
        print(f"Final policy saved to {output_dir / 'policy_best.pth'}")
        print("=" * 60 + "\n")

    finally:
        csv_file.close()
        train_envs.close()
        eval_envs.close()

    # Save final policy
    torch.save(policy.state_dict(), output_dir / "policy_final.pth")
    print(f"Final policy saved to {output_dir / 'policy_final.pth'}")

    return policy


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for therapy environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file or individual arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If provided, other arguments are ignored."
    )

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="therapy_ppo",
        help="Experiment name for logging and saving"
    )

    # Environment parameters
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["cold_stuck", "dominant_stuck", "submissive_stuck", "mixed_random", "cold_warm"],
        help="Client behavior patterns"
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default="frequency_amplifier",
        choices=[
            "bond_only",
            "frequency_amplifier",
            "conditional_amplifier",
            "bond_weighted_frequency_amplifier",
            "bond_weighted_conditional_amplifier"
        ],
        help="Client expectation mechanism"
    )
    parser.add_argument("--threshold", type=float, default=0.9, help="Success threshold")
    parser.add_argument("--max-sessions", type=int, default=100, help="Max sessions per episode")
    parser.add_argument("--entropy", type=float, default=0.5, help="Client action entropy")

    # RL parameters
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")

    # Network architecture
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Resume training
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models. Defaults to models/{experiment_name}"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config is not None:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Override config values with explicitly provided CLI arguments
        # This allows resuming with different settings (e.g., extended total_timesteps)
        overrides_applied = False

        if args.total_timesteps != 500_000:  # Default is 500k
            print(f"  Overriding total_timesteps: {config.total_timesteps} → {args.total_timesteps}")
            config.total_timesteps = args.total_timesteps
            overrides_applied = True

        if args.learning_rate != 3e-4:  # Default is 3e-4
            print(f"  Overriding learning_rate: {config.learning_rate} → {args.learning_rate}")
            config.learning_rate = args.learning_rate
            overrides_applied = True

        if args.batch_size != 64:  # Default is 64
            print(f"  Overriding batch_size: {config.batch_size} → {args.batch_size}")
            config.batch_size = args.batch_size
            overrides_applied = True

        if args.n_envs != 8:  # Default is 8
            print(f"  Overriding n_envs: {config.n_envs} → {args.n_envs}")
            config.n_envs = args.n_envs
            overrides_applied = True

        if args.hidden_size != 256:  # Default is 256
            print(f"  Overriding hidden_size: {config.hidden_size} → {args.hidden_size}")
            config.hidden_size = args.hidden_size
            overrides_applied = True

        if overrides_applied:
            print()  # Add blank line after overrides
    else:
        config = TrainingConfig(
            experiment_name=args.experiment_name,
            patterns=args.patterns,
            mechanism=args.mechanism,
            threshold=args.threshold,
            max_sessions=args.max_sessions,
            entropy=args.entropy,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            hidden_size=args.hidden_size,
            log_dir=args.log_dir,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            seed=args.seed
        )

    # Parse output directory
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Parse resume path
    resume_from = Path(args.resume_from) if args.resume_from else None

    # Train
    try:
        train(config, output_dir=output_dir, resume_from=resume_from)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
