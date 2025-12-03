"""Example usage of CSVLogger for RL training."""

from pathlib import Path
from src.training import CSVLogger, plot_training_curves


def example_usage():
    """Demonstrate CSVLogger usage."""

    # Create logger
    log_dir = Path("logs/example")
    logger = CSVLogger(log_dir, "example_experiment")

    print("Logging example data...")

    # Log some example episodes
    for episode in range(1, 11):
        logger.log_episode({
            "episode_num": episode,
            "reward": 100 - episode * 5,  # Decreasing reward
            "length": 30 + episode,
            "success": episode % 3 == 0,  # Every 3rd episode succeeds
            "dropout": episode % 5 == 0,  # Every 5th episode has dropout
            "pattern": "cold_stuck" if episode % 2 == 0 else "dominant_stuck"
        })

    # Log some training steps
    for step in range(0, 1000, 100):
        logger.log_training_step({
            "timestep": step,
            "loss": 0.5 - step / 2000,  # Decreasing loss
            "entropy": 0.8 - step / 2500,  # Decreasing entropy
            "value_loss": 0.3 - step / 3000,
            "policy_loss": 0.2 - step / 5000,
            "learning_rate": 3e-4
        })

    # Log some evaluation metrics
    for step in range(0, 1000, 250):
        logger.log_evaluation({
            "timestep": step,
            "mean_reward": 50 + step / 20,  # Increasing reward
            "success_rate": 0.3 + step / 3000,  # Increasing success
            "dropout_rate": 0.2 - step / 5000,  # Decreasing dropout
            "mean_sessions": 40 - step / 100
        })

    # Close logger
    logger.close()

    print(f"✓ Logged data to {log_dir}/")
    print(f"  - {logger.episodes_path.name}")
    print(f"  - {logger.training_path.name}")
    print(f"  - {logger.evaluation_path.name}")

    # Generate plots
    print("\nGenerating plots...")
    plot_training_curves(log_dir, "example_experiment")


def example_context_manager():
    """Demonstrate using CSVLogger as context manager."""

    log_dir = Path("logs/example_context")

    with CSVLogger(log_dir, "context_example") as logger:
        logger.log_episode({
            "episode_num": 1,
            "reward": 100.0,
            "length": 50,
            "success": True,
            "dropout": False,
            "pattern": "cold_stuck"
        })

    print(f"✓ Context manager example completed: {log_dir}/")


if __name__ == "__main__":
    print("=" * 60)
    print("CSVLogger Example Usage")
    print("=" * 60 + "\n")

    example_usage()
    print()
    example_context_manager()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
