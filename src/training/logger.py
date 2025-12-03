"""Simple CSV-based logging for cross-platform RL training."""

import csv
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt


class CSVLogger:
    """
    Simple CSV logger for RL training metrics.

    Logs training metrics to separate CSV files for easy analysis
    and cross-platform compatibility.

    Parameters
    ----------
    log_dir : Path
        Directory to save log files
    experiment_name : str
        Name of the experiment (used in filenames)

    Examples
    --------
    >>> logger = CSVLogger(Path("logs"), "my_experiment")
    >>> logger.log_episode({"episode_num": 1, "reward": 100, "length": 50})
    >>> logger.close()
    """

    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.episodes_path = self.log_dir / f"{experiment_name}_episodes.csv"
        self.training_path = self.log_dir / f"{experiment_name}_training.csv"
        self.evaluation_path = self.log_dir / f"{experiment_name}_evaluation.csv"

        # File handles
        self._episodes_file: Optional[Any] = None
        self._training_file: Optional[Any] = None
        self._evaluation_file: Optional[Any] = None

        # CSV writers
        self._episodes_writer: Optional[csv.DictWriter] = None
        self._training_writer: Optional[csv.DictWriter] = None
        self._evaluation_writer: Optional[csv.DictWriter] = None

        # Initialize episode log
        self._init_episodes_log()
        # Training and evaluation logs initialized on first write

    def _init_episodes_log(self) -> None:
        """Initialize episodes CSV file with headers."""
        fieldnames = [
            'episode_num', 'reward', 'length', 'success',
            'dropout', 'pattern'
        ]

        # Check if file exists to determine if we need to write header
        file_exists = self.episodes_path.exists()

        self._episodes_file = open(self.episodes_path, 'a', newline='')
        self._episodes_writer = csv.DictWriter(
            self._episodes_file,
            fieldnames=fieldnames
        )

        if not file_exists:
            self._episodes_writer.writeheader()
            self._episodes_file.flush()

    def _init_training_log(self) -> None:
        """Initialize training CSV file with headers."""
        fieldnames = [
            'timestep', 'loss', 'entropy', 'value_loss',
            'policy_loss', 'learning_rate'
        ]

        file_exists = self.training_path.exists()

        self._training_file = open(self.training_path, 'a', newline='')
        self._training_writer = csv.DictWriter(
            self._training_file,
            fieldnames=fieldnames
        )

        if not file_exists:
            self._training_writer.writeheader()
            self._training_file.flush()

    def _init_evaluation_log(self) -> None:
        """Initialize evaluation CSV file with headers."""
        fieldnames = [
            'timestep', 'mean_reward', 'success_rate',
            'dropout_rate', 'mean_sessions'
        ]

        file_exists = self.evaluation_path.exists()

        self._evaluation_file = open(self.evaluation_path, 'a', newline='')
        self._evaluation_writer = csv.DictWriter(
            self._evaluation_file,
            fieldnames=fieldnames
        )

        if not file_exists:
            self._evaluation_writer.writeheader()
            self._evaluation_file.flush()

    def log_episode(self, episode_info: Dict[str, Any]) -> None:
        """
        Log information from a single episode.

        Parameters
        ----------
        episode_info : dict
            Episode information with keys:
            - episode_num: int
            - reward: float
            - length: int
            - success: bool
            - dropout: bool
            - pattern: str

        Examples
        --------
        >>> logger.log_episode({
        ...     "episode_num": 1,
        ...     "reward": 100.0,
        ...     "length": 50,
        ...     "success": True,
        ...     "dropout": False,
        ...     "pattern": "cold_stuck"
        ... })
        """
        if self._episodes_writer is None or self._episodes_file is None:
            self._init_episodes_log()

        self._episodes_writer.writerow(episode_info)  # type: ignore[union-attr]
        self._episodes_file.flush()  # type: ignore[union-attr]

    def log_training_step(self, step_info: Dict[str, Any]) -> None:
        """
        Log training step metrics.

        Parameters
        ----------
        step_info : dict
            Training step information with keys:
            - timestep: int
            - loss: float
            - entropy: float
            - value_loss: float
            - policy_loss: float
            - learning_rate: float

        Examples
        --------
        >>> logger.log_training_step({
        ...     "timestep": 1000,
        ...     "loss": 0.5,
        ...     "entropy": 0.8,
        ...     "value_loss": 0.3,
        ...     "policy_loss": 0.2,
        ...     "learning_rate": 3e-4
        ... })
        """
        if self._training_writer is None or self._training_file is None:
            self._init_training_log()

        self._training_writer.writerow(step_info)  # type: ignore[union-attr]
        self._training_file.flush()  # type: ignore[union-attr]

    def log_evaluation(self, eval_info: Dict[str, Any]) -> None:
        """
        Log evaluation metrics.

        Parameters
        ----------
        eval_info : dict
            Evaluation information with keys:
            - timestep: int
            - mean_reward: float
            - success_rate: float
            - dropout_rate: float
            - mean_sessions: float

        Examples
        --------
        >>> logger.log_evaluation({
        ...     "timestep": 10000,
        ...     "mean_reward": 85.5,
        ...     "success_rate": 0.75,
        ...     "dropout_rate": 0.10,
        ...     "mean_sessions": 42.3
        ... })
        """
        if self._evaluation_writer is None or self._evaluation_file is None:
            self._init_evaluation_log()

        self._evaluation_writer.writerow(eval_info)  # type: ignore[union-attr]
        self._evaluation_file.flush()  # type: ignore[union-attr]

    def close(self) -> None:
        """Close all open file handles."""
        if self._episodes_file is not None:
            self._episodes_file.close()
        if self._training_file is not None:
            self._training_file.close()
        if self._evaluation_file is not None:
            self._evaluation_file.close()

    def __enter__(self) -> "CSVLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close files."""
        self.close()


def plot_training_curves(log_dir: Path, experiment_name: Optional[str] = None) -> None:
    """
    Generate training curve plots from CSV logs.

    Reads CSV files and creates matplotlib plots for training metrics,
    saving them to log_dir/plots/.

    Parameters
    ----------
    log_dir : Path
        Directory containing log CSV files
    experiment_name : str, optional
        Experiment name to filter files. If None, uses all CSV files in directory.

    Examples
    --------
    >>> plot_training_curves(Path("logs"), "my_experiment")
    """
    log_dir = Path(log_dir)
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    if experiment_name:
        episodes_path = log_dir / f"{experiment_name}_episodes.csv"
        training_path = log_dir / f"{experiment_name}_training.csv"
        evaluation_path = log_dir / f"{experiment_name}_evaluation.csv"
    else:
        # Find any episode/training/evaluation CSVs
        csv_files = list(log_dir.glob("*_episodes.csv"))
        if not csv_files:
            print(f"No episode CSV files found in {log_dir}")
            return
        episodes_path = csv_files[0]
        experiment_name = episodes_path.stem.removesuffix("_episodes")
        training_path = log_dir / f"{experiment_name}_training.csv"
        evaluation_path = log_dir / f"{experiment_name}_evaluation.csv"

    # Plot episodes
    if episodes_path.exists():
        try:
            df = pd.read_csv(episodes_path)
            if not df.empty:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Episode Metrics - {experiment_name}")

                # Reward over episodes
                axes[0, 0].plot(df['episode_num'], df['reward'])
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].set_title('Episode Reward')
                axes[0, 0].grid(True)

                # Length over episodes
                axes[0, 1].plot(df['episode_num'], df['length'])
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Length (sessions)')
                axes[0, 1].set_title('Episode Length')
                axes[0, 1].grid(True)

                # Success rate (rolling average)
                if 'success' in df.columns:
                    window = min(100, len(df) // 10)
                    success_rate = df['success'].rolling(window=window, min_periods=1).mean()
                    axes[1, 0].plot(df['episode_num'], success_rate)
                    axes[1, 0].set_xlabel('Episode')
                    axes[1, 0].set_ylabel('Success Rate')
                    axes[1, 0].set_title(f'Success Rate (rolling avg, window={window})')
                    axes[1, 0].grid(True)

                # Dropout rate (rolling average)
                if 'dropout' in df.columns:
                    window = min(100, len(df) // 10)
                    dropout_rate = df['dropout'].rolling(window=window, min_periods=1).mean()
                    axes[1, 1].plot(df['episode_num'], dropout_rate)
                    axes[1, 1].set_xlabel('Episode')
                    axes[1, 1].set_ylabel('Dropout Rate')
                    axes[1, 1].set_title(f'Dropout Rate (rolling avg, window={window})')
                    axes[1, 1].grid(True)

                plt.tight_layout()
                plt.savefig(plots_dir / f"{experiment_name}_episodes.png", dpi=150)
                plt.close()
                print(f"Saved episode plots to {plots_dir / f'{experiment_name}_episodes.png'}")
        except Exception as e:
            print(f"Error plotting episodes: {e}")

    # Plot training metrics
    if training_path.exists():
        try:
            df = pd.read_csv(training_path)
            if not df.empty:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Training Metrics - {experiment_name}")

                # Total loss
                if 'loss' in df.columns:
                    axes[0, 0].plot(df['timestep'], df['loss'])
                    axes[0, 0].set_xlabel('Timestep')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].set_title('Total Loss')
                    axes[0, 0].grid(True)

                # Policy and value loss
                if 'policy_loss' in df.columns and 'value_loss' in df.columns:
                    axes[0, 1].plot(df['timestep'], df['policy_loss'], label='Policy Loss')
                    axes[0, 1].plot(df['timestep'], df['value_loss'], label='Value Loss')
                    axes[0, 1].set_xlabel('Timestep')
                    axes[0, 1].set_ylabel('Loss')
                    axes[0, 1].set_title('Policy and Value Loss')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)

                # Entropy
                if 'entropy' in df.columns:
                    axes[1, 0].plot(df['timestep'], df['entropy'])
                    axes[1, 0].set_xlabel('Timestep')
                    axes[1, 0].set_ylabel('Entropy')
                    axes[1, 0].set_title('Policy Entropy')
                    axes[1, 0].grid(True)

                # Learning rate
                if 'learning_rate' in df.columns:
                    axes[1, 1].plot(df['timestep'], df['learning_rate'])
                    axes[1, 1].set_xlabel('Timestep')
                    axes[1, 1].set_ylabel('Learning Rate')
                    axes[1, 1].set_title('Learning Rate')
                    axes[1, 1].grid(True)

                plt.tight_layout()
                plt.savefig(plots_dir / f"{experiment_name}_training.png", dpi=150)
                plt.close()
                print(f"Saved training plots to {plots_dir / f'{experiment_name}_training.png'}")
        except Exception as e:
            print(f"Error plotting training metrics: {e}")

    # Plot evaluation metrics
    if evaluation_path.exists():
        try:
            df = pd.read_csv(evaluation_path)
            if not df.empty:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Evaluation Metrics - {experiment_name}")

                # Mean reward
                if 'mean_reward' in df.columns:
                    axes[0, 0].plot(df['timestep'], df['mean_reward'])
                    axes[0, 0].set_xlabel('Timestep')
                    axes[0, 0].set_ylabel('Mean Reward')
                    axes[0, 0].set_title('Mean Evaluation Reward')
                    axes[0, 0].grid(True)

                # Success rate
                if 'success_rate' in df.columns:
                    axes[0, 1].plot(df['timestep'], df['success_rate'])
                    axes[0, 1].set_xlabel('Timestep')
                    axes[0, 1].set_ylabel('Success Rate')
                    axes[0, 1].set_title('Success Rate')
                    axes[0, 1].grid(True)
                    axes[0, 1].set_ylim([0, 1])

                # Dropout rate
                if 'dropout_rate' in df.columns:
                    axes[1, 0].plot(df['timestep'], df['dropout_rate'])
                    axes[1, 0].set_xlabel('Timestep')
                    axes[1, 0].set_ylabel('Dropout Rate')
                    axes[1, 0].set_title('Dropout Rate')
                    axes[1, 0].grid(True)
                    axes[1, 0].set_ylim([0, 1])

                # Mean sessions
                if 'mean_sessions' in df.columns:
                    axes[1, 1].plot(df['timestep'], df['mean_sessions'])
                    axes[1, 1].set_xlabel('Timestep')
                    axes[1, 1].set_ylabel('Mean Sessions')
                    axes[1, 1].set_title('Mean Number of Sessions')
                    axes[1, 1].grid(True)

                plt.tight_layout()
                plt.savefig(plots_dir / f"{experiment_name}_evaluation.png", dpi=150)
                plt.close()
                print(f"Saved evaluation plots to {plots_dir / f'{experiment_name}_evaluation.png'}")
        except Exception as e:
            print(f"Error plotting evaluation metrics: {e}")

    print(f"\nAll plots saved to {plots_dir}/")
