"""Training configuration for RL agent training."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    """
    Training configuration for therapist RL agents.

    Combines environment parameters, RL algorithm hyperparameters,
    and experiment tracking settings.
    """

    # Experiment metadata
    experiment_name: str = "therapy_agent_training"

    # === Environment Parameters ===
    # Client configuration
    patterns: List[str] = field(default_factory=lambda: [
        "cold_stuck", "dominant_stuck", "submissive_stuck",
        "mixed_random", "cold_warm"
    ])
    mechanism: str = "frequency_amplifier"
    threshold: float = 0.9
    max_sessions: int = 100
    entropy: float = 0.5
    bond_alpha: Optional[float] = None  # None = use config.BOND_ALPHA
    bond_offset: float = 0.7
    history_weight: float = 1.0
    enable_perception: bool = True
    baseline_accuracy: float = 0.5

    # === RL Algorithm Parameters (PPO defaults) ===
    total_timesteps: int = 500_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5  # Value function loss coefficient
    max_grad_norm: Optional[float] = 0.5  # Gradient clipping norm (None = disabled)

    # Network architecture
    hidden_size: int = 256
    lstm_hidden_size: int = 128

    # === Omniscient-Specific Parameters ===
    use_omniscient_wrapper: bool = False  # Enable omniscient observation wrapper

    # === Logging and Checkpointing ===
    log_dir: str = "logs"
    save_freq: int = 10_000
    eval_freq: int = 5_000
    eval_episodes: int = 100
    seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate threshold
        if not (0 < self.threshold <= 1):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

        # Validate max_sessions
        if self.max_sessions <= 0:
            raise ValueError(f"max_sessions must be positive, got {self.max_sessions}")

        # Validate entropy
        if self.entropy <= 0:
            raise ValueError(f"entropy must be positive, got {self.entropy}")

        # Validate pattern list
        if not self.patterns:
            raise ValueError("patterns list cannot be empty")

        # Validate mechanism
        valid_mechanisms = [
            'bond_only', 'frequency_amplifier', 'conditional_amplifier',
            'bond_weighted_frequency_amplifier', 'bond_weighted_conditional_amplifier'
        ]
        if self.mechanism not in valid_mechanisms:
            raise ValueError(f"mechanism must be one of {valid_mechanisms}")

        # Validate RL parameters
        if self.n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {self.n_envs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def get_env_kwargs(self) -> Dict[str, Any]:
        """Get TherapyEnv constructor kwargs from this config."""
        return {
            'mechanism': self.mechanism,
            'pattern': self.patterns,
            'threshold': self.threshold,
            'max_sessions': self.max_sessions,
            'entropy': self.entropy,
            'history_weight': self.history_weight,
            'bond_alpha': self.bond_alpha,
            'bond_offset': self.bond_offset,
            'enable_perception': self.enable_perception,
            'baseline_accuracy': self.baseline_accuracy,
        }


def load_config(path: str | Path) -> TrainingConfig:
    """
    Load training configuration from YAML file.

    Parameters
    ----------
    path : str or Path
        Path to YAML config file

    Returns
    -------
    TrainingConfig
        Loaded configuration

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If YAML is malformed or contains invalid parameters

    Examples
    --------
    >>> config = load_config("configs/experiment1.yaml")
    >>> config.experiment_name
    'experiment1'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle None case (empty YAML)
    if config_dict is None:
        config_dict = {}

    try:
        return TrainingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid config parameters in {path}: {e}")


def save_config(config: TrainingConfig, path: str | Path) -> None:
    """
    Save training configuration to YAML file.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to save
    path : str or Path
        Output path for YAML file

    Examples
    --------
    >>> config = get_default_config()
    >>> save_config(config, "configs/default.yaml")
    """
    path = Path(path)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> TrainingConfig:
    """
    Create default training configuration.

    Uses defaults from src.config module for environment parameters
    and standard PPO hyperparameters for RL training.

    Returns
    -------
    TrainingConfig
        Default configuration instance

    Examples
    --------
    >>> config = get_default_config()
    >>> config.mechanism
    'frequency_amplifier'
    >>> config.learning_rate
    3e-4
    """
    return TrainingConfig()
