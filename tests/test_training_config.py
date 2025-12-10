"""Tests for training configuration module."""

import pytest
import tempfile
from pathlib import Path
from src.training import (
    TrainingConfig,
    load_config,
    save_config,
    get_default_config,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Test that default config is created successfully."""
        config = get_default_config()

        assert config.experiment_name == "therapy_agent_training"
        assert config.mechanism == "frequency_amplifier"
        assert config.threshold == 0.9
        assert config.max_sessions == 100
        assert config.entropy == 0.5
        assert config.bond_alpha is None
        assert config.bond_offset == 0.7
        assert config.history_weight == 1.0
        assert config.enable_parataxic is True
        assert config.baseline_accuracy == 0.5
        assert config.total_timesteps == 500_000
        assert config.n_envs == 8
        assert config.learning_rate == 3e-4
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.01
        assert config.hidden_size == 256
        assert config.lstm_hidden_size == 128
        assert config.log_dir == "logs"
        assert config.save_freq == 10_000
        assert config.eval_freq == 5_000
        assert config.eval_episodes == 100
        assert config.seed == 42
        assert len(config.patterns) == 5

    def test_custom_config(self):
        """Test creating config with custom parameters."""
        config = TrainingConfig(
            experiment_name="test_experiment",
            patterns=["cold_stuck"],
            threshold=0.8,
            learning_rate=1e-4
        )

        assert config.experiment_name == "test_experiment"
        assert config.patterns == ["cold_stuck"]
        assert config.threshold == 0.8
        assert config.learning_rate == 1e-4

    def test_config_validation_threshold(self):
        """Test threshold validation."""
        # Invalid: threshold = 0
        with pytest.raises(ValueError, match="threshold must be in"):
            TrainingConfig(threshold=0.0)

        # Invalid: threshold > 1
        with pytest.raises(ValueError, match="threshold must be in"):
            TrainingConfig(threshold=1.5)

        # Invalid: threshold < 0
        with pytest.raises(ValueError, match="threshold must be in"):
            TrainingConfig(threshold=-0.1)

        # Valid: threshold = 0.5
        config = TrainingConfig(threshold=0.5)
        assert config.threshold == 0.5

        # Valid: threshold = 1.0
        config = TrainingConfig(threshold=1.0)
        assert config.threshold == 1.0

    def test_config_validation_max_sessions(self):
        """Test max_sessions validation."""
        # Invalid: max_sessions <= 0
        with pytest.raises(ValueError, match="max_sessions must be positive"):
            TrainingConfig(max_sessions=0)

        with pytest.raises(ValueError, match="max_sessions must be positive"):
            TrainingConfig(max_sessions=-10)

        # Valid: max_sessions > 0
        config = TrainingConfig(max_sessions=50)
        assert config.max_sessions == 50

    def test_config_validation_entropy(self):
        """Test entropy validation."""
        # Invalid: entropy <= 0
        with pytest.raises(ValueError, match="entropy must be positive"):
            TrainingConfig(entropy=0.0)

        with pytest.raises(ValueError, match="entropy must be positive"):
            TrainingConfig(entropy=-0.5)

        # Valid: entropy > 0
        config = TrainingConfig(entropy=0.3)
        assert config.entropy == 0.3

    def test_config_validation_patterns(self):
        """Test patterns list validation."""
        # Invalid: empty patterns list
        with pytest.raises(ValueError, match="patterns list cannot be empty"):
            TrainingConfig(patterns=[])

    def test_config_validation_mechanism(self):
        """Test mechanism validation."""
        # Valid mechanisms
        valid_mechanisms = [
            'bond_only',
            'frequency_amplifier',
            'conditional_amplifier',
            'bond_weighted_frequency_amplifier',
            'bond_weighted_conditional_amplifier'
        ]

        for mechanism in valid_mechanisms:
            config = TrainingConfig(mechanism=mechanism)
            assert config.mechanism == mechanism

        # Invalid mechanism
        with pytest.raises(ValueError, match="mechanism must be one of"):
            TrainingConfig(mechanism="invalid_mechanism")

    def test_config_validation_n_envs(self):
        """Test n_envs validation."""
        # Invalid: n_envs <= 0
        with pytest.raises(ValueError, match="n_envs must be positive"):
            TrainingConfig(n_envs=0)

        with pytest.raises(ValueError, match="n_envs must be positive"):
            TrainingConfig(n_envs=-1)

    def test_config_validation_learning_rate(self):
        """Test learning_rate validation."""
        # Invalid: learning_rate <= 0
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.001)

    def test_config_validation_gamma(self):
        """Test gamma validation."""
        # Invalid: gamma < 0
        with pytest.raises(ValueError, match="gamma must be in"):
            TrainingConfig(gamma=-0.1)

        # Invalid: gamma > 1
        with pytest.raises(ValueError, match="gamma must be in"):
            TrainingConfig(gamma=1.5)

        # Valid: gamma in [0, 1]
        config = TrainingConfig(gamma=0.0)
        assert config.gamma == 0.0

        config = TrainingConfig(gamma=1.0)
        assert config.gamma == 1.0

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(
            experiment_name="test",
            patterns=["cold_stuck"],
            threshold=0.8
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['experiment_name'] == "test"
        assert config_dict['patterns'] == ["cold_stuck"]
        assert config_dict['threshold'] == 0.8
        assert 'learning_rate' in config_dict
        assert 'mechanism' in config_dict

    def test_get_env_kwargs(self):
        """Test extracting TherapyEnv kwargs from config."""
        config = TrainingConfig(
            patterns=["cold_stuck", "dominant_stuck"],
            mechanism="frequency_amplifier",
            threshold=0.85,
            max_sessions=50,
            entropy=0.4,
            bond_alpha=0.3,
            bond_offset=0.6,
            history_weight=0.8,
            enable_parataxic=False,
            baseline_accuracy=0.7
        )

        env_kwargs = config.get_env_kwargs()

        assert env_kwargs['mechanism'] == "frequency_amplifier"
        assert env_kwargs['pattern'] == ["cold_stuck", "dominant_stuck"]
        assert env_kwargs['threshold'] == 0.85
        assert env_kwargs['max_sessions'] == 50
        assert env_kwargs['entropy'] == 0.4
        assert env_kwargs['bond_alpha'] == 0.3
        assert env_kwargs['bond_offset'] == 0.6
        assert env_kwargs['history_weight'] == 0.8
        assert env_kwargs['enable_parataxic'] is False
        assert env_kwargs['baseline_accuracy'] == 0.7

        # Should not include RL or logging parameters
        assert 'learning_rate' not in env_kwargs
        assert 'n_envs' not in env_kwargs
        assert 'log_dir' not in env_kwargs
        assert 'total_timesteps' not in env_kwargs


class TestConfigSaveLoad:
    """Tests for config save/load functionality."""

    def test_save_and_load_config(self):
        """Test YAML round-trip (save then load)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Create and save config
            original_config = TrainingConfig(
                experiment_name="round_trip_test",
                patterns=["cold_stuck", "dominant_stuck"],
                mechanism="conditional_amplifier",
                threshold=0.75,
                max_sessions=80,
                learning_rate=1e-4,
                n_envs=4,
                seed=123
            )

            save_config(original_config, config_path)

            # Verify file exists
            assert config_path.exists()

            # Load config
            loaded_config = load_config(config_path)

            # Verify all fields match
            assert loaded_config.experiment_name == original_config.experiment_name
            assert loaded_config.patterns == original_config.patterns
            assert loaded_config.mechanism == original_config.mechanism
            assert loaded_config.threshold == original_config.threshold
            assert loaded_config.max_sessions == original_config.max_sessions
            assert loaded_config.learning_rate == original_config.learning_rate
            assert loaded_config.n_envs == original_config.n_envs
            assert loaded_config.seed == original_config.seed

    def test_save_creates_parent_directory(self):
        """Test that save_config creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "nested" / "config.yaml"

            config = get_default_config()
            save_config(config, config_path)

            assert config_path.exists()
            assert config_path.parent.exists()

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_empty_yaml(self):
        """Test loading empty YAML file uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "empty.yaml"

            # Create empty YAML file
            config_path.write_text("")

            # Should load with default values
            config = load_config(config_path)
            default_config = get_default_config()

            assert config.experiment_name == default_config.experiment_name
            assert config.patterns == default_config.patterns

    def test_load_invalid_yaml(self):
        """Test loading malformed YAML raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"

            # Create YAML with invalid parameter
            config_path.write_text("invalid_parameter: 123\nthreshold: 0.9")

            # Should raise ValueError for unexpected keyword argument
            with pytest.raises(ValueError, match="Invalid config parameters"):
                load_config(config_path)

    def test_load_invalid_values(self):
        """Test loading YAML with invalid values raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid_values.yaml"

            # Create YAML with invalid threshold
            config_path.write_text("threshold: 1.5")

            # Should raise ValueError from __post_init__ validation
            with pytest.raises(ValueError, match="threshold must be in"):
                load_config(config_path)

    def test_save_preserves_types(self):
        """Test that save/load preserves data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "types.yaml"

            original_config = TrainingConfig(
                patterns=["cold_stuck"],
                threshold=0.8,
                max_sessions=100,
                enable_parataxic=True,
                bond_alpha=None,
                learning_rate=3e-4
            )

            save_config(original_config, config_path)
            loaded_config = load_config(config_path)

            # Check types
            assert isinstance(loaded_config.patterns, list)
            assert isinstance(loaded_config.threshold, float)
            assert isinstance(loaded_config.max_sessions, int)
            assert isinstance(loaded_config.enable_parataxic, bool)
            assert loaded_config.bond_alpha is None
            assert isinstance(loaded_config.learning_rate, float)


class TestConfigIntegration:
    """Integration tests for config usage."""

    def test_config_yaml_format(self):
        """Test that saved YAML is human-readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "readable.yaml"

            config = TrainingConfig(experiment_name="test")
            save_config(config, config_path)

            yaml_content = config_path.read_text()

            # Check that YAML contains expected fields
            assert "experiment_name: test" in yaml_content
            assert "mechanism:" in yaml_content
            assert "patterns:" in yaml_content
            assert "learning_rate:" in yaml_content

            # Should not use flow style (compact format)
            assert not yaml_content.startswith("{")

    def test_partial_config_override(self):
        """Test loading YAML with partial parameters uses defaults for rest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "partial.yaml"

            # Save YAML with only a few parameters
            config_path.write_text("""
experiment_name: partial_test
patterns:
  - cold_stuck
threshold: 0.7
""")

            config = load_config(config_path)

            # Check overridden values
            assert config.experiment_name == "partial_test"
            assert config.patterns == ["cold_stuck"]
            assert config.threshold == 0.7

            # Check default values are used for rest
            default = get_default_config()
            assert config.mechanism == default.mechanism
            assert config.learning_rate == default.learning_rate
            assert config.n_envs == default.n_envs
