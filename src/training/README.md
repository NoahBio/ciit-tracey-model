# PPO Training for Therapy RL Agents

This module provides tools for training reinforcement learning agents using Proximal Policy Optimization (PPO) with the Tianshou library.

## Components

### 1. Configuration (`config.py`)
- `TrainingConfig`: Dataclass for training parameters
- `load_config()`: Load configuration from YAML
- `save_config()`: Save configuration to YAML
- `get_default_config()`: Create default configuration

### 2. Networks (`networks.py`)
- `TherapyNet`: Feature extraction network for Dict observation space
- `Actor`: Policy network for discrete actions
- `Critic`: Value function network
- `make_therapy_networks()`: Factory function to create actor-critic pair

### 3. Training Script (`train_ppo.py`)
Main training script with CLI interface.

## Installation

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using a Config File

Create a YAML config file (see `configs/example_experiment.yaml`):

```bash
python -m src.training.train_ppo --config configs/example_experiment.yaml
```

### Option 2: Using CLI Arguments

```bash
python -m src.training.train_ppo \
  --experiment-name "my_experiment" \
  --patterns cold_stuck dominant_stuck submissive_stuck \
  --mechanism frequency_amplifier \
  --threshold 0.9 \
  --total-timesteps 500000 \
  --n-envs 8 \
  --learning-rate 0.0003 \
  --batch-size 64 \
  --hidden-size 256 \
  --seed 42
```

### Option 3: Quick Test Run

For quick testing with minimal settings:

```bash
python -m src.training.train_ppo \
  --config configs/test_training.yaml
```

This runs a short training session (10k timesteps) with 2 environments.

## CLI Arguments

### Experiment Settings
- `--config`: Path to YAML config (overrides other args if provided)
- `--experiment-name`: Name for this experiment
- `--output-dir`: Output directory for models (default: `models/{experiment_name}`)
- `--resume-from`: Path to checkpoint to resume training from

### Environment Parameters
- `--patterns`: Client behavior patterns (space-separated list)
- `--mechanism`: Client expectation mechanism
  - Choices: `bond_only`, `frequency_amplifier`, `conditional_amplifier`, `bond_weighted_frequency_amplifier`, `bond_weighted_conditional_amplifier`
- `--threshold`: Success threshold (0-1)
- `--max-sessions`: Maximum sessions per episode
- `--entropy`: Client action entropy/temperature

### RL Hyperparameters
- `--total-timesteps`: Total training timesteps
- `--n-envs`: Number of parallel environments
- `--learning-rate`: Learning rate for optimizer
- `--batch-size`: Batch size for updates
- `--n-epochs`: Number of epochs per update
- `--gamma`: Discount factor
- `--gae-lambda`: GAE lambda parameter
- `--clip-range`: PPO clipping parameter
- `--ent-coef`: Entropy coefficient

### Network Architecture
- `--hidden-size`: Size of hidden layers

### Logging
- `--log-dir`: Directory for logs
- `--save-freq`: Checkpoint save frequency (timesteps)
- `--eval-freq`: Evaluation frequency (timesteps)
- `--eval-episodes`: Number of episodes per evaluation
- `--seed`: Random seed

## Outputs

### Directory Structure

Training creates the following directory structure:

```
models/{experiment_name}/
├── config.yaml              # Saved configuration
├── policy_best.pth          # Best policy (by eval reward)
├── policy_final.pth         # Final policy after training
└── checkpoint_*.pth         # Periodic checkpoints

logs/{experiment_name}/
└── {experiment_name}_progress.csv  # Training metrics
```

### Training Metrics

The CSV log contains:
- `epoch`: Training epoch
- `step`: Environment step count
- `train_reward_mean`: Mean training episode reward
- `train_reward_std`: Std dev of training rewards
- `train_length_mean`: Mean episode length
- `test_reward_mean`: Mean evaluation reward
- `test_reward_std`: Std dev of evaluation rewards
- `test_length_mean`: Mean evaluation episode length
- `duration`: Time elapsed

## Examples

### Example 1: Training on Cold Stuck Pattern

```bash
python -m src.training.train_ppo \
  --experiment-name "cold_stuck_baseline" \
  --patterns cold_stuck \
  --mechanism frequency_amplifier \
  --threshold 0.8 \
  --total-timesteps 1000000 \
  --n-envs 8
```

### Example 2: Multi-Pattern Training

```bash
python -m src.training.train_ppo \
  --experiment-name "multi_pattern" \
  --patterns cold_stuck dominant_stuck submissive_stuck mixed_random cold_warm \
  --mechanism frequency_amplifier \
  --total-timesteps 2000000 \
  --n-envs 16 \
  --batch-size 128
```

### Example 3: Resume Training from Checkpoint

```bash
python -m src.training.train_ppo \
  --config configs/experiment.yaml \
  --resume-from models/experiment/checkpoint_100000.pth
```

## Network Architecture

The default architecture consists of:

1. **Feature Extraction (TherapyNet)**
   - Client action embedding (8 actions → 16-dim)
   - History embedding (50 timesteps × 9 values → 50 × 8-dim → flattened to 400-dim)
   - Session number (1-dim normalized)
   - Concatenated features → MLP (256 → 256)

2. **Actor Network**
   - TherapyNet features → Linear(256, 8)
   - Outputs action probabilities (Categorical distribution)

3. **Critic Network**
   - TherapyNet features → Linear(256, 1)
   - Outputs state value estimate

Total parameters: ~200k (with default hidden_size=256)

## Tips for Training

### Hyperparameter Tuning
- Start with default parameters for initial experiments
- Increase `n_envs` (parallel environments) for faster training
- Adjust `learning_rate` if training is unstable (decrease) or too slow (increase)
- Increase `total_timesteps` for difficult patterns or mechanisms
- Use `--threshold 0.8` for easier tasks, `0.9` for standard difficulty

### Monitoring Training
- Check `logs/{experiment_name}_progress.csv` for training curves
- Look for increasing `test_reward_mean` over time
- If rewards plateau early, try:
  - Decreasing learning rate
  - Increasing entropy coefficient
  - Using more training timesteps

### GPU Usage
- Training automatically uses GPU if available
- For CPU-only: networks still work but training is slower
- Use fewer parallel environments (`n_envs`) on CPU

## Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install numpy torch gymnasium tianshou tensorboard PyYAML
```

### Out of Memory
- Reduce `n_envs` (parallel environments)
- Reduce `batch_size`
- Reduce `hidden_size`

### Training Not Converging
- Check that patterns and mechanism are compatible
- Verify threshold is achievable (try 0.8 first)
- Increase `total_timesteps`
- Try different random seeds
