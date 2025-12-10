# CIIT-Tracey Computational Model

Computational validation of Tracey's Three-Step Model of therapy using Contemporary Integrative Interpersonal Theory (CIIT) and Reinforcement Learning.

## Research Questions (10.12.2025)

1. Specifying the assumption space for either side of the great psychotherapy debate through agent-based modeling.
  -> Under which conditions does a strategic therapist outperform a simply complementary therapist?
2. Under which conditions does the three-step pattern emerge even in a computationally simulated therapeutic dyad? 
3. What is a parsimonious computational simulation of interpersonal transactional cycles to study the above question?
4. What parts of CIIT relevant to Tracey’s-Three-Step model are formally underspecified? What parts of Tracey’s Model are too vague to test through ABM?
5. Why was it so difficult to stop the always-complementary therapist from being successful?



## Technical Stack

- **Language**: Python 3.10+
- **RL Framework**: Tianshou 1.x
- **Environment**: Custom Gymnasium environment (TherapyEnv)
- **Training Modes**:
  - Standard (model-free): Partial observability
  - Omniscient: Perfect client state information
- **Client Mechanisms**: 5 expectation mechanisms with optional perception distortion
- **Optimization**: Optuna for hyperparameter tuning

## Project Structure
```
src/
├── agents/
│   └── client_agents/              # 5 client mechanisms + perception
│       ├── base_client.py          # Shared client functionality
│       ├── bond_only_client.py
│       ├── frequency_amplifier_client.py
│       ├── conditional_amplifier_client.py
│       ├── bond_weighted_frequency_amplifier_client.py
│       ├── bond_weighted_conditional_amplifier_client.py
│       └── perceptual_distortion.py  # Perception mixin
├── environment/                     # Gymnasium environments
│   ├── therapy_env.py              # Main TherapyEnv
│   └── omniscient_wrapper.py       # Omniscient observation wrapper
├── training/                        # PPO training with Tianshou
│   ├── config.py                   # TrainingConfig dataclass
│   ├── networks.py                 # Standard neural networks
│   ├── omniscient_networks.py      # Omniscient networks
│   └── train_ppo.py                # Main training script
├── evaluation/                      # Policy evaluation tools
└── config.py                        # Global parameters and constants

configs/                             # YAML configuration files
├── example_experiment.yaml
├── omniscient_experiment.yaml
├── omniscient_RL_vs_Complementary.yaml
└── ...

during development/                  # Development tools
├── run_multi_seed_simulation.py    # Statistical validation
├── test_verbose_session_trace.py   # Single-run debugging
├── optuna_hyperparameter_search.py # Hyperparameter tuning
└── optuna_utils.py                 # Optuna analysis tools

tests/                               # Comprehensive test suite
├── unit/                           # Unit tests
├── integration/                    # Integration tests
├── session_flow/                   # Session flow tests
├── test_omniscient_*.py            # Omniscient RL tests
└── ...

docs/                                # Documentation files
├── INDEX.md                        # Documentation index
├── GENERAL_README.md               # This file
├── CLIENT_ARCHITECTURE_README.md   # Client architecture guide
├── THERAPYENV_README.md            # Environment API reference
├── SRC_TRAINING_README.md          # Training guide
├── RUN_OMNISCIENT_RL_README.md     # Omniscient RL guide
├── MULTI_SEED_USAGE_README.md      # Multi-seed evaluation
├── OPTUNA_STUDY_README.md          # Hyperparameter optimization
└── OMNISCIENT_TESTS_README.md      # Test suite documentation

optuna_studies/                      # Optuna optimization databases
models/                              # Trained model checkpoints
results/                             # Evaluation results
logs/                                # Training logs and TensorBoard data
```

For detailed documentation on each component, see [docs/INDEX.md](INDEX.md).

## Setup
```bash
# Clone repository
git clone https://github.com/NoahBio/ciit-tracey-model.git
cd ciit-tracey-model

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

After setup, explore the documentation:
- **New users**: Start with [docs/INDEX.md](INDEX.md) for a guided tour
- **Training**: See [docs/SRC_TRAINING_README.md](SRC_TRAINING_README.md)
- **Evaluation**: See [docs/MULTI_SEED_USAGE_README.md](MULTI_SEED_USAGE_README.md)
- **Understanding clients**: See [docs/CLIENT_ARCHITECTURE_README.md](CLIENT_ARCHITECTURE_README.md)

## Status

🚧 Under active development
