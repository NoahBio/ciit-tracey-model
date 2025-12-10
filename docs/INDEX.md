# CIIT-Tracey Model Documentation Index

Welcome to the CIIT-Tracey computational model documentation. This index will help you navigate to the documentation relevant to your needs.

## Getting Started

- **[GENERAL_README.md](GENERAL_README.md)** - Project overview, installation, and setup
  - Research questions and technical stack
  - Project structure and directory organization
  - Quick setup instructions

## Core Components

### Client Architecture
- **[CLIENT_ARCHITECTURE_README.md](CLIENT_ARCHITECTURE_README.md)** - Complete guide to client agent architecture
  - 5 client expectation mechanisms (bond-only, frequency, conditional, bond-weighted variants)
  - Perceptual distortion system
  - Detailed calculation flow diagrams
  - Memory, RS, and bond calculations

### Environment
- **[THERAPYENV_README.md](THERAPYENV_README.md)** - Gymnasium environment API reference
  - Observation and action spaces
  - Reward structure
  - Termination conditions
  - Usage examples and integration with RL libraries

## Training & Evaluation

### Standard RL Training
- **[SRC_TRAINING_README.md](SRC_TRAINING_README.md)** - PPO training with Tianshou
  - Configuration system
  - Network architectures (standard and omniscient)
  - CLI arguments and examples
  - Hyperparameter tuning tips

### Omniscient RL
- **[RUN_OMNISCIENT_RL_README.md](RUN_OMNISCIENT_RL_README.md)** - Training with perfect client state information
  - What is omniscient RL and why use it
  - Configuration differences from standard training
  - Expected results and performance comparisons
  - Monitoring and troubleshooting

### Multi-Seed Evaluation
- **[MULTI_SEED_USAGE_README.md](MULTI_SEED_USAGE_README.md)** - Statistical validation across random seeds
  - Running multi-seed simulations
  - Strategic therapist with plateau-triggered interventions
  - Comprehensive statistics and analysis
  - Integration with verbose session tracing

## Advanced Topics

### Hyperparameter Optimization
- **[OPTUNA_STUDY_README.md](OPTUNA_STUDY_README.md)** - Automated hyperparameter search
  - Single and multi-objective optimization
  - Available objectives and parameters
  - Study organization and resumption
  - Interactive dashboards and analysis

### Testing Framework
- **[OMNISCIENT_TESTS_README.md](OMNISCIENT_TESTS_README.md)** - Comprehensive test suite
  - Unit tests for wrapper and networks
  - Integration tests for training pipeline
  - Smoke tests for end-to-end validation
  - Test coverage and CI/CD integration

## Development Tools

All development scripts are located in the `during development/` directory:

- **`run_multi_seed_simulation.py`** - Statistical validation across random seeds
  - See [MULTI_SEED_USAGE_README.md](MULTI_SEED_USAGE_README.md) for details

- **`test_verbose_session_trace.py`** - Single-run debugging with detailed session-by-session output
  - Useful for understanding specific simulation behaviors
  - Complements multi-seed statistical analysis

- **`optuna_hyperparameter_search.py`** - Automated hyperparameter tuning
  - See [OPTUNA_STUDY_README.md](OPTUNA_STUDY_README.md) for usage

- **`optuna_utils.py`** - Analysis tools for Optuna studies
  - Export results to CSV
  - Generate visualizations

## Quick Navigation by Task

### I want to...

**...understand how clients work**
→ Start with [CLIENT_ARCHITECTURE_README.md](CLIENT_ARCHITECTURE_README.md)

**...train a therapist agent**
→ Start with [SRC_TRAINING_README.md](SRC_TRAINING_README.md) for standard training
→ Or [RUN_OMNISCIENT_RL_README.md](RUN_OMNISCIENT_RL_README.md) for omniscient training

**...evaluate agent performance**
→ Use [MULTI_SEED_USAGE_README.md](MULTI_SEED_USAGE_README.md)

**...tune hyperparameters**
→ See [OPTUNA_STUDY_README.md](OPTUNA_STUDY_README.md)

**...understand the environment API**
→ Read [THERAPYENV_README.md](THERAPYENV_README.md)

**...run tests**
→ See [OMNISCIENT_TESTS_README.md](OMNISCIENT_TESTS_README.md)

**...understand the project structure**
→ Start with [GENERAL_README.md](GENERAL_README.md)

## Documentation Conventions

- All code examples assume you're running from the project root directory
- Virtual environment activation: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Development scripts are in `during development/` and should be referenced with that path
- Config files are in `configs/` directory

## Contributing

When adding new features:
1. Update relevant documentation files
2. Add examples and usage instructions
3. Update this index if creating new documentation
4. Ensure all code examples work from project root

## Questions or Issues?

- Check the relevant README for detailed information
- Review test files in `tests/` for usage examples
- See `during development/` for debugging and analysis tools
