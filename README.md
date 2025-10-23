# CIIT-Tracey Computational Model

Computational validation of Tracey's Three-Step Model of therapy using Contemporary Integrative Interpersonal Theory (CIIT) and Reinforcement Learning.

## Research Questions

1. **Model-Free (RQ1)**: Does the Three-Step Model emerge naturally from a client-therapist interaction based in CIIT?
2. **Model-Based (RQ2)**: Does Tracey's strategy work optimally given certain client dynamics?

## Technical Stack

- **Language**: Python 3.12
- **RL Framework**: Tianshou 1.x
- **Algorithms**: PPO+LSTM (model-free), MBPO (model-based)
- **Environment**: Custom Gymnasium environment

## Project Structure
```
src/
├── agents/          # Client and therapist agent implementations
├── environment/     # Therapy simulation environment
├── models/          # RL model implementations (PPO, MBPO)
├── analysis/        # Metrics and visualization
└── config.py        # Global parameters and constants

experiments/         # Training configurations and results
tests/              # Unit tests
notebooks/          # Jupyter notebooks for exploration
```

## Setup
```bash
# Clone repository
git clone https://github.com/NoahBio/ciit-tracey-model.git
cd ciit-tracey-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Status

🚧 Under active development
