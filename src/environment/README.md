# TherapyEnv - Gymnasium Environment for Therapy Simulation

A Gymnasium-compatible reinforcement learning environment for training therapist agents. The goal is to discover whether Tracey's proposed Three-Stage-Structure emerges.

## Overview

`TherapyEnv` simulates a therapy session where:
1. **Client** selects behavior based on relationship satisfaction (RS) and bond
2. **Therapist** (RL agent) observes client behavior and responds with an interpersonal action
3. **Client** updates internal state based on the interaction
4. Episode continues until success (RS >= threshold), dropout (session 10), or max sessions

## Installation

```python
from src.environment import TherapyEnv
```

## Quick Start

```python
import numpy as np
from src.environment import TherapyEnv

# Create environment
env = TherapyEnv(
    mechanism="frequency_amplifier",
    pattern="cold_stuck",
    threshold=0.8,
    max_sessions=100,
    entropy=0.5
)

# Reset environment
obs, info = env.reset(seed=42)

# Simple complementary policy
complement_map = {0:4, 1:3, 2:2, 3:1, 4:0, 5:7, 6:6, 7:5}

# Run episode
for _ in range(100):
    client_action = obs['client_action']
    therapist_action = complement_map[client_action]
    obs, reward, terminated, truncated, info = env.step(therapist_action)

    if terminated or truncated:
        print(f"Episode ended: Success={info['success']}, Reward={reward}")
        break
```

## Observation Space

`Dict` with three components:

| Component | Type | Shape | Range | Description |
|-----------|------|-------|-------|-------------|
| `client_action` | Discrete(8) | - | 0-7 | Client's current interpersonal octant |
| `session_number` | Box(float32) | (1,) | [0, 1] | Normalized session count |
| `history` | MultiDiscrete([9]*50) | (50,) | 0-8 | Last 25 client-therapist interactions (padded with 8) |

### Interpersonal Octants

| Index | Octant | Description |
|-------|--------|-------------|
| 0 | D | Dominant |
| 1 | WD | Warm-Dominant |
| 2 | W | Warm |
| 3 | WS | Warm-Submissive |
| 4 | S | Submissive |
| 5 | CS | Cold-Submissive |
| 6 | C | Cold |
| 7 | CD | Cold-Dominant |

### History Format

- Flattened array: `[c0, t0, c1, t1, ..., c24, t24]` (50 elements)
- Contains only **shared therapy interactions** (excludes client's initial memory)
- Early sessions are padded with value `8` (not `0`, to avoid confusion with Dominant octant)
- Example: After 2 sessions, history might be `[6, 2, 7, 5, 8, 8, 8, ..., 8]`

## Action Space

`Discrete(8)` - Therapist selects one of 8 interpersonal octants (0-7).

## Rewards

**Terminal-only reward structure:**

| Outcome | Reward Formula | Example |
|---------|---------------|---------|
| Success | `100 + (max_sessions - sessions) × 2` | Success at session 30: 100 + (100-30)×2 = **240** |
| Dropout | `-150` | Dropout at session 10: **-150** |
| Max sessions | `0` | Reached 100 sessions without success: **0** |
| Non-terminal | `0` | All intermediate steps: **0** |

**Key insight:** Earlier success = higher reward (efficiency bonus)

## Termination Conditions

| Condition | Type | Description |
|-----------|------|-------------|
| Success | `terminated=True` | Client's RS reaches or exceeds threshold |
| Dropout | `terminated=True` | At session 10, if RS decreased from initial |
| Max sessions | `truncated=True` | Reached `max_sessions` without success/dropout |

## Parameters

```python
TherapyEnv(
    mechanism="frequency_amplifier",           # Client expectation mechanism
    pattern=["cold_stuck", "dominant_stuck"],  # Initial memory pattern(s)
    threshold=0.9,                             # Success threshold (percentile)
    max_sessions=100,                          # Maximum sessions per episode
    entropy=0.5,                               # Client temperature parameter
    history_weight=1.0,                        # History influence weight
    bond_alpha=None,                           # Bond sigmoid steepness (None = use config)
    bond_offset=0.7,                           # Bond sigmoid inflection (70th percentile)
    enable_perception=True,                    # Enable perceptual distortion
    baseline_accuracy=0.5,                     # Perception accuracy
    random_state=None                          # Random seed
)
```

### Client Mechanisms

| Mechanism | Description |
|-----------|-------------|
| `bond_only` | Baseline: Expectations based on bond level only |
| `frequency_amplifier` | **Default**: Marginal history amplifies utilities |
| `conditional_amplifier` | Conditional history (P(T\|C)) amplifies utilities |
| `bond_weighted_frequency_amplifier` | Frequency with bond-scaled influence |
| `bond_weighted_conditional_amplifier` | Conditional with bond-scaled influence |

### Initial Memory Patterns

| Pattern | Description |
|---------|-------------|
| `cold_stuck` | Client stuck in cold behaviors (CS, C, CD) |
| `dominant_stuck` | Client stuck in dominant (D, WD, CD) |
| `submissive_stuck` | Client stuck in submissive (WS, S, CS) |
| `cold_warm` | Client always Cold, therapist always Warm |
| `complementary_perfect` | High-quality interactions (70% warm, all complementary) |
| `conflictual` | D→D power struggle |
| `mixed_random` | Fully random interactions |

**Pattern List:** Pass a list to randomly sample one per episode (curriculum learning)

```python
env = TherapyEnv(pattern=["cold_stuck", "dominant_stuck", "submissive_stuck"])
```

## Info Dictionary

### reset() returns:

```python
{
    'pattern': str,           # Sampled initial memory pattern
    'entropy': float,         # Client entropy value
    'rs_threshold': float,    # Success threshold for this client
    'initial_rs': float,      # Starting relationship satisfaction
    'initial_bond': float,    # Starting bond level
    'episode_seed': int       # Reproducibility seed
}
```

### step() returns:

```python
{
    'session': int,           # Current session number
    'rs': float,              # Current relationship satisfaction
    'bond': float,            # Current bond level [0, 1]
    'client_action': int,     # Client octant this step
    'therapist_action': int,  # Therapist octant this step
    'success': bool,          # Reached threshold?
    'dropped_out': bool,      # Client dropped out?
    'max_reached': bool,      # Hit max sessions?
    'perception_stats': dict  # Optional: perception metrics
}
```

## Examples

### Example 1: Random Policy

```python
env = TherapyEnv()
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Example 2: Complementary Policy

```python
complement = {0:4, 1:3, 2:2, 3:1, 4:0, 5:7, 6:6, 7:5}

env = TherapyEnv(pattern="cold_stuck", threshold=0.8)
obs, info = env.reset(seed=42)

while True:
    therapist_action = complement[obs['client_action']]
    obs, reward, terminated, truncated, info = env.step(therapist_action)
    if terminated or truncated:
        print(f"Success: {info['success']}, Sessions: {info['session']}")
        break
```

### Example 3: Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create vectorized environment
def make_env():
    return TherapyEnv(
        pattern=["cold_stuck", "dominant_stuck", "submissive_stuck"],
        threshold=0.8,
        entropy=0.5
    )

env = DummyVecEnv([make_env for _ in range(8)])

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

# Evaluate
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
```

### Example 4: Curriculum Learning

```python
# Start with easier patterns, gradually add harder ones
easy_patterns = ["complementary_perfect"]
medium_patterns = ["mixed_random", "cold_warm"]
hard_patterns = ["cold_stuck", "dominant_stuck", "submissive_stuck"]

# Stage 1: Easy
env = TherapyEnv(pattern=easy_patterns, threshold=0.7)
# ... train ...

# Stage 2: Medium
env = TherapyEnv(pattern=easy_patterns + medium_patterns, threshold=0.8)
# ... continue training ...

# Stage 3: Hard
env = TherapyEnv(pattern=easy_patterns + medium_patterns + hard_patterns, threshold=0.9)
# ... final training ...
```

## Design Notes

### Client Acts First

**Critical:** The client selects their action BEFORE the therapist responds. This reflects realistic therapy dynamics:

1. Client arrives with a behavior/presentation
2. Therapist observes and responds
3. Client updates internal state
4. Client selects next behavior

Implementation:
- `reset()`: Client selects initial action → stored in observation
- `step(therapist_action)`: Use client's previous action → update memory → client selects next action

### Partial Observability

The therapist **cannot directly observe:**
- Relationship satisfaction (RS)
- Bond level
- Success threshold
- Client's initial memory (pre-therapy history)

The therapist **can observe:**
- Client's current action
- Session number
- Last 25 shared interactions

This forces the agent to **infer** the client's internal state from behavior patterns.

### Terminal-Only Rewards

Rewards are only given at episode end. This:
- Simplifies credit assignment
- Matches therapy outcomes (success/failure at end)
- Encourages long-term planning

## Gymnasium Compatibility

`TherapyEnv` is fully compatible with Gymnasium 0.28+:

```python
from gymnasium.utils.env_checker import check_env

env = TherapyEnv()
check_env(env, skip_render_check=True)  # Passes all checks
```

Compatible with:
- Stable-Baselines3
- Tianshou
- RLlib
- CleanRL
- Any Gymnasium-compatible library

## References

1. Tracey, T. J. (1993). An interpersonal stage model of the therapeutic process. *Journal of Counseling Psychology, 40*(4), 396-409.
2. Kiesler, D. J. (1996). *Contemporary Interpersonal Theory and Research*. Wiley.
3. Leary, T. (1957). *Interpersonal Diagnosis of Personality*. Ronald Press.

## Citation

If you use this environment in research, please cite:

```bibtex
@software{therapy_env_2025,
  title={TherapyEnv: A Gymnasium Environment for Therapy Simulation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ciit-tracey-model}
}
```

## License

[Your license here]
