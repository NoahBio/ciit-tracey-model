# Model Overview V7 - Frequency Amplifier + V2 Therapist + Optuna Optimization

**Status**: Current as of 2025-01-10
**Version**: 7.0
**Focus**: OmniscientStrategicTherapist V2, Frequency Amplifier mechanism, Optuna parameter optimization

---

## Document Purpose

This document describes the active components of the CIIT-Tracey computational model used for the V2 omniscient therapist optimization study. Components outside this scope (RL training, alternative mechanisms) are listed in the Cleanup Appendix.

---

## 1. Frequency Amplifier Client Mechanism

**Source**: `src/agents/client_agents/frequency_amplifier_client.py`

### Mathematical Formula

```
adjusted[i,j] = U[i,j] + (U[i,j] * P(j) * k)
```

Where:
- `U[i,j]` = raw utility from the client's u_matrix (client action i, therapist action j)
- `P(j)` = marginal probability of therapist action j (recency-weighted from memory)
- `k` = `history_weight` parameter (default: 1.0)

### How It Works

1. **Calculate Marginal Frequencies**: `_calculate_marginal_frequencies()` computes P(therapist_j) from the client's memory using recency-weighted counts
2. **Amplify Utilities**: For each client action, raw utilities are amplified by `raw + (raw * frequency * history_weight)`
3. **Sort and Select**: Amplified utilities are sorted; bond-based percentile interpolation selects the expected payoff
4. **Softmax Selection**: Expected payoffs are converted to action probabilities via softmax with entropy (temperature)

### Key Properties

| Condition | Effect |
|-----------|--------|
| Unobserved responses (P(j)=0) | Utilities unchanged |
| High-frequency positive utilities | Amplified upward |
| High-frequency negative utilities | Amplified MORE negative |

### Parameters

| Parameter | Default | Suggested Range | Description |
|-----------|---------|-----------------|-------------|
| `history_weight` | 1.0 | 0.5 - 2.0 | Strength of frequency amplification |
| `entropy` | varies | 1.5 - 5.0 | Temperature for action selection |

---

## 2. OmniscientStrategicTherapist V2

**Source**: `src/agents/therapist_agents/omniscient_therapist_v2.py`

### Overview

The V2 therapist has omniscient access to client internals and uses strategic "perceptual seeding" to help clients escape maladaptive patterns. V2 adds real-time feedback monitoring to adaptively adjust seeding strategy.

### Three-Phase Strategy

```
RELATIONSHIP BUILDING  -->  LADDER-CLIMBING  -->  CONSOLIDATION
      (Pure complementarity)     (Strategic seeding)     (Complement at new level)
```

| Phase | Entry Condition | Strategy | Exit Condition |
|-------|-----------------|----------|----------------|
| **Relationship Building** | Start of therapy | Pure complementarity | Session >= 10, bond > 0.1, beneficial target found |
| **Ladder-Climbing** | From above | Cost-benefit seeding | Client takes target action OR target aborted |
| **Consolidation** | Target achieved/aborted | Pure complementarity | New ladder step available |

### V2-Specific Hyperparameters

| Parameter | Default | Optuna Range | Purpose |
|-----------|---------|--------------|---------|
| `seeding_benefit_scaling` | 0.3 | [0.1, 2.0] | Scaling factor for expected seeding benefit calculation |
| `skip_seeding_accuracy_threshold` | 0.9 | [0.75, 0.95] | Skip seeding if perception accuracy above this |
| `quick_seed_actions_threshold` | 3 | [1, 5] | "Just do it" seeding if actions_needed <= this |
| `abort_consecutive_failures_threshold` | 5 | [4, 9] | Abort target after this many consecutive failures |

### Feedback Monitoring System

V2 introduces real-time tracking of seeding effectiveness:

- **SeedingMonitor**: Tracks per-target metrics (attempts, successes, failures, competitor boosting)
- **FeedbackRecord**: Captures outcome of each seeding action (perceived vs. intended, memory changes)

### Intelligent Seeding Decision Logic

The `_is_seeding_beneficial()` method evaluates:

1. **Free seeding**: If complement equals target action, always seed
2. **High accuracy**: If perception accuracy > threshold, skip seeding
3. **Quick seed**: If actions_needed <= threshold, just do it
4. **Cost-benefit analysis**: Compare expected future benefit vs. immediate utility cost

### Four Abort Criteria

The therapist abandons a seeding target when:

1. **Consecutive failures**: Failures >= `abort_consecutive_failures_threshold`
2. **Competitor boosting**: A competitor action is being boosted faster than target
3. **Time pressure**: Insufficient sessions remaining to benefit
4. **Low success rate**: Success rate falls below baseline_accuracy - 10%

### Two-Stage Execution

**Critical**: V2 requires a two-stage call pattern per session:

```python
# Stage 1: Decision
therapist_action, metadata = therapist.decide_action(client_action, session)

# Environment update
client.update_memory(client_action, therapist_action)

# Stage 2: Feedback (MUST be after update_memory!)
therapist.process_feedback_after_memory_update(session, client_action)
```

---

## 3. Parataxic Distortion System

**Source**: `src/agents/client_agents/parataxic_distortion.py`

### Theoretical Basis

Implements Sullivan's parataxic distortion: clients perceive therapist actions through the lens of recent interaction history.

### Algorithm

```
If random() < baseline_accuracy:
    perceive ACTUAL action correctly
Else:
    perceive MOST COMMON action in recent memory
    (If tied: choose most recently enacted among tied actions)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PARATAXIC_WINDOW` | 15 | Number of recent interactions considered |
| `PARATAXIC_BASELINE_ACCURACY` | 0.5 | Probability of correct perception |

### Integration

Applied via mixin pattern:

```python
from src.agents.client_agents.parataxic_distortion import with_parataxic

ParataxicFrequencyAmplifier = with_parataxic(FrequencyAmplifierClient)
client = ParataxicFrequencyAmplifier(
    u_matrix=...,
    entropy=...,
    initial_memory=...,
    baseline_accuracy=0.3,
    enable_parataxic=True
)
```

---

## 4. Optuna V2 Optimization Study

**Source**: `scripts/optimize_omniscient_v2_advantage.py`

### Objective Function

```
advantage = omniscient_success_rate - complementary_success_rate
```

Direction: **MAXIMIZE** (find parameters where V2 therapist outperforms simple complementarity)

### Search Space (11 Parameters)

#### Client Parameters (6)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `baseline_accuracy` | float | [0.21, 0.9] | Parataxic distortion baseline |
| `threshold` | float | [0.8, 0.99] | Success threshold percentile |
| `bond_alpha` | float | [1.0, 15.0] | Bond sigmoid steepness |
| `bond_offset` | float | [0.5, 0.95] | Bond sigmoid inflection point |
| `recency_weighting_factor` | float | [1.0, 5.0] | Memory recency newest:oldest weight ratio |
| `max_sessions` | int | [100, 2000] | Maximum therapy sessions |

#### Therapist V2 Parameters (5)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `perception_window` | int | [7, 20] | Size of perception window |
| `seeding_benefit_scaling` | float | [0.1, 2.0] | Benefit scaling factor |
| `skip_seeding_accuracy_threshold` | float | [0.75, 0.95] | Accuracy skip threshold |
| `quick_seed_actions_threshold` | int | [1, 5] | Quick seed trigger |
| `abort_consecutive_failures_threshold` | int | [4, 9] | Abort trigger |

### Fixed Constraints

| Parameter | Fixed Value | Rationale |
|-----------|-------------|-----------|
| `mechanism` | `'frequency_amplifier'` | Study focuses on this mechanism |
| `enable_parataxic` | `True` | Required for realistic difficulty |
| `pattern` | `'cold_stuck'` | Initial memory pattern |
| `entropy` | `1.0` | Fixed temperature |

### Study Configuration

- **Sampler**: TPESampler (Tree-structured Parzen Estimator) with seed=42
- **Storage**: SQLite database (resumable)
- **Seeds per trial**: 50 (default)
- **Output**: Top-K configs as JSON + visualization HTML files

### Usage

```bash
# Run 100 trials with 50 seeds each
python scripts/optimize_omniscient_v2_advantage.py \
  --n-trials 100 \
  --study-name freq_amp_v2_optimization \
  --n-seeds 50 \
  --therapist-version v2

# View results in browser
optuna-dashboard sqlite:///optuna_studies/freq_amp_v2_optimization.db
```

---

## 5. Key Files Reference

| File | Purpose |
|------|---------|
| `src/agents/therapist_agents/omniscient_therapist_v2.py` | V2 strategic therapist implementation |
| `src/agents/client_agents/frequency_amplifier_client.py` | Frequency amplifier mechanism |
| `src/agents/client_agents/parataxic_distortion.py` | Perception distortion system |
| `src/config.py` | Global configuration and utilities |
| `scripts/optimize_omniscient_v2_advantage.py` | Optuna V2 optimization script |
| `scripts/evaluate_omniscient_therapist.py` | Simulation infrastructure |
| `src/environment/therapy_env.py` | Therapy environment wrapper |

---

## 6. Supporting Systems

### Bond Calculation

**Source**: `src/config.py:rs_to_bond()`

```
bond = sigmoid(alpha * (2 * (rs_normalized - offset)))
```

Where:
- `rs_normalized = (rs - rs_min) / (rs_max - rs_min)`
- Default `alpha = 5`, `offset = 0.8`

### Memory Weighting

**Source**: `src/config.py:get_memory_weights()`

Square-root shaped recency bias:
- `recency_weighting_factor` is the newest:oldest weight ratio directly (float, >= 1.0)
- E.g., 2.0 means the newest interaction is weighted 2x the oldest

### Complementary Action Mapping

```python
COMPLEMENT_MAP = {
    0: 4,  # D -> S
    1: 3,  # WD -> WS
    2: 2,  # W -> W
    3: 1,  # WS -> WD
    4: 0,  # S -> D
    5: 7,  # CS -> CD
    6: 6,  # C -> C
    7: 5,  # CD -> CS
}
```

---

## Appendix: Cleanup List

Items outside the current V2 optimization focus. Review for removal or archiving.

### `during development/` Directory

| File | Status | Notes |
|------|--------|-------|
| `deprecated/client_agent_legacy.py` | Archive | Old client implementation |
| `optuna_hyperparameter_search.py` | Review | Replaced by V2-specific script |
| `optuna_utils.py` | Review | May need consolidation |
| `run_multi_seed_simulation.py` | Check | Verify if still used |
| `test_perception_comparison.py` | Archive | Development test |
| `test_verbose_session_trace.py` | Archive | Development test |

### `src/training/` Directory (RL-related, abandoned)

| File | Status | Notes |
|------|--------|-------|
| `train_ppo.py` | Archive | RL training script |
| `networks.py` | Archive | RL neural networks |
| `omniscient_networks.py` | Archive | RL networks |
| `example_logger_usage.py` | Remove | Example code |

### `src/evaluation/` Directory

| File | Status | Notes |
|------|--------|-------|
| `evaluate_policy.py` | Archive | RL policy evaluation |
| `baseline_comparison.py` | Review | May still be useful |

### Documentation

| File | Status | Notes |
|------|--------|-------|
| `docs/RUN_OMNISCIENT_RL_README.md` | Archive | RL-related |
| `docs/SRC_TRAINING_README.md` | Archive | RL training docs |

### Client Mechanisms (not in V2 focus)

| File | Status | Notes |
|------|--------|-------|
| `bond_only_client.py` | Keep | Used for baseline comparison |
| `conditional_amplifier_client.py` | Keep | Alternative mechanism |
| `bond_weighted_conditional_amplifier_client.py` | Keep | Alternative mechanism |

### Results Directories

| Directory | Status | Notes |
|-----------|--------|-------|
| `results/test_ppo_training/` | Archive | RL experiment |
| `results/smoke_test/` | Remove | Old test |
| `results/RL_vs_Complementary_*/` | Archive | RL comparison experiments |

---

*Document generated from source code verification. All formulas, parameters, and line references validated against implementation.*
