# Multi-Seed Simulation Runner - Usage Guide

## Overview

The `run_multi_seed_simulation.py` script runs therapy simulations across multiple random seeds and provides comprehensive statistical analysis. This is useful for assessing the reliability and success rate of different configurations.

## Quick Start

### Basic Usage

Run 10 simulations with default settings:
```bash
python run_multi_seed_simulation.py --n-seeds 10
```

### Common Examples

**Test conditional amplifier with cold_warm pattern:**
```bash
python run_multi_seed_simulation.py \
    --n-seeds 20 \
    --mechanism conditional_amplifier \
    --pattern cold_warm \
    --threshold 0.8 \
    --entropy 3.0
```

**Test with perception enabled:**
```bash
python run_multi_seed_simulation.py \
    --n-seeds 20 \
    --mechanism conditional_amplifier \
    --pattern cold_warm \
    --threshold 0.8 \
    --enable-perception \
    --baseline-accuracy 0.2
```

**Test bond-weighted mechanism:**
```bash
python run_multi_seed_simulation.py \
    --n-seeds 20 \
    --mechanism bond_weighted_conditional_amplifier \
    --pattern cold_warm \
    --threshold 0.8 \
    --bond-power 2.0 \
    --entropy 3.0
```

**Quick test with minimal output:**
```bash
python run_multi_seed_simulation.py \
    --n-seeds 5 \
    --mechanism conditional_amplifier \
    --pattern cold_warm \
    --quiet \
    --no-trajectories \
    --no-actions
```

**Test strategic therapist with plateau-triggered interventions:**
```bash
python run_multi_seed_simulation.py \
    --n-seeds 20 \
    --mechanism conditional_amplifier \
    --pattern cold_warm \
    --enable-strategic-therapist \
    --rs-plateau-threshold 5.0 \
    --plateau-window 15 \
    --intervention-duration 10
```

## Command-Line Arguments

### Required Arguments

- `--n-seeds, -n` : Number of random seeds to run (seeds 0 to N-1)

### Configuration Arguments

- `--mechanism, -m` : Client expectation mechanism (default: `conditional_amplifier`)
  - Options: `bond_only`, `frequency_amplifier`, `conditional_amplifier`, `bond_weighted_frequency_amplifier`, `bond_weighted_conditional_amplifier`

- `--pattern, -p` : Initial memory pattern (default: `cold_warm`)
  - Options: `cold_warm`, `complementary_perfect`, `conflictual`, `mixed_random`, `cold_stuck`, `dominant_stuck`, `submissive_stuck`

- `--threshold, -t` : Success threshold percentile 0.0-1.0 (default: `0.8`)

- `--entropy, -e` : Client entropy/exploration parameter (default: `3.0`)

- `--max-sessions, -s` : Maximum therapy sessions per run (default: `100`)

### Perception Arguments

- `--enable-perception` : Enable perceptual distortion (default: disabled)

- `--baseline-accuracy` : Baseline perception accuracy (default: `0.2`)

### Mechanism-Specific Arguments

- `--history-weight, -hw` : History weight for amplifier mechanisms (default: `1.0`)

- `--bond-power, -bp` : Bond power for bond_weighted mechanisms (default: `1.0`)

- `--bond-alpha, -ba` : Bond sigmoid steepness (default: `5.0`)

- `--bond-offset, -bo` : Bond sigmoid inflection point (default: `0.8`)

### Strategic Therapist Arguments

- `--enable-strategic-therapist` : Enable strategic therapist with plateau-triggered optimal interventions (default: disabled)

- `--rs-plateau-threshold` : RS range threshold to detect plateau (default: `5.0` RS points)

- `--plateau-window` : Number of consecutive sessions to check for plateau (default: `15`)

- `--intervention-duration` : Number of sessions to enact optimal action during intervention (default: `10`)

### Display Arguments

- `--no-trajectories` : Hide trajectory statistics in output

- `--no-actions` : Hide action distribution in output

- `--quiet, -q` : Suppress progress messages during runs

## Output Statistics

The script provides comprehensive statistics including:

### 1. Success Metrics
- **Success rate**: Percentage of runs that reached the threshold
- **Dropout rate**: Percentage of runs where client dropped out
- **Failure rate**: Percentage of runs that neither succeeded nor dropped out

### 2. Success Timing (for successful runs)
- Mean, median, std dev, min, max session number when success occurred
- Mode (most common success session)

### 3. Relationship Satisfaction (RS) Statistics
- **Final State**: Mean, std, min, quartiles, median, max of final RS values
- **Total Change**: Statistics on RS improvement from initial to final state

### 4. Bond Statistics
- **Final State**: Mean, std, min, quartiles, median, max of final bond values
- **Total Change**: Statistics on bond development from initial to final state

### 5. Perception Statistics (if enabled)
- Misperception rate across runs (mean, std, min, max)
- Stage 1 override rate (history-based perception)
- Computed accuracy statistics

### 6. Near-Miss Analysis (for failed runs)
- How close failed runs got to the threshold
- Mean and range of gaps to threshold
- Useful for understanding "almost successful" configurations

### 7. Trajectory Analysis (optional)
- Mean and std dev of RS and Bond at each session number
- Shows how metrics evolve over therapy sessions
- Displayed every 10 sessions

### 8. Action Distribution (optional)
- Aggregated client and therapist action frequencies across all runs
- Shows which octants are most commonly chosen
- Useful for understanding behavioral patterns

### 9. Intervention Analysis (if strategic therapist enabled)
- **Intervention Rate**: Percentage of runs that triggered at least one intervention
- **Intervention Count**: Mean, median, and max interventions per run
- **RS During Intervention**: Descriptive statistics of relationship satisfaction during intervention periods
- **Bond During Intervention**: Descriptive statistics of bond during intervention periods
- Shows effectiveness of plateau-triggered optimal action interventions

## Strategic Therapist Feature

### Overview

The strategic therapist feature implements a sophisticated intervention strategy that monitors client relationship satisfaction (RS) for stability and triggers optimal action interventions when RS has plateaued.

### How It Works

1. **Normal Behavior**: By default, the therapist uses complementary responses (matching the client's interpersonal style)

2. **Plateau Detection**: The system continuously monitors RS values across a sliding window of sessions
   - Default window: 15 consecutive sessions
   - Plateau threshold: RS range ≤ 5.0 points
   - When `max(RS history) - min(RS history) ≤ threshold`, a plateau is detected

3. **Intervention Trigger**: When a plateau is detected:
   - The therapist identifies the globally optimal action (maximum utility in client's u_matrix)
   - Switches from complementary to optimal action for a fixed duration (default: 10 sessions)
   - Once triggered, the intervention completes regardless of RS/bond changes

4. **Return to Normal**: After intervention completes, therapist returns to complementary behavior and resumes plateau monitoring

5. **Multiple Interventions**: Can trigger multiple times in a single simulation if RS plateaus again after an intervention

### Key Parameters

- `--rs-plateau-threshold`: How much RS can vary to still be considered "stable" (default: 5.0)
- `--plateau-window`: How many sessions to check for stability (default: 15)
- `--intervention-duration`: How many sessions to enact optimal action (default: 10)

### Use Cases

- **Expectation Building**: Works synergistically with perceptual distortion to build client expectations
- **Breaking Through Stagnation**: When therapy has reached a stable but suboptimal state
- **Testing Optimal Action Effects**: Compare outcomes with and without strategic interventions

### Example Comparison

```bash
# Baseline: complementary therapist only
python run_multi_seed_simulation.py \
    --n-seeds 50 \
    --mechanism conditional_amplifier \
    --pattern cold_warm \
    --enable-perception

# Strategic therapist with interventions
python run_multi_seed_simulation.py \
    --n-seeds 50 \
    --mechanism conditional_amplifier \
    --pattern cold_warm \
    --enable-perception \
    --enable-strategic-therapist
```

## Tips and Best Practices

### Choosing Number of Seeds

- **Quick test**: 5-10 seeds for rapid prototyping
- **Preliminary analysis**: 20-30 seeds for initial insights
- **Publication-quality**: 50-100+ seeds for reliable statistics

### Performance Considerations

- Each simulation runs up to `max_sessions` sessions
- With default settings (100 sessions, no perception): ~0.1-0.2 seconds per seed
- With perception enabled: ~0.2-0.4 seconds per seed
- 100 seeds with perception: ~30-40 seconds total

### Output Management

Use `--quiet`, `--no-trajectories`, and `--no-actions` flags to reduce output when running many seeds:

```bash
python run_multi_seed_simulation.py --n-seeds 100 --quiet --no-actions
```

### Using Strategic Therapist

- **Combine with Perception**: The strategic therapist works best with `--enable-perception` to build client expectations
- **Longer Sessions**: Consider using higher `--max-sessions` (e.g., 150-200) to allow multiple interventions
- **Pattern Selection**: Some patterns (e.g., `complementary_perfect`) may plateau faster than others
- **Tuning Sensitivity**: Adjust `--rs-plateau-threshold` if interventions trigger too often (increase) or not at all (decrease)
- **Window Size**: Shorter `--plateau-window` triggers interventions sooner but may respond to temporary fluctuations

### Comparing Configurations

To compare different configurations, run the script multiple times with different parameters and redirect output to files:

```bash
# Test different mechanisms
python run_multi_seed_simulation.py --n-seeds 50 --mechanism conditional_amplifier > results_conditional.txt
python run_multi_seed_simulation.py --n-seeds 50 --mechanism bond_weighted_conditional_amplifier > results_bond_weighted.txt

# Test different memory patterns
python run_multi_seed_simulation.py --n-seeds 50 --pattern cold_warm > results_cold_warm.txt
python run_multi_seed_simulation.py --n-seeds 50 --pattern conflictual > results_conflictual.txt

# Test perception vs no perception
python run_multi_seed_simulation.py --n-seeds 50 > results_no_perception.txt
python run_multi_seed_simulation.py --n-seeds 50 --enable-perception > results_with_perception.txt

# Test strategic therapist vs complementary
python run_multi_seed_simulation.py --n-seeds 50 --enable-perception > results_complementary.txt
python run_multi_seed_simulation.py --n-seeds 50 --enable-perception --enable-strategic-therapist > results_strategic.txt
```

## Integration with Existing Tools

This script is designed to work alongside `test_verbose_session_trace.py`:

- Use **`test_verbose_session_trace.py`** for:
  - Detailed inspection of a single simulation run
  - Debugging specific behaviors
  - Understanding step-by-step dynamics
  - Verifying implementation correctness

- Use **`run_multi_seed_simulation.py`** for:
  - Assessing success rates across configurations
  - Understanding variability in outcomes
  - Statistical validation of theoretical predictions
  - Publication-ready aggregate statistics

## Example Workflow

1. **Initial exploration** with verbose trace:
   ```bash
   python test_verbose_session_trace.py --mechanism conditional_amplifier --pattern cold_warm
   ```

2. **Statistical validation** with multi-seed runner:
   ```bash
   python run_multi_seed_simulation.py --n-seeds 50 --mechanism conditional_amplifier --pattern cold_warm
   ```

3. **Detailed investigation** of specific seeds:
   ```bash
   # If seed 17 was interesting from multi-seed run
   python test_verbose_session_trace.py --mechanism conditional_amplifier --pattern cold_warm --seed 17
   ```

## Return Value

The script returns a `MultiSeedStatistics` object containing all computed statistics. This can be used for programmatic access when importing the module:

```python
from run_multi_seed_simulation import run_multi_seed_simulation

stats = run_multi_seed_simulation(
    n_seeds=20,
    mechanism='conditional_amplifier',
    pattern='cold_warm',
    threshold=0.8,
    verbose=False
)

print(f"Success rate: {stats.success_rate:.1%}")
print(f"Mean final RS: {stats.final_rs_stats['mean']:.2f}")
```

## Troubleshooting

**Issue**: Script runs slowly with perception enabled
- **Solution**: Reduce `--max-sessions` or `--n-seeds`, or disable trajectories/actions output

**Issue**: All runs fail
- **Solution**: Lower the `--threshold` value or increase `--max-sessions` to give therapy more time

**Issue**: High variability in results
- **Solution**: Increase `--n-seeds` for more stable statistics, or investigate specific seeds with verbose trace

**Issue**: Want to focus on specific sessions
- **Solution**: Use `--max-sessions` to limit runtime and `--no-trajectories` to reduce output clutter
