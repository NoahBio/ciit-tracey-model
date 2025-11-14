# Parameter Sweep Testing Framework

## Overview

This directory contains a comprehensive testing framework for evaluating the success rate of the always-complementary therapist strategy across different parameter configurations.

## Scripts

### 1. `test_parameter_sweep.py` (Main Framework)

The core module that implements the parameter sweep functionality.

**Key Functions:**
- `generate_memory_pattern()` - Creates different initial memory patterns
- `simulate_therapy_episode()` - Runs a single therapy episode
- `run_parameter_configuration()` - Tests one parameter configuration across multiple trials
- `run_parameter_sweep()` - Full factorial parameter sweep
- `print_results_summary()` - Displays results with analysis
- `save_results()` - Saves results to JSON

**Memory Patterns:**
- `cw_50_50` - Pure anticomplementary (C→W)
- `complementary_perfect` - Perfect complementarity (D→S)
- `conflictual` - Worst case (D→D)
- `mixed_random` - Random interactions
- `cold_dominant` - Client stuck in CD with varied therapist responses

**Success Criteria:**
- No dropout during MAX_SESSIONS
- Final bond ≥ success_threshold

### 2. `test_parameter_sweep_quick.py` (Quick Test)

Reduced parameter space for rapid testing.

**Configuration:**
- 2 mechanisms (conditional_amplifier, bond_weighted_conditional_amplifier)
- 2 entropies (1.0, 2.0)
- 2 history_weights (1.0, 2.0)
- 2 bond_powers (1.0, 2.0)
- 2 initial memory patterns (cw_50_50, complementary_perfect)
- **Total: 32 configurations × 30 trials = 960 simulations**
- **Runtime: ~2-3 minutes**

**Usage:**
```bash
python "during development/test_parameter_sweep_quick.py"
```

**Purpose:** Verify the framework works correctly before running larger sweeps.

### 3. `test_parameter_sweep_focused.py` (Recommended)

Targeted testing of likely failure conditions rather than full factorial.

**Test Scenarios:**
1. **Memory pattern effects** - All 4 patterns with standard settings
2. **Entropy effects** - 3 levels (0.3, 1.0, 3.0) with conflictual memory
3. **Success threshold effects** - 4 levels (0.3, 0.5, 0.7, 0.9) with C→W memory
4. **History weight effects** - 5 levels (0.1, 0.5, 1.0, 2.0, 5.0) with C→W memory
5. **Bond power effects** - 4 levels (0.5, 1.0, 2.0, 3.0) with C→W memory
6. **Bond alpha effects** - 5 levels (1.0, 2.0, 5.0, 10.0, 20.0) with conflictual memory
7. **Extreme combinations** - Hand-picked challenging scenarios

**Total: ~45-50 unique configurations × 50 trials = ~2,500 simulations**
**Runtime: ~10-15 minutes**

**Usage:**
```bash
python "during development/test_parameter_sweep_focused.py"
```

**Output:**
- Console summary showing lowest/highest success rates
- JSON file: `parameter_sweep_focused_results.json`
- Failure pattern analysis by mechanism, memory pattern, threshold

### 4. `test_parameter_sweep_challenging.py` (Comprehensive)

Full factorial sweep with extensive parameter ranges.

**Warning:** This generates >10,000 configurations and may take hours to run!

**Parameters:**
- 4 mechanisms
- 3 entropies
- 3 history_weights
- 3 smoothing_alphas
- 3 bond_powers
- 3 bond_alphas
- 3 success_thresholds
- 4 initial memory patterns

**Total: ~3,888+ configurations × 30 trials = ~116,640 simulations**
**Runtime: Several hours**

**Usage:**
```bash
python "during development/test_parameter_sweep_challenging.py"
```

## Parameter Descriptions

### Client-Level Parameters

**entropy** (Temperature for action selection)
- Low (0.3-0.5): Deterministic, exploits best action
- Medium (1.0): Balanced exploration/exploitation
- High (2.0-3.0): Stochastic, more random actions

**history_weight** (Strength of history influence)
- Low (0.1-0.5): Weak history amplification
- Medium (1.0): Standard influence
- High (2.0-5.0): Strong history dependence

**smoothing_alpha** (Laplace smoothing for conditional mechanisms)
- Low (0.01): Data-driven, sharp distributions
- Medium (0.1): Balanced
- High (0.5-1.0): More uniform prior

**bond_power** (Exponent for bond-weighted mechanisms)
- Low (0.5): Weak bond modulation
- Linear (1.0): Direct scaling
- High (2.0-3.0): Strong bond effect

### System-Level Parameters

**bond_alpha** (Steepness of RS→bond sigmoid)
- Low (1.0-2.0): Gentle transition
- Medium (4.0-5.0): Moderate (default~5.0)
- High (10.0-20.0): Very steep, binary-like

**success_threshold** (Minimum final bond for success)
- Lenient (0.3-0.4): Easy to achieve
- Moderate (0.5-0.6): Standard
- Strict (0.7-0.9): Difficult

**memory_size** (Note: Currently fixed at 50 due to base_client validation)
- Cannot be varied without modifying base_client.py
- Hardcoded in MEMORY_SIZE constant

### Initial Memory Patterns

**cw_50_50** - Pure C→W anticomplementary pattern
- Client always Cold (6), therapist always Warm (2)
- Tests recovery from negative pattern

**complementary_perfect** - Ideal starting point
- All D→S (perfect complementarity)
- Baseline for best-case scenario

**conflictual** - Worst starting point
- All D→D (conflictual)
- Tests recovery from very poor bond

**mixed_random** - Unpredictable pattern
- Random client and therapist actions
- Tests robustness to noise

## Output Format

### Console Output

**Results Table:**
```
Mech       Ent   HW   SA   BP   BA   MS  Thr  Memory           Succ%  Drop%  Bond
------------------------------------------------------------------------
...
```

**Overall Statistics:**
- Mean/median/std of success rates
- Min/max success rates
- Number of 0% and 100% success configurations
- Mean dropout rate

### JSON Output

Saved files contain array of configuration objects:
```json
[
  {
    "mechanism": "conditional_amplifier",
    "entropy": 1.0,
    "history_weight": 1.0,
    ...
    "success_rate": 0.98,
    "avg_sessions_completed": 100.0,
    "avg_final_bond": 0.85,
    "dropout_rate": 0.02
  },
  ...
]
```

## Interpreting Results

### Success Rate
- **100%**: Always-complementary strategy works perfectly under these conditions
- **80-99%**: Generally successful, occasional failures
- **50-79%**: Moderate success, substantial failure rate
- **<50%**: Strategy often fails under these conditions

### Key Patterns to Look For

1. **Initial memory effects**: Which starting patterns are hardest to overcome?
2. **Entropy sensitivity**: Does the strategy fail with very low or very high entropy?
3. **Threshold effects**: What bond level is realistic to achieve?
4. **Mechanism differences**: Do bond-weighted mechanisms perform better/worse?
5. **Parameter interactions**: Do certain combinations create failure modes?

## Example Workflow

1. **Quick validation:**
   ```bash
   python "during development/test_parameter_sweep_quick.py"
   ```
   Verify framework works, see if any failures occur

2. **Focused analysis:**
   ```bash
   python "during development/test_parameter_sweep_focused.py"
   ```
   Identify specific failure conditions

3. **Deep dive (optional):**
   Modify parameter ranges in focused script to zoom in on interesting regions

4. **Analysis:**
   - Load JSON results
   - Plot success rate vs. individual parameters
   - Identify failure patterns
   - Generate hypotheses about mechanism behavior

## Limitations

1. **Memory size cannot be varied** due to base_client validation
2. **Computational cost** - Full sweeps can take hours
3. **Random variation** - Some configs may show variation across runs
4. **Simplified therapist** - Only tests always-complementary strategy

## Future Extensions

- Test other therapist strategies (random, adversarial, etc.)
- Add multi-objective optimization (success + sessions + bond improvement)
- Implement parallel execution for faster sweeps
- Add visualization tools for result analysis
- Test with real client pattern types (cold_stuck, etc.) instead of fixed memories
