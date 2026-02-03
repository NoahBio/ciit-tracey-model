# Aggregated Complementarity Visualization Across Configs

## Overview

This script (`aggregate_complementarity_across_configs.py`) samples multiple configs from an Optuna database and aggregates complementarity dynamics across all of them. It creates a unified visualization showing how V2 advantage, baseline advantage, and remaining seeds differ in their complementarity patterns.

## What It Does

1. **Loads configs from Optuna database**: Samples configs uniformly across the v2_advantage spectrum
2. **Runs massive simulations**: For each config, runs both baseline complementary and V2 omniscient therapists for thousands of seeds
3. **Categorizes seeds**: Divides seeds into 4 categories:
   - **V2 Advantage** (green): V2 succeeded, baseline failed
   - **Baseline Advantage** (red): Baseline succeeded, V2 failed
   - **Both Success** (blue): Both therapists succeeded
   - **Both Fail** (gray): Neither therapist succeeded
4. **Aggregates trajectories**: Combines complementarity distance trajectories across all configs
5. **Visualizes**: Creates a single plot showing the green, red, and black (remaining = both_success + both_fail) lines

## Usage

### Basic Command

```bash
python scripts/aggregate_complementarity_across_configs.py \
    --db-path optuna_studies/freq_amp_v2_optimization.db \
    --study-name freq_amp_v2_optimization \
    --n-configs 1000 \
    --n-seeds 10000 \
    --max-sessions 300
```

### Full Command (as requested)

```bash
python scripts/aggregate_complementarity_across_configs.py \
    --db-path optuna_studies/freq_amp_v2_optimization.db \
    --study-name freq_amp_v2_optimization \
    --n-configs 1000 \
    --n-seeds 10000 \
    --max-sessions 300 \
    --mechanism frequency_amplifier \
    --pattern cold_stuck \
    --window-size 10 \
    --n-workers $(nproc)
```

### Test Run (recommended first)

Before running the full 20 million simulations, test with smaller parameters:

```bash
python scripts/aggregate_complementarity_across_configs.py \
    --n-configs 10 \
    --n-seeds 100 \
    --max-sessions 300 \
    --n-workers 4
```

## Performance Estimates

### Full Run (1000 configs × 10000 seeds)

- **Total simulations**: 20,000,000 (1000 configs × 10000 seeds × 2 therapists)
- **Estimated time**:
  - On 1 core: ~2000 hours (83 days)
  - On 40 cores: ~50 hours (2 days)
  - On 100 cores: ~20 hours
- **Memory usage**: ~500 MB peak (using online aggregation)
- **Disk space**:
  - ~260 GB for checkpoints (temporary, can be deleted after)
  - ~10-20 GB for final results

### Recommendations for Large Runs

1. **Use a cluster**: This is designed for HPC environments
2. **Start small**: Test with 10-100 configs first
3. **Use all cores**: Set `--n-workers $(nproc)` to use all available CPUs
4. **Monitor progress**: The script shows a progress bar
5. **Save checkpoints**: Results are saved incrementally

## Parameters

### Database Parameters
- `--db-path`: Path to Optuna SQLite database (default: `optuna_studies/freq_amp_v2_optimization.db`)
- `--study-name`: Name of the Optuna study (default: `freq_amp_v2_optimization`)

### Sampling Parameters
- `--n-configs`: Number of configs to sample from database (default: 1000)
- `--n-seeds`: Number of seeds per config (default: 10000)

### Simulation Parameters
- `--max-sessions`: Maximum therapy sessions (default: 300)
- `--window-size`: Complementarity tracking window (default: 10)
- `--mechanism`: Client mechanism (default: `frequency_amplifier`)
- `--pattern`: Initial memory pattern (default: `cold_stuck`)

### Performance Parameters
- `--n-workers`: Number of parallel workers (default: CPU count)
- `--batch-size`: Batch size for processing (default: 10)

### Output
- `--output-dir`: Custom output directory (default: timestamped in `results/`)

## Output Files

The script creates a timestamped directory with:

1. **aggregated_complementarity.png**: Main visualization showing green/red/black lines
2. **aggregated_trajectories.npz**: Numpy arrays of mean/std trajectories
3. **config_info.json**: Information about sampled configs
4. **summary.json**: Summary statistics and breakdown

## Interpretation

- **Green line (V2 Advantage)**: Shows complementarity patterns for seeds where V2's omniscience provided unique value
- **Red line (Baseline Advantage)**: Shows where always-complementary therapist succeeded but V2 failed
- **Black line (Remaining)**: Aggregates both_success (both won) and both_fail (both lost)

Lower octant distance = more complementary behavior (0 = perfect complementarity, 4 = maximally anti-complementary)

## Memory Optimization

The script uses **online aggregation with Welford's algorithm** to compute statistics incrementally without storing all results in memory.

### How It Works

1. **Online Statistics**: Instead of accumulating all ConfigResults in RAM, the script updates running mean/variance statistics as each config completes
2. **Welford's Algorithm**: Numerically stable one-pass algorithm for computing mean and standard deviation
3. **Checkpointing**: Each ConfigResult is saved to disk immediately after completion for recovery and verification
4. **Memory Footprint**: O(max_sessions) instead of O(n_configs × n_seeds × max_sessions)

### Memory Usage Comparison

| Approach | Memory Usage (1000 configs × 10000 seeds) |
|----------|------------------------------------------|
| Old (accumulate all) | ~260 GB → OOM killed |
| New (online aggregation) | ~500 MB peak |

### Checkpointing System

Results are automatically saved to `{output_dir}/checkpoints/` as they complete:
- Format: `config_XXXX_trial_YYYY.pkl` (pickle format)
- Purpose: Recovery from failures, verification, post-hoc analysis
- Size: ~260 MB per config
- Total: ~260 GB for 1000 configs (can be deleted after aggregation completes)

### Monitoring Memory Usage

The script logs memory usage at key points:
```
[MEMORY] After initializing online stats: 150.2 MB
[MEMORY] After 100 configs: 420.5 MB
[MEMORY] After 200 configs: 380.1 MB
[MEMORY] After aggregation complete: 200.3 MB
```

Note: `psutil` is required for memory monitoring. Install with `pip install psutil` if needed.

## Troubleshooting

### Out of Memory (OOM)
The script should now handle large runs without OOM issues. If you still encounter OOM:
- Ensure you're running the latest version with online aggregation
- Check that checkpoints are being created (this confirms online mode is active)
- Verify memory logs show constant usage (not growing)
- If using an older version, update to the memory-efficient implementation

### Too Slow
- Increase `--n-workers` to use more CPU cores
- Use a cluster or HPC system
- Reduce `--n-configs` or `--n-seeds` for testing
- First config may take longer due to worker initialization with spawn context (~10-20 seconds)

### No Results
- Check that the database exists and has completed trials
- Verify the study name matches
- Run a small test first

### Disk Space
- Checkpoints require ~260 GB for 1000 configs
- Checkpoints can be safely deleted after the script completes
- To manually clean up: `rm -rf {output_dir}/checkpoints/`

## Technical Details

### Architecture
- **Multiprocessing**: Uses `spawn` context for clean worker isolation (avoids global state issues)
- **Parallelization**: Processes configs in parallel (one config per worker)
- **Online Aggregation**: Welford's algorithm for incremental mean/std computation
- **Checkpointing**: Pickle-based serialization for recovery and verification

### Memory Characteristics
- **Peak usage**: One ConfigResult (~260 MB) + accumulators (~16 KB) = ~500 MB
- **Constant footprint**: Memory does not grow with number of configs processed
- **Garbage collection**: Explicit cleanup after each config and every 100 configs

### Compatibility
- Requires Python 3.7+ for dataclasses
- Works with all numpy/matplotlib versions
- `psutil` optional (for memory monitoring only)
- Compatible with HPC/cluster environments
