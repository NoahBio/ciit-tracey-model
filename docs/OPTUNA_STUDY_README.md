# Optuna Hyperparameter Optimization Studies

This directory contains SQLite databases for Optuna hyperparameter optimization studies.

## Quick Start

### Running an Optimization

Single-objective (maximize success rate):
```bash
cd "during development"
../venv/bin/python optuna_hyperparameter_search.py \
  --study-name "strategic_optimization" \
  --mechanism conditional_amplifier \
  --pattern cold_warm \
  --enable-strategic-therapist \
  --n-trials 100 \
  --n-seeds 50 \
  --objectives maximize:success_rate \
  --optimize-params entropy rs_plateau_threshold plateau_window intervention_duration
```

Multi-objective (maximize success, minimize sessions):
```bash
cd "during development"
../venv/bin/python optuna_hyperparameter_search.py \
  --study-name "multi_obj_study" \
  --mechanism conditional_amplifier \
  --enable-strategic-therapist \
  --n-trials 100 \
  --n-seeds 50 \
  --objectives maximize:success_rate minimize:avg_sessions_to_success \
  --optimize-params entropy mechanism pattern plateau_window intervention_duration
```

### Viewing Results

**Interactive Dashboard (recommended):**
```bash
../venv/bin/optuna-dashboard sqlite:///optuna_studies/your_study_name.db
```
Then open your browser to the URL shown (usually http://localhost:8080)

**Command-line Analysis:**
```bash
cd "during development"
../venv/bin/python optuna_utils.py \
  --study-name "your_study_name" \
  --storage-path ../optuna_studies/your_study_name.db \
  --export-csv ../optuna_studies/your_study_name/trials.csv
```

**View Results JSON:**
```bash
cat optuna_studies/your_study_name/results.json
```

## Available Objectives

- `success_rate` - Proportion of runs reaching RS threshold (maximize)
- `avg_sessions_to_success` - Average sessions until success (minimize)
- `dropout_rate` - Proportion of runs with dropout (minimize)
- `final_rs_mean` - Mean final relationship satisfaction (maximize)
- `final_bond_mean` - Mean final bond strength (maximize)

## Optimizable Parameters

### Numerical Parameters
- `entropy` (0.1 - 10.0, log scale) - Client exploration parameter
- `history_weight` (0.1 - 5.0) - For amplifier mechanisms
- `bond_power` (0.5 - 3.0) - For bond_weighted mechanisms
- `bond_alpha` (1.0 - 15.0) - Sigmoid steepness
- `bond_offset` (0.5 - 0.95) - Sigmoid inflection point
- `rs_plateau_threshold` (2.0 - 15.0) - RS range for plateau detection
- `plateau_window` (5 - 30) - Sessions to check for plateau
- `intervention_duration` (3 - 20) - Sessions of optimal intervention
- `threshold` (0.5 - 0.95) - Success threshold percentile
- `baseline_accuracy` (0.1 - 0.5) - Perception accuracy (if enabled)
- `max_sessions` (50 - 200) - Maximum therapy sessions

### Categorical Parameters
- `mechanism` - Client expectation mechanism
  - `bond_only`
  - `frequency_amplifier`
  - `conditional_amplifier`
  - `bond_weighted_frequency_amplifier`
  - `bond_weighted_conditional_amplifier`
- `pattern` - Initial memory pattern
  - `cw_50_50`, `cold_warm`, `complementary_perfect`
  - `conflictual`, `mixed_random`
  - `cold_stuck`, `dominant_stuck`, `submissive_stuck`

## Parameter Constraints

Automatically enforced:
- `intervention_duration < plateau_window` (can't intervene longer than detection window)
- `bond_offset >= 0.5` (sigmoid inflects in upper half)
- Mechanism-specific parameters only used when relevant

## Study Organization

Each study creates:
```
optuna_studies/
├── your_study_name.db           # SQLite database (resumable)
└── your_study_name/
    ├── results.json              # Best parameters or Pareto front
    ├── optimization_history.html # Plotly visualization
    ├── param_importances.html    # Parameter importance plot
    └── pareto_front.html         # Multi-objective trade-offs (if applicable)
```

## Resuming Studies

Studies are automatically resumed if you use the same `--study-name`:
```bash
# Run initial 50 trials
../venv/bin/python optuna_hyperparameter_search.py --study-name "my_study" --n-trials 50 ...

# Resume and run 50 more trials (total: 100)
../venv/bin/python optuna_hyperparameter_search.py --study-name "my_study" --n-trials 50 ...
```

## Tips

1. **Start small**: Use `--n-trials 10 --n-seeds 20` to test your configuration
2. **Increase gradually**: Move to `--n-trials 100 --n-seeds 50` for real optimization
3. **Multi-objective**: Use when you care about multiple metrics (e.g., success AND speed)
4. **Dashboard**: The optuna-dashboard is the best way to explore results interactively
5. **Categorical params**: Include `mechanism` or `pattern` in `--optimize-params` to find best mechanism/pattern

## Database Files

`.db`, `.db-journal`, `.db-shm`, and `.db-wal` files are ignored by git (can be large).
Results JSON files and visualizations are small and can be committed.
