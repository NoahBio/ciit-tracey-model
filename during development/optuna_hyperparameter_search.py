"""Optuna hyperparameter optimization for strategic therapist.

This script uses Optuna to find optimal parameter configurations that maximize
(or minimize) therapy success rates and other objectives.

Features:
- Single-objective and multi-objective optimization
- Support for both numerical and categorical parameters
- Parameter constraint enforcement
- Persistent SQLite storage with optuna-dashboard visualization
- Resumable studies

Example usage:

# Single-objective: Maximize success rate
python optuna_hyperparameter_search.py \
  --study-name "strategic_optimization" \
  --mechanism conditional_amplifier \
  --pattern cold_warm \
  --enable-strategic-therapist \
  --n-trials 100 \
  --objectives maximize:success_rate \
  --optimize-params entropy rs_plateau_threshold plateau_window intervention_duration

# Multi-objective: Maximize success rate AND minimize sessions
python optuna_hyperparameter_search.py \
  --study-name "multi_obj_optimization" \
  --mechanism conditional_amplifier \
  --enable-strategic-therapist \
  --n-trials 100 \
  --objectives maximize:success_rate minimize:avg_sessions_to_success \
  --optimize-params entropy mechanism pattern plateau_window

# View results in browser:
optuna-dashboard sqlite:///optuna_studies/strategic_optimization.db
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_pareto_front,
)

from run_multi_seed_simulation import run_multi_seed_simulation, MultiSeedStatistics


# Define comprehensive search space for all parameters
SEARCH_SPACE_DEFINITIONS = {
    # Numerical parameters
    'entropy': ('float', 0.01, 10.0, 'log'),
    'history_weight': ('float', 0.1, 5.0, 'uniform'),
    'bond_power': ('float', 0.5, 3.0, 'uniform'),
    'bond_alpha': ('float', 1.0, 15.0, 'uniform'),
    'bond_offset': ('float', 0.5, 0.95, 'uniform'),
    'rs_plateau_threshold': ('float', 2.0, 25.0, 'uniform'),
    'plateau_window': ('int', 5, 30),
    'intervention_duration': ('int', 3, 20),
    'threshold': ('float', 0.5, 0.95, 'uniform'),
    'baseline_accuracy': ('float', 0.1, 0.7, 'uniform'),
    'max_sessions': ('int', 50, 200),

    # Categorical parameters
    'mechanism': ('categorical', [
        'bond_only',
        'frequency_amplifier',
        'conditional_amplifier',
        'bond_weighted_frequency_amplifier',
        'bond_weighted_conditional_amplifier'
    ]),
    'pattern': ('categorical', [
        'cw_50_50',
        'cold_warm',
        'complementary_perfect',
        'conflictual',
        'mixed_random',
        'cold_stuck',
        'dominant_stuck',
        'submissive_stuck'
    ]),
}

# Map parameter names to run_multi_seed_simulation argument names
PARAM_NAME_MAPPING = {
    'threshold': 'success_threshold_percentile',
    'pattern': 'initial_memory_pattern',
}


def suggest_parameter(trial: optuna.Trial, param_name: str) -> Any:
    """Suggest a parameter value from the search space."""
    if param_name not in SEARCH_SPACE_DEFINITIONS:
        raise ValueError(f"Unknown parameter: {param_name}")

    spec = SEARCH_SPACE_DEFINITIONS[param_name]
    param_type = spec[0]

    if param_type == 'float':
        low, high = spec[1], spec[2]
        log = len(spec) > 3 and spec[3] == 'log'
        return trial.suggest_float(param_name, low, high, log=log)

    elif param_type == 'int':
        low, high = spec[1], spec[2]
        return trial.suggest_int(param_name, low, high)

    elif param_type == 'categorical':
        choices = spec[1]
        return trial.suggest_categorical(param_name, choices)

    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def enforce_constraints(params: Dict[str, Any], trial: optuna.Trial) -> None:
    """Enforce parameter constraints.

    Raises optuna.TrialPruned if constraints are violated.
    """
    # Constraint: intervention_duration < plateau_window
    if 'intervention_duration' in params and 'plateau_window' in params:
        if params['intervention_duration'] >= params['plateau_window']:
            raise optuna.TrialPruned(
                f"Constraint violated: intervention_duration ({params['intervention_duration']}) "
                f">= plateau_window ({params['plateau_window']})"
            )


def extract_objectives(
    stats: 'MultiSeedStatistics',
    objective_specs: List[Tuple[str, str]]
) -> Tuple[float, ...]:
    """Extract objective values from MultiSeedStatistics.

    Parameters
    ----------
    stats : MultiSeedStatistics
        Simulation results
    objective_specs : List[Tuple[str, str]]
        List of (direction, metric_name) tuples

    Returns
    -------
    Tuple[float, ...]
        Objective values (one per objective)
    """
    objectives = []

    for direction, metric in objective_specs:
        if metric == 'success_rate':
            value = stats.success_rate
        elif metric == 'avg_sessions_to_success':
            # Use mean of successful runs, or max_sessions if no successes
            value = stats.success_sessions_mean if stats.success_sessions_mean is not None else 999.0
        elif metric == 'dropout_rate':
            value = stats.dropout_rate
        elif metric == 'final_rs_mean':
            value = stats.final_rs_stats.get('mean', 0.0)
        elif metric == 'final_bond_mean':
            value = stats.final_bond_stats.get('mean', 0.0)
        else:
            raise ValueError(f"Unknown objective metric: {metric}")

        objectives.append(value)

    return tuple(objectives)


def objective(
    trial: optuna.Trial,
    fixed_config: Dict[str, Any],
    optimize_params: List[str],
    objective_specs: List[Tuple[str, str]],
    n_seeds: int = 50
) -> Tuple[float, ...] | float:
    """Objective function for Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    fixed_config : Dict[str, Any]
        Fixed parameters for the simulation
    optimize_params : List[str]
        List of parameters to optimize
    objective_specs : List[Tuple[str, str]]
        List of (direction, metric_name) tuples
    n_seeds : int
        Number of random seeds per trial

    Returns
    -------
    float or Tuple[float, ...]
        Single value for single-objective, tuple for multi-objective
    """
    # Sample parameters to optimize
    trial_params = {}
    for param in optimize_params:
        trial_params[param] = suggest_parameter(trial, param)

    # Enforce constraints
    combined_params = {**fixed_config, **trial_params}
    enforce_constraints(combined_params, trial)

    # Map parameter names to simulation argument names
    sim_kwargs = {}
    for key, value in combined_params.items():
        mapped_key = PARAM_NAME_MAPPING.get(key, key)
        sim_kwargs[mapped_key] = value

    # Add fixed arguments
    sim_kwargs['n_seeds'] = n_seeds
    sim_kwargs['show_trajectories'] = False
    sim_kwargs['show_actions'] = False
    sim_kwargs['verbose'] = False

    # Run simulation
    try:
        stats = run_multi_seed_simulation(**sim_kwargs)
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned(f"Simulation failed: {e}")

    # Extract objective values
    objective_values = extract_objectives(stats, objective_specs)

    # Log results
    print(f"Trial {trial.number}: {dict(trial_params)} -> Objectives: {objective_values}")

    # Return single value for single-objective, tuple for multi-objective
    if len(objective_values) == 1:
        return objective_values[0]
    else:
        return objective_values


def create_or_load_study(
    study_name: str,
    storage_path: Path,
    directions: List[str]
) -> optuna.Study:
    """Create new study or load existing one.

    Parameters
    ----------
    study_name : str
        Name of the study
    storage_path : Path
        Path to SQLite database file
    directions : List[str]
        List of optimization directions ('maximize' or 'minimize')

    Returns
    -------
    optuna.Study
        Optuna study object
    """
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        directions=directions,
        sampler=optuna.samplers.TPESampler()
    )

    return study


def parse_objectives(objectives_str: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Parse objective specifications.

    Parameters
    ----------
    objectives_str : List[str]
        List of objectives like ['maximize:success_rate', 'minimize:avg_sessions_to_success']

    Returns
    -------
    directions : List[str]
        List of directions ['maximize', 'minimize']
    objective_specs : List[Tuple[str, str]]
        List of (direction, metric) tuples
    """
    directions = []
    objective_specs = []

    for obj_str in objectives_str:
        if ':' in obj_str:
            direction, metric = obj_str.split(':', 1)
        else:
            # Default to maximize
            direction = 'maximize'
            metric = obj_str

        if direction not in ['maximize', 'minimize']:
            raise ValueError(f"Direction must be 'maximize' or 'minimize', got: {direction}")

        directions.append(direction)
        objective_specs.append((direction, metric))

    return directions, objective_specs


def save_results(
    study: optuna.Study,
    output_path: Path,
    objective_specs: List[Tuple[str, str]],
    fixed_config: Dict[str, Any],
    optimize_params: List[str],
    n_seeds: int,
    n_trials: int
) -> None:
    """Save optimization results to JSON.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    output_path : Path
        Path to save results JSON
    objective_specs : List[Tuple[str, str]]
        Objective specifications
    fixed_config : Dict[str, Any]
        Fixed parameters used in the study
    optimize_params : List[str]
        Parameters that were optimized
    n_seeds : int
        Number of seeds per trial
    n_trials : int
        Number of trials run
    """
    results = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat(),
        'objectives': [{'direction': d, 'metric': m} for d, m in objective_specs],
        'configuration': {
            'fixed_parameters': fixed_config,
            'optimized_parameters': optimize_params,
            'n_seeds_per_trial': n_seeds,
            'n_trials_requested': n_trials,
        }
    }

    if len(objective_specs) == 1:
        # Single-objective
        best_trial = study.best_trial
        results['best_trial'] = {
            'number': best_trial.number,
            'params': best_trial.params,
            'value': best_trial.value,
        }
    else:
        # Multi-objective: save Pareto front
        pareto_trials = study.best_trials
        results['pareto_front'] = [
            {
                'number': t.number,
                'params': t.params,
                'values': t.values,
            }
            for t in pareto_trials
        ]
        results['n_pareto_solutions'] = len(pareto_trials)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def display_results(study: optuna.Study, objective_specs: List[Tuple[str, str]]) -> None:
    """Display optimization results."""
    print("\n" + "=" * 100)
    print("OPTIMIZATION RESULTS")
    print("=" * 100)

    if len(objective_specs) == 1:
        # Single-objective
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_trial.value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    else:
        # Multi-objective
        print(f"\nNumber of Pareto optimal solutions: {len(study.best_trials)}")
        print("\nPareto front:")
        for i, trial in enumerate(study.best_trials[:10]):  # Show first 10
            print(f"\nSolution {i+1} (Trial {trial.number}):")
            print(f"  Objectives: {trial.values}")
            print(f"  Parameters: {trial.params}")

        if len(study.best_trials) > 10:
            print(f"\n... and {len(study.best_trials) - 10} more solutions")


def generate_visualizations(
    study: optuna.Study,
    output_dir: Path,
    objective_specs: List[Tuple[str, str]]
) -> None:
    """Generate and save visualization plots.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    output_dir : Path
        Directory to save plots
    objective_specs : List[Tuple[str, str]]
        Objective specifications
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(output_dir / "optimization_history.html")
        print(f"Saved: {output_dir / 'optimization_history.html'}")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    try:
        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_html(output_dir / "param_importances.html")
        print(f"Saved: {output_dir / 'param_importances.html'}")
    except Exception as e:
        print(f"Could not generate parameter importance plot: {e}")

    # Pareto front for multi-objective
    if len(objective_specs) > 1:
        try:
            fig = plot_pareto_front(study)
            fig.write_html(output_dir / "pareto_front.html")
            print(f"Saved: {output_dir / 'pareto_front.html'}")
        except Exception as e:
            print(f"Could not generate Pareto front plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for strategic therapist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Optuna-specific arguments
    parser.add_argument('--study-name', required=True, help='Name for the Optuna study')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--n-seeds', type=int, default=50, help='Number of seeds per trial evaluation')
    parser.add_argument(
        '--objectives',
        nargs='+',
        default=['maximize:success_rate'],
        help='Objectives to optimize (e.g., maximize:success_rate minimize:avg_sessions_to_success)'
    )
    parser.add_argument(
        '--optimize-params',
        nargs='+',
        required=True,
        help='Parameters to optimize (e.g., entropy bond_power mechanism)'
    )
    parser.add_argument('--timeout', type=int, help='Timeout in seconds for optimization')

    # Simulation parameters (fixed values)
    parser.add_argument(
        '--mechanism',
        default='conditional_amplifier',
        choices=['bond_only', 'frequency_amplifier', 'conditional_amplifier',
                 'bond_weighted_frequency_amplifier', 'bond_weighted_conditional_amplifier'],
        help='Client expectation mechanism (can be optimized if in --optimize-params)'
    )
    parser.add_argument(
        '--pattern',
        default='cold_warm',
        choices=['cw_50_50', 'cold_warm', 'complementary_perfect', 'conflictual',
                 'mixed_random', 'cold_stuck', 'dominant_stuck', 'submissive_stuck'],
        help='Initial memory pattern (can be optimized if in --optimize-params)'
    )
    parser.add_argument('--threshold', type=float, default=0.8, help='Success threshold percentile')
    parser.add_argument('--enable-perception', action='store_true', help='Enable perceptual distortion')
    parser.add_argument('--baseline-accuracy', type=float, default=0.2, help='Baseline perception accuracy')
    parser.add_argument('--max-sessions', type=int, default=100, help='Maximum sessions per run')
    parser.add_argument('--entropy', type=float, default=3.0, help='Client entropy')
    parser.add_argument('--history-weight', type=float, default=1.0, help='History weight for amplifiers')
    parser.add_argument('--bond-power', type=float, default=1.0, help='Bond power for bond_weighted mechanisms')
    parser.add_argument('--bond-alpha', type=float, default=5.0, help='Bond alpha (sigmoid steepness)')
    parser.add_argument('--bond-offset', type=float, default=0.8, help='Bond offset for sigmoid')
    parser.add_argument('--enable-strategic-therapist', action='store_true', help='Enable strategic therapist')
    parser.add_argument('--rs-plateau-threshold', type=float, default=5.0, help='RS plateau threshold')
    parser.add_argument('--plateau-window', type=int, default=15, help='Plateau detection window')
    parser.add_argument('--intervention-duration', type=int, default=10, help='Intervention duration')

    args = parser.parse_args()

    # Parse objectives
    directions, objective_specs = parse_objectives(args.objectives)

    # Validate optimize_params
    for param in args.optimize_params:
        if param not in SEARCH_SPACE_DEFINITIONS:
            raise ValueError(
                f"Unknown parameter to optimize: {param}\n"
                f"Available parameters: {list(SEARCH_SPACE_DEFINITIONS.keys())}"
            )

    # Build fixed config (parameters NOT being optimized)
    fixed_config = {}
    all_params = {
        'mechanism': args.mechanism,
        'pattern': args.pattern,
        'threshold': args.threshold,
        'enable_perception': args.enable_perception,
        'baseline_accuracy': args.baseline_accuracy,
        'max_sessions': args.max_sessions,
        'entropy': args.entropy,
        'history_weight': args.history_weight,
        'bond_power': args.bond_power,
        'bond_alpha': args.bond_alpha,
        'bond_offset': args.bond_offset,
        'enable_strategic_therapist': args.enable_strategic_therapist,
        'rs_plateau_threshold': args.rs_plateau_threshold,
        'plateau_window': args.plateau_window,
        'intervention_duration': args.intervention_duration,
    }

    for key, value in all_params.items():
        if key not in args.optimize_params:
            fixed_config[key] = value

    # Create study
    storage_path = project_root / "optuna_studies" / f"{args.study_name}.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    study = create_or_load_study(args.study_name, storage_path, directions)

    print("=" * 100)
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 100)
    print(f"Study name: {args.study_name}")
    print(f"Storage: {storage_path}")
    print(f"Objectives: {[f'{d}:{m}' for d, m in objective_specs]}")
    print(f"Parameters to optimize: {args.optimize_params}")
    print(f"Fixed parameters: {list(fixed_config.keys())}")
    print(f"Trials: {args.n_trials}")
    print(f"Seeds per trial: {args.n_seeds}")
    print("=" * 100)
    print()

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            fixed_config,
            args.optimize_params,
            objective_specs,
            args.n_seeds
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Display results
    display_results(study, objective_specs)

    # Save results
    results_dir = project_root / "optuna_studies" / args.study_name
    results_dir.mkdir(parents=True, exist_ok=True)

    save_results(
        study, 
        results_dir / "results.json", 
        objective_specs,
        fixed_config,
        args.optimize_params,
        args.n_seeds,
        args.n_trials
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(study, results_dir, objective_specs)

    # Print dashboard command
    print("\n" + "=" * 100)
    print("To view interactive dashboard, run:")
    print(f"optuna-dashboard sqlite:///{storage_path}")
    print("=" * 100)


if __name__ == '__main__':
    main()
