"""Optuna optimization to find configs where omniscient therapist outperforms baseline.

This script uses Bayesian optimization (Optuna) to search for parameter configurations
where the omniscient strategic therapist achieves significant improvements over pure
complementary therapy.

Fixed constraints (user-specified):
- enable_parataxic = True
- baseline_accuracy > 0.2
- entropy < 1
- pattern = cold_stuck

Example usage:

# Run 50 trials
python scripts/optimize_omniscient_advantage.py \
  --n-trials 50 \
  --study-name omniscient_advantage_v1 \
  --n-seeds 50

# View results in browser:
optuna-dashboard sqlite:///optuna_studies/omniscient_advantage_v1.db
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from typing import Any
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)

# Import evaluation infrastructure
sys.path.insert(0, str(project_root / "scripts"))
from evaluate_omniscient_therapist import run_single_simulation, compute_statistics


# Define search space (with constraints enforced)
SEARCH_SPACE = {
    # Categorical
    'mechanism': ('categorical', [
        'bond_only',
        'frequency_amplifier',
        'conditional_amplifier',
        'bond_weighted_frequency_amplifier',
        'bond_weighted_conditional_amplifier'
    ]),

    # Constrained parameters (user requirements)
    'baseline_accuracy': ('float', 0.21, 0.9, 'uniform'),  # > 0.2
    'entropy': ('float', 0.05, 0.99, 'log'),  # < 1

    # Variable parameters
    'threshold': ('float', 0.5, 0.95, 'uniform'),
    'bond_alpha': ('float', 1.0, 15.0, 'uniform'),
    'bond_offset': ('float', 0.5, 0.95, 'uniform'),
    'recency_weighting_factor': ('int', 1, 5),
    'perception_window': ('int', 5, 30),
    'max_sessions': ('int', 50, 200),

    # Mechanism-specific (conditionally sampled)
    'history_weight': ('float', 0.1, 5.0, 'uniform'),
    'bond_power': ('float', 0.5, 3.0, 'uniform'),
}


def suggest_parameter(trial: optuna.Trial, param_name: str) -> Any:
    """Suggest a parameter value from the search space."""
    if param_name not in SEARCH_SPACE:
        raise ValueError(f"Unknown parameter: {param_name}")

    spec = SEARCH_SPACE[param_name]
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


def objective(trial: optuna.Trial, n_seeds: int = 50, therapist_version: str = 'v2') -> float:
    """Objective function that maximizes (omniscient_success - complementary_success).

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    n_seeds : int
        Number of random seeds per trial (default: 50)
    therapist_version : str
        Therapist version to use ('v1' or 'v2', default: 'v2')

    Returns
    -------
    float
        Advantage: omniscient_success_rate - complementary_success_rate
    """
    # Fixed constraints (user-specified)
    enable_parataxic = True
    pattern = 'cold_stuck'

    # Sample core parameters
    mechanism = suggest_parameter(trial, 'mechanism')
    baseline_accuracy = suggest_parameter(trial, 'baseline_accuracy')
    entropy = suggest_parameter(trial, 'entropy')
    threshold = suggest_parameter(trial, 'threshold')
    bond_alpha = suggest_parameter(trial, 'bond_alpha')
    bond_offset = suggest_parameter(trial, 'bond_offset')
    recency_weighting_factor = suggest_parameter(trial, 'recency_weighting_factor')
    perception_window = suggest_parameter(trial, 'perception_window')
    max_sessions = suggest_parameter(trial, 'max_sessions')

    # Mechanism-specific parameters
    if 'amplifier' in mechanism:
        history_weight = suggest_parameter(trial, 'history_weight')
    else:
        history_weight = 1.0

    if 'bond_weighted' in mechanism:
        bond_power = suggest_parameter(trial, 'bond_power')
    else:
        bond_power = 1.0

    # Common simulation kwargs
    sim_kwargs = {
        'mechanism': mechanism,
        'initial_memory_pattern': pattern,
        'success_threshold_percentile': threshold,
        'enable_parataxic': enable_parataxic,
        'baseline_accuracy': baseline_accuracy,
        'perception_window': perception_window,
        'max_sessions': max_sessions,
        'entropy': entropy,
        'history_weight': history_weight,
        'bond_power': bond_power,
        'bond_alpha': bond_alpha,
        'bond_offset': bond_offset,
        'recency_weighting_factor': recency_weighting_factor,
    }

    # Run BOTH therapists with same configuration
    omniscient_results = []
    complementary_results = []

    for seed in range(n_seeds):
        # Omniscient therapist
        omni_result = run_single_simulation(
            seed=seed,
            therapist_type='omniscient',
            therapist_version=therapist_version,
            **sim_kwargs
        )
        omniscient_results.append(omni_result)

        # Complementary therapist (baseline)
        comp_result = run_single_simulation(
            seed=seed,
            therapist_type='complementary',
            therapist_version=therapist_version,
            **sim_kwargs
        )
        complementary_results.append(comp_result)

    # Compute statistics
    omniscient_stats = compute_statistics(omniscient_results)
    complementary_stats = compute_statistics(complementary_results)

    # Calculate advantage
    advantage = omniscient_stats['success_rate'] - complementary_stats['success_rate']

    # Store detailed results as user attributes for analysis
    trial.set_user_attr('omniscient_success', omniscient_stats['success_rate'])
    trial.set_user_attr('complementary_success', complementary_stats['success_rate'])
    trial.set_user_attr('advantage', advantage)
    trial.set_user_attr('omniscient_dropout', omniscient_stats['dropout_rate'])
    trial.set_user_attr('complementary_dropout', complementary_stats['dropout_rate'])

    # Log progress
    print(f"\nTrial {trial.number}:")
    print(f"  Mechanism: {mechanism}")
    print(f"  Baseline Acc: {baseline_accuracy:.2f}, Entropy: {entropy:.3f}")
    print(f"  Omniscient: {omniscient_stats['success_rate']:.1%}")
    print(f"  Complementary: {complementary_stats['success_rate']:.1%}")
    print(f"  Advantage: {advantage:+.1%}")

    return advantage


def create_or_load_study(
    study_name: str,
    storage_path: Path
) -> optuna.Study:
    """Create new study or load existing one.

    Parameters
    ----------
    study_name : str
        Name of the study
    storage_path : Path
        Path to SQLite database file

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
        direction='maximize',  # Maximize advantage
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    return study


def export_results(study: optuna.Study, output_dir: Path, top_k: int = 15):
    """Export top results and visualizations.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    output_dir : Path
        Directory to save results
    top_k : int
        Number of top trials to export (default: 15)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get best trials (filter out trials without user attributes and with valid values)
    valid_trials = [t for t in study.best_trials
                    if 'omniscient_success' in t.user_attrs and t.value is not None]
    best_trials = sorted(valid_trials, key=lambda t: t.value or 0.0, reverse=True)[:top_k]

    if not best_trials:
        print("\nWarning: No valid trials found with user attributes.")
        return

    # Export best configs
    best_configs = []
    for i, trial in enumerate(best_trials):
        config = {
            'rank': i + 1,
            'trial_number': trial.number,
            'advantage': trial.value,
            'omniscient_success': trial.user_attrs.get('omniscient_success', 0.0),
            'complementary_success': trial.user_attrs.get('complementary_success', 0.0),
            'parameters': trial.params,
        }
        best_configs.append(config)

    # Save to JSON
    results_file = output_dir / f"top_{top_k}_configs.json"
    with open(results_file, 'w') as f:
        json.dump(best_configs, f, indent=2)

    print(f"\nTop {top_k} configs saved to: {results_file}")

    # Generate visualizations
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        print(f"Optimization history saved to: {output_dir / 'optimization_history.html'}")

        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        print(f"Parameter importance saved to: {output_dir / 'param_importances.html'}")

    except Exception as e:
        print(f"Warning: Failed to generate visualizations: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Study name: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best advantage: {study.best_value:+.1%}")
    print(f"\nTop {min(5, top_k)} Configurations:")
    print(f"{'-'*80}")

    for i, config in enumerate(best_configs[:5]):
        print(f"\nRank {i+1}:")
        print(f"  Advantage: {config['advantage']:+.1%} "
              f"(Omni: {config['omniscient_success']:.1%}, "
              f"Comp: {config['complementary_success']:.1%})")
        print(f"  Mechanism: {config['parameters']['mechanism']}")
        print(f"  Baseline Acc: {config['parameters']['baseline_accuracy']:.2f}, "
              f"Entropy: {config['parameters']['entropy']:.3f}")
        print(f"  Threshold: {config['parameters']['threshold']:.2f}, "
              f"Bond Offset: {config['parameters']['bond_offset']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Find configs where omniscient therapist outperforms baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--study-name', '-s',
        type=str,
        default='omniscient_advantage_v1',
        help='Name of the optimization study'
    )

    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='Number of optimization trials to run'
    )

    parser.add_argument(
        '--n-seeds',
        type=int,
        default=50,
        help='Number of random seeds per trial'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for the entire study'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=15,
        help='Number of top configs to export'
    )

    parser.add_argument(
        '--therapist-version',
        type=str,
        default='v2',
        choices=['v1', 'v2'],
        help='Omniscient therapist version (v1=original, v2=with feedback monitoring)'
    )

    args = parser.parse_args()

    # Setup paths
    storage_dir = project_root / "optuna_studies"
    storage_dir.mkdir(exist_ok=True)
    storage_path = storage_dir / f"{args.study_name}.db"

    output_dir = storage_dir / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create or load study
    print(f"{'='*80}")
    print(f"OMNISCIENT ADVANTAGE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Study name: {args.study_name}")
    print(f"Database: {storage_path}")
    print(f"Trials: {args.n_trials}")
    print(f"Seeds per trial: {args.n_seeds}")
    print(f"{'='*80}\n")

    study = create_or_load_study(args.study_name, storage_path)

    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, n_seeds=args.n_seeds, therapist_version=args.therapist_version),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")

    # Export results
    export_results(study, output_dir, top_k=args.top_k)

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTo visualize results in browser:")
    print(f"  optuna-dashboard sqlite:///{storage_path}")
    print(f"\nTo analyze top configs:")
    print(f"  cat {output_dir / f'top_{args.top_k}_configs.json'}")


if __name__ == "__main__":
    main()
