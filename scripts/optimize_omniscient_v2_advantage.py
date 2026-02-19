"""Optuna optimization for frequency_amplifier with OmniscientStrategicTherapist V2.

This script uses Bayesian optimization (Optuna) to search for parameter configurations
where the omniscient strategic therapist achieves significant improvements over pure
complementary therapy.

Fixed constraints:
- mechanism = 'frequency_amplifier'
- enable_parataxic = True
- entropy = 1.0
- pattern = 'cold_stuck'

Optimized parameters:
- Client: baseline_accuracy, threshold, bond_alpha, bond_offset, recency_weighting_factor, max_sessions
- Therapist: perception_window, seeding_benefit_scaling, skip_seeding_accuracy_threshold,
             quick_seed_actions_threshold, abort_consecutive_failures_threshold

Example usage:

# Run 100 trials with 50 seeds each
python scripts/optimize_omniscient_v2_advantage.py \
  --n-trials 100 \
  --study-name freq_amp_v2_optimization \
  --n-seeds 50 \
  --therapist-version v2

# View results in browser:
optuna-dashboard sqlite:///optuna_studies/freq_amp_v2_optimization.db
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import evaluation infrastructure
sys.path.insert(0, str(project_root / "scripts"))
from evaluate_omniscient_therapist import run_single_simulation, compute_statistics


# Define search space per user specifications
SEARCH_SPACE = {
    # FIXED: mechanism = 'frequency_amplifier' (not in search space)
    # FIXED: entropy = 1.0 (not in search space)

    # Client parameters
    'baseline_accuracy': ('float', 0.21, 0.9, 'uniform'),
    'threshold': ('float', 0.8, 0.99, 'uniform'),
    'bond_alpha': ('float', 1.0, 15.0, 'uniform'),
    'bond_offset': ('float', 0.5, 0.95, 'uniform'),
    'recency_weighting_factor': ('float', 1.0, 5.0, 'uniform'),
    'max_sessions': ('int', 100, 2000),

    # Therapist parameters
    'perception_window': ('int', 7, 20),
    'seeding_benefit_scaling': ('float', 0.1, 2.0, 'uniform'),
    'skip_seeding_accuracy_threshold': ('float', 0.75, 0.95, 'uniform'),
    'quick_seed_actions_threshold': ('int', 1, 5),
    'abort_consecutive_failures_threshold': ('int', 4, 9),
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


def run_seed_therapist_pair(args):
    """Helper function to run a single (seed, therapist_type) simulation.

    Must be module-level for ProcessPoolExecutor pickling.

    Parameters
    ----------
    args : tuple
        (seed, therapist_type, therapist_version, sim_kwargs)

    Returns
    -------
    SimulationResult
        Result of the simulation
    """
    seed, therapist_type, therapist_version, sim_kwargs = args
    return run_single_simulation(
        seed=seed,
        therapist_type=therapist_type,
        therapist_version=therapist_version,
        **sim_kwargs
    )


def objective(trial: optuna.Trial, n_seeds: int = 50, therapist_version: str = 'v2', n_workers: int | None = None) -> float:
    """Objective function that maximizes (omniscient_success - complementary_success).

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    n_seeds : int
        Number of random seeds per trial (default: 50)
    therapist_version : str
        Therapist version to use ('v1' or 'v2', default: 'v2')
    n_workers : int
        Number of parallel workers (default: CPU count)

    Returns
    -------
    float
        Advantage: omniscient_success_rate - complementary_success_rate
    """
    # Fixed constraints
    mechanism = 'frequency_amplifier'
    enable_parataxic = True
    pattern = 'cold_stuck'
    entropy = 1.0

    # Sample all searchable parameters
    baseline_accuracy = suggest_parameter(trial, 'baseline_accuracy')
    threshold = suggest_parameter(trial, 'threshold')
    bond_alpha = suggest_parameter(trial, 'bond_alpha')
    bond_offset = suggest_parameter(trial, 'bond_offset')
    recency_weighting_factor = suggest_parameter(trial, 'recency_weighting_factor')
    max_sessions = suggest_parameter(trial, 'max_sessions')
    perception_window = suggest_parameter(trial, 'perception_window')
    seeding_benefit_scaling = suggest_parameter(trial, 'seeding_benefit_scaling')
    skip_seeding_accuracy_threshold = suggest_parameter(trial, 'skip_seeding_accuracy_threshold')
    quick_seed_actions_threshold = suggest_parameter(trial, 'quick_seed_actions_threshold')
    abort_consecutive_failures_threshold = suggest_parameter(trial, 'abort_consecutive_failures_threshold')

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
        'history_weight': 1.0,  # Not used by frequency_amplifier, but required by signature
        'bond_power': 1.0,      # Not used by frequency_amplifier, but required by signature
        'bond_alpha': bond_alpha,
        'bond_offset': bond_offset,
        'recency_weighting_factor': recency_weighting_factor,
        'seeding_benefit_scaling': seeding_benefit_scaling,
        'skip_seeding_accuracy_threshold': skip_seeding_accuracy_threshold,
        'quick_seed_actions_threshold': quick_seed_actions_threshold,
        'abort_consecutive_failures_threshold': abort_consecutive_failures_threshold,
    }

    # Run BOTH therapists with same configuration (parallelized)
    if n_workers is None:
        n_workers = mp.cpu_count()

    ctx = mp.get_context('spawn')

    # Prepare all tasks (omniscient + complementary for each seed)
    tasks = []
    for seed in range(n_seeds):
        tasks.append((seed, 'omniscient', therapist_version, sim_kwargs))
        tasks.append((seed, 'complementary', therapist_version, sim_kwargs))

    # Execute in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(run_seed_therapist_pair, task): i
                   for i, task in enumerate(tasks)}

        for future in as_completed(futures):
            try:
                result = future.result(timeout=600)  # 10 min timeout per sim
                all_results.append(result)
            except Exception as e:
                print(f"Simulation failed with error: {e}")
                raise

    # Split results back into omniscient and complementary
    omniscient_results = [r for r in all_results if r.therapist_type == 'omniscient']
    complementary_results = [r for r in all_results if r.therapist_type == 'complementary']

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
        print(f"  Mechanism: frequency_amplifier (fixed)")
        print(f"  Baseline Acc: {config['parameters']['baseline_accuracy']:.2f}, "
              f"Entropy: 1.000 (fixed)")
        print(f"  Threshold: {config['parameters']['threshold']:.2f}, "
              f"Bond Offset: {config['parameters']['bond_offset']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize frequency_amplifier with OmniscientStrategicTherapist V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--study-name', '-s',
        type=str,
        default='freq_amp_v2_optimization',
        help='Name of the optimization study'
    )

    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=100,
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

    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
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
    print(f"FREQUENCY AMPLIFIER V2 OPTIMIZATION")
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
            lambda trial: objective(trial, n_seeds=args.n_seeds,
                                   therapist_version=args.therapist_version,
                                   n_workers=args.n_workers),
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
