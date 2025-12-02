"""Utility functions for Optuna hyperparameter optimization.

This module provides helper functions for parameter validation, result extraction,
and analysis of Optuna studies.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Dict, List, Any, Optional
import optuna
import pandas as pd


def validate_param_compatibility(
    mechanism: str,
    optimize_params: List[str]
) -> List[str]:
    """Validate that optimize_params are compatible with the chosen mechanism.

    Parameters
    ----------
    mechanism : str
        Client mechanism name
    optimize_params : List[str]
        List of parameters to optimize

    Returns
    -------
    List[str]
        List of warnings (empty if all params are valid)
    """
    warnings = []

    # Parameters only relevant for amplifier mechanisms
    amplifier_params = {'history_weight'}
    # Parameters only relevant for bond_weighted mechanisms
    bond_weighted_params = {'bond_power'}

    for param in optimize_params:
        if param in amplifier_params and 'amplifier' not in mechanism:
            warnings.append(
                f"Parameter '{param}' is only used by amplifier mechanisms, "
                f"but mechanism is '{mechanism}'"
            )

        if param in bond_weighted_params and 'bond_weighted' not in mechanism:
            warnings.append(
                f"Parameter '{param}' is only used by bond_weighted mechanisms, "
                f"but mechanism is '{mechanism}'"
            )

    return warnings


def compare_studies(study_paths: List[Path]) -> pd.DataFrame:
    """Compare results across multiple Optuna studies.

    Parameters
    ----------
    study_paths : List[Path]
        List of paths to study database files

    Returns
    -------
    pd.DataFrame
        Comparison table with best trials from each study
    """
    results = []

    for study_path in study_paths:
        storage_url = f"sqlite:///{study_path}"
        study_summaries = optuna.study.get_all_study_summaries(storage_url)

        for summary in study_summaries:
            study = optuna.load_study(
                study_name=summary.study_name,
                storage=storage_url
            )

            if len(study.directions) == 1:
                # Single-objective
                best_trial = study.best_trial
                result = {
                    'study_name': study.study_name,
                    'n_trials': len(study.trials),
                    'best_value': best_trial.value,
                    **best_trial.params
                }
            else:
                # Multi-objective: report first Pareto solution
                best_trials = study.best_trials
                if best_trials:
                    result = {
                        'study_name': study.study_name,
                        'n_trials': len(study.trials),
                        'n_pareto_solutions': len(best_trials),
                        'best_values': best_trials[0].values,
                        **best_trials[0].params
                    }
                else:
                    continue

            results.append(result)

    return pd.DataFrame(results)


def analyze_parameter_effects(
    study: optuna.Study,
    param_name: str,
    objective_idx: int = 0
) -> pd.DataFrame:
    """Analyze the effect of a parameter on objectives.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    param_name : str
        Parameter to analyze
    objective_idx : int
        Index of objective to analyze (for multi-objective)

    Returns
    -------
    pd.DataFrame
        Analysis of parameter effects
    """
    trials_df = study.trials_dataframe()

    # Filter to completed trials
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']

    if param_name not in trials_df.columns:
        raise ValueError(f"Parameter '{param_name}' not found in study")

    # Get parameter values and objectives
    param_col = f'params_{param_name}'
    if len(study.directions) == 1:
        value_col = 'value'
    else:
        value_col = f'values_{objective_idx}'

    if param_col not in trials_df.columns:
        param_col = param_name  # Try without prefix

    analysis = trials_df[[param_col, value_col]].copy()
    analysis.columns = ['parameter_value', 'objective_value']

    # Sort by parameter value
    analysis = analysis.sort_values('parameter_value')

    return analysis


def get_best_params_for_mechanism(
    study: optuna.Study,
    mechanism: str
) -> Optional[Dict[str, Any]]:
    """Get best parameters for a specific mechanism from a study.

    Useful when optimizing over mechanisms to see best config per mechanism.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    mechanism : str
        Mechanism name to filter by

    Returns
    -------
    Optional[Dict[str, Any]]
        Best parameters for this mechanism, or None if not found
    """
    # Filter trials by mechanism
    mechanism_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.params.get('mechanism') == mechanism
        and t.value is not None
    ]

    if not mechanism_trials:
        return None

    # Find best trial (for single-objective only)
    if len(study.directions) != 1:
        raise ValueError("This function only works for single-objective studies")

    # Cumbersome notation due to pylance errors
    if study.directions[0] == optuna.study.StudyDirection.MAXIMIZE:
        best_trial = max(mechanism_trials, key=lambda t: t.value if t.value is not None else float('-inf'))
    else:
        best_trial = min(mechanism_trials, key=lambda t: t.value if t.value is not None else float('inf'))

    return {
        'value': best_trial.value,
        'params': best_trial.params,
        'trial_number': best_trial.number
    }


def export_trial_history(study: optuna.Study, output_path: Path) -> None:
    """Export complete trial history to CSV.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    output_path : Path
        Path to save CSV file
    """
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_path, index=False)
    print(f"Trial history saved to: {output_path}")


def print_study_summary(study: optuna.Study) -> None:
    """Print a comprehensive summary of a study.

    Parameters
    ----------
    study : optuna.Study
        Study to summarize
    """
    print("=" * 80)
    print(f"Study: {study.study_name}")
    print("=" * 80)

    # Basic info
    print(f"\nTotal trials: {len(study.trials)}")
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Complete trials: {len(complete_trials)}")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Pruned trials: {len(pruned_trials)}")

    # Objectives
    print(f"\nObjectives: {len(study.directions)}")
    for i, direction in enumerate(study.directions):
        print(f"  Objective {i}: {direction.name}")

    # Best results
    if len(study.directions) == 1:
        print(f"\nBest value: {study.best_trial.value:.4f}")
        print(f"Best trial: {study.best_trial.number}")
        print("\nBest parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nPareto optimal solutions: {len(study.best_trials)}")
        print("\nTop 3 solutions:")
        for i, trial in enumerate(study.best_trials[:3]):
            print(f"\n  Solution {i+1} (Trial {trial.number}):")
            print(f"    Values: {trial.values}")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

    print("\n" + "=" * 80)


def load_and_analyze_study(
    study_name: str,
    storage_path: Path
) -> Dict[str, Any]:
    """Load a study and return comprehensive analysis.

    Parameters
    ----------
    study_name : str
        Name of the study
    storage_path : Path
        Path to SQLite database

    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results
    """
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    analysis = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'directions': [d.name for d in study.directions],
    }

    if len(study.directions) == 1:
        analysis['best_value'] = study.best_trial.value
        analysis['best_params'] = study.best_trial.params
        analysis['best_trial_number'] = study.best_trial.number
    else:
        analysis['n_pareto_solutions'] = len(study.best_trials)
        analysis['pareto_front'] = [
            {
                'trial_number': t.number,
                'values': t.values,
                'params': t.params
            }
            for t in study.best_trials
        ]

    return analysis


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Optuna studies")
    parser.add_argument('--study-name', required=True, help='Name of study to analyze')
    parser.add_argument('--storage-path', required=True, type=Path, help='Path to .db file')
    parser.add_argument('--export-csv', type=Path, help='Export trial history to CSV')

    args = parser.parse_args()

    # Load and analyze
    analysis = load_and_analyze_study(args.study_name, args.storage_path)

    # Print summary
    storage_url = f"sqlite:///{args.storage_path}"
    study = optuna.load_study(study_name=args.study_name, storage=storage_url)
    print_study_summary(study)

    # Export if requested
    if args.export_csv:
        export_trial_history(study, args.export_csv)

    # Save analysis to JSON
    output_path = args.storage_path.parent / f"{args.study_name}_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {output_path}")
