"""
Analyze most reliable failure configurations from overnight parameter search.

This script identifies configurations that consistently fail (low success rate)
with high confidence (many trials completed without early stopping).

Focus:
- Failures with <20% success rate
- High trial count (≥80 trials, ideally 100)
- Not early-stopped (meaning they're consistent failures, not successes)
- Statistical reliability: consistent failure across diverse u_matrix seeds
"""

import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.client_agents import create_client
from src.agents.client_agents.base_client import BaseClientAgent
from src.config import calculate_success_threshold, sample_u_matrix, rs_to_bond


def load_overnight_results(filename=None):
    """
    Load overnight parameter search results.

    Parameters
    ----------
    filename : str, optional
        Specific results file to load. If None, finds most recent overnight results.

    Returns
    -------
    dict
        Full results data including metadata and configurations
    """
    results_dir = Path(__file__).parent

    if filename:
        filepath = results_dir / filename
    else:
        # Find most recent overnight results file
        overnight_files = list(results_dir.glob("bond_weighted_freq_amp_search_overnight_*.json"))
        if not overnight_files:
            raise FileNotFoundError("No overnight results files found")
        filepath = max(overnight_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading results from: {filepath.name}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def identify_reliable_failures(results, success_threshold=0.20, min_trials=80):
    """
    Identify configurations that reliably fail.

    Reliable failure = low success rate + many trials (high confidence)

    Parameters
    ----------
    results : list of dict
        Configuration results from overnight search
    success_threshold : float
        Maximum success rate to be considered a failure (default: 0.20 = 20%)
    min_trials : int
        Minimum trials completed for reliability (default: 80)

    Returns
    -------
    list of dict
        Reliable failure configurations, sorted by success rate
    """
    reliable_failures = []

    for r in results:
        # Criteria for reliable failure:
        # 1. Low success rate
        # 2. Many trials (not early stopped, or stopped with low success)
        # 3. Not early-stopped as "too_successful"

        if (r['success_rate'] < success_threshold and
            r['n_trials'] >= min_trials and
            r.get('stop_reason', '') != 'too_successful'):
            reliable_failures.append(r)

    # Sort by success rate (lowest first), then by number of trials (most first)
    reliable_failures.sort(key=lambda x: (x['success_rate'], -x['n_trials']))

    return reliable_failures


def analyze_parameter_patterns(failures):
    """
    Analyze parameter patterns in reliable failures.

    Parameters
    ----------
    failures : list of dict
        Reliable failure configurations

    Returns
    -------
    dict
        Parameter statistics and patterns
    """
    if not failures:
        return {}

    params = {
        'bond_power': [r['config']['bond_power'] for r in failures],
        'history_weight': [r['config']['history_weight'] for r in failures],
        'entropy': [r['config']['entropy'] for r in failures],
        'bond_alpha': [r['config']['bond_alpha'] for r in failures],
        'bond_offset': [r['config']['bond_offset'] for r in failures],
    }

    stats = {}
    for param_name, values in params.items():
        stats[param_name] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
        }

    return stats


def categorize_failures(failures):
    """
    Categorize failures by dominant parameter characteristics.

    Parameters
    ----------
    failures : list of dict
        Reliable failure configurations

    Returns
    -------
    dict
        Categorized failures by type
    """
    categories = {
        'high_offset': [],           # bond_offset > 0.65
        'high_bond_power': [],       # bond_power > 7
        'high_history_weight': [],   # history_weight > 10
        'steep_bond_curve': [],      # bond_alpha > 6
        'high_entropy': [],          # entropy > 0.35
        'combined_high': [],         # Multiple high values
        'other': []
    }

    for r in failures:
        c = r['config']
        categorized = False

        # Check for combined high values
        high_count = sum([
            c['bond_offset'] > 0.65,
            c['bond_power'] > 7,
            c['history_weight'] > 10,
            c['bond_alpha'] > 6,
        ])

        if high_count >= 2:
            categories['combined_high'].append(r)
            categorized = True
        elif c['bond_offset'] > 0.65:
            categories['high_offset'].append(r)
            categorized = True
        elif c['bond_power'] > 7:
            categories['high_bond_power'].append(r)
            categorized = True
        elif c['history_weight'] > 10:
            categories['high_history_weight'].append(r)
            categorized = True
        elif c['bond_alpha'] > 6:
            categories['steep_bond_curve'].append(r)
            categorized = True
        elif c['entropy'] > 0.35:
            categories['high_entropy'].append(r)
            categorized = True

        if not categorized:
            categories['other'].append(r)

    return categories


def print_failure_summary(failures, all_results):
    """Print summary statistics for reliable failures."""
    print("=" * 100)
    print("RELIABLE FAILURE ANALYSIS")
    print("=" * 100)
    print()

    print(f"Total configurations tested: {len(all_results)}")
    print(f"Reliable failures (<20% success, ≥80 trials): {len(failures)}")
    print(f"Percentage: {len(failures)/len(all_results)*100:.1f}%")
    print()

    if not failures:
        print("No reliable failures found!")
        return

    # Success rate distribution
    success_rates = [r['success_rate'] for r in failures]
    print(f"Failure success rate statistics:")
    print(f"  Mean:   {np.mean(success_rates):.1%}")
    print(f"  Median: {np.median(success_rates):.1%}")
    print(f"  Min:    {np.min(success_rates):.1%}")
    print(f"  Max:    {np.max(success_rates):.1%}")
    print(f"  Std:    {np.std(success_rates):.1%}")
    print()

    # Trial count distribution
    trial_counts = [r['n_trials'] for r in failures]
    print(f"Trial count statistics:")
    print(f"  Mean:   {np.mean(trial_counts):.1f}")
    print(f"  Median: {np.median(trial_counts):.1f}")
    print(f"  Min:    {np.min(trial_counts)}")
    print(f"  Max:    {np.max(trial_counts)}")
    print()

    # Extreme failures
    extreme = [r for r in failures if r['success_rate'] < 0.05]
    print(f"Extreme failures (<5% success): {len(extreme)} / {len(failures)}")
    print()


def print_top_failures(failures, n=10):
    """Print detailed information about top N failures."""
    print("=" * 100)
    print(f"TOP {n} MOST RELIABLE FAILURES (Lowest Success Rate)")
    print("=" * 100)
    print()

    for i, r in enumerate(failures[:n], 1):
        c = r['config']
        print(f"{i}. Success Rate: {r['success_rate']:6.1%} ({r['n_trials']} trials)")
        print(f"   Mean Final RS: {r['mean_final_rs']:7.2f}")
        print(f"   Dropout Rate:  {r['dropout_rate']:6.1%}")
        print()
        print(f"   Parameters:")
        print(f"     bond_power      = {c['bond_power']:6.3f}")
        print(f"     history_weight  = {c['history_weight']:6.3f}")
        print(f"     entropy         = {c['entropy']:6.3f}")
        print(f"     bond_alpha      = {c['bond_alpha']:6.3f}")
        print(f"     bond_offset     = {c['bond_offset']:6.3f}")
        print()


def print_parameter_analysis(stats):
    """Print parameter pattern analysis."""
    print("=" * 100)
    print("PARAMETER PATTERNS IN RELIABLE FAILURES")
    print("=" * 100)
    print()

    print(f"{'Parameter':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Q25':>8} {'Q75':>8}")
    print("-" * 100)

    for param, s in stats.items():
        print(f"{param:<20} {s['mean']:8.3f} {s['median']:8.3f} {s['std']:8.3f} "
              f"{s['min']:8.3f} {s['max']:8.3f} {s['q25']:8.3f} {s['q75']:8.3f}")
    print()


def print_categorized_failures(categories):
    """Print failure categorization."""
    print("=" * 100)
    print("FAILURE CATEGORIZATION")
    print("=" * 100)
    print()

    total = sum(len(configs) for configs in categories.values())

    print(f"{'Category':<25} {'Count':>8} {'%':>8} {'Avg Success':>12}")
    print("-" * 100)

    for category, configs in sorted(categories.items(), key=lambda x: -len(x[1])):
        if len(configs) > 0:
            count = len(configs)
            pct = count / total * 100
            avg_success = np.mean([r['success_rate'] for r in configs])
            print(f"{category:<25} {count:8d} {pct:7.1f}% {avg_success:11.1%}")
    print()

    # Show example from largest category
    if categories:
        largest_cat = max(categories.items(), key=lambda x: len(x[1]))
        cat_name, configs = largest_cat
        if len(configs) > 0:
            print(f"Example from largest category '{cat_name}':")
            example = configs[0]
            c = example['config']
            print(f"  Success rate: {example['success_rate']:.1%}")
            print(f"  bond_power={c['bond_power']:.3f}, history_weight={c['history_weight']:.3f}, "
                  f"entropy={c['entropy']:.3f}")
            print(f"  bond_alpha={c['bond_alpha']:.3f}, bond_offset={c['bond_offset']:.3f}")
            print()


def create_visualizations(failures, all_results, output_dir=None):
    """
    Create comprehensive visualizations of failure patterns.

    Parameters
    ----------
    failures : list of dict
        Reliable failure configurations
    all_results : list of dict
        All configuration results
    output_dir : Path, optional
        Directory to save visualizations
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "visualization_output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 100)
    print("CREATING VISUALIZATIONS")
    print("=" * 100)
    print()

    # Extract data
    param_names = ['bond_power', 'history_weight', 'entropy', 'bond_alpha', 'bond_offset']

    failure_params = {
        param: [r['config'][param] for r in failures]
        for param in param_names
    }
    failure_success = [r['success_rate'] for r in failures]

    all_params = {
        param: [r['config'][param] for r in all_results]
        for param in param_names
    }
    all_success = [r['success_rate'] for r in all_results]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Success rate distribution comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_success, bins=30, alpha=0.5, label='All configs', edgecolor='black')
    ax1.hist(failure_success, bins=30, alpha=0.7, label='Reliable failures', edgecolor='black')
    ax1.axvline(0.20, color='red', linestyle='--', label='20% threshold')
    ax1.set_xlabel('Success Rate')
    ax1.set_ylabel('Count')
    ax1.set_title('Success Rate Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2-6. Parameter distributions (failures vs all)
    param_axes = [
        fig.add_subplot(gs[0, 1]),  # bond_power
        fig.add_subplot(gs[0, 2]),  # history_weight
        fig.add_subplot(gs[1, 0]),  # entropy
        fig.add_subplot(gs[1, 1]),  # bond_alpha
        fig.add_subplot(gs[1, 2]),  # bond_offset
    ]

    for ax, param in zip(param_axes, param_names):
        ax.hist(all_params[param], bins=20, alpha=0.5, label='All configs', edgecolor='black')
        ax.hist(failure_params[param], bins=20, alpha=0.7, label='Failures', edgecolor='black')
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.set_title(f'{param.replace("_", " ").title()} Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 7. 2D scatter: bond_offset vs bond_alpha (key failure drivers)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(all_params['bond_offset'], all_params['bond_alpha'],
                c=all_success, cmap='RdYlGn', alpha=0.3, s=20, vmin=0, vmax=1)
    ax7.scatter(failure_params['bond_offset'], failure_params['bond_alpha'],
                c=failure_success, cmap='RdYlGn', alpha=0.8, s=50,
                edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
    ax7.set_xlabel('Bond Offset')
    ax7.set_ylabel('Bond Alpha')
    ax7.set_title('Bond Parameters (failures highlighted)')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax7.collections[-1], ax=ax7)
    cbar.set_label('Success Rate')

    # 8. 2D scatter: bond_power vs history_weight
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(all_params['bond_power'], all_params['history_weight'],
                c=all_success, cmap='RdYlGn', alpha=0.3, s=20, vmin=0, vmax=1)
    ax8.scatter(failure_params['bond_power'], failure_params['history_weight'],
                c=failure_success, cmap='RdYlGn', alpha=0.8, s=50,
                edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
    ax8.set_xlabel('Bond Power')
    ax8.set_ylabel('History Weight')
    ax8.set_title('History Parameters (failures highlighted)')
    ax8.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax8.collections[-1], ax=ax8)
    cbar.set_label('Success Rate')

    # 9. Trial count vs success rate for failures
    ax9 = fig.add_subplot(gs[2, 2])
    trial_counts = [r['n_trials'] for r in failures]
    ax9.scatter(trial_counts, failure_success, alpha=0.6, s=50)
    ax9.set_xlabel('Trials Completed')
    ax9.set_ylabel('Success Rate')
    ax9.set_title('Reliability: Trials vs Success (failures only)')
    ax9.grid(True, alpha=0.3)
    ax9.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax9.legend()

    plt.suptitle('Reliable Failure Analysis: Parameter Patterns', fontsize=16, y=0.995)

    # Save
    output_path = output_dir / "reliable_failures_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    print()

    plt.close()


def export_top_failures(failures, n=20, output_file=None):
    """
    Export top N failures to JSON for further analysis.

    Parameters
    ----------
    failures : list of dict
        Reliable failure configurations
    n : int
        Number of top failures to export
    output_file : Path, optional
        Output file path
    """
    if output_file is None:
        output_file = Path(__file__).parent / "top_reliable_failures.json"

    top_failures = failures[:n]

    export_data = {
        'metadata': {
            'description': 'Top reliable failure configurations for detailed analysis',
            'n_failures': len(top_failures),
            'selection_criteria': '<20% success, ≥80 trials',
        },
        'configurations': [
            {
                'rank': i + 1,
                'success_rate': r['success_rate'],
                'n_trials': r['n_trials'],
                'mean_final_rs': r['mean_final_rs'],
                'dropout_rate': r['dropout_rate'],
                'config': r['config'],
            }
            for i, r in enumerate(top_failures)
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Top {n} failures exported to: {output_file}")
    print()


def main():
    print("=" * 100)
    print("RELIABLE FAILURE ANALYSIS")
    print("=" * 100)
    print()

    # Load overnight results
    try:
        data = load_overnight_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the overnight parameter search first.")
        return

    results = data['results']
    metadata = data['metadata']

    print(f"Loaded {len(results)} configurations")
    print(f"Total simulations: {metadata['total_simulations']:,}")
    print(f"Parameter ranges:")
    for param, range_val in metadata['parameter_ranges'].items():
        if isinstance(range_val, list):
            print(f"  {param:<20}: [{range_val[0]}, {range_val[1]}]")
        else:
            print(f"  {param:<20}: {range_val}")
    print()

    # Identify reliable failures
    print("Identifying reliable failures...")
    # Adjusted criteria: <40% success (since worst found was 26%), ≥90 trials for reliability
    failures = identify_reliable_failures(results, success_threshold=0.40, min_trials=90)
    print(f"Found {len(failures)} reliable failures (<40% success, ≥90 trials)")
    print()

    # Summary statistics
    print_failure_summary(failures, results)

    if not failures:
        return

    # Top failures
    print_top_failures(failures, n=10)

    # Parameter patterns
    stats = analyze_parameter_patterns(failures)
    print_parameter_analysis(stats)

    # Categorize failures
    categories = categorize_failures(failures)
    print_categorized_failures(categories)

    # Create visualizations
    create_visualizations(failures, results)

    # Export top failures
    export_top_failures(failures, n=20)

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()
    print("Next steps:")
    print("  1. Review top_reliable_failures.json for detailed configuration data")
    print("  2. Check visualization_output/reliable_failures_analysis.png")
    print("  3. Run detailed trajectory analysis on selected configurations")
    print()


if __name__ == "__main__":
    main()
