"""
Analyze parameter search results for bond_weighted_frequency_amplifier.

Focus on identifying configurations with <50% success rate and understanding
what parameter combinations lead to failure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(filename="bond_weighted_freq_amp_search_results.json"):
    """Load results from JSON file."""
    filepath = Path(__file__).parent / filename
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def analyze_overall_statistics(results):
    """Print overall statistics."""
    print("=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)
    print()

    success_rates = [r['success_rate'] for r in results]

    print(f"Total configurations: {len(results)}")
    print(f"Mean success rate:    {np.mean(success_rates):.1%}")
    print(f"Median success rate:  {np.median(success_rates):.1%}")
    print(f"Std deviation:        {np.std(success_rates):.1%}")
    print(f"Min success rate:     {np.min(success_rates):.1%}")
    print(f"Max success rate:     {np.max(success_rates):.1%}")
    print()

    # Distribution by success bins
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    bin_labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]

    print("Success rate distribution:")
    for i in range(len(bins)-1):
        count = sum(1 for sr in success_rates if bins[i] <= sr < bins[i+1])
        pct = count / len(results) * 100
        print(f"  {bin_labels[i]:<10}: {count:3d} configs ({pct:5.1f}%)")
    print()


def analyze_failures(results, threshold=0.5):
    """Analyze configurations with success rate < threshold."""
    failures = [r for r in results if r['success_rate'] < threshold]

    print("=" * 100)
    print(f"FAILURE ANALYSIS (success rate < {threshold:.0%})")
    print("=" * 100)
    print()

    print(f"Number of failure configurations: {len(failures)} / {len(results)} ({len(failures)/len(results)*100:.1f}%)")
    print()

    if len(failures) == 0:
        print("No failures found!")
        return

    # Sort by success rate
    failures_sorted = sorted(failures, key=lambda x: x['success_rate'])

    print("All failure configurations:")
    print("-" * 100)
    print(f"{'#':<4} {'Success':<8} {'bond_p':<8} {'hist_w':<8} {'entropy':<8} {'b_alpha':<8} {'b_offset':<8} {'mean_RS':<8}")
    print("-" * 100)

    for i, r in enumerate(failures_sorted):
        c = r['config']
        print(f"{i+1:<4} {r['success_rate']:<8.1%} "
              f"{c['bond_power']:<8.2f} {c['history_weight']:<8.2f} {c['entropy']:<8.2f} "
              f"{c['bond_alpha']:<8.2f} {c['bond_offset']:<8.3f} {r['mean_final_rs']:<8.1f}")
    print()

    # Parameter statistics for failures
    print("Parameter statistics for failures:")
    print("-" * 100)

    param_names = ['bond_power', 'history_weight', 'entropy', 'bond_alpha', 'bond_offset']

    for param in param_names:
        values = [r['config'][param] for r in failures]
        print(f"  {param:<18}: mean={np.mean(values):6.2f}, "
              f"median={np.median(values):6.2f}, "
              f"min={np.min(values):6.2f}, "
              f"max={np.max(values):6.2f}")
    print()

    # Compare to successes
    successes = [r for r in results if r['success_rate'] >= 0.75]

    print("Parameter comparison: Failures vs High-Success (≥75%):")
    print("-" * 100)
    print(f"{'Parameter':<18} {'Failures (mean)':<18} {'High-Success (mean)':<18} {'Difference':<12}")
    print("-" * 100)

    for param in param_names:
        fail_mean = np.mean([r['config'][param] for r in failures])
        succ_mean = np.mean([r['config'][param] for r in successes])
        diff = fail_mean - succ_mean
        print(f"{param:<18} {fail_mean:<18.2f} {succ_mean:<18.2f} {diff:+12.2f}")
    print()


def parameter_correlations(results):
    """Calculate correlations between parameters and success rate."""
    print("=" * 100)
    print("PARAMETER CORRELATIONS WITH SUCCESS RATE")
    print("=" * 100)
    print()

    success_rates = np.array([r['success_rate'] for r in results])
    param_names = ['bond_power', 'history_weight', 'entropy', 'bond_alpha', 'bond_offset']

    print("Pearson correlation (r):")
    print("-" * 100)
    print(f"{'Parameter':<18} {'Correlation (r)':<18} {'--':<12} {'Interpretation'}")
    print("-" * 100)

    correlations = {}

    for param in param_names:
        values = np.array([r['config'][param] for r in results])
        # Calculate correlation coefficient using numpy
        corr_matrix = np.corrcoef(values, success_rates)
        r = corr_matrix[0, 1]
        correlations[param] = r

        # Interpretation
        if abs(r) < 0.1:
            interp = "negligible"
        elif abs(r) < 0.3:
            interp = "weak"
        elif abs(r) < 0.5:
            interp = "moderate"
        else:
            interp = "strong"

        direction = "positive" if r > 0 else "negative"

        print(f"{param:<18} {r:<+18.3f} {'N/A':<12} {interp} {direction}")
    print()

    # Most important parameters
    sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Parameters ranked by correlation strength:")
    for param, r in sorted_params:
        print(f"  {param:<18}: r={r:+.3f}")
    print()

    return correlations


def identify_failure_modes(results, threshold=0.5):
    """Identify distinct failure modes."""
    failures = [r for r in results if r['success_rate'] < threshold]

    if len(failures) == 0:
        print("No failures to analyze.")
        return

    print("=" * 100)
    print("FAILURE MODE IDENTIFICATION")
    print("=" * 100)
    print()

    # Categorize by dominant parameter extremes
    print("Categorizing failures by extreme parameter values:")
    print("-" * 100)
    print()

    # Get parameter distributions across all configs for reference
    all_params = {
        param: [r['config'][param] for r in results]
        for param in ['bond_power', 'history_weight', 'entropy', 'bond_alpha', 'bond_offset']
    }

    # Define "extreme" as top/bottom 20% of range
    categories = {
        'high_bond_offset': [],
        'low_bond_offset': [],
        'high_bond_power': [],
        'low_entropy': [],
        'high_entropy': [],
        'high_history_weight': [],
        'other': []
    }

    for r in failures:
        c = r['config']
        categorized = False

        # High bond_offset (top 20%)
        if c['bond_offset'] > np.percentile(all_params['bond_offset'], 80):
            categories['high_bond_offset'].append(r)
            categorized = True

        # Low bond_offset (bottom 20%)
        elif c['bond_offset'] < np.percentile(all_params['bond_offset'], 20):
            categories['low_bond_offset'].append(r)
            categorized = True

        # High bond_power (top 20%)
        elif c['bond_power'] > np.percentile(all_params['bond_power'], 80):
            categories['high_bond_power'].append(r)
            categorized = True

        # Low entropy (bottom 20%)
        elif c['entropy'] < np.percentile(all_params['entropy'], 20):
            categories['low_entropy'].append(r)
            categorized = True

        # High entropy (top 20%)
        elif c['entropy'] > np.percentile(all_params['entropy'], 80):
            categories['high_entropy'].append(r)
            categorized = True

        # High history_weight (top 20%)
        elif c['history_weight'] > np.percentile(all_params['history_weight'], 80):
            categories['high_history_weight'].append(r)
            categorized = True

        if not categorized:
            categories['other'].append(r)

    for category, configs in categories.items():
        if len(configs) > 0:
            mean_sr = np.mean([r['success_rate'] for r in configs])
            print(f"  {category:<25}: {len(configs):2d} configs (mean success: {mean_sr:.1%})")

            # Show parameter means for this category
            for param in ['bond_power', 'history_weight', 'entropy', 'bond_alpha', 'bond_offset']:
                mean_val = np.mean([r['config'][param] for r in configs])
                print(f"    {param:<18}: {mean_val:.2f}")
            print()


def create_visualizations(results):
    """Create visualization plots."""
    print("=" * 100)
    print("CREATING VISUALIZATIONS")
    print("=" * 100)
    print()

    # Extract data
    success_rates = [r['success_rate'] for r in results]
    params = {
        'bond_power': [r['config']['bond_power'] for r in results],
        'history_weight': [r['config']['history_weight'] for r in results],
        'entropy': [r['config']['entropy'] for r in results],
        'bond_alpha': [r['config']['bond_alpha'] for r in results],
        'bond_offset': [r['config']['bond_offset'] for r in results],
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Parameter Search Results: bond_weighted_frequency_amplifier', fontsize=16)

    # 1. Success rate histogram
    ax = axes[0, 0]
    ax.hist(success_rates, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', label='50% threshold')
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Success Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2-6. Parameter vs success rate scatter plots
    param_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    param_names = ['bond_power', 'history_weight', 'entropy', 'bond_alpha', 'bond_offset']

    for ax, param in zip(param_axes, param_names):
        # Color by success/failure
        colors = ['red' if sr < 0.5 else 'green' if sr >= 0.75 else 'orange'
                  for sr in success_rates]

        ax.scatter(params[param], success_rates, c=colors, alpha=0.6, s=30)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Success vs {param.replace("_", " ").title()}')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0.75, color='green', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Failure (<50%)'),
        Patch(facecolor='orange', label='Moderate (50-75%)'),
        Patch(facecolor='green', label='Success (≥75%)')
    ]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "visualization_output" / "parameter_search_analysis.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    print()

    plt.close()


def main():
    # Load results
    print("Loading results...")
    data = load_results()
    results = data['results']
    print(f"Loaded {len(results)} configurations")
    print()

    # Overall statistics
    analyze_overall_statistics(results)

    # Failure analysis (main focus)
    analyze_failures(results, threshold=0.5)

    # Parameter correlations
    correlations = parameter_correlations(results)

    # Identify failure modes
    identify_failure_modes(results, threshold=0.5)

    # Create visualizations
    create_visualizations(results)

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
