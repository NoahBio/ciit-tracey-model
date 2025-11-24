"""Targeted parameter sweep focusing on observed failure patterns.

Based on analysis showing:
- conflictual failures: mainly frequency_amplifier + low entropy + high threshold
- mixed_random failures: similar pattern
- cw_50_50 failures: rare but present
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main sweep module
from test_parameter_sweep import run_parameter_sweep, print_results_summary, save_results


def main():
    """Run targeted parameter sweep on failure-prone combinations."""

    print("=" * 100)
    print("TARGETED PARAMETER SWEEP (Failure-prone combinations)")
    print("=" * 100)

    # Based on observed failures, test combinations most likely to fail
    mechanisms = [
        'frequency_amplifier',  # Most failures
        'conditional_amplifier',  # Some failures
        'bond_weighted_conditional_amplifier',  # Some failures
    ]

    # Low entropy had most failures (41.4%)
    entropies = [0.3, 1.0]

    # High history_weight showed in failures
    history_weights = [5.0]

    # Varied smoothing
    smoothing_alphas = [0.01, 0.1, 1.0]

    # All bond powers
    bond_powers = [0.5, 1.0, 3.0]

    # Bond alphas that appeared in failures
    bond_alphas = [2.0, 5.0, 10.0]

    memory_sizes = [50]

    # High threshold had most failures (69.9%)
    success_thresholds = [0.8]

    # ONLY non-complementary_perfect patterns
    initial_memory_patterns = [
        'cw_50_50',
        'conflictual',
        'mixed_random',
    ]

    pattern_types = ['none']

    print("\nParameter ranges:")
    print(f"  Mechanisms: {mechanisms}")
    print(f"  Entropies: {entropies}")
    print(f"  History weights: {history_weights}")
    print(f"  Smoothing alphas: {smoothing_alphas}")
    print(f"  Bond powers: {bond_powers}")
    print(f"  Bond alphas: {bond_alphas}")
    print(f"  Memory sizes: {memory_sizes}")
    print(f"  Success thresholds: {success_thresholds}")
    print(f"  Initial memory patterns: {initial_memory_patterns}")
    print()

    # Calculate total combinations
    total = (len(mechanisms) * len(entropies) * len(history_weights) *
             len(smoothing_alphas) * len(bond_powers) * len(bond_alphas) *
             len(memory_sizes) * len(success_thresholds) *
             len(initial_memory_patterns) * len(pattern_types))

    print(f"Total parameter combinations: {total}")
    print(f"Trials per configuration: 50")
    print(f"Total simulations: {total * 50}")
    print(f"Estimated time: ~{total * 50 / 3600:.1f} hours (at 1 sec/sim)")
    print()

    # Run sweep
    results = run_parameter_sweep(
        mechanisms=mechanisms,
        entropies=entropies,
        history_weights=history_weights,
        smoothing_alphas=smoothing_alphas,
        bond_powers=bond_powers,
        bond_alphas=bond_alphas,
        memory_sizes=memory_sizes,
        success_thresholds=success_thresholds,
        initial_memory_patterns=initial_memory_patterns,
        pattern_types=pattern_types,
        n_trials=50,
        random_seed=42,
    )

    # Print results
    print_results_summary(results, top_n=20)

    # Save to file
    save_results(results, 'parameter_sweep_targeted_results.json')

    # Failure analysis
    print("\n" + "=" * 100)
    print("FAILURE ANALYSIS")
    print("=" * 100)

    failures = [r for r in results if r.success_rate < 1.0]
    if failures:
        print(f"\nFound {len(failures)} configurations with <100% success rate")

        from collections import Counter

        # Analyze by initial memory pattern
        print("\nFailures by initial memory pattern:")
        pattern_counts = Counter(r.initial_memory_pattern for r in failures)
        for pattern, count in pattern_counts.most_common():
            print(f"  {pattern}: {count} failures")

        # Analyze by mechanism
        print("\nFailures by mechanism:")
        mech_counts = Counter(r.mechanism for r in failures)
        for mech, count in mech_counts.most_common():
            print(f"  {mech}: {count} failures")

        # Analyze by entropy
        print("\nFailures by entropy:")
        ent_counts = Counter(r.entropy for r in failures)
        for ent, count in sorted(ent_counts.items()):
            print(f"  {ent:.1f}: {count} failures")

        print(f"\nWorst failure rate: {min(r.success_rate for r in failures)*100:.1f}%")

    else:
        print("\n✓ NO FAILURES - 100% success across all configurations!")

    print("\n" + "=" * 100)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
