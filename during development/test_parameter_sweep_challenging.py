"""Parameter sweep focused on finding failure modes of always-complementary strategy.

Tests more challenging conditions to identify where the strategy breaks down.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main sweep module
from test_parameter_sweep import run_parameter_sweep, print_results_summary, save_results


def main():
    """Run challenging parameter sweep to find failure modes."""

    print("=" * 100)
    print("CHALLENGING PARAMETER SWEEP (Finding failure modes)")
    print("=" * 100)

    # More challenging parameter ranges
    mechanisms = [
        'bond_only',  # No history influence
        'conditional_amplifier',
        'bond_weighted_conditional_amplifier',
        'frequency_amplifier',
    ]

    entropies = [0.3, 1.0, 3.0]  # Very low (deterministic), normal, very high (random)

    history_weights = [0.1, 1.0, 5.0]  # Very weak, normal, very strong

    smoothing_alphas = [0.01, 0.1, 1.0]  # Data-driven, normal, uniform prior

    bond_powers = [0.5, 1.0, 3.0]  # Weak bond effect, linear, strong bond effect

    bond_alphas = [2.0, 5.0, 10.0]  # Gentle, normal (default is 5), very steep

    memory_sizes = [25, 50, 100]  # Short, normal, long

    success_thresholds = [0.4, 0.6, 0.8]  # Lenient, moderate, strict

    initial_memory_patterns = [
        'cw_50_50',  # The problematic C→W pattern
        'complementary_perfect',  # Ideal starting point
        'conflictual',  # Worst starting point
        'mixed_random',  # Unpredictable pattern
    ]

    pattern_types = ['none']  # Placeholder

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
    print(f"Trials per configuration: 30")
    print(f"Total simulations: {total * 30}")
    print()

    if total > 1000:
        print("WARNING: This will take a very long time!")
        response = input(f"Continue with {total} configurations? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Consider using a smaller parameter space.")
            return

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
        n_trials=30,
        random_seed=42,
    )

    # Print results
    print_results_summary(results, top_n=20)

    # Save to file
    save_results(results, 'parameter_sweep_challenging_results.json')

    # Additional analysis: Find patterns in failures
    print("\n" + "=" * 100)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 100)

    failures = [r for r in results if r.success_rate < 0.5]
    if failures:
        print(f"\nFound {len(failures)} configurations with <50% success rate")

        # Analyze by initial memory pattern
        print("\nFailures by initial memory pattern:")
        from collections import Counter
        pattern_counts = Counter(r.initial_memory_pattern for r in failures)
        for pattern, count in pattern_counts.most_common():
            print(f"  {pattern}: {count} failures")

        # Analyze by mechanism
        print("\nFailures by mechanism:")
        mech_counts = Counter(r.mechanism for r in failures)
        for mech, count in mech_counts.most_common():
            print(f"  {mech}: {count} failures")

        # Analyze by success threshold
        print("\nFailures by success threshold:")
        thresh_counts = Counter(r.success_threshold for r in failures)
        for thresh, count in thresh_counts.most_common():
            print(f"  {thresh:.1f}: {count} failures")

    else:
        print("\nNo configurations with <50% success rate found!")
        print("The always-complementary strategy is very robust!")

    print("\n" + "=" * 100)
    print("CHALLENGING PARAMETER SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
