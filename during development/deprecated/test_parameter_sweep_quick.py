"""Quick parameter sweep test with reduced parameter space.

Tests a smaller subset to verify functionality before running full sweep.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main sweep module
from test_parameter_sweep import run_parameter_sweep, print_results_summary, save_results


def main():
    """Run quick parameter sweep with reduced ranges."""

    print("=" * 100)
    print("QUICK PARAMETER SWEEP (Reduced parameter space for testing)")
    print("=" * 100)

    # Reduced parameter ranges for quick testing
    mechanisms = [
        'conditional_amplifier',
        'bond_weighted_conditional_amplifier',
    ]

    entropies = [1.0, 2.0]  # Medium, high

    history_weights = [1.0, 2.0]  # Normal, strong

    smoothing_alphas = [0.1]  # Normal only

    bond_powers = [1.0, 2.0]  # Linear, quadratic

    bond_alphas = [4.0]  # Default only

    memory_sizes = [50]  # Normal only

    success_thresholds = [0.5]  # Normal only

    initial_memory_patterns = [
        'cw_50_50',  # The problematic C→W pattern
        'complementary_perfect',  # Ideal starting point
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
    print(f"Trials per configuration: 30 (reduced for quick test)")
    print(f"Total simulations: {total * 30}")
    print()

    # Run sweep with fewer trials
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
        n_trials=30,  # Reduced for quick test
        random_seed=42,
    )

    # Print results
    print_results_summary(results, top_n=10)

    # Save to file
    save_results(results, 'parameter_sweep_quick_results.json')

    print("\n" + "=" * 100)
    print("QUICK PARAMETER SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
