"""Test bond-weighted amplifier mechanisms with C→W memory pattern."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.agents.client_agents import create_client
from src.config import sample_u_matrix, OCTANTS


def generate_cw_memory(n_interactions=50):
    """
    Generate memory of pure anticomplementary pattern: C → W.

    Client consistently exhibits Cold behavior (octant 6).
    Therapist consistently responds with Warm behavior (octant 2).

    This represents a therapist who reliably responds with warmth
    to a cold/withdrawn client.
    """
    return [(6, 2)] * n_interactions  # C → W


def test_bond_weighted_amplifiers():
    """
    Compare standard amplifiers vs bond-weighted variants.

    Test scenario: 50 C→W interactions
    - Client stuck in Cold behavior (octant 6)
    - Therapist consistently Warm (octant 2)

    Expected behavior:
    - Standard amplifiers: Full history influence regardless of bond
    - Bond-weighted (power=1.0): History scales linearly with bond
    - Bond-weighted (power=2.0): History scales quadratically with bond

    With expected low bond (~0.3), bond_weighted should show:
    - Much reduced history influence
    - Closer to bond_only baseline expectations
    """

    # Setup
    rng = np.random.RandomState(42)
    u_matrix = sample_u_matrix(random_state=42)
    memory = generate_cw_memory(50)

    print("=" * 80)
    print("BOND-WEIGHTED AMPLIFIER MECHANISM TEST")
    print("=" * 80)
    print(f"Memory: 50 interactions of C→W (anticomplementary)")
    print(f"Client octant: C (6), Therapist octant: W (2)")
    print()

    # Configuration for each test
    test_configs = [
        # Conditional Amplifier variants
        {
            'name': 'Conditional Amplifier (baseline)',
            'mechanism': 'conditional_amplifier',
            'kwargs': {'history_weight': 1.0},
        },
        {
            'name': 'Bond-Weighted Conditional (power=1.0)',
            'mechanism': 'bond_weighted_conditional_amplifier',
            'kwargs': {'history_weight': 1.0, 'bond_power': 1.0},
        },
        {
            'name': 'Bond-Weighted Conditional (power=2.0)',
            'mechanism': 'bond_weighted_conditional_amplifier',
            'kwargs': {'history_weight': 1.0, 'bond_power': 2.0},
        },
        # Frequency Amplifier variants
        {
            'name': 'Frequency Amplifier (baseline)',
            'mechanism': 'frequency_amplifier',
            'kwargs': {'history_weight': 1.0},
        },
        {
            'name': 'Bond-Weighted Frequency (power=1.0)',
            'mechanism': 'bond_weighted_frequency_amplifier',
            'kwargs': {'history_weight': 1.0, 'bond_power': 1.0},
        },
        {
            'name': 'Bond-Weighted Frequency (power=2.0)',
            'mechanism': 'bond_weighted_frequency_amplifier',
            'kwargs': {'history_weight': 1.0, 'bond_power': 2.0},
        },
    ]

    results = {}

    for config in test_configs:
        print(f"\n{'=' * 80}")
        print(f"{config['name']}")
        print(f"{'=' * 80}")

        # Create client
        client = create_client(
            mechanism=config['mechanism'],
            u_matrix=u_matrix,
            entropy=1.0,
            initial_memory=memory,
            random_state=42,
            **config['kwargs']
        )

        # Display client state
        print(f"Bond: {client.bond:.4f}")
        print(f"Relationship Satisfaction: {client.relationship_satisfaction:.2f}")
        print(f"History Weight (base): {client.history_weight:.2f}")

        # Show effective weight for bond-weighted variants
        if hasattr(client, '_get_effective_history_weight'):
            effective_weight = client._get_effective_history_weight()
            print(f"Effective History Weight: {effective_weight:.4f}", end="")
            if hasattr(client, 'bond_power'):
                reduction_pct = (1 - effective_weight / client.history_weight) * 100
                print(f" (bond^{client.bond_power:.1f} × {client.history_weight:.2f} = {reduction_pct:.1f}% reduction)")
            else:
                print()

        # Get expected payoffs
        expected = client._calculate_expected_payoffs()

        print(f"\nExpected payoffs:")
        for i in range(8):
            print(f"  {OCTANTS[i]:3s} ({i}): {expected[i]:8.2f}")

        # Get action probabilities
        probs = client._softmax(expected)
        print(f"\nAction probabilities (entropy={client.entropy}):")
        for i in range(8):
            bar = '█' * int(probs[i] * 100)
            print(f"  {OCTANTS[i]:3s} ({i}): {probs[i]:6.3f} ({probs[i]*100:5.1f}%) {bar}")

        # Most likely action
        most_likely = np.argmax(probs)
        print(f"\nMost likely action: {OCTANTS[int(most_likely)]} ({most_likely}), prob={probs[most_likely]:.3f}")

        # Store results
        results[config['name']] = {
            'expected_payoffs': expected,
            'probabilities': probs,
            'bond': client.bond,
            'rs': client.relationship_satisfaction,
            'effective_weight': effective_weight if hasattr(client, '_get_effective_history_weight') else client.history_weight,
        }

    # Comparison Summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    # Compare conditional variants
    print("\nCONDITIONAL AMPLIFIER VARIANTS:")
    print("-" * 80)
    baseline_name = 'Conditional Amplifier (baseline)'
    baseline_expected = results[baseline_name]['expected_payoffs']

    for name in ['Conditional Amplifier (baseline)',
                 'Bond-Weighted Conditional (power=1.0)',
                 'Bond-Weighted Conditional (power=2.0)']:
        print(f"\n{name}:")
        print(f"  Effective weight: {results[name]['effective_weight']:.4f}")
        print(f"  Bond: {results[name]['bond']:.4f}")
        most_likely = int(np.argmax(results[name]['probabilities']))
        prob = results[name]['probabilities'][most_likely]
        print(f"  Most likely action: {OCTANTS[most_likely]} ({most_likely}), prob={prob:.3f}")

        if name != baseline_name:
            diffs = results[name]['expected_payoffs'] - baseline_expected
            print(f"  Payoff differences from baseline:")
            for i in range(8):
                if abs(diffs[i]) > 0.01:  # Only show non-trivial differences
                    sign = '+' if diffs[i] >= 0 else ''
                    print(f"    {OCTANTS[i]:3s}: {sign}{diffs[i]:7.2f}")

    # Compare frequency variants
    print("\n\nFREQUENCY AMPLIFIER VARIANTS:")
    print("-" * 80)
    baseline_name = 'Frequency Amplifier (baseline)'
    baseline_expected = results[baseline_name]['expected_payoffs']

    for name in ['Frequency Amplifier (baseline)',
                 'Bond-Weighted Frequency (power=1.0)',
                 'Bond-Weighted Frequency (power=2.0)']:
        print(f"\n{name}:")
        print(f"  Effective weight: {results[name]['effective_weight']:.4f}")
        print(f"  Bond: {results[name]['bond']:.4f}")
        most_likely = int(np.argmax(results[name]['probabilities']))
        prob = results[name]['probabilities'][most_likely]
        print(f"  Most likely action: {OCTANTS[most_likely]} ({most_likely}), prob={prob:.3f}")

        if name != baseline_name:
            diffs = results[name]['expected_payoffs'] - baseline_expected
            print(f"  Payoff differences from baseline:")
            for i in range(8):
                if abs(diffs[i]) > 0.01:  # Only show non-trivial differences
                    sign = '+' if diffs[i] >= 0 else ''
                    print(f"    {OCTANTS[i]:3s}: {sign}{diffs[i]:7.2f}")

    # Key insights
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS")
    print(f"{'=' * 80}")

    bond = results['Conditional Amplifier (baseline)']['bond']

    print(f"\nWith bond = {bond:.4f}:")
    print(f"  - power=1.0: effective_weight = {bond:.4f} × 1.0 = {bond:.4f}")
    print(f"  - power=2.0: effective_weight = {bond:.4f}² × 1.0 = {bond**2:.4f}")
    print()
    print("Interpretation:")
    print("  - Low bond → history has reduced influence on expectations")
    print("  - Higher power → more aggressive reduction at low bond")
    print("  - Bond-weighted variants should show less deviation from raw utilities")
    print("  - This prevents over-reliance on history when trust is low")

    return results


if __name__ == "__main__":
    results = test_bond_weighted_amplifiers()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
