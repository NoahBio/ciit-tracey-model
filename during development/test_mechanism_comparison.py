"""Compare all client agent expectation mechanisms."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.agents.client_agents import create_client, BaseClientAgent
from src.config import sample_u_matrix, OCTANTS


def generate_test_memory():
    """Generate standard test memory for comparison."""
    # Pure anticomplementary: cold → warm
    memory = []
    for _ in range(50):
        memory.append((6, 2))  # C → W
    return memory


def compare_mechanisms():
    """Run identical scenarios through all mechanisms."""

    # Setup
    rng = np.random.RandomState(42)
    u_matrix = sample_u_matrix(random_state=42)
    memory = generate_test_memory()

    mechanisms = [
        'bond_only',
        'frequency_filter',
        'frequency_amplifier',
        'conditional_filter',
        'conditional_amplifier'
    ]
    results = {}

    print("MECHANISM COMPARISON")
    print("=" * 80)
    print(f"Memory: 50 interactions of C→W (pure anticomplementarity)")
    print()

    for mech in mechanisms:
        print(f"\nMechanism: {mech}")
        print("-" * 80)

        # Create client
        if mech == 'frequency_amplifier':
            client = create_client(
                mechanism=mech,
                u_matrix=u_matrix,
                entropy=1.0,
                initial_memory=memory,
                history_weight=1.0,
                random_state=42
            )
        else:
            client = create_client(
                mechanism=mech,
                u_matrix=u_matrix,
                entropy=1.0,
                initial_memory=memory,
                random_state=42
            )

        # Get expected payoffs for all actions
        expected = client._calculate_expected_payoffs()

        print(f"Bond: {client.bond:.3f}")
        print(f"RS: {client.relationship_satisfaction:.2f}")
        print(f"Expected payoffs:")
        for i in range(8):
            print(f"  {OCTANTS[i]:3s} ({i}): {expected[i]:7.2f}")

        # Get action probabilities
        probs = client._softmax(expected)
        print(f"\nAction probabilities (with entropy={client.entropy}):")
        for i in range(8):
            print(f"  {OCTANTS[i]:3s} ({i}): {probs[i]:6.3f} ({probs[i]*100:5.1f}%)")

        # Most likely action
        most_likely = np.argmax(probs)
        print(f"\nMost likely action: {OCTANTS[int(most_likely)]} (prob={probs[most_likely]:.3f})")

        results[mech] = {
            'expected_payoffs': expected,
            'probabilities': probs,
            'bond': client.bond,
            'rs': client.relationship_satisfaction,
        }

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Show differences in expected payoffs
    print("\nExpected Payoff Differences (relative to bond_only):")
    baseline = results['bond_only']['expected_payoffs']
    for mech in ['frequency_filter', 'frequency_amplifier', 'conditional_filter', 'conditional_amplifier']:
        print(f"\n{mech}:")
        diffs = results[mech]['expected_payoffs'] - baseline
        for i in range(8):
            sign = '+' if diffs[i] >= 0 else ''
            print(f"  {OCTANTS[i]:3s}: {sign}{diffs[i]:6.2f}")

    # Show which action is most likely for each mechanism
    print("\nMost Likely Actions:")
    for mech in mechanisms:
        most_likely = int(np.argmax(results[mech]['probabilities']))
        prob = results[mech]['probabilities'][most_likely]
        print(f"  {mech:20s}: {OCTANTS[most_likely]:3s} (prob={prob:.3f})")

    return results


def compare_with_varied_memories():
    """Compare mechanisms across different memory patterns."""

    print("\n" + "=" * 80)
    print("COMPARISON ACROSS DIFFERENT MEMORY PATTERNS")
    print("=" * 80)

    # Different memory patterns to test
    patterns = {
        'pure_complementary': [(0, 4)] * 50,  # D → S (always complementary)
        'pure_anticomplementary': [(6, 2)] * 50,  # C → W (always anticomplementary)
        'mixed': [(i % 8, (i+4) % 8) for i in range(50)],  # Mixed pattern
    }

    u_matrix = sample_u_matrix(random_state=42)
    mechanisms = [
        'bond_only',
        'frequency_filter',
        'frequency_amplifier',
        'conditional_filter',
        'conditional_amplifier'
    ]

    for pattern_name, memory in patterns.items():
        print(f"\n{'='*80}")
        print(f"Pattern: {pattern_name}")
        print(f"{'='*80}")

        for mech in mechanisms:
            client = create_client(
                mechanism=mech,
                u_matrix=u_matrix,
                entropy=1.0,
                initial_memory=memory,
                random_state=42
            )

            expected = client._calculate_expected_payoffs()
            probs = client._softmax(expected)
            most_likely = np.argmax(probs)

            print(f"\n{mech}:")
            print(f"  Bond: {client.bond:.3f}, RS: {client.relationship_satisfaction:.2f}")
            print(f"  Most likely action: {OCTANTS[int(most_likely)]} ({most_likely}) = {probs[most_likely]:.3f}")
            print(f"  Expected payoff for that action: {expected[most_likely]:.2f}")


def detailed_mechanism_explanation():
    """Show detailed step-by-step calculation for each mechanism."""

    print("\n" + "=" * 80)
    print("DETAILED MECHANISM CALCULATIONS")
    print("=" * 80)

    u_matrix = sample_u_matrix(random_state=42)
    memory = generate_test_memory()  # C → W pattern

    # Calculate therapist frequencies
    therapist_counts = np.zeros(8)
    for _, therapist_oct in memory:
        therapist_counts[therapist_oct] += 1
    therapist_freq = therapist_counts / len(memory)

    print("\nTherapist Frequency Distribution:")
    for i in range(8):
        if therapist_freq[i] > 0:
            print(f"  {OCTANTS[i]:3s}: {therapist_freq[i]:.3f} ({int(therapist_counts[i])} occurrences)")

    # Pick one client action to analyze
    client_action = 6  # C (cold)
    print(f"\n{'='*80}")
    print(f"Analyzing Expected Payoff for Client Action: {OCTANTS[client_action]} ({client_action})")
    print(f"{'='*80}")

    raw_utilities = u_matrix[client_action, :]
    print(f"\nRaw utilities (u_matrix row {client_action}):")
    for j in range(8):
        print(f"  vs {OCTANTS[j]:3s}: {raw_utilities[j]:7.2f}")

    # Create clients with same state
    bond_only_client = create_client('bond_only', u_matrix=u_matrix, entropy=1.0, initial_memory=memory, random_state=42)
    filter_client = create_client('frequency_filter', u_matrix=u_matrix, entropy=1.0, initial_memory=memory, random_state=42)
    amplifier_client = create_client('frequency_amplifier', u_matrix=u_matrix, entropy=1.0, initial_memory=memory, random_state=42)

    bond = bond_only_client.bond
    print(f"\nBond level: {bond:.4f}")

    # Show bond_only calculation
    print(f"\n{'BOND_ONLY Mechanism':-^80}")
    print("Steps: Sort raw utilities → Select percentile based on bond")
    sorted_raw = np.sort(raw_utilities)
    print(f"Sorted utilities: {sorted_raw}")
    position = bond * 7
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, 7)
    interp_weight = position - lower_idx
    expected_bond_only = (1 - interp_weight) * sorted_raw[lower_idx] + interp_weight * sorted_raw[upper_idx]
    print(f"Position: {position:.4f} → indices [{lower_idx}, {upper_idx}], weight={interp_weight:.4f}")
    print(f"Expected payoff: {expected_bond_only:.4f}")

    # Show frequency_filter calculation
    print(f"\n{'FREQUENCY_FILTER Mechanism':-^80}")
    print("Steps: Multiply utilities by frequencies → Sort → Select percentile")
    adjusted_filter = raw_utilities * therapist_freq
    print(f"Adjusted utilities (raw * freq):")
    for j in range(8):
        print(f"  vs {OCTANTS[j]:3s}: {raw_utilities[j]:7.2f} * {therapist_freq[j]:.3f} = {adjusted_filter[j]:7.4f}")
    sorted_filter = np.sort(adjusted_filter)
    print(f"Sorted adjusted: {sorted_filter}")
    expected_filter = (1 - interp_weight) * sorted_filter[lower_idx] + interp_weight * sorted_filter[upper_idx]
    print(f"Expected payoff: {expected_filter:.4f}")

    # Show frequency_amplifier calculation
    print(f"\n{'FREQUENCY_AMPLIFIER Mechanism':-^80}")
    print("Steps: Add frequency-weighted boost → Sort → Select percentile")
    history_weight = 1.0
    adjusted_amplifier = raw_utilities + (raw_utilities * therapist_freq * history_weight)
    print(f"Adjusted utilities (raw + raw * freq * {history_weight}):")
    for j in range(8):
        boost = raw_utilities[j] * therapist_freq[j] * history_weight
        print(f"  vs {OCTANTS[j]:3s}: {raw_utilities[j]:7.2f} + {boost:7.4f} = {adjusted_amplifier[j]:7.4f}")
    sorted_amplifier = np.sort(adjusted_amplifier)
    print(f"Sorted adjusted: {sorted_amplifier}")
    expected_amplifier = (1 - interp_weight) * sorted_amplifier[lower_idx] + interp_weight * sorted_amplifier[upper_idx]
    print(f"Expected payoff: {expected_amplifier:.4f}")

    # Summary
    print(f"\n{'SUMMARY':-^80}")
    print(f"Expected payoffs for action {OCTANTS[client_action]} ({client_action}):")
    print(f"  Bond Only:           {expected_bond_only:7.2f}")
    print(f"  Frequency Filter:    {expected_filter:7.2f} (diff: {expected_filter - expected_bond_only:+7.2f})")
    print(f"  Frequency Amplifier: {expected_amplifier:7.2f} (diff: {expected_amplifier - expected_bond_only:+7.2f})")


if __name__ == "__main__":
    # Run basic comparison
    results = compare_mechanisms()

    # Compare across different patterns
    compare_with_varied_memories()

    # Show detailed calculations
    detailed_mechanism_explanation()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
