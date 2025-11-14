"""
Comprehensive debugging of FrequencyFilterClient mechanism.
Shows all under-the-hood calculations step-by-step.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.agents.client_agents import create_client, FrequencyFilterClient
from src.config import sample_u_matrix, MEMORY_SIZE, get_memory_weights, OCTANTS


def analyze_memory_contents(memory):
    """Analyze and display memory contents."""
    print("\n" + "="*80)
    print("MEMORY CONTENTS ANALYSIS")
    print("="*80)

    print(f"\nTotal interactions in memory: {len(memory)}")

    # Count client actions
    client_counts = {}
    for client_oct, _ in memory:
        client_counts[client_oct] = client_counts.get(client_oct, 0) + 1

    print("\nClient action distribution:")
    for i in range(8):
        count = client_counts.get(i, 0)
        pct = (count / len(memory)) * 100 if memory else 0
        if count > 0:
            print(f"  {OCTANTS[i]:3s} ({i}): {count:2d} / {len(memory)} ({pct:5.1f}%)")

    # Count therapist actions
    therapist_counts = {}
    for _, therapist_oct in memory:
        therapist_counts[therapist_oct] = therapist_counts.get(therapist_oct, 0) + 1

    print("\nTherapist action distribution:")
    for i in range(8):
        count = therapist_counts.get(i, 0)
        pct = (count / len(memory)) * 100 if memory else 0
        if count > 0:
            print(f"  {OCTANTS[i]:3s} ({i}): {count:2d} / {len(memory)} ({pct:5.1f}%)")

    # Show interaction pairs
    pair_counts = {}
    for client_oct, therapist_oct in memory:
        pair = (client_oct, therapist_oct)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    print("\nInteraction pair distribution:")
    for (c, t), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(memory)) * 100
        print(f"  {OCTANTS[c]:3s} → {OCTANTS[t]:3s}: {count:2d} / {len(memory)} ({pct:5.1f}%)")

    # Show memory with recency weights
    weights = get_memory_weights(len(memory))
    print("\nMemory with recency weights (last 10 interactions):")
    print(f"  {'Idx':>3s}  {'Client':>6s}  {'Therapist':>9s}  {'Weight':>8s}")
    for i in range(max(0, len(memory) - 10), len(memory)):
        client_oct, therapist_oct = memory[i]
        print(f"  {i:3d}  {OCTANTS[client_oct]:>6s}  {OCTANTS[therapist_oct]:>9s}  {weights[i]:8.4f}")

    return client_counts, therapist_counts, pair_counts


def calculate_marginal_frequencies_debug(client):
    """Calculate marginal frequencies with detailed debug output."""
    print("\n" + "="*80)
    print("MARGINAL FREQUENCY CALCULATION")
    print("="*80)

    memory_weights = get_memory_weights(len(client.memory))
    weighted_counts = np.zeros(8)

    print("\nWeighted counting process:")
    print(f"  {'Idx':>3s}  {'Client':>6s}  {'Therapist':>9s}  {'Weight':>8s}  {'Contribution':>12s}")

    # Show contribution of each interaction
    for idx, (client_oct, therapist_oct) in enumerate(client.memory):
        weight = memory_weights[idx]
        weighted_counts[therapist_oct] += weight
        if idx >= len(client.memory) - 10:  # Show last 10
            print(f"  {idx:3d}  {OCTANTS[client_oct]:>6s}  {OCTANTS[therapist_oct]:>9s}  {weight:8.4f}  → count[{OCTANTS[therapist_oct]}]")

    total_weight = sum(memory_weights)

    print(f"\nWeighted counts (before normalization, total_weight={total_weight:.4f}):")
    for i in range(8):
        if weighted_counts[i] > 0:
            print(f"  {OCTANTS[i]:3s} ({i}): {weighted_counts[i]:8.4f}")

    # Normalize to get frequencies
    if total_weight == 0:
        frequencies = np.ones(8) / 8
        print("\n⚠️  WARNING: Total weight is 0, using uniform distribution")
    else:
        frequencies = weighted_counts / total_weight

    print(f"\nMarginal therapist frequencies P(therapist_j):")
    for i in range(8):
        print(f"  P({OCTANTS[i]:3s}) = {frequencies[i]:.6f}", end="")
        if frequencies[i] == 0:
            print("  ← ZERO (will kill utilities!)")
        elif frequencies[i] == 1.0:
            print("  ← MAXIMUM (only this action observed)")
        else:
            print()

    # Show number of zero vs non-zero frequencies
    num_zeros = np.sum(frequencies == 0)
    num_nonzeros = np.sum(frequencies > 0)
    print(f"\nFrequency distribution summary:")
    print(f"  Non-zero frequencies: {num_nonzeros} / 8")
    print(f"  Zero frequencies: {num_zeros} / 8")

    if num_nonzeros == 1:
        print(f"\n  ⚠️  CRITICAL: Only ONE non-zero frequency!")
        print(f"      This will cause degenerate distribution in frequency_filter!")

    return frequencies


def analyze_utility_adjustment(client, client_action, frequencies):
    """Analyze how utilities are adjusted by frequencies."""
    print("\n" + "="*80)
    print(f"UTILITY ADJUSTMENT FOR CLIENT ACTION: {OCTANTS[client_action]} ({client_action})")
    print("="*80)

    raw_utilities = client.u_matrix[client_action, :]

    print(f"\nRaw utilities from u_matrix[{client_action}, :]:")
    for j in range(8):
        print(f"  U[{OCTANTS[client_action]}, {OCTANTS[j]}] = {raw_utilities[j]:8.2f}")

    print(f"\nMultiplication process: adjusted[j] = U[{client_action}, j] × P(j)")
    adjusted_utilities = raw_utilities * frequencies

    for j in range(8):
        print(f"  {OCTANTS[j]:3s}: {raw_utilities[j]:8.2f} × {frequencies[j]:.6f} = {adjusted_utilities[j]:8.4f}", end="")
        if frequencies[j] == 0:
            print("  ← ZEROED OUT")
        elif frequencies[j] == 1.0:
            print("  ← PRESERVED (freq=1)")
        else:
            print()

    # Sort and show distribution
    sorted_adjusted = np.sort(adjusted_utilities)
    print(f"\nSorted adjusted utilities:")
    print(f"  {sorted_adjusted}")

    # Analyze the distribution
    num_zeros = np.sum(adjusted_utilities == 0)
    num_negative_zeros = np.sum(adjusted_utilities == -0.0)
    num_positive = np.sum(adjusted_utilities > 0)
    num_negative = np.sum(adjusted_utilities < 0)

    print(f"\nAdjusted utility distribution:")
    print(f"  Positive values: {num_positive}")
    print(f"  Negative values: {num_negative}")
    print(f"  Zeros (including -0.0): {num_zeros + num_negative_zeros}")

    if num_zeros + num_negative_zeros >= 6:
        print(f"\n  ⚠️  CRITICAL: {num_zeros + num_negative_zeros} out of 8 values are zero!")
        print(f"      Bond-based selection will likely pick from zeros!")

    return raw_utilities, adjusted_utilities, sorted_adjusted


def analyze_bond_selection(client, sorted_adjusted):
    """Analyze bond-based percentile selection."""
    print("\n" + "="*80)
    print("BOND-BASED PERCENTILE SELECTION")
    print("="*80)

    bond = client.bond
    print(f"\nClient bond: {bond:.6f}")
    print(f"Relationship satisfaction: {client.relationship_satisfaction:.2f}")
    print(f"RS bounds: [{client.rs_min:.2f}, {client.rs_max:.2f}]")

    # Calculate position
    position = bond * 7
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, 7)
    interpolation_weight = position - lower_idx

    print(f"\nPercentile calculation:")
    print(f"  Position = bond × 7 = {bond:.6f} × 7 = {position:.6f}")
    print(f"  Lower index: {lower_idx}")
    print(f"  Upper index: {upper_idx}")
    print(f"  Interpolation weight: {interpolation_weight:.6f}")

    print(f"\nValues at selection indices:")
    print(f"  sorted_adjusted[{lower_idx}] = {sorted_adjusted[lower_idx]:.6f}")
    print(f"  sorted_adjusted[{upper_idx}] = {sorted_adjusted[upper_idx]:.6f}")

    # Calculate expected payoff
    expected_payoff = (
        (1 - interpolation_weight) * sorted_adjusted[lower_idx] +
        interpolation_weight * sorted_adjusted[upper_idx]
    )

    print(f"\nInterpolation:")
    print(f"  (1 - {interpolation_weight:.6f}) × {sorted_adjusted[lower_idx]:.6f} + {interpolation_weight:.6f} × {sorted_adjusted[upper_idx]:.6f}")
    print(f"  = {(1 - interpolation_weight) * sorted_adjusted[lower_idx]:.6f} + {interpolation_weight * sorted_adjusted[upper_idx]:.6f}")
    print(f"  = {expected_payoff:.6f}")

    if abs(expected_payoff) < 1e-10:
        print(f"\n  ⚠️  CRITICAL: Expected payoff is essentially ZERO!")
        print(f"      This means both selection indices point to zeros in the array!")

    # Show what would happen at different bond levels
    print(f"\nExpected payoff at different bond levels (for comparison):")
    for test_bond in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        pos = test_bond * 7
        low = int(pos)
        high = min(low + 1, 7)
        weight = pos - low
        exp = (1 - weight) * sorted_adjusted[low] + weight * sorted_adjusted[high]
        print(f"  Bond={test_bond:.1f}: position={pos:.2f} → [{low},{high}] → expected={exp:8.4f}")

    return expected_payoff


def full_analysis_all_actions(client, frequencies):
    """Analyze expected payoffs for all client actions."""
    print("\n" + "="*80)
    print("EXPECTED PAYOFFS FOR ALL CLIENT ACTIONS")
    print("="*80)

    expected_payoffs = np.zeros(8)

    for client_action in range(8):
        raw_utilities = client.u_matrix[client_action, :]
        adjusted = raw_utilities * frequencies
        sorted_adj = np.sort(adjusted)

        position = client.bond * 7
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, 7)
        interp_weight = position - lower_idx

        expected = (1 - interp_weight) * sorted_adj[lower_idx] + interp_weight * sorted_adj[upper_idx]
        expected_payoffs[client_action] = expected

        num_zeros = np.sum(np.abs(adjusted) < 1e-10)
        print(f"\n  {OCTANTS[client_action]:3s} ({client_action}):")
        print(f"    Raw utilities: {raw_utilities}")
        print(f"    Adjusted (×freq): {adjusted}")
        print(f"    Zeros in adjusted: {num_zeros}/8")
        print(f"    Expected payoff: {expected:8.4f}")

    print(f"\n{'='*80}")
    print("SUMMARY: Expected payoffs for all actions")
    print(f"{'='*80}")
    for i in range(8):
        print(f"  {OCTANTS[i]:3s} ({i}): {expected_payoffs[i]:8.4f}")

    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(expected_payoffs):8.4f}")
    print(f"  Std:  {np.std(expected_payoffs):8.4f}")
    print(f"  Min:  {np.min(expected_payoffs):8.4f}")
    print(f"  Max:  {np.max(expected_payoffs):8.4f}")

    if np.std(expected_payoffs) < 1e-6:
        print(f"\n  ⚠️  CRITICAL: All expected payoffs are nearly identical!")
        print(f"      This leads to UNIFORM action probabilities!")

    return expected_payoffs


def analyze_action_probabilities(client, expected_payoffs):
    """Analyze softmax probabilities from expected payoffs."""
    print("\n" + "="*80)
    print("SOFTMAX ACTION PROBABILITIES")
    print("="*80)

    print(f"\nEntropy (temperature): {client.entropy:.6f}")

    # Manual softmax calculation with debug
    scaled = expected_payoffs / client.entropy
    print(f"\nScaled payoffs (payoffs / entropy):")
    for i in range(8):
        print(f"  {OCTANTS[i]:3s}: {expected_payoffs[i]:8.4f} / {client.entropy:.4f} = {scaled[i]:8.4f}")

    # Subtract max for numerical stability
    max_scaled = np.max(scaled)
    scaled_stable = scaled - max_scaled
    print(f"\nAfter subtracting max ({max_scaled:.4f}) for stability:")
    print(f"  {scaled_stable}")

    # Exponentiate
    exp_scaled = np.exp(scaled_stable)
    print(f"\nExponentiated:")
    for i in range(8):
        print(f"  {OCTANTS[i]:3s}: exp({scaled_stable[i]:8.4f}) = {exp_scaled[i]:12.6e}")

    # Normalize
    sum_exp = np.sum(exp_scaled)
    probs = exp_scaled / sum_exp

    print(f"\nSum of exponentials: {sum_exp:.6e}")
    print(f"\nFinal probabilities (after normalization):")
    for i in range(8):
        print(f"  {OCTANTS[i]:3s} ({i}): {probs[i]:.6f} ({probs[i]*100:5.2f}%)")

    # Check if uniform
    uniform_prob = 1.0 / 8
    max_deviation = np.max(np.abs(probs - uniform_prob))

    if max_deviation < 0.01:
        print(f"\n  ⚠️  CRITICAL: Probabilities are nearly UNIFORM!")
        print(f"      Maximum deviation from 1/8: {max_deviation:.6f}")
        print(f"      This means the mechanism provides NO guidance!")

    most_likely = np.argmax(probs)
    print(f"\nMost likely action: {OCTANTS[most_likely]} ({most_likely}) with prob={probs[most_likely]:.6f}")

    return probs


def test_scenario(scenario_name, memory, u_matrix=None):
    """Test frequency_filter mechanism with a specific scenario."""
    print("\n" + "█"*80)
    print(f"█  SCENARIO: {scenario_name}")
    print("█"*80)

    if u_matrix is None:
        u_matrix = sample_u_matrix(random_state=42)

    # Create client
    client = create_client(
        mechanism='frequency_filter',
        u_matrix=u_matrix,
        entropy=1.0,
        initial_memory=memory,
        random_state=42
    )

    print(f"\nClient created with FrequencyFilterClient mechanism")
    print(f"  Entropy: {client.entropy}")
    print(f"  Bond: {client.bond:.6f}")
    print(f"  RS: {client.relationship_satisfaction:.2f}")

    # Analyze memory
    analyze_memory_contents(client.get_memory())

    # Calculate frequencies with debug
    frequencies = calculate_marginal_frequencies_debug(client)

    # Analyze utility adjustment for one action (action 6 = C)
    client_action = 6  # C (cold)
    raw, adjusted, sorted_adj = analyze_utility_adjustment(client, client_action, frequencies)

    # Analyze bond selection
    expected_single = analyze_bond_selection(client, sorted_adj)

    # Get all expected payoffs
    expected_all = full_analysis_all_actions(client, frequencies)

    # Verify against client's method
    client_expected = client._calculate_expected_payoffs()
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")
    print(f"Manual calculation matches client method: {np.allclose(expected_all, client_expected)}")
    if not np.allclose(expected_all, client_expected):
        print("  Differences:")
        for i in range(8):
            diff = expected_all[i] - client_expected[i]
            if abs(diff) > 1e-6:
                print(f"    {OCTANTS[i]}: manual={expected_all[i]:.6f}, client={client_expected[i]:.6f}, diff={diff:.6f}")

    # Analyze action probabilities
    probs = analyze_action_probabilities(client, expected_all)

    return client, frequencies, expected_all, probs


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE FREQUENCY_FILTER MECHANISM DEBUG")
    print("="*80)

    # Scenario 1: Pure consistent therapist (problematic case)
    print("\n\n")
    memory1 = [(6, 2)] * 50  # C → W (always)
    test_scenario("Pure Consistent Therapist (C→W always)", memory1)

    # Scenario 2: Mixed but biased
    print("\n\n")
    memory2 = [(6, 2)] * 40 + [(6, 1)] * 10  # Mostly C→W, some C→WD
    test_scenario("Biased but Not Pure (80% C→W, 20% C→WD)", memory2)

    # Scenario 3: Varied therapist responses
    print("\n\n")
    memory3 = [(6, i % 8) for i in range(50)]  # C → all octants
    test_scenario("Fully Varied Therapist Responses", memory3)

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
