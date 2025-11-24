"""
Test bond_weighted_frequency_amplifier with extreme parameters.

Configuration:
- bond_power = 3 (history influence scales with bond^3)
- history_weight = 20 (high base weight)
- entropy = 0.2 (some exploration)
- bond_alpha = 3 (gentler bond curve)
- initial memory: all C→W (cold_warm pattern)
- therapist: always complements
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.client_agents import create_client
from src.agents.client_agents.base_client import BaseClientAgent
from src.config import calculate_success_threshold, sample_u_matrix, BOND_ALPHA

OCTANT_NAMES = {
    0: 'D  (Dominant)',
    1: 'WD (Warm-Dominant)',
    2: 'W  (Warm)',
    3: 'WS (Warm-Submissive)',
    4: 'S  (Submissive)',
    5: 'CS (Cold-Submissive)',
    6: 'C  (Cold)',
    7: 'CD (Cold-Dominant)',
}


def always_complement(client_action: int) -> int:
    """
    Simple complementary strategy:
    - Dominant ↔ Submissive (0↔4, 1↔3, 7↔5)
    - Warm ↔ Warm (2↔2)
    - Cold ↔ Cold (6↔6)
    """
    complement_map = {
        0: 4,  # D → S
        1: 3,  # WD → WS
        2: 2,  # W → W
        3: 1,  # WS → WD
        4: 0,  # S → D
        5: 7,  # CS → CD
        6: 6,  # C → C
        7: 5,  # CD → CS
    }
    return complement_map[client_action]


def main():
    print("=" * 100)
    print("BOND-WEIGHTED FREQUENCY AMPLIFIER: EXTREME PARAMETERS TEST")
    print("=" * 100)
    print()

    # Configuration
    config = {
        'mechanism': 'bond_weighted_frequency_amplifier',
        'bond_power': 3,
        'history_weight': 20,
        'entropy': 0.2,
        'random_state': 42,
        'bond_alpha': 3,
        'success_threshold': 0.8,
        'max_sessions': 100,
    }

    print("CONFIGURATION:")
    print("-" * 100)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Generate initial memory: all C→W (cold_warm pattern)
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type='cold_warm',
        n_interactions=50,
        random_state=42
    )

    print("INITIAL MEMORY:")
    print("-" * 100)
    print(f"  Pattern: cold_warm")
    print(f"  All 50 interactions: Client=C (6), Therapist=W (2)")
    print(f"  Interaction: (6, 2) × 50")
    print()

    # Sample client utility matrix
    rng = np.random.RandomState(42)
    u_matrix = sample_u_matrix(random_state=rng)

    print("CLIENT UTILITY MATRIX:")
    print("-" * 100)
    print(f"{'Client Action':<25} {'Mean U':>10} {'Max U':>10} {'Min U':>10}")
    print("-" * 100)
    for i in range(8):
        mean_u = u_matrix[i, :].mean()
        max_u = u_matrix[i, :].max()
        min_u = u_matrix[i, :].min()
        print(f"{OCTANT_NAMES[i]:<25} {mean_u:10.2f} {max_u:10.2f} {min_u:10.2f}")
    print()

    best_action = np.argmax([u_matrix[i, :].mean() for i in range(8)])
    print(f"Best action (by mean utility): {OCTANT_NAMES[best_action]}")
    print()

    # Create client
    client = create_client(
        mechanism=config['mechanism'],
        initial_memory=initial_memory,
        u_matrix=u_matrix,
        entropy=config['entropy'],
        history_weight=config['history_weight'],
        bond_power=config['bond_power'],
        random_state=rng,
    )

    # Override bond_alpha if needed
    if hasattr(client, 'bond_alpha'):
        client.bond_alpha = config['bond_alpha']

    # Calculate success threshold
    rs_threshold = calculate_success_threshold(u_matrix, config['success_threshold'])

    print("INITIAL STATE:")
    print("-" * 100)
    print(f"  Initial RS: {client.relationship_satisfaction:.4f}")
    print(f"  Initial bond: {client.bond:.6f}")
    print(f"  RS threshold (80th percentile): {rs_threshold:.4f}")
    print(f"  RS range: [{u_matrix.min():.2f}, {u_matrix.max():.2f}]")
    print(f"  Gap to threshold: {rs_threshold - client.relationship_satisfaction:.4f} points")
    print()

    # Check if client has expected_payoffs calculation we can inspect
    print("=" * 100)
    print("SESSION-BY-SESSION TRACKING")
    print("=" * 100)
    print()

    session_data = []

    for session in range(1, config['max_sessions'] + 1):
        # Capture pre-action state
        pre_rs = client.relationship_satisfaction
        pre_bond = client.bond

        # Get expected payoffs (before action selection)
        expected_payoffs = client._calculate_expected_payoffs()

        # Calculate probabilities
        # For entropy near 0, this will be nearly deterministic
        scaled_payoffs = expected_payoffs / config['entropy']
        scaled_payoffs = scaled_payoffs - np.max(scaled_payoffs)
        exp_payoffs = np.exp(scaled_payoffs)
        probabilities = exp_payoffs / np.sum(exp_payoffs)

        # Select action
        client_action = client.select_action()

        # Get therapist response
        therapist_action = always_complement(client_action)

        # Get utility
        utility = u_matrix[client_action, therapist_action]

        # Update client
        client.update_memory(client_action, therapist_action)

        # Capture post-action state
        post_rs = client.relationship_satisfaction
        post_bond = client.bond

        # Store session data
        session_data.append({
            'session': session,
            'pre_rs': pre_rs,
            'post_rs': post_rs,
            'pre_bond': pre_bond,
            'post_bond': post_bond,
            'client_action': client_action,
            'therapist_action': therapist_action,
            'utility': utility,
            'expected_payoffs': expected_payoffs,
            'probabilities': probabilities,
        })

        # Verbose output for select sessions
        if session <= 10 or session % 10 == 0 or session >= config['max_sessions'] - 5:
            print(f"SESSION {session}")
            print("-" * 100)
            print(f"  Pre-action:  RS={pre_rs:8.4f}, Bond={pre_bond:.6f}")
            print(f"  Action:      Client={OCTANT_NAMES[client_action]}, Therapist={OCTANT_NAMES[therapist_action]}")
            print(f"  Utility:     {utility:.4f}")
            print(f"  Post-action: RS={post_rs:8.4f}, Bond={post_bond:.6f}")
            print(f"  Changes:     ΔRS={post_rs-pre_rs:+.4f}, ΔBond={post_bond-pre_bond:+.6f}")
            print()

            # Show expected payoffs and probabilities
            print(f"  Expected Payoffs:")
            for i in range(8):
                print(f"    {OCTANT_NAMES[i]:<25} Payoff={expected_payoffs[i]:8.4f}, Prob={probabilities[i]:.6f}")
            print()

            # Show which action had max probability
            max_prob_action = np.argmax(probabilities)
            print(f"  → Selected: {OCTANT_NAMES[client_action]} (max prob: {OCTANT_NAMES[max_prob_action]})")
            print()

        # Check dropout
        if client.check_dropout():
            print(f"\n⚠️  CLIENT DROPPED OUT at session {session}")
            break

    # Final outcome
    print("\n" + "=" * 100)
    print("FINAL OUTCOME")
    print("=" * 100)
    print()

    final_rs = client.relationship_satisfaction
    final_bond = client.bond
    success = final_rs >= rs_threshold

    print(f"Sessions completed: {len(session_data)}")
    print(f"Final RS: {final_rs:.4f}")
    print(f"Final bond: {final_bond:.6f}")
    print(f"RS threshold: {rs_threshold:.4f}")
    print(f"Gap to threshold: {final_rs - rs_threshold:+.4f} points")
    print(f"Success: {'✓ YES' if success else '✗ NO'}")
    print()

    # Action distribution
    from collections import Counter
    actions = [s['client_action'] for s in session_data]
    action_counts = Counter(actions)

    print("ACTION DISTRIBUTION:")
    print("-" * 100)
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = count / len(actions) * 100
        bar = '█' * int(pct / 2)
        print(f"  {OCTANT_NAMES[action]:<25} {count:4d} ({pct:5.1f}%) {bar}")
    print()

    # RS trajectory visualization
    print("RS TRAJECTORY:")
    print("-" * 100)
    print(f"{'Session':>8} {'RS':>10} {'Bond':>10} {'Visual (relative to threshold)'}")
    print("-" * 100)

    for i in range(0, len(session_data), 5):
        s = session_data[i]
        rs = s['post_rs']
        bond = s['post_bond']

        # Visual relative to threshold
        relative_pos = rs - rs_threshold
        if relative_pos >= 0:
            bar = ' ' * 40 + '|' + '█' * min(40, int(relative_pos / 2))
            marker = '✓'
        else:
            bar_len = min(40, int(-relative_pos / 2))
            bar = ' ' * (40 - bar_len) + '░' * bar_len + '|'
            marker = ' '

        print(f"{s['session']:8d} {rs:10.4f} {bond:10.6f} {bar} {marker}")

    # Final
    final_s = session_data[-1]
    rs = final_s['post_rs']
    bond = final_s['post_bond']
    relative_pos = rs - rs_threshold
    if relative_pos >= 0:
        bar = ' ' * 40 + '|' + '█' * min(40, int(relative_pos / 2))
        marker = '✓'
    else:
        bar_len = min(40, int(-relative_pos / 2))
        bar = ' ' * (40 - bar_len) + '░' * bar_len + '|'
        marker = ' '
    print(f"{final_s['session']:8d} {rs:10.4f} {bond:10.6f} {bar} {marker}")

    print()
    print(f"Legend: | = threshold ({rs_threshold:.4f}), ░ = below, █ = above")
    print()

    # Key insights
    print("=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print()

    print("Bond Power Effect (bond^3):")
    print(f"  At bond=0.10: history weight effectively = 20 × 0.10^3 = {20 * 0.10**3:.6f}")
    print(f"  At bond=0.50: history weight effectively = 20 × 0.50^3 = {20 * 0.50**3:.6f}")
    print(f"  At bond=0.90: history weight effectively = 20 × 0.90^3 = {20 * 0.90**3:.6f}")
    print()
    print("→ Early in therapy (low bond=0.1), history has minimal influence (×0.02)")
    print("→ Mid therapy (bond=0.5), history has moderate influence (×2.5)")
    print("→ Late in therapy (high bond=0.9), history has strong influence (×14.6)")
    print()

    # Check when bond crossed thresholds
    bond_crossed_50 = None
    bond_crossed_90 = None
    for s in session_data:
        if bond_crossed_50 is None and s['post_bond'] >= 0.5:
            bond_crossed_50 = s['session']
        if bond_crossed_90 is None and s['post_bond'] >= 0.9:
            bond_crossed_90 = s['session']

    if bond_crossed_50:
        print(f"Bond crossed 0.5 at session {bond_crossed_50}")
    if bond_crossed_90:
        print(f"Bond crossed 0.9 at session {bond_crossed_90}")
    print()

    print("Entropy Effect (0.2):")
    print("  With entropy = 0.2, action selection has some stochasticity")
    print("  Client usually picks action with highest expected payoff")
    print("  But allows for some exploration of suboptimal actions")
    print()

    print("Bond-Weighted Frequency Amplifier Mechanism:")
    print("  Tracks marginal P(therapist_j) ignoring client action")
    print("  History influence scales with bond: effective_weight = bond^3 × 20")
    print("  Low bond → minimal history influence (data-driven)")
    print("  High bond → strong history influence (expectation-driven)")
    print()


if __name__ == "__main__":
    main()
