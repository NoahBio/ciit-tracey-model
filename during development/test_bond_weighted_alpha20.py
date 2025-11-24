"""
Test bond_weighted_conditional_amplifier with STEEP bond curve (alpha=7).

# Modified from alpha=20 to alpha=7 to stay within [3,7] range for consistency

Configuration:
- bond_power = 5 (history weight scales with bond^5)
- history_weight = 10 (very high when bond is high)
- entropy = 0.0001 (completely deterministic)
- smoothing_alpha = 0.01 (very data-driven)
- bond_alpha = 7 (STEEP RS→bond mapping, was 20 before range restriction)
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
from src.config import calculate_success_threshold, sample_u_matrix, rs_to_bond

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
    """Simple complementary strategy."""
    complement_map = {
        0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5
    }
    return complement_map[client_action]


def main():
    print("=" * 100)
    print("BOND-WEIGHTED WITH STEEP BOND CURVE (alpha=7)")
    print("=" * 100)
    print()

    # Configuration
    config = {
        'mechanism': 'bond_weighted_conditional_amplifier',
        'bond_power': 5,
        'history_weight': 10,
        'entropy': 0.0001,
        'smoothing_alpha': 0.01,
        'random_state': 42,
        'bond_alpha': 7,  # STEEP curve! (was 20, clamped to [3,7] range)
        'success_threshold': 0.8,
        'max_sessions': 100,
    }

    print("CONFIGURATION:")
    print("-" * 100)
    for key, value in config.items():
        if key == 'bond_alpha':
            print(f"  {key}: {value} ← STEEP! (default is 5, was 20 before range restriction)")
        else:
            print(f"  {key}: {value}")
    print()

    # Generate initial memory
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type='cold_warm',
        n_interactions=50,
        random_state=42
    )

    print("INITIAL MEMORY:")
    print("-" * 100)
    print(f"  Pattern: cold_warm")
    print(f"  All 50 interactions: Client=C (6), Therapist=W (2)")
    print()

    # Sample client utility matrix
    rng = np.random.RandomState(42)
    u_matrix = sample_u_matrix(random_state=rng)

    print("CLIENT UTILITY MATRIX:")
    print("-" * 100)
    print(f"{'Client Action':<25} {'Mean U':>10} {'U[i,C]':>10} {'U[i,W]':>10}")
    print("-" * 100)
    for i in range(8):
        mean_u = u_matrix[i, :].mean()
        u_cold = u_matrix[i, 6]  # Utility when therapist responds with Cold
        u_warm = u_matrix[i, 2]  # Utility when therapist responds with Warm
        print(f"{OCTANT_NAMES[i]:<25} {mean_u:10.2f} {u_cold:10.2f} {u_warm:10.2f}")
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
        smoothing_alpha=config['smoothing_alpha'],
        bond_power=config['bond_power'],
        random_state=rng,
    )

    # Calculate success threshold
    rs_threshold = calculate_success_threshold(u_matrix, config['success_threshold'])

    # Override bond calculation to use alpha=7
    rs_min = float(u_matrix.min())
    rs_max = float(u_matrix.max())

    print("INITIAL STATE:")
    print("-" * 100)
    print(f"  Initial RS: {client.relationship_satisfaction:.4f}")

    # Calculate bond with alpha=7
    initial_bond_alpha7 = rs_to_bond(
        rs=client.relationship_satisfaction,
        rs_min=rs_min,
        rs_max=rs_max,
        alpha=7
    )
    initial_bond_alpha5 = client.bond  # Default alpha=5

    print(f"  Initial bond (alpha=5): {initial_bond_alpha5:.6f}")
    print(f"  Initial bond (alpha=7): {initial_bond_alpha7:.6f} ← Using this!")
    print(f"  RS threshold (80th percentile): {rs_threshold:.4f}")
    print(f"  RS range: [{rs_min:.2f}, {rs_max:.2f}]")
    print()

    print("BOND COMPARISON (alpha=5 vs alpha=7):")
    print("-" * 100)
    print(f"{'RS':>10} {'Bond α=5':>12} {'Bond α=7':>12} {'Difference':>12}")
    print("-" * 100)
    for rs in [-10, 0, 10, 20, 30, 40]:
        bond5 = rs_to_bond(rs, rs_min, rs_max, alpha=5)
        bond7 = rs_to_bond(rs, rs_min, rs_max, alpha=7)
        diff = bond7 - bond5
        print(f"{rs:10.1f} {bond5:12.6f} {bond7:12.6f} {diff:+12.6f}")
    print()

    print("=" * 100)
    print("SESSION-BY-SESSION TRACKING")
    print("=" * 100)
    print()

    session_data = []

    for session in range(1, config['max_sessions'] + 1):
        # Capture pre-action state
        pre_rs = client.relationship_satisfaction

        # Calculate bond with alpha=7
        pre_bond = rs_to_bond(pre_rs, rs_min, rs_max, alpha=7)

        # Get expected payoffs
        expected_payoffs = client._calculate_expected_payoffs()

        # Need to recalculate with alpha=7 bond
        # The payoffs depend on bond, so we need to temporarily modify the calculation
        # Actually, the client's _calculate_expected_payoffs() uses client.bond internally
        # We need to monkey-patch it

        original_bond = client.bond
        client.bond = pre_bond  # Override with alpha=7 bond
        expected_payoffs = client._calculate_expected_payoffs()

        # Calculate probabilities
        scaled_payoffs = expected_payoffs / config['entropy']
        scaled_payoffs = scaled_payoffs - np.max(scaled_payoffs)
        exp_payoffs = np.exp(scaled_payoffs)
        probabilities = exp_payoffs / np.sum(exp_payoffs)

        # Select action (will use overridden bond)
        client_action = client.select_action()

        # Get therapist response
        therapist_action = always_complement(client_action)

        # Get utility
        utility = u_matrix[client_action, therapist_action]

        # Update client (this will recalculate bond with alpha=5, but we'll override again)
        client.update_memory(client_action, therapist_action)

        # Capture post-action state with alpha=7
        post_rs = client.relationship_satisfaction
        post_bond = rs_to_bond(post_rs, rs_min, rs_max, alpha=7)

        # Override again
        client.bond = post_bond

        # Effective history weight
        effective_hw = config['history_weight'] * (post_bond ** config['bond_power'])

        # Store session data
        session_data.append({
            'session': session,
            'pre_rs': pre_rs,
            'post_rs': post_rs,
            'pre_bond': pre_bond,
            'post_bond': post_bond,
            'effective_hw': effective_hw,
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
            print(f"  Pre:  RS={pre_rs:8.4f}, Bond={pre_bond:.6f}, EffectiveHW={config['history_weight'] * (pre_bond**5):7.3f}")
            print(f"  Action: C={OCTANT_NAMES[client_action]}, T={OCTANT_NAMES[therapist_action]}, U={utility:.2f}")
            print(f"  Post: RS={post_rs:8.4f}, Bond={post_bond:.6f}, EffectiveHW={effective_hw:7.3f}")
            print(f"  Changes: ΔRS={post_rs-pre_rs:+.4f}, ΔBond={post_bond-pre_bond:+.6f}")
            print()

            # Show top 3 expected payoffs
            top3_idx = np.argsort(expected_payoffs)[::-1][:3]
            print(f"  Top 3 Expected Payoffs:")
            for idx in top3_idx:
                print(f"    {OCTANT_NAMES[idx]:<25} Payoff={expected_payoffs[idx]:8.2f}, Prob={probabilities[idx]:.6f}")
            print(f"  → Selected: {OCTANT_NAMES[client_action]}")
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
    final_bond = rs_to_bond(final_rs, rs_min, rs_max, alpha=7)
    success = final_rs >= rs_threshold

    print(f"Sessions completed: {len(session_data)}")
    print(f"Final RS: {final_rs:.4f}")
    print(f"Final bond (alpha=7): {final_bond:.6f}")
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

    # Analyze when history started dominating
    print("HISTORY INFLUENCE OVER TIME:")
    print("-" * 100)
    print(f"{'Session':>8} {'Bond':>10} {'Bond^5':>10} {'EffHW':>10} {'% of Total':>12}")
    print("-" * 100)

    milestones = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for milestone in milestones:
        if milestone <= len(session_data):
            s = session_data[milestone-1]
            bond = s['post_bond']
            bond5 = bond ** 5
            eff_hw = s['effective_hw']
            pct = (eff_hw / (1 + eff_hw)) * 100  # Rough estimate of history's weight
            marker = " ← HISTORY DOMINATES!" if pct > 50 else ""
            print(f"{milestone:8d} {bond:10.6f} {bond5:10.6f} {eff_hw:10.3f} {pct:11.1f}%{marker}")
    print()

    # Key transitions
    print("KEY INSIGHTS:")
    print("-" * 100)

    # When did bond cross key thresholds?
    bond_thresholds = [0.5, 0.7, 0.9, 0.95]
    for thresh in bond_thresholds:
        for s in session_data:
            if s['post_bond'] >= thresh:
                eff_hw = s['effective_hw']
                print(f"  Bond crossed {thresh:.2f} at session {s['session']:3d} (effective HW = {eff_hw:6.2f})")
                break
    print()

    # Did client stay in Cold?
    cold_sessions = sum(1 for s in session_data if s['client_action'] == 6)
    cold_pct = cold_sessions / len(session_data) * 100

    if cold_pct > 50:
        print(f"  ✓ Client stayed mostly COLD ({cold_pct:.1f}% of sessions)")
        print(f"  → Expected cold lock-in scenario OCCURRED!")
    else:
        print(f"  ✗ Client did NOT stay in Cold ({cold_pct:.1f}% of sessions)")
        print(f"  → Expected cold lock-in scenario DID NOT occur")


if __name__ == "__main__":
    main()
