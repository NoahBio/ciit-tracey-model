"""
Detailed analysis of representative failure configurations.

Analyzes 5 diverse failure modes to understand the specific mechanisms
causing each type of failure.

# Note: bond_alpha values clamped to [3,7] range for future consistency.
# Original values were outside this range but have been adjusted while
# keeping other parameters unchanged for historical reference.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
    """Always-complementary therapist strategy."""
    complement_map = {
        0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5
    }
    return complement_map[client_action]


def run_configuration_test(name, config, random_seed=42, verbose=True):
    """
    Run a single configuration with detailed tracking.

    Parameters
    ----------
    name : str
        Name/description of this configuration
    config : dict
        Parameter configuration
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Print detailed session-by-session output

    Returns
    -------
    dict
        Results including trajectory data
    """
    print("=" * 100)
    print(f"TESTING: {name}")
    print("=" * 100)
    print()

    print("CONFIGURATION:")
    print("-" * 100)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Generate initial memory
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type='cold_warm',
        n_interactions=50,
        random_state=42
    )

    # Sample client utility matrix
    rng = np.random.RandomState(random_seed)
    u_matrix = sample_u_matrix(random_state=rng)

    # Create client
    client = create_client(
        mechanism='bond_weighted_frequency_amplifier',
        initial_memory=initial_memory,
        u_matrix=u_matrix,
        entropy=config['entropy'],
        history_weight=config['history_weight'],
        bond_power=config['bond_power'],
        random_state=rng,
    )

    # Override bond parameters
    rs_min = float(u_matrix.min())
    rs_max = float(u_matrix.max())

    # Calculate initial bond with custom parameters
    client.bond = rs_to_bond(
        client.relationship_satisfaction,
        rs_min,
        rs_max,
        alpha=config['bond_alpha'],
        offset=config['bond_offset']
    )

    # Calculate success threshold
    rs_threshold = calculate_success_threshold(u_matrix, config['success_threshold'])

    print("INITIAL STATE:")
    print("-" * 100)
    print(f"  Initial RS: {client.relationship_satisfaction:.4f}")
    print(f"  Initial bond: {client.bond:.6f}")
    print(f"  RS threshold: {rs_threshold:.4f}")
    print(f"  Gap to threshold: {rs_threshold - client.relationship_satisfaction:.4f} points")
    print()

    # Run therapy sessions
    session_data = []
    max_sessions = 100

    for session in range(1, max_sessions + 1):
        # Capture pre-action state
        pre_rs = client.relationship_satisfaction
        pre_bond = client.bond
        pre_effective_hw = (pre_bond ** config['bond_power']) * config['history_weight']

        # Select client action
        client_action = client.select_action()

        # Get therapist response
        therapist_action = always_complement(client_action)

        # Get utility
        utility = u_matrix[client_action, therapist_action]

        # Update client
        client.update_memory(client_action, therapist_action)

        # Recalculate bond with custom parameters
        post_rs = client.relationship_satisfaction
        client.bond = rs_to_bond(
            post_rs,
            rs_min,
            rs_max,
            alpha=config['bond_alpha'],
            offset=config['bond_offset']
        )
        post_bond = client.bond
        post_effective_hw = (post_bond ** config['bond_power']) * config['history_weight']

        # Store session data
        session_data.append({
            'session': session,
            'pre_rs': pre_rs,
            'post_rs': post_rs,
            'pre_bond': pre_bond,
            'post_bond': post_bond,
            'pre_effective_hw': pre_effective_hw,
            'post_effective_hw': post_effective_hw,
            'client_action': client_action,
            'therapist_action': therapist_action,
            'utility': utility,
        })

        # Verbose output for select sessions
        if verbose and (session <= 10 or session % 10 == 0 or session >= max_sessions - 5):
            print(f"SESSION {session}")
            print(f"  Pre:  RS={pre_rs:8.4f}, Bond={pre_bond:.6f}, EffHW={pre_effective_hw:7.3f}")
            print(f"  Action: C={OCTANT_NAMES[client_action]}, T={OCTANT_NAMES[therapist_action]}, U={utility:.2f}")
            print(f"  Post: RS={post_rs:8.4f}, Bond={post_bond:.6f}, EffHW={post_effective_hw:7.3f}")
            print(f"  ΔRS={post_rs-pre_rs:+.4f}, ΔBond={post_bond-pre_bond:+.6f}")
            print()

        # Check dropout
        if client.check_dropout():
            print(f"⚠️  CLIENT DROPPED OUT at session {session}")
            break

    # Final results
    final_rs = client.relationship_satisfaction
    final_bond = client.bond
    success = final_rs >= rs_threshold
    sessions_completed = len(session_data)

    print("=" * 100)
    print("FINAL OUTCOME")
    print("=" * 100)
    print()
    print(f"Sessions completed: {sessions_completed}")
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
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = count / len(actions) * 100
        bar = '█' * int(pct / 2)
        print(f"  {OCTANT_NAMES[action]:<25} {count:3d} ({pct:5.1f}%) {bar}")
    print()

    # Key transitions
    bond_50_session = next((s['session'] for s in session_data if s['post_bond'] >= 0.5), None)
    bond_90_session = next((s['session'] for s in session_data if s['post_bond'] >= 0.9), None)
    hw_1_session = next((s['session'] for s in session_data if s['post_effective_hw'] >= 1.0), None)

    print("KEY TRANSITIONS:")
    if bond_50_session:
        print(f"  Bond crossed 0.5 at session {bond_50_session}")
    if bond_90_session:
        print(f"  Bond crossed 0.9 at session {bond_90_session}")
    if hw_1_session:
        print(f"  Effective history weight crossed 1.0 at session {hw_1_session}")
    print()

    return {
        'name': name,
        'config': config,
        'success': success,
        'final_rs': final_rs,
        'final_bond': final_bond,
        'rs_threshold': rs_threshold,
        'sessions_completed': sessions_completed,
        'session_data': session_data,
        'action_counts': dict(action_counts),
        'bond_50_session': bond_50_session,
        'bond_90_session': bond_90_session,
        'hw_1_session': hw_1_session,
    }


def compare_configurations(all_results):
    """
    Compare results across all configurations.
    """
    print("\n" + "=" * 100)
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)
    print()

    # Summary table
    print("Summary Table:")
    print("-" * 100)
    print(f"{'Config':<20} {'Success':<8} {'Final RS':<10} {'Bond@50':<10} {'Bond@90':<10} {'HW>1':<10}")
    print("-" * 100)

    for r in all_results:
        bond_50 = r['bond_50_session'] if r['bond_50_session'] else 'Never'
        bond_90 = r['bond_90_session'] if r['bond_90_session'] else 'Never'
        hw_1 = r['hw_1_session'] if r['hw_1_session'] else 'Never'

        print(f"{r['name']:<20} {'✓' if r['success'] else '✗':<8} "
              f"{r['final_rs']:<10.2f} {str(bond_50):<10} {str(bond_90):<10} {str(hw_1):<10}")
    print()

    # Parameter comparison
    print("Parameter Comparison:")
    print("-" * 100)
    print(f"{'Config':<20} {'bond_pwr':<10} {'hist_wgt':<10} {'entropy':<10} {'b_alpha':<10} {'b_offset':<10}")
    print("-" * 100)

    for r in all_results:
        c = r['config']
        print(f"{r['name']:<20} {c['bond_power']:<10.2f} {c['history_weight']:<10.2f} "
              f"{c['entropy']:<10.3f} {c['bond_alpha']:<10.2f} {c['bond_offset']:<10.3f}")
    print()


def create_trajectory_visualization(all_results):
    """
    Create visualization of RS and bond trajectories.
    """
    print("=" * 100)
    print("CREATING TRAJECTORY VISUALIZATIONS")
    print("=" * 100)
    print()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Failure Configuration Trajectories', fontsize=16)

    # Plot each configuration
    colors = ['red', 'orange', 'yellow', 'green', 'blue']

    for idx, r in enumerate(all_results):
        color = colors[idx]
        sessions = [s['session'] for s in r['session_data']]
        rs_values = [s['post_rs'] for s in r['session_data']]
        bond_values = [s['post_bond'] for s in r['session_data']]
        eff_hw_values = [s['post_effective_hw'] for s in r['session_data']]

        # RS trajectory
        axes[0, 0].plot(sessions, rs_values, label=r['name'], color=color, alpha=0.7, linewidth=2)

        # Bond trajectory
        axes[0, 1].plot(sessions, bond_values, label=r['name'], color=color, alpha=0.7, linewidth=2)

        # Effective history weight trajectory
        axes[0, 2].plot(sessions, eff_hw_values, label=r['name'], color=color, alpha=0.7, linewidth=2)

    # Configure axes
    axes[0, 0].set_xlabel('Session')
    axes[0, 0].set_ylabel('Relationship Satisfaction')
    axes[0, 0].set_title('RS Trajectory')
    axes[0, 0].axhline(all_results[0]['rs_threshold'], color='black', linestyle='--', label='Threshold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Session')
    axes[0, 1].set_ylabel('Bond')
    axes[0, 1].set_title('Bond Trajectory')
    axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_xlabel('Session')
    axes[0, 2].set_ylabel('Effective History Weight')
    axes[0, 2].set_title('History Influence Over Time')
    axes[0, 2].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='HW=1')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Individual configuration details
    detail_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]

    for idx, (ax, r) in enumerate(zip(detail_axes, all_results[:3])):
        sessions = [s['session'] for s in r['session_data']]
        rs_values = [s['post_rs'] for s in r['session_data']]
        bond_values = [s['post_bond'] for s in r['session_data']]

        ax2 = ax.twinx()

        # Plot RS on left axis
        line1 = ax.plot(sessions, rs_values, 'b-', label='RS', linewidth=2)
        ax.axhline(r['rs_threshold'], color='blue', linestyle='--', alpha=0.5)
        ax.set_xlabel('Session')
        ax.set_ylabel('RS', color='b')
        ax.tick_params(axis='y', labelcolor='b')

        # Plot bond on right axis
        line2 = ax2.plot(sessions, bond_values, 'r-', label='Bond', linewidth=2)
        ax2.set_ylabel('Bond', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title(f"{r['name']}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "visualization_output" / "failure_trajectories.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    print()

    plt.close()


def main():
    # Define test configurations
    # bond_alpha values clamped to [3,7] range (original values in comments)
    test_configs = [
        ("Worst (10%)", {
            'bond_power': 1.520584,
            'history_weight': 14.551988,
            'entropy': 0.453514,
            'bond_alpha': 7.0,  # was 14.060780, clamped to max=7
            'bond_offset': 0.698472,
            'success_threshold': 0.9,
        }),
        ("High bond_power (15%)", {
            'bond_power': 9.863571,
            'history_weight': 12.664022,
            'entropy': 0.149865,
            'bond_alpha': 7.0,  # was 13.970944, clamped to max=7
            'bond_offset': 0.660969,
            'success_threshold': 0.9,
        }),
        ("Mid-range (25%)", {
            'bond_power': 9.354867,
            'history_weight': 6.708670,
            'entropy': 0.486662,
            'bond_alpha': 7.0,  # was 14.527060, clamped to max=7
            'bond_offset': 0.655903,
            'success_threshold': 0.9,
        }),
        ("Borderline (40%)", {
            'bond_power': 5.513647,
            'history_weight': 12.075280,
            'entropy': 0.359986,
            'bond_alpha': 7.0,  # was 11.125569, clamped to max=7
            'bond_offset': 0.638738,
            'success_threshold': 0.9,
        }),
        ("Lowest offset (30%)", {
            'bond_power': 1.284275,
            'history_weight': 14.075078,
            'entropy': 0.120789,
            'bond_alpha': 7.0,  # was 9.036852, clamped to max=7
            'bond_offset': 0.612718,
            'success_threshold': 0.9,
        }),
    ]

    # Run all configurations
    all_results = []
    for name, config in test_configs:
        result = run_configuration_test(name, config, random_seed=42, verbose=False)
        all_results.append(result)
        print()

    # Compare configurations
    compare_configurations(all_results)

    # Create visualizations
    create_trajectory_visualization(all_results)

    print("=" * 100)
    print("TESTING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
