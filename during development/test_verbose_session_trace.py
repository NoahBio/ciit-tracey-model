"""Verbose diagnostic trace of client behavior across therapy sessions.

Shows detailed step-by-step information about:
- Initial client state
- Expected payoffs and action probabilities each session
- Selected actions and therapist responses
- Memory updates, RS and bond changes
- Progress toward success threshold
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.agents.client_agents import (
    with_perception,
    BondOnlyClient,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
    BondWeightedConditionalAmplifier,
    BondWeightedFrequencyAmplifier,
    BaseClientAgent,
)
from src import config
from src.config import (
    sample_u_matrix,
    OCTANTS,
    calculate_success_threshold,
    PERCEPTION_BASELINE_ACCURACY,
    PERCEPTION_ADJACENCY_NOISE,
)
import argparse


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


# Map legacy pattern names to BaseClientAgent names
PATTERN_ALIASES = {
    'cw_50_50': 'cold_warm',
}


def verbose_session_trace(
    mechanism: str,
    initial_memory_pattern: str,
    success_threshold_percentile: float,
    enable_perception: bool = False,
    baseline_accuracy: float = 0.2,
    verbose_sessions: range = range(1, 6),
    max_sessions: int = 100,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 5.0,  
    bond_offset: float = 0.8,  
    random_state: int = 42,
):
    """
    Run therapy sessions with verbose diagnostic output for specified sessions.

    Runs all sessions to completion (or dropout), but only shows detailed
    verbose output for sessions in the verbose_sessions range.

    Parameters
    ----------
    mechanism : str
        Client mechanism to test
    initial_memory_pattern : str
        Initial memory pattern (cw_50_50, complementary_perfect, conflictual, mixed_random)
    success_threshold_percentile : float
        Percentile of client's achievable RS range (0-1)
    verbose_sessions : range or list, default=range(1, 6)
        Which sessions to show verbose output for (e.g., range(1, 6) for sessions 1-5)
        Set to [] to suppress all verbose output and show only summary
    max_sessions : int, default=100
        Maximum number of sessions to run
    entropy : float
        Client entropy (exploration parameter)
    history_weight : float
        History weight for amplifier mechanisms
    bond_power : float
        Bond power for bond_weighted mechanisms
    bond_alpha : float
        Bond alpha (sigmoid steepness parameter)
    bond_offset : float
        Bond offset for sigmoid inflection point (0.0-1.0)
    random_state : int
        Random seed
    """

    print("=" * 100)
    print(f"VERBOSE SESSION TRACE")
    print("=" * 100)
    print()

    # Setup
    rng = np.random.RandomState(random_state)
    u_matrix = sample_u_matrix(random_state=random_state)

    # Map legacy pattern names and generate memory
    pattern_type = PATTERN_ALIASES.get(initial_memory_pattern, initial_memory_pattern)
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type=pattern_type,
        n_interactions=50,
        random_state=random_state,
    )

    # Set global bond parameters BEFORE creating client
    config.BOND_ALPHA = bond_alpha
    config.BOND_OFFSET = bond_offset

    # Create client
    client_kwargs = {
        'u_matrix': u_matrix,
        'entropy': entropy,
        'initial_memory': initial_memory,
        'random_state': random_state,
    }

    if 'amplifier' in mechanism:
        client_kwargs['history_weight'] = history_weight

    if 'bond_weighted' in mechanism:
        client_kwargs['bond_power'] = bond_power

    if enable_perception:
        client_kwargs['baseline_accuracy'] = baseline_accuracy
        client_kwargs['enable_perception'] = True
    
    mechanisms = {
        'bond_only': BondOnlyClient,
        'frequency_amplifier': FrequencyAmplifierClient,
        'conditional_amplifier': ConditionalAmplifierClient,
        'bond_weighted_conditional_amplifier': BondWeightedConditionalAmplifier,
        'bond_weighted_frequency_amplifier': BondWeightedFrequencyAmplifier,
    }

    ClientClass = mechanisms[mechanism]

    if enable_perception:
        ClientClass = with_perception(ClientClass)

    # Create client directly with the class (not create_client)
    client = ClientClass(**client_kwargs)

    # Calculate RS threshold
    rs_threshold = calculate_success_threshold(u_matrix, success_threshold_percentile)

    # Display initial state
    print("CONFIGURATION")
    print("-" * 100)
    print(f"Mechanism: {mechanism}")
    print(f"Initial memory pattern: {initial_memory_pattern}")
    print(f"Success threshold percentile: {success_threshold_percentile:.1f} ({success_threshold_percentile*100:.0f}%)")
    print(f"Entropy: {entropy:.2f}")
    if 'amplifier' in mechanism:
        print(f"History weight: {history_weight:.2f}")
    if 'bond_weighted' in mechanism:
        print(f"Bond power: {bond_power:.2f}")
    print(f"Bond alpha: {bond_alpha:.2f}")
    print(f"Bond offset: {bond_offset:.2f}")
    if enable_perception:
        print(f"Perception enabled: Yes")
        print(f"Baseline accuracy: {baseline_accuracy:.1%}")
    else:
        print(f"Perception enabled: No (perfect perception)")
    print()

    print("CLIENT U_MATRIX BOUNDS")
    print("-" * 100)
    print(f"Min possible RS: {u_matrix.min():7.2f}")
    print(f"Max possible RS: {u_matrix.max():7.2f}")
    print(f"RS range: {u_matrix.max() - u_matrix.min():7.2f}")
    print(f"RS threshold (at {success_threshold_percentile:.0%}): {rs_threshold:7.2f}")
    print()

    print("INITIAL CLIENT STATE (Before any therapy sessions)")
    print("-" * 100)
    print(f"Initial RS: {client.relationship_satisfaction:7.2f}")
    print(f"Initial bond: {client.bond:.4f}")
    print(f"RS needed to succeed: {rs_threshold:7.2f}")
    print(f"RS gap to threshold: {rs_threshold - client.relationship_satisfaction:7.2f}")
    print()

    # Show initial expected payoffs
    initial_expected = client._calculate_expected_payoffs()
    print("Initial expected payoffs (before therapy):")
    for i in range(8):
        print(f"  {OCTANTS[i]:3s} ({i}): {initial_expected[i]:8.2f}")
    print()

    # Track session history
    session_history = []
    initial_rs = client.relationship_satisfaction
    initial_bond = client.bond

    # Run sessions
    print("=" * 100)
    print("THERAPY SESSIONS")
    print("=" * 100)
    if verbose_sessions:
        print(f"Showing verbose output for sessions: {list(verbose_sessions)[:10]}{'...' if len(list(verbose_sessions)) > 10 else ''}")
    else:
        print("Running all sessions (verbose output suppressed)")
    print()

    session = 0
    dropped_out = False

    for session in range(1, max_sessions + 1):
        # Get current state
        current_rs = client.relationship_satisfaction
        current_bond = client.bond

        # Determine if this session should be verbose
        is_verbose = session in verbose_sessions

        if is_verbose:
            print(f"{'SESSION ' + str(session):=^100}")
            print()
            print(f"Pre-session state:")
            print(f"  RS: {current_rs:7.2f} | Bond: {current_bond:.4f}")
            print(f"  Gap to threshold: {rs_threshold - current_rs:7.2f}")
            print()

        # Calculate expected payoffs
        expected_payoffs = client._calculate_expected_payoffs()

        if is_verbose:
            print("Expected payoffs for each client action:")
            for i in range(8):
                print(f"  {OCTANTS[i]:3s} ({i}): {expected_payoffs[i]:8.2f}")
            print()

        # Calculate action probabilities
        action_probs = client._softmax(expected_payoffs)

        if is_verbose:
            print(f"Action probabilities (entropy={entropy:.2f}):")
            for i in range(8):
                bar = '█' * int(action_probs[i] * 50)  # 50 chars max
                print(f"  {OCTANTS[i]:3s} ({i}): {action_probs[i]:6.3f} ({action_probs[i]*100:5.1f}%) {bar}")
            print()

        # Select action
        client_action = client.select_action()
        therapist_action = always_complement(client_action)

        # Get utility of this interaction
        interaction_utility = u_matrix[client_action, therapist_action]

        if is_verbose:
            print(f"Client selects: {OCTANTS[client_action]:3s} ({client_action})")
            print(f"Therapist responds: {OCTANTS[therapist_action]:3s} ({therapist_action}) [complementary]")
            print(f"Utility of interaction: {interaction_utility:7.2f}")
            print()

        # Update memory
        client.update_memory(client_action, therapist_action)

        # Show perception details if enabled
        if is_verbose and enable_perception:
            if hasattr(client, 'perception_history') and client.perception_history:
                record = client.perception_history[-1]

                print("Perception Details:")
                print(f"  Actual therapist action: {OCTANTS[record.actual_therapist_action]:3s} ({record.actual_therapist_action})")
                print(f"  Perceived by client:     {OCTANTS[record.perceived_therapist_action]:3s} ({record.perceived_therapist_action})", end="")

                if record.perceived_therapist_action != record.actual_therapist_action:
                    print(" ⚠ MISPERCEPTION")
                else:
                    print(" ✓ Correct")

                # Show perception stages
                print(f"  Stage 1 (history-based): {OCTANTS[record.stage1_result]:3s} ({record.stage1_result})", end="")
                if record.baseline_path_succeeded:
                    print(f" [baseline path: {PERCEPTION_BASELINE_ACCURACY:.0%}]")
                else:
                    print(f" [frequency path: {record.computed_accuracy:.2%}]")

                if record.stage1_changed_from_actual:
                    print(f"    → Changed from actual (history override)")

                if record.stage2_shifted:
                    print(f"  Stage 2 (adjacency):     Shifted ±1 ({PERCEPTION_ADJACENCY_NOISE:.0%} chance)")

                print()

        # Get new state
        new_rs = client.relationship_satisfaction
        new_bond = client.bond
        rs_change = new_rs - current_rs
        bond_change = new_bond - current_bond

        # Record session data
        session_history.append({
            'session': session,
            'client_action': client_action,
            'therapist_action': therapist_action,
            'interaction_utility': interaction_utility,
            'rs_before': current_rs,
            'rs_after': new_rs,
            'rs_change': rs_change,
            'bond_before': current_bond,
            'bond_after': new_bond,
            'bond_change': bond_change,
            'reached_threshold': new_rs >= rs_threshold,
        })

        if is_verbose:
            print(f"Post-session state:")
            print(f"  RS: {new_rs:7.2f} (change: {rs_change:+7.2f})")
            print(f"  Bond: {new_bond:.4f} (change: {bond_change:+.4f})")
            print(f"  Gap to threshold: {rs_threshold - new_rs:7.2f}")

            if new_rs >= rs_threshold:
                print(f"  ✓ SUCCESS THRESHOLD REACHED!")
            else:
                remaining_gap = rs_threshold - new_rs
                print(f"  → Still need {remaining_gap:7.2f} more RS to reach threshold")

            print()

        # Check dropout
        would_dropout = client.check_dropout()
        if would_dropout:
            if is_verbose or not verbose_sessions:
                print(f"Session {session}: ⚠ Client dropped out!")
                print()
            dropped_out = True
            break

        # Show progress updates for non-verbose sessions (every 10 sessions)
        if not is_verbose and session % 10 == 0:
            print(f"Session {session:3d}: RS={new_rs:7.2f}, Bond={new_bond:.4f}, Gap to threshold={rs_threshold - new_rs:7.2f}")

    # Show final non-verbose sessions if last session wasn't shown
    if verbose_sessions and session > max(verbose_sessions):
        print()
        print(f"... (sessions {max(verbose_sessions)+1}-{session} completed)")
        print()

    # Final summary
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print()

    final_rs = client.relationship_satisfaction
    final_bond = client.bond
    total_rs_change = final_rs - initial_rs
    total_bond_change = final_bond - initial_bond

    print("OVERALL OUTCOME")
    print("-" * 100)
    print(f"Total sessions completed: {session}")
    print(f"Dropped out: {'Yes' if dropped_out else 'No'}")
    print()

    print("RELATIONSHIP SATISFACTION (RS)")
    print("-" * 100)
    print(f"Initial RS: {initial_rs:7.2f}")
    print(f"Final RS:   {final_rs:7.2f}")
    print(f"Total change: {total_rs_change:+7.2f}")
    print(f"Average per session: {total_rs_change/session:+7.2f}")
    print()
    print(f"Client RS range: [{u_matrix.min():.2f}, {u_matrix.max():.2f}]")
    print(f"RS threshold ({success_threshold_percentile:.0%}): {rs_threshold:7.2f}")
    print(f"Final gap to threshold: {rs_threshold - final_rs:7.2f}")
    print()

    if final_rs >= rs_threshold:
        print(f"✓ SUCCESS: Client reached {success_threshold_percentile:.0%} percentile of their RS range")
        # Find when threshold was first reached
        threshold_session = next((h['session'] for h in session_history if h['reached_threshold']), None)
        if threshold_session:
            print(f"  Threshold first reached at session {threshold_session}")
    else:
        shortfall = rs_threshold - final_rs
        print(f"✗ FAILURE: Client fell short by {shortfall:7.2f} RS")
    print()

    print("BOND DEVELOPMENT")
    print("-" * 100)
    print(f"Initial bond: {initial_bond:.4f}")
    print(f"Final bond:   {final_bond:.4f}")
    print(f"Total change: {total_bond_change:+.4f}")
    print(f"Average per session: {total_bond_change/session:+.4f}")
    print()

    # Perception summary
    if enable_perception and hasattr(client, 'get_perception_stats'):
        print("PERCEPTION SUMMARY")
        print("-" * 100)
        pstats = client.get_perception_stats()
        print(f"Total interactions: {pstats['total_interactions']}")
        print(f"Overall misperception rate: {pstats['overall_misperception_rate']:.1%}")
        print(f"Stage 1 override rate: {pstats['stage1_override_rate']:.1%} (history-based changes)")
        print(f"Stage 2 shift rate: {pstats['stage2_shift_rate']:.1%} (adjacency noise)")
        print(f"Mean computed accuracy: {pstats['mean_computed_accuracy']:.3f}")
        print(f"Baseline path successes: {pstats['baseline_correct_count']} times")
        print()

    # Action distribution
    print("ACTION DISTRIBUTION")
    print("-" * 100)
    from collections import Counter
    client_actions = [h['client_action'] for h in session_history]
    action_counts = Counter(client_actions)

    print("Client actions:")
    for action in range(8):
        count = action_counts.get(action, 0)
        pct = count / len(session_history) * 100 if session_history else 0
        bar = '█' * int(pct / 2)  # 50 chars = 100%
        print(f"  {OCTANTS[action]:3s} ({action}): {count:3d} ({pct:5.1f}%) {bar}")
    print()

    # Trajectory visualization (every 10 sessions or key milestones)
    print("RS TRAJECTORY (every 10 sessions)")
    print("-" * 100)
    print(f"{'Session':<10} {'RS':>10} {'Change':>10} {'Bond':>10} {'Status':<20}")
    print("-" * 100)
    print(f"Initial    {initial_rs:10.2f} {'':>10} {initial_bond:10.4f}")

    for i in range(9, len(session_history), 10):  # Every 10 sessions
        h = session_history[i]
        status = "✓ Threshold reached" if h['reached_threshold'] else ""
        print(f"{h['session']:<10} {h['rs_after']:10.2f} {h['rs_change']:+10.2f} {h['bond_after']:10.4f} {status:<20}")

    # Always show final session if not already shown
    if session_history and (len(session_history) - 1) % 10 != 9:
        h = session_history[-1]
        status = "✓ Threshold reached" if h['reached_threshold'] else ""
        print(f"{h['session']:<10} {h['rs_after']:10.2f} {h['rs_change']:+10.2f} {h['bond_after']:10.4f} {status:<20}")

    print()

    # Summary statistics
    if session_history:
        avg_utility = np.mean([h['interaction_utility'] for h in session_history])
        avg_rs_change = np.mean([h['rs_change'] for h in session_history])
        avg_bond_change = np.mean([h['bond_change'] for h in session_history])

        print("SESSION STATISTICS")
        print("-" * 100)
        print(f"Average interaction utility: {avg_utility:7.2f}")
        print(f"Average RS change per session: {avg_rs_change:+7.2f}")
        print(f"Average bond change per session: {avg_bond_change:+.4f}")
        print()


def run_comparison_tests():
    """Run multiple scenarios for comparison."""

    print("=" * 100)
    print("VERBOSE SESSION TRACE COMPARISON")
    print("=" * 100)
    print()

    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Standard conditional_amplifier with C→W memory, 50% threshold',
            'mechanism': 'conditional_amplifier',
            'pattern': 'cw_50_50',
            'threshold': 0.5,
            'entropy': 1.0,
        },
        {
            'name': 'Scenario 2: Same but with 80% threshold (more stringent)',
            'mechanism': 'conditional_amplifier',
            'pattern': 'cw_50_50',
            'threshold': 0.8,
            'entropy': 1.0,
        },
        {
            'name': 'Scenario 3: Conflictual memory with 50% threshold',
            'mechanism': 'conditional_amplifier',
            'pattern': 'conflictual',
            'threshold': 0.5,
            'entropy': 1.0,
        },
        {
            'name': 'Scenario 4: Bond-weighted with C→W memory, 80% threshold',
            'mechanism': 'bond_weighted_conditional_amplifier',
            'pattern': 'cw_50_50',
            'threshold': 0.8,
            'entropy': 1.0,
            'bond_power': 2.0,
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'#' * 100}")
        print(f"# {scenario['name']}")
        print(f"{'#' * 100}\n")

        kwargs = {
            'mechanism': scenario['mechanism'],
            'initial_memory_pattern': scenario['pattern'],
            'success_threshold_percentile': scenario['threshold'],
            'entropy': scenario.get('entropy', 1.0),
            'verbose_sessions': range(1, 6),  # Show verbose output for first 5 sessions
            'max_sessions': 100,  # Run up to 100 sessions total
            'random_state': 42,
        }

        if 'bond_power' in scenario:
            kwargs['bond_power'] = scenario['bond_power']

        verbose_session_trace(**kwargs)

        if i < len(scenarios):
            print("\n" + "=" * 100)
            print("Press Enter to continue to next scenario...")
            print("=" * 100)
            # input()  # Uncomment to pause between scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verbose trace of client behavior across therapy sessions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core configuration
    parser.add_argument(
        '--mechanism', '-m',
        type=str,
        default='conditional_amplifier',
        choices=[
            'bond_only',
            'frequency_filter',
            'frequency_amplifier',
            'conditional_filter',
            'conditional_amplifier',
            'bond_weighted_frequency_amplifier',
            'bond_weighted_conditional_amplifier'
        ],
        help='Client expectation mechanism'
    )

    parser.add_argument(
    '--enable-perception',
    action='store_true',
    help='Enable perceptual distortion (imperfect client perception of therapist actions)'
    )

    parser.add_argument(
        '--baseline-accuracy',
        type=float,
        default=0.2,
        help='Baseline perception accuracy (default: 0.2 = 20%% chance of correct perception via baseline path)'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='cw_50_50',
        choices=['cw_50_50', 'cold_warm', 'complementary_perfect', 'conflictual',
                 'mixed_random', 'cold_stuck', 'dominant_stuck', 'submissive_stuck'],
        help='Initial memory pattern'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.8,
        help='Success threshold percentile (0.0-1.0)'
    )
    
    parser.add_argument(
        '--entropy', '-e',
        type=float,
        default=3.0,
        help='Client entropy (exploration parameter)'
    )
    
    parser.add_argument(
        '--history-weight', '-hw',
        type=float,
        default=1.0,
        help='History weight for amplifier mechanisms'
    )
    
    parser.add_argument(
        '--bond-power', '-bp',
        type=float,
        default=1.0,
        help='Bond power for bond_weighted mechanisms'
    )
    
    parser.add_argument(
        '--bond-alpha', '-ba',
        type=float,
        default=5.0,
        help='Bond alpha (sigmoid steepness parameter)'
    )
    
    parser.add_argument(
        '--bond-offset', '-bo',
        type=float,
        default=0.8,
        help='Bond offset for sigmoid inflection point (0.0-1.0)'
    )
    
    parser.add_argument(
        '--max-sessions', '-s',
        type=int,
        default=100,
        help='Maximum number of therapy sessions'
    )
    
    parser.add_argument(
        '--verbose-start',
        type=int,
        default=1,
        help='First session to show verbose output for'
    )
    
    parser.add_argument(
        '--verbose-end',
        type=int,
        default=5,
        help='Last session to show verbose output for'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Suppress verbose output, show only summary'
    )
    
    parser.add_argument(
        '--seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparison tests instead of single trace'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison tests
        run_comparison_tests()
    else:
        # Run single trace with specified parameters
        verbose_sessions = [] if args.no_verbose else range(args.verbose_start, args.verbose_end + 1)
        
        kwargs = {
            'mechanism': args.mechanism,
            'enable_perception': args.enable_perception,
            'baseline_accuracy': args.baseline_accuracy,  
            'initial_memory_pattern': args.pattern,
            'success_threshold_percentile': args.threshold,
            'entropy': args.entropy,
            'history_weight': args.history_weight,
            'bond_power': args.bond_power,
            'bond_alpha': args.bond_alpha,  
            'bond_offset': args.bond_offset,
            'verbose_sessions': verbose_sessions,
            'max_sessions': args.max_sessions,
            'random_state': args.seed,
        }
        
        verbose_session_trace(**kwargs)
        
        print("\n" + "=" * 100)
        print("VERBOSE SESSION TRACE COMPLETE")
        print("=" * 100)
