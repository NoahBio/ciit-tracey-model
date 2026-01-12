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
    with_parataxic,
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
    PARATAXIC_BASELINE_ACCURACY,
)
from src.agents.therapist_agents.omniscient_therapist_v2 import OmniscientStrategicTherapist
import argparse


# Map legacy pattern names to BaseClientAgent names
PATTERN_ALIASES = {
    'cw_50_50': 'cold_warm',
}


def verbose_session_trace(
    mechanism: str,
    initial_memory_pattern: str,
    success_threshold_percentile: float,
    enable_parataxic: bool = False,
    baseline_accuracy: float = 0.5549619551286054,
    verbose_sessions: range = range(1, 6),
    max_sessions: int = 1940,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 11.847676335038303,
    bond_offset: float = 0.624462461360537,
    random_state: int = 42,
    # V2 Therapist parameters
    perception_window: int = 10,
    seeding_benefit_scaling: float = 1.8658722646107764,
    skip_seeding_accuracy_threshold: float = 0.814677493978211,
    quick_seed_actions_threshold: int = 1,
    abort_consecutive_failures_threshold: int = 4,
):
    """
    Run therapy sessions with verbose diagnostic output for specified sessions.

    Runs all sessions to completion (or dropout), but only shows detailed
    verbose output for sessions in the verbose_sessions range.

    Success is determined by whether the client reaches the success threshold
    AT ANY POINT during therapy, consistent with the criterion used in
    test_perception_comparison.py. Even if RS later declines below threshold,
    reaching it indicates successful therapeutic progress.

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
    perception_window : int, default=10
        Memory window size for parataxic distortion (V2 therapist)
    seeding_benefit_scaling : float, default=1.8658722646107764
        Scaling factor for expected seeding benefit (0.1-2.0)
    skip_seeding_accuracy_threshold : float, default=0.814677493978211
        Skip seeding if accuracy above this threshold (0.75-0.95)
    quick_seed_actions_threshold : int, default=1
        "Just do it" if actions_needed <= this (1-5)
    abort_consecutive_failures_threshold : int, default=4
        Abort after this many consecutive failures (4-9)
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

    if enable_parataxic:
        client_kwargs['baseline_accuracy'] = baseline_accuracy
        client_kwargs['enable_parataxic'] = True
    
    mechanisms = {
        'bond_only': BondOnlyClient,
        'frequency_amplifier': FrequencyAmplifierClient,
        'conditional_amplifier': ConditionalAmplifierClient,
        'bond_weighted_conditional_amplifier': BondWeightedConditionalAmplifier,
        'bond_weighted_frequency_amplifier': BondWeightedFrequencyAmplifier,
    }

    ClientClass = mechanisms[mechanism]

    if enable_parataxic:
        ClientClass = with_parataxic(ClientClass)

    # Create client directly with the class (not create_client)
    client = ClientClass(**client_kwargs)

    # Create V2 therapist with omniscient access
    therapist = OmniscientStrategicTherapist(
        client_ref=client,
        perception_window=perception_window,
        baseline_accuracy=baseline_accuracy,
        seeding_benefit_scaling=seeding_benefit_scaling,
        skip_seeding_accuracy_threshold=skip_seeding_accuracy_threshold,
        quick_seed_actions_threshold=quick_seed_actions_threshold,
        abort_consecutive_failures_threshold=abort_consecutive_failures_threshold,
    )

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
    if enable_parataxic:
        print(f"Parataxic distortion enabled: Yes")
        print(f"Baseline accuracy: {baseline_accuracy:.1%}")
    else:
        print(f"Parataxic distortion enabled: No (perfect perception)")

    print(f"V2 Therapist parameters:")
    print(f"  Perception window: {perception_window}")
    print(f"  Seeding benefit scaling: {seeding_benefit_scaling:.4f}")
    print(f"  Skip seeding accuracy threshold: {skip_seeding_accuracy_threshold:.4f}")
    print(f"  Quick seed actions threshold: {quick_seed_actions_threshold}")
    print(f"  Abort consecutive failures threshold: {abort_consecutive_failures_threshold}")
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

    # Track if threshold is ever reached (matching comparison test criterion)
    threshold_ever_reached = False
    first_threshold_session = None

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

        # Get therapist decision using V2 strategy
        therapist_action, metadata = therapist.decide_action(client_action, session)

        # Get utility of this interaction
        interaction_utility = u_matrix[client_action, therapist_action]

        if is_verbose:
            print(f"Client selects: {OCTANTS[client_action]:3s} ({client_action})")
            print(f"Therapist responds: {OCTANTS[therapist_action]:3s} ({therapist_action}) [Phase: {metadata['phase']}]")
            print(f"Rationale: {metadata['rationale']}")
            print(f"Utility of interaction: {interaction_utility:7.2f}")
            print()

        # Update memory
        client.update_memory(client_action, therapist_action)

        # Process feedback for seeding monitoring
        therapist.process_feedback_after_memory_update(session, client_action)

        # Show parataxic distortion details if enabled
        if is_verbose and enable_parataxic:
            if hasattr(client, 'parataxic_history') and client.parataxic_history:
                record = client.parataxic_history[-1]

                print("Parataxic Distortion Details:")
                print(f"  Actual therapist action: {OCTANTS[record.actual_therapist_action]:3s} ({record.actual_therapist_action})")
                print(f"  Perceived by client:     {OCTANTS[record.perceived_therapist_action]:3s} ({record.perceived_therapist_action})", end="")

                if record.perceived_therapist_action != record.actual_therapist_action:
                    print(" ⚠ MISPERCEPTION")
                else:
                    print(" ✓ Correct")

                # Show perception stages
                print(f"  Stage 1 (history-based): {OCTANTS[record.stage1_result]:3s} ({record.stage1_result})", end="")
                if record.baseline_path_succeeded:
                    print(f" [baseline path: {PARATAXIC_BASELINE_ACCURACY:.0%}]")
                else:
                    print(f" [frequency path: {record.computed_accuracy:.2%}]")

                if record.stage1_changed_from_actual:
                    print(f"    → Changed from actual (history override)")

                print()

        # Get new state
        new_rs = client.relationship_satisfaction
        new_bond = client.bond
        rs_change = new_rs - current_rs
        bond_change = new_bond - current_bond

        # Check if threshold reached this session
        reached_threshold_this_session = new_rs >= rs_threshold

        # Track first time threshold is reached (matching comparison test criterion)
        if reached_threshold_this_session and not threshold_ever_reached:
            threshold_ever_reached = True
            first_threshold_session = session

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
            'reached_threshold': reached_threshold_this_session,
        })

        if is_verbose:
            print(f"Post-session state:")
            print(f"  RS: {new_rs:7.2f} (change: {rs_change:+7.2f})")
            print(f"  Bond: {new_bond:.4f} (change: {bond_change:+.4f})")
            print(f"  Gap to threshold: {rs_threshold - new_rs:7.2f}")

            if reached_threshold_this_session:
                if first_threshold_session == session:
                    print(f"  ✓ SUCCESS THRESHOLD REACHED FOR THE FIRST TIME!")
                else:
                    print(f"  ✓ Threshold still met (first reached at session {first_threshold_session})")
            else:
                if threshold_ever_reached:
                    print(f"  ⚠ Below threshold (previously reached at session {first_threshold_session})")
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

    # Success based on "ever reached" criterion (matching comparison test)
    if threshold_ever_reached:
        print(f"✓ SUCCESS: Client reached {success_threshold_percentile:.0%} percentile of their RS range")
        print(f"  Threshold first reached at session {first_threshold_session}")

        # Also report final state for trajectory understanding
        if final_rs >= rs_threshold:
            print(f"  Final state: Above threshold (RS = {final_rs:.2f})")
        else:
            decline = rs_threshold - final_rs
            print(f"  Final state: Below threshold (RS = {final_rs:.2f}, declined by {decline:.2f})")
            print(f"  Note: Despite later decline, therapy successfully reached target at session {first_threshold_session}")
    else:
        shortfall = rs_threshold - final_rs
        closest_rs = max([h['rs_after'] for h in session_history]) if session_history else final_rs
        print(f"✗ FAILURE: Client never reached threshold")
        print(f"  Final RS: {final_rs:.2f}")
        print(f"  Fell short by: {shortfall:7.2f}")
        if closest_rs > final_rs:
            print(f"  Closest approach: {closest_rs:.2f} (still {rs_threshold - closest_rs:.2f} below threshold)")
    print()

    print("BOND DEVELOPMENT")
    print("-" * 100)
    print(f"Initial bond: {initial_bond:.4f}")
    print(f"Final bond:   {final_bond:.4f}")
    print(f"Total change: {total_bond_change:+.4f}")
    print(f"Average per session: {total_bond_change/session:+.4f}")
    print()

    # Parataxic distortion summary
    if enable_parataxic and hasattr(client, 'get_parataxic_stats'):
        print("PARATAXIC DISTORTION SUMMARY")
        print("-" * 100)
        pstats = client.get_parataxic_stats()
        print(f"Total interactions: {pstats['total_interactions']}")
        print(f"Overall misperception rate: {pstats['overall_misperception_rate']:.1%}")
        print(f"Stage 1 override rate: {pstats['stage1_override_rate']:.1%} (history-based changes)")
        print(f"Mean computed accuracy: {pstats['mean_computed_accuracy']:.3f}")
        print(f"Baseline path successes: {pstats['baseline_correct_count']} times")
        print()

    # V2 Therapist strategy summary
    print("THERAPIST STRATEGY SUMMARY (OmniscientStrategicTherapist V2)")
    print("-" * 100)
    phase_summary = therapist.get_phase_summary()
    print(f"Total sessions: {phase_summary['total_sessions']}")
    print(f"Current phase: {phase_summary['current_phase']}")
    print()
    print("Time in each phase:")
    for phase, count in phase_summary['phase_counts'].items():
        pct = count / phase_summary['total_sessions'] * 100 if phase_summary['total_sessions'] > 0 else 0
        print(f"  {phase:30s}: {count:3d} sessions ({pct:5.1f}%)")
    print()

    seeding_summary = therapist.get_seeding_summary()
    if seeding_summary['total_seeding_sessions'] > 0:
        print("Seeding activity:")
        print(f"  Total seeding sessions: {seeding_summary['total_seeding_sessions']}")
        print(f"  Actions seeded: {seeding_summary['seeding_actions']}")
        print()

    feedback_summary = therapist.get_feedback_monitoring_summary()
    if feedback_summary:
        print("Feedback monitoring:")
        print(f"  Total seeding sessions: {feedback_summary.get('total_seeding_sessions', 0)}")
        print(f"  Recalculations: {feedback_summary.get('recalculations', 0)}")
        print(f"  Aborts: {feedback_summary.get('aborts', 0)}")
        print(f"  Avg success rate: {feedback_summary.get('avg_success_rate', 0):.1%}")
        print(f"  Competitor boost rate: {feedback_summary.get('competitor_boost_rate', 0):.1%}")
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
        '--enable-parataxic',
        action='store_true',
        dest='enable_parataxic',
        help='Enable parataxic distortion (Sullivan\'s concept of imperfect client perception)'
    )

    parser.add_argument(
        '--enable-perception',
        action='store_true',
        dest='enable_parataxic',
        help='[DEPRECATED] Use --enable-parataxic instead'
    )

    parser.add_argument(
        '--baseline-accuracy',
        type=float,
        default=0.5549619551286054,
        help='Baseline parataxic distortion accuracy (default: 0.555 from optimized trial)'
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
        default=0.9358603798762596,
        help='Success threshold percentile (0.0-1.0, default: 0.936 from optimized trial)'
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
        default=11.847676335038303,
        help='Bond alpha (sigmoid steepness parameter, default: 11.85 from optimized trial)'
    )
    
    parser.add_argument(
        '--bond-offset', '-bo',
        type=float,
        default=0.624462461360537,
        help='Bond offset for sigmoid inflection point (0.0-1.0)'
    )

    # V2 Therapist arguments
    parser.add_argument(
        '--perception-window',
        type=int,
        default=10,
        help='Memory window size for parataxic distortion (V2 therapist)'
    )

    parser.add_argument(
        '--seeding-benefit-scaling',
        type=float,
        default=1.8658722646107764,
        help='Scaling factor for expected seeding benefit (0.1-2.0)'
    )

    parser.add_argument(
        '--skip-seeding-accuracy-threshold',
        type=float,
        default=0.814677493978211,
        help='Skip seeding if accuracy above this (0.75-0.95)'
    )

    parser.add_argument(
        '--quick-seed-actions-threshold',
        type=int,
        default=1,
        help='"Just do it" if actions_needed <= this (1-5)'
    )

    parser.add_argument(
        '--abort-consecutive-failures-threshold',
        type=int,
        default=4,
        help='Abort after this many consecutive failures (4-9)'
    )

    parser.add_argument(
        '--max-sessions', '-s',
        type=int,
        default=1940,
        help='Maximum number of therapy sessions (default: 1940 from optimized trial)'
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
            'enable_parataxic': args.enable_parataxic,
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
            # V2 Therapist parameters
            'perception_window': args.perception_window,
            'seeding_benefit_scaling': args.seeding_benefit_scaling,
            'skip_seeding_accuracy_threshold': args.skip_seeding_accuracy_threshold,
            'quick_seed_actions_threshold': args.quick_seed_actions_threshold,
            'abort_consecutive_failures_threshold': args.abort_consecutive_failures_threshold,
        }
        
        verbose_session_trace(**kwargs)
        
        print("\n" + "=" * 100)
        print("VERBOSE SESSION TRACE COMPLETE")
        print("=" * 100)
