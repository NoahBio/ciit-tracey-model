"""
Compare BondOnlyClient performance with and without perception mixin.

Tests the effect of imperfect perception on BondOnlyClient with always_complement
therapist strategy across multiple initial memory patterns.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from src.agents.client_agents import BondOnlyClient, BaseClientAgent
from src.agents.client_agents.perceptual_distortion import with_perception
from src.config import (
    sample_u_matrix,
    calculate_success_threshold,
    OCTANTS,
    PERCEPTION_BASELINE_ACCURACY,
    PERCEPTION_ADJACENCY_NOISE,
    PERCEPTION_WINDOW,
    CLIENT_ENTROPY_MEAN,
    MAX_SESSIONS,
)


def display_configuration():
    """Display standard configuration values at startup."""
    print("=" * 100)
    print("BONDONLY CLIENT PERCEPTION COMPARISON")
    print("=" * 100)
    print()
    print("CONFIGURATION")
    print("-" * 100)
    print("Standard Config Values:")
    print(f"  PERCEPTION_BASELINE_ACCURACY: {PERCEPTION_BASELINE_ACCURACY:.2f}")
    print(f"  PERCEPTION_ADJACENCY_NOISE: {PERCEPTION_ADJACENCY_NOISE:.2f}")
    print(f"  PERCEPTION_WINDOW: {PERCEPTION_WINDOW}")
    print(f"  CLIENT_ENTROPY: {CLIENT_ENTROPY_MEAN:.2f}")
    print(f"  MAX_SESSIONS: {MAX_SESSIONS}")
    print(f"  SUCCESS_THRESHOLD: 80th percentile")
    print()


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


def run_single_trial(
    enable_perception: bool,
    pattern: str,
    entropy: float,
    threshold_percentile: float,
    random_state: int,
) -> Dict:
    """
    Run a single therapy episode.

    Parameters
    ----------
    enable_perception : bool
        Whether to use perception mixin
    pattern : str
        Initial memory pattern name
    entropy : float
        Client exploration parameter
    threshold_percentile : float
        Success threshold percentile (0-1)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Trial results including success, sessions, RS, bond, perception stats
    """
    # Setup
    rng = np.random.RandomState(random_state)
    u_matrix = sample_u_matrix(random_state=rng)

    # Map legacy pattern names and generate memory
    pattern_type = PATTERN_ALIASES.get(pattern, pattern)
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type=pattern_type,
        n_interactions=50,
        random_state=random_state,
    )
    rs_threshold = calculate_success_threshold(u_matrix, threshold_percentile)

    # Create client
    if enable_perception:
        PerceptualBondOnly = with_perception(BondOnlyClient)
        client = PerceptualBondOnly(
            u_matrix=u_matrix,
            entropy=entropy,
            initial_memory=initial_memory,
            baseline_accuracy=PERCEPTION_BASELINE_ACCURACY,
            enable_perception=True,
            random_state=rng,
        )
    else:
        client = BondOnlyClient(
            u_matrix=u_matrix,
            entropy=entropy,
            initial_memory=initial_memory,
            random_state=rng,
        )

    # Track metrics
    success = False
    dropped_out = False
    action_counts = Counter()

    # Run therapy sessions
    for session in range(1, MAX_SESSIONS + 1):
        # Select action
        client_action = client.select_action()
        action_counts[client_action] += 1

        # Therapist responds
        therapist_action = always_complement(client_action)

        # Update memory (perception happens here if enabled)
        client.update_memory(client_action, therapist_action)

        # Check success
        if client.relationship_satisfaction >= rs_threshold:
            success = True
            break

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            break

    # Collect results
    results = {
        'success': success,
        'sessions': session,
        'final_rs': client.relationship_satisfaction,
        'final_bond': client.bond,
        'dropped_out': dropped_out,
        'action_counts': action_counts,
        'perception_stats': None,
    }

    # Get perception statistics if available
    if hasattr(client, 'get_perception_stats'):
        results['perception_stats'] = client.get_perception_stats()  # type: ignore[attr-defined]

    return results


def run_condition(
    enable_perception: bool,
    pattern: str,
    n_trials: int,
    entropy: float,
    threshold_percentile: float,
) -> Dict:
    """
    Run multiple trials for one condition.

    Parameters
    ----------
    enable_perception : bool
        Whether to use perception mixin
    pattern : str
        Initial memory pattern name
    n_trials : int
        Number of trials to run
    entropy : float
        Client exploration parameter
    threshold_percentile : float
        Success threshold percentile (0-1)

    Returns
    -------
    dict
        Aggregated results across all trials
    """
    condition_name = "Imperfect Perception" if enable_perception else "Perfect Perception"

    # Run trials with progress bar
    all_results = []
    for trial in tqdm(range(n_trials), desc=f"  {condition_name}", leave=False):
        result = run_single_trial(
            enable_perception=enable_perception,
            pattern=pattern,
            entropy=entropy,
            threshold_percentile=threshold_percentile,
            random_state=trial,
        )
        all_results.append(result)

    # Aggregate results
    successes = [r for r in all_results if r['success']]
    dropouts = [r for r in all_results if r['dropped_out']]

    # Calculate statistics
    success_rate = len(successes) / n_trials if n_trials > 0 else 0.0
    dropout_rate = len(dropouts) / n_trials if n_trials > 0 else 0.0

    # Mean sessions for successful trials only
    if successes:
        sessions_success = [r['sessions'] for r in successes]
        mean_sessions_success = np.mean(sessions_success)
        std_sessions_success = np.std(sessions_success)
    else:
        mean_sessions_success = 0.0
        std_sessions_success = 0.0

    # Mean final RS and bond across all trials
    final_rs_all = [r['final_rs'] for r in all_results]
    final_bond_all = [r['final_bond'] for r in all_results]

    # Aggregate action counts
    total_action_counts = Counter()
    for r in all_results:
        total_action_counts.update(r['action_counts'])

    # Aggregate perception stats if available
    perception_stats_agg = None
    if enable_perception and all_results[0]['perception_stats'] is not None:
        perception_stats_list = [r['perception_stats'] for r in all_results]
        perception_stats_agg = {
            'overall_misperception_rate': np.mean([p['overall_misperception_rate'] for p in perception_stats_list]),
            'overall_misperception_rate_std': np.std([p['overall_misperception_rate'] for p in perception_stats_list]),
            'stage1_override_rate': np.mean([p['stage1_override_rate'] for p in perception_stats_list]),
            'stage1_override_rate_std': np.std([p['stage1_override_rate'] for p in perception_stats_list]),
            'stage2_shift_rate': np.mean([p['stage2_shift_rate'] for p in perception_stats_list]),
            'stage2_shift_rate_std': np.std([p['stage2_shift_rate'] for p in perception_stats_list]),
            'mean_computed_accuracy': np.mean([p['mean_computed_accuracy'] for p in perception_stats_list]),
            'mean_computed_accuracy_std': np.std([p['mean_computed_accuracy'] for p in perception_stats_list]),
            'baseline_correct_count': np.mean([p['baseline_correct_count'] for p in perception_stats_list]),
            'baseline_correct_count_std': np.std([p['baseline_correct_count'] for p in perception_stats_list]),
        }

    return {
        'success_rate': success_rate,
        'dropout_rate': dropout_rate,
        'mean_sessions_success': mean_sessions_success,
        'std_sessions_success': std_sessions_success,
        'mean_final_rs': np.mean(final_rs_all),
        'std_final_rs': np.std(final_rs_all),
        'mean_final_bond': np.mean(final_bond_all),
        'std_final_bond': np.std(final_bond_all),
        'action_counts': total_action_counts,
        'perception_stats': perception_stats_agg,
    }


def compare_conditions(
    patterns: List[str],
    n_trials_per_pattern: int,
    entropy: float,
    threshold_percentile: float,
):
    """
    Main comparison function - runs both conditions for each pattern.

    Parameters
    ----------
    patterns : list of str
        Memory patterns to test
    n_trials_per_pattern : int
        Number of trials per pattern per condition
    entropy : float
        Client exploration parameter
    threshold_percentile : float
        Success threshold percentile (0-1)
    """
    print("Test Parameters:")
    print(f"  Patterns: {patterns}")
    print(f"  Trials per condition: {n_trials_per_pattern}")
    print(f"  Total trials: {len(patterns) * 2 * n_trials_per_pattern}")
    print(f"  Entropy: {entropy:.2f}")
    print()

    print("=" * 100)
    print("RUNNING COMPARISONS")
    print("=" * 100)
    print()

    # Run comparisons for each pattern
    for pattern in patterns:
        print(f"Pattern: {pattern}")
        print("-" * 100)

        # Run both conditions
        perfect_results = run_condition(
            enable_perception=False,
            pattern=pattern,
            n_trials=n_trials_per_pattern,
            entropy=entropy,
            threshold_percentile=threshold_percentile,
        )

        imperfect_results = run_condition(
            enable_perception=True,
            pattern=pattern,
            n_trials=n_trials_per_pattern,
            entropy=entropy,
            threshold_percentile=threshold_percentile,
        )

        # Display results
        print()
        print("=" * 100)
        print(f"RESULTS: {pattern} Pattern")
        print("=" * 100)
        print()

        # Outcome metrics table
        print("OUTCOME METRICS")
        print("-" * 100)
        print(f"{'Metric':<30} {'Perfect':<20} {'Imperfect':<20} {'Difference':<20}")
        print(f"{'':30} {'Perception':<20} {'Perception':<20} {'(Imperfect - Perfect)':<20}")
        print("-" * 100)

        # Success rate
        perfect_success = perfect_results['success_rate'] * 100
        imperfect_success = imperfect_results['success_rate'] * 100
        diff_success = imperfect_success - perfect_success
        print(f"{'Success Rate':<30} {perfect_success:>6.1f}%{'':<13} {imperfect_success:>6.1f}%{'':<13} {diff_success:>+6.1f}%")

        # Mean sessions (success only)
        perfect_sess = perfect_results['mean_sessions_success']
        perfect_sess_std = perfect_results['std_sessions_success']
        imperfect_sess = imperfect_results['mean_sessions_success']
        imperfect_sess_std = imperfect_results['std_sessions_success']
        diff_sess = imperfect_sess - perfect_sess
        print(f"{'Mean Sessions (success)':<30} {perfect_sess:>6.1f} ± {perfect_sess_std:<6.1f}  {imperfect_sess:>6.1f} ± {imperfect_sess_std:<6.1f}  {diff_sess:>+6.1f}")

        # Mean final RS
        perfect_rs = perfect_results['mean_final_rs']
        perfect_rs_std = perfect_results['std_final_rs']
        imperfect_rs = imperfect_results['mean_final_rs']
        imperfect_rs_std = imperfect_results['std_final_rs']
        diff_rs = imperfect_rs - perfect_rs
        print(f"{'Mean Final RS':<30} {perfect_rs:>+6.1f} ± {perfect_rs_std:<6.1f}  {imperfect_rs:>+6.1f} ± {imperfect_rs_std:<6.1f}  {diff_rs:>+6.1f}")

        # Mean final bond
        perfect_bond = perfect_results['mean_final_bond']
        perfect_bond_std = perfect_results['std_final_bond']
        imperfect_bond = imperfect_results['mean_final_bond']
        imperfect_bond_std = imperfect_results['std_final_bond']
        diff_bond = imperfect_bond - perfect_bond
        print(f"{'Mean Final Bond':<30} {perfect_bond:>7.4f} ± {perfect_bond_std:<6.4f} {imperfect_bond:>7.4f} ± {imperfect_bond_std:<6.4f} {diff_bond:>+7.4f}")

        # Dropout rate
        perfect_dropout = perfect_results['dropout_rate'] * 100
        imperfect_dropout = imperfect_results['dropout_rate'] * 100
        diff_dropout = imperfect_dropout - perfect_dropout
        print(f"{'Dropout Rate':<30} {perfect_dropout:>6.1f}%{'':<13} {imperfect_dropout:>6.1f}%{'':<13} {diff_dropout:>+6.1f}%")
        print()

        # Action distributions
        print("ACTION DISTRIBUTION (Perfect Perception)")
        print("-" * 100)
        total_perfect = sum(perfect_results['action_counts'].values())
        for action in range(8):
            count = perfect_results['action_counts'].get(action, 0)
            pct = count / total_perfect * 100 if total_perfect > 0 else 0
            bar = '█' * int(pct / 2)  # 50 chars = 100%
            print(f"  {OCTANTS[action]:3s} ({action}): {count:5d} ({pct:5.1f}%) {bar}")
        print()

        print("ACTION DISTRIBUTION (Imperfect Perception)")
        print("-" * 100)
        total_imperfect = sum(imperfect_results['action_counts'].values())
        for action in range(8):
            count = imperfect_results['action_counts'].get(action, 0)
            pct = count / total_imperfect * 100 if total_imperfect > 0 else 0
            bar = '█' * int(pct / 2)  # 50 chars = 100%
            print(f"  {OCTANTS[action]:3s} ({action}): {count:5d} ({pct:5.1f}%) {bar}")
        print()

        # Perception statistics
        if imperfect_results['perception_stats'] is not None:
            print("PERCEPTION STATISTICS (Imperfect Perception Only)")
            print("-" * 100)
            pstats = imperfect_results['perception_stats']
            print(f"{'Metric':<40} {'Mean ± Std':<20}")
            print("-" * 100)
            print(f"{'Overall Misperception Rate':<40} {pstats['overall_misperception_rate']*100:>5.1f}% ± {pstats['overall_misperception_rate_std']*100:<4.1f}%")
            print(f"{'Stage 1 Override Rate':<40} {pstats['stage1_override_rate']*100:>5.1f}% ± {pstats['stage1_override_rate_std']*100:<4.1f}%")
            print(f"{'Stage 2 Shift Rate':<40} {pstats['stage2_shift_rate']*100:>5.1f}% ± {pstats['stage2_shift_rate_std']*100:<4.1f}%")
            print(f"{'Mean Computed Accuracy':<40} {pstats['mean_computed_accuracy']:>6.3f} ± {pstats['mean_computed_accuracy_std']:<5.3f}")
            print(f"{'Baseline Correct Count':<40} {pstats['baseline_correct_count']:>6.1f} ± {pstats['baseline_correct_count_std']:<5.1f}")
            print()

    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    # Display configuration
    display_configuration()

    # Run comparison
    compare_conditions(
        patterns=['cw_50_50', 'complementary_perfect'],
        n_trials_per_pattern=100,
        entropy=CLIENT_ENTROPY_MEAN,
        threshold_percentile=0.8,
    )
