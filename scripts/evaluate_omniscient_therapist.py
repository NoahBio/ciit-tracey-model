"""Evaluation script for OmniscientStrategicTherapist.

Compares the omniscient strategic therapist against the complementary baseline
across multiple random seeds and provides statistical analysis.

Usage:
    python scripts/evaluate_omniscient_therapist.py --n-seeds 100 --pattern cold_stuck

For baseline comparison with optimized defaults (from top >20% advantage Optuna trials):
    python scripts/evaluate_omniscient_therapist.py \
        --n-seeds 100 \
        --pattern cold_stuck \
        --enable-parataxic \
        --mechanism frequency_amplifier
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
import argparse

from src.agents.client_agents import (
    with_parataxic,
    BondOnlyClient,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
    BondWeightedConditionalAmplifier,
    BondWeightedFrequencyAmplifier,
    BaseClientAgent,
)
from src.agents.therapist_agents import (
    OmniscientStrategicTherapist,
    OmniscientStrategicTherapistV1,
    OmniscientStrategicTherapistV2,
)
from src import config
from src.config import (
    sample_u_matrix,
    OCTANTS,
    calculate_success_threshold,
    PARATAXIC_WINDOW,
    PARATAXIC_BASELINE_ACCURACY,
)


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    seed: int
    therapist_type: str  # 'omniscient' or 'complementary'
    success: bool
    dropped_out: bool
    total_sessions: int
    first_success_session: Optional[int]

    # Final state
    final_rs: float
    final_bond: float
    initial_rs: float
    initial_bond: float

    # Trajectory data
    rs_trajectory: List[float]
    bond_trajectory: List[float]

    # Threshold info
    rs_threshold: float
    closest_rs: float
    gap_to_threshold: float

    # Action distribution
    client_actions: List[int]
    therapist_actions: List[int]

    # Perception stats (if enabled)
    perception_stats: Optional[Dict[str, Any]] = None

    # Omniscient therapist specific
    phase_summary: Optional[Dict[str, Any]] = None
    seeding_summary: Optional[Dict[str, Any]] = None


def always_complement(client_action: int) -> int:
    """Simple complementary therapist strategy."""
    complement_map = {
        0: 4, 1: 3, 2: 2, 3: 1,
        4: 0, 5: 7, 6: 6, 7: 5,
    }
    return complement_map[client_action]


def run_single_simulation(
    seed: int,
    therapist_type: str,  # 'omniscient' or 'complementary'
    mechanism: str,
    initial_memory_pattern: str,
    success_threshold_percentile: float,
    enable_parataxic: bool = False,
    baseline_accuracy: float = 0.4477,
    perception_window: int = 17,
    max_sessions: int = 132,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 13.4426,
    bond_offset: float = 0.5122,
    recency_weighting_factor: float = 2.0,
    seeding_benefit_scaling: float = 1.4928,
    skip_seeding_accuracy_threshold: float = 0.8924,
    quick_seed_actions_threshold: int = 2,
    abort_consecutive_failures_threshold: int = 4,
    therapist_version: str = 'v2',  # 'v1' or 'v2'
) -> SimulationResult:
    """Run a single therapy simulation with specified therapist type."""

    # Setup
    rng = np.random.RandomState(seed)
    u_matrix = sample_u_matrix(random_state=seed)

    # Generate initial memory
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type=initial_memory_pattern,
        n_interactions=50,
        random_state=seed,
    )

    # Set global bond parameters
    config.BOND_ALPHA = bond_alpha
    config.BOND_OFFSET = bond_offset
    config.RECENCY_WEIGHTING_FACTOR = recency_weighting_factor

    # Create client
    client_kwargs = {
        'u_matrix': u_matrix,
        'entropy': entropy,
        'initial_memory': initial_memory,
        'random_state': seed,
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

    client = ClientClass(**client_kwargs)

    # Calculate RS threshold
    rs_threshold = calculate_success_threshold(u_matrix, success_threshold_percentile)

    # Store success_threshold on client for omniscient therapist
    client.success_threshold = rs_threshold

    # Create therapist
    omniscient_therapist = None
    if therapist_type == 'omniscient':
        # Select therapist version
        TherapistClass = OmniscientStrategicTherapistV2 if therapist_version == 'v2' else OmniscientStrategicTherapistV1
        omniscient_therapist = TherapistClass(
            client_ref=client,
            perception_window=perception_window,
            baseline_accuracy=baseline_accuracy,
            seeding_benefit_scaling=seeding_benefit_scaling,
            skip_seeding_accuracy_threshold=skip_seeding_accuracy_threshold,
            quick_seed_actions_threshold=quick_seed_actions_threshold,
            abort_consecutive_failures_threshold=abort_consecutive_failures_threshold,
        )

    # Track initial state
    initial_rs = client.relationship_satisfaction
    initial_bond = client.bond

    # Track trajectories
    rs_trajectory = [initial_rs]
    bond_trajectory = [initial_bond]

    # Track actions
    client_actions = []
    therapist_actions = []

    # Track success
    threshold_ever_reached = False
    first_threshold_session = None
    closest_rs = initial_rs

    # Run sessions
    session = 0
    dropped_out = False

    for session in range(1, max_sessions + 1):
        # Client selects action
        client_action = client.select_action()

        # Therapist responds
        if therapist_type == 'omniscient' and omniscient_therapist is not None:
            therapist_action, _ = omniscient_therapist.decide_action(client_action, session)
        else:
            therapist_action = always_complement(client_action)

        # Record actions
        client_actions.append(client_action)
        therapist_actions.append(therapist_action)

        # Update client memory
        client.update_memory(client_action, therapist_action)

        # Process feedback (only for v2 omniscient therapist)
        if therapist_type == 'omniscient' and omniscient_therapist is not None:
            # v2 has feedback monitoring capability
            feedback_method = getattr(omniscient_therapist, 'process_feedback_after_memory_update', None)
            if feedback_method is not None:
                feedback_method(session, client_action)

        # Get new state
        new_rs = client.relationship_satisfaction
        new_bond = client.bond

        # Track trajectories
        rs_trajectory.append(new_rs)
        bond_trajectory.append(new_bond)

        # Update closest RS
        if new_rs > closest_rs:
            closest_rs = new_rs

        # Check if threshold reached
        if new_rs >= rs_threshold and not threshold_ever_reached:
            threshold_ever_reached = True
            first_threshold_session = session

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            break

    # Get final state
    final_rs = client.relationship_satisfaction
    final_bond = client.bond

    # Get parataxic stats if enabled
    perception_stats = None
    if enable_parataxic and hasattr(client, 'get_parataxic_stats'):
        perception_stats = client.get_parataxic_stats()

    # Get omniscient therapist stats
    phase_summary = None
    seeding_summary = None
    if omniscient_therapist is not None:
        phase_summary = omniscient_therapist.get_phase_summary()
        seeding_summary = omniscient_therapist.get_seeding_summary()

    # Calculate gap to threshold
    gap_to_threshold = rs_threshold - closest_rs

    return SimulationResult(
        seed=seed,
        therapist_type=therapist_type,
        success=threshold_ever_reached,
        dropped_out=dropped_out,
        total_sessions=session,
        first_success_session=first_threshold_session,
        final_rs=final_rs,
        final_bond=final_bond,
        initial_rs=initial_rs,
        initial_bond=initial_bond,
        rs_trajectory=rs_trajectory,
        bond_trajectory=bond_trajectory,
        rs_threshold=rs_threshold,
        closest_rs=closest_rs,
        gap_to_threshold=gap_to_threshold,
        client_actions=client_actions,
        therapist_actions=therapist_actions,
        perception_stats=perception_stats,
        phase_summary=phase_summary,
        seeding_summary=seeding_summary,
    )


def compute_statistics(results: List[SimulationResult]) -> Dict[str, Any]:
    """Compute statistics for a set of simulation results."""
    n_runs = len(results)

    # Success metrics
    n_success = sum(1 for r in results if r.success)
    n_dropout = sum(1 for r in results if r.dropped_out)
    n_failure = n_runs - n_success

    success_rate = n_success / n_runs if n_runs > 0 else 0.0
    dropout_rate = n_dropout / n_runs if n_runs > 0 else 0.0

    # Session timing for successful runs
    success_sessions = [r.first_success_session for r in results
                       if r.success and r.first_success_session is not None]

    success_sessions_mean = float(np.mean(success_sessions)) if success_sessions else None
    success_sessions_median = float(np.median(success_sessions)) if success_sessions else None
    success_sessions_std = float(np.std(success_sessions)) if success_sessions else None

    # Final RS stats
    final_rs_values = [r.final_rs for r in results]
    final_rs_mean = float(np.mean(final_rs_values))
    final_rs_std = float(np.std(final_rs_values))

    # Final bond stats
    final_bond_values = [r.final_bond for r in results]
    final_bond_mean = float(np.mean(final_bond_values))
    final_bond_std = float(np.std(final_bond_values))

    # Phase summary (for omniscient therapist)
    phase_stats = None
    if results[0].phase_summary is not None:
        phase_counts = {
            'relationship_building': [],
            'ladder_climbing': [],
            'consolidation': [],
        }
        for r in results:
            if r.phase_summary:
                for phase, count in r.phase_summary['phase_counts'].items():
                    phase_counts[phase].append(count)

        phase_stats = {
            phase: {
                'mean': float(np.mean(counts)) if counts else 0,
                'std': float(np.std(counts)) if counts else 0,
            }
            for phase, counts in phase_counts.items()
        }

    return {
        'n_runs': n_runs,
        'success_rate': success_rate,
        'dropout_rate': dropout_rate,
        'n_success': n_success,
        'n_dropout': n_dropout,
        'n_failure': n_failure,
        'success_sessions_mean': success_sessions_mean,
        'success_sessions_median': success_sessions_median,
        'success_sessions_std': success_sessions_std,
        'final_rs_mean': final_rs_mean,
        'final_rs_std': final_rs_std,
        'final_bond_mean': final_bond_mean,
        'final_bond_std': final_bond_std,
        'phase_stats': phase_stats,
    }


def display_comparison(
    omniscient_stats: Dict[str, Any],
    complementary_stats: Dict[str, Any],
    config_summary: Dict[str, Any],
):
    """Display comparison between omniscient and complementary therapists."""

    print("=" * 100)
    print("OMNISCIENT STRATEGIC THERAPIST EVALUATION")
    print("=" * 100)
    print()

    # Configuration
    print("CONFIGURATION")
    print("-" * 100)
    for key, value in config_summary.items():
        print(f"{key:30s}: {value}")
    print()

    # Comparison table
    print("=" * 100)
    print("COMPARISON: Omniscient vs Complementary Baseline")
    print("=" * 100)
    print()

    print(f"{'Metric':<40} {'Omniscient':>20} {'Complementary':>20} {'Difference':>15}")
    print("-" * 100)

    # Success rate
    omni_sr = omniscient_stats['success_rate']
    comp_sr = complementary_stats['success_rate']
    diff_sr = omni_sr - comp_sr
    print(f"{'Success Rate':<40} {omni_sr:>19.1%} {comp_sr:>19.1%} {diff_sr:>+14.1%}")

    # Dropout rate
    omni_dr = omniscient_stats['dropout_rate']
    comp_dr = complementary_stats['dropout_rate']
    diff_dr = omni_dr - comp_dr
    print(f"{'Dropout Rate':<40} {omni_dr:>19.1%} {comp_dr:>19.1%} {diff_dr:>+14.1%}")

    # Sessions to success (if available)
    omni_sess = omniscient_stats['success_sessions_mean']
    comp_sess = complementary_stats['success_sessions_mean']
    if omni_sess and comp_sess:
        diff_sess = omni_sess - comp_sess
        print(f"{'Mean Sessions to Success':<40} {omni_sess:>20.1f} {comp_sess:>20.1f} {diff_sess:>+15.1f}")
    else:
        print(f"{'Mean Sessions to Success':<40} {'N/A':>20} {'N/A':>20} {'N/A':>15}")

    # Final RS
    omni_rs = omniscient_stats['final_rs_mean']
    comp_rs = complementary_stats['final_rs_mean']
    diff_rs = omni_rs - comp_rs
    print(f"{'Final RS (mean)':<40} {omni_rs:>20.2f} {comp_rs:>20.2f} {diff_rs:>+15.2f}")

    # Final bond
    omni_bond = omniscient_stats['final_bond_mean']
    comp_bond = complementary_stats['final_bond_mean']
    diff_bond = omni_bond - comp_bond
    print(f"{'Final Bond (mean)':<40} {omni_bond:>20.4f} {comp_bond:>20.4f} {diff_bond:>+15.4f}")

    print()

    # Phase breakdown (omniscient only)
    if omniscient_stats['phase_stats']:
        print("OMNISCIENT THERAPIST PHASE BREAKDOWN")
        print("-" * 100)
        for phase, stats in omniscient_stats['phase_stats'].items():
            print(f"{phase:<30}: {stats['mean']:>10.1f} sessions (std: {stats['std']:.1f})")
        print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    improvement = omni_sr - comp_sr
    if improvement > 0:
        print(f"Omniscient therapist IMPROVED success rate by {improvement:.1%} "
              f"({comp_sr:.1%} → {omni_sr:.1%})")
    elif improvement < 0:
        print(f"Omniscient therapist DECREASED success rate by {-improvement:.1%} "
              f"({comp_sr:.1%} → {omni_sr:.1%})")
    else:
        print(f"No change in success rate ({omni_sr:.1%})")

    if omni_sess and comp_sess:
        sess_improvement = comp_sess - omni_sess  # Lower is better
        if sess_improvement > 0:
            print(f"Omniscient therapist achieved success {sess_improvement:.1f} sessions FASTER")
        elif sess_improvement < 0:
            print(f"Omniscient therapist was {-sess_improvement:.1f} sessions SLOWER")

    print()


def run_evaluation(
    n_seeds: int,
    mechanism: str = 'frequency_amplifier',
    initial_memory_pattern: str = 'cold_stuck',
    success_threshold_percentile: float = 0.9755,
    enable_parataxic: bool = True,
    baseline_accuracy: float = 0.4477,
    perception_window: int = 17,
    max_sessions: int = 132,
    entropy: float = 0.1,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 13.4426,
    bond_offset: float = 0.5122,
    recency_weighting_factor: float = 2.0,
    verbose: bool = True,
    therapist_version: str = 'v2',
):
    """Run evaluation comparing omniscient vs complementary therapist."""

    config_summary = {
        'n_seeds': n_seeds,
        'mechanism': mechanism,
        'initial_memory_pattern': initial_memory_pattern,
        'success_threshold_percentile': success_threshold_percentile,
        'therapist_version': therapist_version,
        'enable_parataxic': enable_parataxic,
        'baseline_accuracy': baseline_accuracy if enable_parataxic else 'N/A',
        'perception_window': perception_window if enable_parataxic else 'N/A',
        'max_sessions': max_sessions,
        'entropy': entropy,
        'history_weight': history_weight if 'amplifier' in mechanism else 'N/A',
        'bond_power': bond_power if 'bond_weighted' in mechanism else 'N/A',
        'bond_alpha': bond_alpha,
        'bond_offset': bond_offset,
        'recency_weighting_factor': recency_weighting_factor,
    }

    if verbose:
        print("=" * 100)
        print(f"Running evaluation with {n_seeds} seeds...")
        print("=" * 100)
        print()

    # Run omniscient therapist
    omniscient_results = []
    if verbose:
        print("Running OMNISCIENT therapist simulations...")
    for seed in range(n_seeds):
        if verbose and (seed + 1) % 10 == 0:
            print(f"  Seed {seed + 1}/{n_seeds}...")
        result = run_single_simulation(
            seed=seed,
            therapist_type='omniscient',
            mechanism=mechanism,
            initial_memory_pattern=initial_memory_pattern,
            success_threshold_percentile=success_threshold_percentile,
            enable_parataxic=enable_parataxic,
            baseline_accuracy=baseline_accuracy,
            perception_window=perception_window,
            max_sessions=max_sessions,
            entropy=entropy,
            history_weight=history_weight,
            bond_power=bond_power,
            bond_alpha=bond_alpha,
            bond_offset=bond_offset,
            recency_weighting_factor=recency_weighting_factor,
            therapist_version=therapist_version,
        )
        omniscient_results.append(result)

    # Run complementary therapist (baseline)
    complementary_results = []
    if verbose:
        print("Running COMPLEMENTARY therapist simulations (baseline)...")
    for seed in range(n_seeds):
        if verbose and (seed + 1) % 10 == 0:
            print(f"  Seed {seed + 1}/{n_seeds}...")
        result = run_single_simulation(
            seed=seed,
            therapist_type='complementary',
            mechanism=mechanism,
            initial_memory_pattern=initial_memory_pattern,
            success_threshold_percentile=success_threshold_percentile,
            enable_parataxic=enable_parataxic,
            baseline_accuracy=baseline_accuracy,
            perception_window=perception_window,
            max_sessions=max_sessions,
            entropy=entropy,
            history_weight=history_weight,
            bond_power=bond_power,
            bond_alpha=bond_alpha,
            bond_offset=bond_offset,
            recency_weighting_factor=recency_weighting_factor,
            therapist_version=therapist_version,
        )
        complementary_results.append(result)

    # Compute statistics
    omniscient_stats = compute_statistics(omniscient_results)
    complementary_stats = compute_statistics(complementary_results)

    # Display comparison
    if verbose:
        print()
        display_comparison(omniscient_stats, complementary_stats, config_summary)

    return omniscient_stats, complementary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OmniscientStrategicTherapist against complementary baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--n-seeds', '-n',
        type=int,
        default=100,
        help='Number of random seeds to run'
    )

    parser.add_argument(
        '--mechanism', '-m',
        type=str,
        default='frequency_amplifier',
        choices=[
            'bond_only',
            'frequency_amplifier',
            'conditional_amplifier',
            'bond_weighted_frequency_amplifier',
            'bond_weighted_conditional_amplifier'
        ],
        help='Client expectation mechanism'
    )

    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='cold_stuck',
        choices=['cold_warm', 'complementary_perfect', 'conflictual',
                 'mixed_random', 'cold_stuck', 'dominant_stuck', 'submissive_stuck'],
        help='Initial memory pattern'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.9755,
        help='Success threshold percentile (0.0-1.0, default: 0.9755 from top >20%% advantage trials)'
    )

    parser.add_argument(
        '--enable-parataxic',
        action='store_true',
        default=False,
        help='Enable parataxic distortion'
    )

    parser.add_argument(
        '--baseline-accuracy',
        type=float,
        default=0.4477,
        help='Baseline accuracy for parataxic distortion (default: 0.4477 from top >20%% advantage trials)'
    )

    parser.add_argument(
        '--perception-window',
        type=int,
        default=17,
        help='Perception window size for parataxic distortion (default: 17 from top >20%% advantage trials)'
    )

    parser.add_argument(
        '--max-sessions', '-s',
        type=int,
        default=132,
        help='Maximum number of therapy sessions (default: 132 from top >20%% advantage trials)'
    )

    parser.add_argument(
        '--entropy', '-e',
        type=float,
        default=0.1,
        help='Client entropy'
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
        default=13.4426,
        help='Bond alpha (sigmoid steepness, default: 13.4426 from top >20%% advantage trials)'
    )

    parser.add_argument(
        '--bond-offset', '-bo',
        type=float,
        default=0.5122,
        help='Bond offset for sigmoid (default: 0.5122 from top >20%% advantage trials)'
    )

    parser.add_argument(
        '--recency-weighting-factor', '-rwf',
        type=float,
        default=2.0,
        help='Recency weighting factor (newest:oldest weight ratio, must be >= 1.0, default: 2.0)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    parser.add_argument(
        '--therapist-version',
        type=str,
        default='v2',
        choices=['v1', 'v2'],
        help='Omniscient therapist version (v1=original, v2=with feedback monitoring)'
    )

    args = parser.parse_args()

    enable_parataxic = args.enable_parataxic

    run_evaluation(
        n_seeds=args.n_seeds,
        mechanism=args.mechanism,
        initial_memory_pattern=args.pattern,
        success_threshold_percentile=args.threshold,
        enable_parataxic=enable_parataxic,
        baseline_accuracy=args.baseline_accuracy,
        perception_window=args.perception_window,
        max_sessions=args.max_sessions,
        entropy=args.entropy,
        history_weight=args.history_weight,
        bond_power=args.bond_power,
        bond_alpha=args.bond_alpha,
        bond_offset=args.bond_offset,
        recency_weighting_factor=args.recency_weighting_factor,
        verbose=not args.quiet,
        therapist_version=args.therapist_version,
    )

    print()
    print("=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)
