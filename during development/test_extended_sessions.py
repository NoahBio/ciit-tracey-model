"""
Test whether reliable failures succeed with extended sessions (1000 vs 100).

Goal: Determine if failures are structural or just slow convergence.

This script tests the top 20-28 failure configurations from the overnight
parameter search to see if they would succeed given more time (1000 sessions
instead of the original 100).

For each configuration:
- Run 100 trials (seeds 0-99)
- Allow up to 1000 sessions per trial
- Track success at checkpoints: 100, 200, 500, 1000 sessions
- Early stopping: stop when threshold reached or client drops out
- Compare success rates across session lengths

Expected runtime: ~5-6 hours (20-28 configs × 100 trials × ~500 avg sessions)
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.client_agents import create_client
from src.agents.client_agents.base_client import BaseClientAgent
from src.config import calculate_success_threshold, sample_u_matrix, rs_to_bond


def always_complement(client_action: int) -> int:
    """Always-complementary therapist strategy."""
    complement_map = {
        0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5
    }
    return complement_map[client_action]


def run_extended_trial(
    config,
    random_seed,
    max_sessions=1000,
    checkpoints=(100, 200, 500, 1000),
):
    """
    Run a single trial with extended sessions and checkpoint tracking.

    Parameters
    ----------
    config : dict
        Parameter configuration
    random_seed : int
        Random seed for this trial
    max_sessions : int
        Maximum number of sessions (default: 1000)
    checkpoints : tuple
        Session counts at which to check success (default: 100, 200, 500, 1000)

    Returns
    -------
    dict
        Results including success at each checkpoint and session where threshold reached
    """
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

    # Calculate RS bounds and success threshold
    rs_min = float(u_matrix.min())
    rs_max = float(u_matrix.max())
    rs_threshold = calculate_success_threshold(u_matrix, config['success_threshold'])

    # Override bond parameters and calculate initial bond
    client.bond = rs_to_bond(
        client.relationship_satisfaction,
        rs_min,
        rs_max,
        alpha=config['bond_alpha'],
        offset=config['bond_offset']
    )

    # Track results at checkpoints
    checkpoint_results = {cp: {'success': False, 'rs': None} for cp in checkpoints}
    threshold_reached_at = None
    dropped_out = False
    dropped_out_at = None

    # Run sessions
    for session in range(1, max_sessions + 1):
        # Select client action
        client_action = client.select_action()

        # Get therapist response
        therapist_action = always_complement(client_action)

        # Update client
        client.update_memory(client_action, therapist_action)

        # Recalculate bond with custom parameters
        client.bond = rs_to_bond(
            client.relationship_satisfaction,
            rs_min,
            rs_max,
            alpha=config['bond_alpha'],
            offset=config['bond_offset']
        )

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            dropped_out_at = session
            break

        # Check if threshold reached (for early stopping)
        if threshold_reached_at is None and client.relationship_satisfaction >= rs_threshold:
            threshold_reached_at = session

        # Record checkpoint results
        for cp in checkpoints:
            if session == cp:
                checkpoint_results[cp]['rs'] = float(client.relationship_satisfaction)
                checkpoint_results[cp]['success'] = (
                    client.relationship_satisfaction >= rs_threshold and not dropped_out
                )

        # Early stopping: if threshold reached and we've passed all checkpoints before it
        if threshold_reached_at is not None and session > max(checkpoints):
            break

    # Fill in final results for checkpoints not reached (due to dropout or early success)
    final_rs = client.relationship_satisfaction
    final_success = final_rs >= rs_threshold and not dropped_out

    for cp in checkpoints:
        if checkpoint_results[cp]['rs'] is None:
            # This checkpoint wasn't reached (dropout or early success)
            checkpoint_results[cp]['rs'] = float(final_rs)
            checkpoint_results[cp]['success'] = final_success

    return {
        'checkpoint_results': checkpoint_results,
        'threshold_reached_at': threshold_reached_at,
        'dropped_out': dropped_out,
        'dropped_out_at': dropped_out_at,
        'final_rs': float(final_rs),
        'final_success': final_success,
        'sessions_run': session if dropped_out else threshold_reached_at if threshold_reached_at else max_sessions,
    }


def test_configuration_extended(config, n_trials=100, checkpoints=(100, 200, 500, 1000)):
    """
    Test a configuration with extended sessions across multiple trials.

    Parameters
    ----------
    config : dict
        Parameter configuration
    n_trials : int
        Number of trials to run (default: 100)
    checkpoints : tuple
        Session counts at which to check success

    Returns
    -------
    dict
        Aggregated results across all trials
    """
    all_trial_results = []

    for seed in range(n_trials):
        trial_result = run_extended_trial(
            config=config,
            random_seed=seed,
            max_sessions=1000,
            checkpoints=checkpoints,
        )
        all_trial_results.append(trial_result)

    # Aggregate results
    success_rates = {}
    mean_rs = {}

    for cp in checkpoints:
        successes = sum(1 for r in all_trial_results if r['checkpoint_results'][cp]['success'])
        success_rates[cp] = successes / n_trials
        mean_rs[cp] = np.mean([r['checkpoint_results'][cp]['rs'] for r in all_trial_results])

    # Calculate when threshold was reached
    threshold_reached_trials = [r for r in all_trial_results if r['threshold_reached_at'] is not None]
    mean_sessions_to_threshold = np.mean([r['threshold_reached_at'] for r in threshold_reached_trials]) if threshold_reached_trials else None

    # Dropout statistics
    dropout_count = sum(1 for r in all_trial_results if r['dropped_out'])
    dropout_rate = dropout_count / n_trials

    return {
        'config': config,
        'n_trials': n_trials,
        'success_rates': success_rates,
        'mean_rs': mean_rs,
        'threshold_reached_count': len(threshold_reached_trials),
        'mean_sessions_to_threshold': mean_sessions_to_threshold,
        'dropout_rate': dropout_rate,
        'all_trial_results': all_trial_results,
    }


def classify_failure_type(result):
    """
    Classify whether a failure is structural or slow convergence.

    Parameters
    ----------
    result : dict
        Results from test_configuration_extended

    Returns
    -------
    str
        'structural' if still fails at 1000 sessions, 'slow_convergence' if succeeds with more time
    """
    success_1000 = result['success_rates'][1000]
    success_100 = result['success_rates'][100]

    if success_1000 < 0.5:
        return 'structural'
    elif success_1000 >= 0.8:
        return 'slow_convergence'
    else:
        return 'partial_improvement'


def main():
    print("=" * 100)
    print("EXTENDED SESSIONS TEST: Do Failures Succeed Given More Time?")
    print("=" * 100)
    print()

    # Load failure configurations
    failures_file = Path(__file__).parent / "top_reliable_failures.json"

    if not failures_file.exists():
        print(f"ERROR: {failures_file} not found!")
        print("Please run analyze_reliable_failures.py first to generate this file.")
        return

    with open(failures_file, 'r') as f:
        failures_data = json.load(f)

    configs_to_test = failures_data['configurations']
    n_configs = len(configs_to_test)

    print(f"Loaded {n_configs} failure configurations from {failures_file.name}")
    print()

    # Testing parameters
    checkpoints = (100, 200, 500, 1000)
    n_trials = 100

    print("TESTING PARAMETERS:")
    print("-" * 100)
    print(f"  Configurations to test: {n_configs}")
    print(f"  Trials per config: {n_trials}")
    print(f"  Max sessions per trial: 1000")
    print(f"  Checkpoints: {checkpoints}")
    print(f"  Early stopping: when threshold reached")
    print()

    # Estimate runtime
    avg_sessions_per_trial = 500  # Conservative estimate with early stopping
    total_trials = n_configs * n_trials
    est_seconds = total_trials * avg_sessions_per_trial * 0.015  # ~15ms per session
    est_hours = est_seconds / 3600

    print(f"ESTIMATED RUNTIME: ~{est_hours:.1f} hours")
    print(f"  Total trials: {total_trials:,}")
    print(f"  Estimated avg sessions/trial: {avg_sessions_per_trial}")
    print()

    # Run tests
    print("=" * 100)
    print("RUNNING EXTENDED SESSIONS TESTS")
    print("=" * 100)
    print()

    all_results = []
    start_time = datetime.now()

    for i, config_data in enumerate(tqdm(configs_to_test, desc="Testing configurations")):
        config = config_data['config']
        original_success_rate = config_data['success_rate']

        result = test_configuration_extended(
            config=config,
            n_trials=n_trials,
            checkpoints=checkpoints,
        )

        # Add metadata
        result['rank'] = config_data['rank']
        result['original_success_rate'] = original_success_rate
        result['failure_type'] = classify_failure_type(result)

        all_results.append(result)

        # Progress update every 5 configs
        if (i + 1) % 5 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time_per_config = elapsed / (i + 1)
            remaining_configs = n_configs - (i + 1)
            est_remaining = remaining_configs * avg_time_per_config / 3600

            print(f"\n  Progress: {i+1}/{n_configs} configs")
            print(f"    Elapsed: {elapsed/3600:.2f} hours")
            print(f"    Est. remaining: {est_remaining:.2f} hours")

    total_time = (datetime.now() - start_time).total_seconds() / 3600

    # Analysis
    print()
    print("=" * 100)
    print("RESULTS ANALYSIS")
    print("=" * 100)
    print()

    print(f"Total runtime: {total_time:.2f} hours")
    print()

    # Success rate improvement
    print("SUCCESS RATE IMPROVEMENT:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Original':<12} {'@ 100':<10} {'@ 200':<10} {'@ 500':<10} {'@ 1000':<10} {'Type':<20}")
    print("-" * 100)

    for r in all_results:
        print(f"{r['rank']:<6} "
              f"{r['original_success_rate']*100:>6.1f}%      "
              f"{r['success_rates'][100]*100:>6.1f}%    "
              f"{r['success_rates'][200]*100:>6.1f}%    "
              f"{r['success_rates'][500]*100:>6.1f}%    "
              f"{r['success_rates'][1000]*100:>6.1f}%    "
              f"{r['failure_type']:<20}")

    print()

    # Classification summary
    failure_types = {}
    for r in all_results:
        ftype = r['failure_type']
        failure_types[ftype] = failure_types.get(ftype, 0) + 1

    print("FAILURE TYPE CLASSIFICATION:")
    print("-" * 100)
    for ftype, count in sorted(failure_types.items()):
        pct = count / len(all_results) * 100
        print(f"  {ftype:<25} {count:3d} / {len(all_results)} ({pct:5.1f}%)")
    print()

    # Overall statistics
    print("OVERALL STATISTICS:")
    print("-" * 100)
    for cp in checkpoints:
        mean_success = np.mean([r['success_rates'][cp] for r in all_results])
        print(f"  Mean success rate @ {cp:4d} sessions: {mean_success:6.1%}")
    print()

    # Mean sessions to threshold (for those that succeeded)
    successful_configs = [r for r in all_results if r['threshold_reached_count'] > 0]
    if successful_configs:
        overall_mean_sessions = np.mean([r['mean_sessions_to_threshold'] for r in successful_configs])
        print(f"Mean sessions to threshold (successful configs): {overall_mean_sessions:.1f}")
        print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).parent / f"extended_session_results_{timestamp}.json"

    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_configurations': n_configs,
            'n_trials_per_config': n_trials,
            'max_sessions': 1000,
            'checkpoints': checkpoints,
            'total_runtime_hours': total_time,
            'source_file': str(failures_file.name),
        },
        'results': [
            {
                'rank': r['rank'],
                'original_success_rate': r['original_success_rate'],
                'success_rates': {str(k): v for k, v in r['success_rates'].items()},
                'mean_rs': {str(k): v for k, v in r['mean_rs'].items()},
                'threshold_reached_count': r['threshold_reached_count'],
                'mean_sessions_to_threshold': r['mean_sessions_to_threshold'],
                'dropout_rate': r['dropout_rate'],
                'failure_type': r['failure_type'],
                'config': r['config'],
            }
            for r in all_results
        ],
        'summary': {
            'failure_types': failure_types,
            'mean_success_rates': {
                str(cp): float(np.mean([r['success_rates'][cp] for r in all_results]))
                for cp in checkpoints
            },
        },
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Generate visualization
    create_visualization(all_results, checkpoints)

    print("=" * 100)
    print("TESTING COMPLETE")
    print("=" * 100)


def create_visualization(all_results, checkpoints):
    """Create visualization of success rate improvements."""
    print()
    print("=" * 100)
    print("CREATING VISUALIZATIONS")
    print("=" * 100)
    print()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Extended Sessions Analysis: Do Failures Succeed With More Time?', fontsize=16)

    # 1. Success rate improvement by checkpoint
    ax = axes[0, 0]
    for r in all_results:
        cp_list = list(checkpoints)
        success_list = [r['success_rates'][cp] * 100 for cp in cp_list]
        color = 'red' if r['failure_type'] == 'structural' else 'orange' if r['failure_type'] == 'partial_improvement' else 'green'
        alpha = 0.3
        ax.plot(cp_list, success_list, color=color, alpha=alpha, linewidth=1)

    # Add mean line
    mean_success = [np.mean([r['success_rates'][cp] for r in all_results]) * 100 for cp in checkpoints]
    ax.plot(checkpoints, mean_success, 'b-', linewidth=3, label='Mean', marker='o', markersize=8)

    ax.set_xlabel('Sessions', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate vs Sessions (Each Line = 1 Config)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 100])

    # 2. Distribution of success rates at each checkpoint
    ax = axes[0, 1]
    positions = [1, 2, 3, 4]
    data = [[r['success_rates'][cp] * 100 for r in all_results] for cp in checkpoints]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{cp}' for cp in checkpoints])
    ax.set_xlabel('Sessions', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Distribution at Checkpoints', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # 3. Failure type classification
    ax = axes[1, 0]
    failure_types = {}
    for r in all_results:
        ftype = r['failure_type']
        failure_types[ftype] = failure_types.get(ftype, 0) + 1

    colors_map = {
        'structural': 'red',
        'partial_improvement': 'orange',
        'slow_convergence': 'green'
    }
    ftypes = list(failure_types.keys())
    counts = [failure_types[ft] for ft in ftypes]
    colors = [colors_map.get(ft, 'gray') for ft in ftypes]

    bars = ax.bar(range(len(ftypes)), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(ftypes)))
    ax.set_xticklabels([ft.replace('_', ' ').title() for ft in ftypes], rotation=15, ha='right')
    ax.set_ylabel('Number of Configurations', fontsize=12)
    ax.set_title('Failure Type Classification', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. Improvement magnitude (1000 sessions vs 100 sessions)
    ax = axes[1, 1]
    improvements = []
    original_rates = []

    for r in all_results:
        improvement = (r['success_rates'][1000] - r['success_rates'][100]) * 100
        improvements.append(improvement)
        original_rates.append(r['original_success_rate'] * 100)

    colors = ['red' if r['failure_type'] == 'structural' else 'orange' if r['failure_type'] == 'partial_improvement' else 'green' for r in all_results]

    ax.scatter(original_rates, improvements, c=colors, s=100, alpha=0.6, edgecolor='black', linewidth=1)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Original Success Rate (@ 100 sessions, %)', fontsize=12)
    ax.set_ylabel('Improvement (1000 sessions - 100 sessions, %)', fontsize=12)
    ax.set_title('Success Rate Improvement: 1000 vs 100 Sessions', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, edgecolor='black', label='Slow Convergence'),
        Patch(facecolor='orange', alpha=0.6, edgecolor='black', label='Partial Improvement'),
        Patch(facecolor='red', alpha=0.6, edgecolor='black', label='Structural Failure'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "visualization_output" / "extended_session_analysis.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    print()

    plt.close()


if __name__ == "__main__":
    main()
