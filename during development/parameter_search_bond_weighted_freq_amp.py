"""
Comprehensive random search for bond_weighted_frequency_amplifier parameters.

Goal: Find parameter configurations that lead to low success rates (<50%)
for the always-complementary therapist strategy, revealing failure modes.

Sampling strategy (overnight run):
- 7,000 random configurations
- Up to 100 trials per configuration (different u_matrix seeds)
- Early stopping: if >40 successes, stop and mark as "too_successful"
- Estimated runtime: ~8 hours
- Focus: configurations with <50% success (ideally <5%)
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.client_agents import create_client
from src.agents.client_agents.base_client import BaseClientAgent
from src.config import calculate_success_threshold, sample_u_matrix


def always_complement(client_action: int) -> int:
    """Always-complementary therapist strategy."""
    complement_map = {
        0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5
    }
    return complement_map[client_action]


def run_single_trial(
    bond_power,
    history_weight,
    entropy,
    bond_alpha,
    bond_offset,
    success_threshold,
    random_seed,
    max_sessions=100,
):
    """
    Run a single therapy simulation trial.

    Returns
    -------
    dict
        Results including success, final_rs, sessions_completed, action_counts
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
        entropy=entropy,
        history_weight=history_weight,
        bond_power=bond_power,
        random_state=rng,
    )

    # Override bond parameters
    if hasattr(client, 'bond_alpha'):
        client.bond_alpha = bond_alpha

    # Apply offset by modifying rs_to_bond (monkey patch)
    from src.config import rs_to_bond as original_rs_to_bond

    def custom_rs_to_bond(rs, rs_min, rs_max, alpha=None, offset=None):
        if alpha is None:
            alpha = bond_alpha
        if offset is None:
            offset = bond_offset
        return original_rs_to_bond(rs, rs_min, rs_max, alpha=alpha, offset=offset)

    # Override the bond calculation
    client._rs_to_bond = custom_rs_to_bond
    # Recalculate initial bond with custom parameters
    client.bond = custom_rs_to_bond(
        client.relationship_satisfaction,
        float(u_matrix.min()),
        float(u_matrix.max()),
        alpha=bond_alpha,
        offset=bond_offset
    )

    # Calculate success threshold
    rs_threshold = calculate_success_threshold(u_matrix, success_threshold)

    # Run therapy sessions
    action_counts = {i: 0 for i in range(8)}
    dropped_out = False

    for session in range(1, max_sessions + 1):
        # Select client action
        client_action = client.select_action()
        action_counts[client_action] += 1

        # Get therapist response
        therapist_action = always_complement(client_action)

        # Update client
        client.update_memory(client_action, therapist_action)

        # Recalculate bond with custom parameters
        client.bond = custom_rs_to_bond(
            client.relationship_satisfaction,
            float(u_matrix.min()),
            float(u_matrix.max()),
            alpha=bond_alpha,
            offset=bond_offset
        )

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            sessions_completed = session
            break
    else:
        sessions_completed = max_sessions

    # Final results
    final_rs = client.relationship_satisfaction
    success = final_rs >= rs_threshold and not dropped_out

    # Calculate behavioral diversity
    unique_actions = sum(1 for count in action_counts.values() if count > 0)

    return {
        'success': success,
        'final_rs': float(final_rs),
        'rs_threshold': float(rs_threshold),
        'sessions_completed': sessions_completed,
        'dropped_out': dropped_out,
        'unique_actions': unique_actions,
        'action_counts': action_counts,
    }


def run_configuration(config, n_trials=100, early_stop_threshold=40):
    """
    Run multiple trials for a single configuration with early stopping.

    Parameters
    ----------
    config : dict
        Parameter configuration
    n_trials : int
        Maximum number of trials to run (default: 100)
    early_stop_threshold : int
        Stop if successes exceed this threshold (default: 40)
        Optimization: configs with >40% success aren't interesting failures

    Returns
    -------
    dict
        Configuration + aggregated results
    """
    results = []
    successes = 0
    early_stopped = False

    for seed in range(n_trials):
        trial_result = run_single_trial(
            bond_power=config['bond_power'],
            history_weight=config['history_weight'],
            entropy=config['entropy'],
            bond_alpha=config['bond_alpha'],
            bond_offset=config['bond_offset'],
            success_threshold=config['success_threshold'],
            random_seed=seed,
        )
        results.append(trial_result)

        if trial_result['success']:
            successes += 1

        # Early stopping: if too successful, this config isn't a failure mode
        if successes > early_stop_threshold:
            early_stopped = True
            break

    # Aggregate results
    n_trials_run = len(results)
    success_rate = sum(r['success'] for r in results) / n_trials_run
    mean_final_rs = np.mean([r['final_rs'] for r in results])
    mean_sessions = np.mean([r['sessions_completed'] for r in results])
    dropout_rate = sum(r['dropped_out'] for r in results) / n_trials_run
    mean_unique_actions = np.mean([r['unique_actions'] for r in results])

    return {
        'config': config,
        'success_rate': success_rate,
        'mean_final_rs': mean_final_rs,
        'mean_sessions': mean_sessions,
        'dropout_rate': dropout_rate,
        'mean_unique_actions': mean_unique_actions,
        'n_trials': n_trials_run,
        'early_stopped': early_stopped,
        'stop_reason': 'too_successful' if early_stopped else 'all_trials_completed',
    }


def sample_random_configurations(n_configs=7000):
    """
    Sample random parameter configurations.

    Parameters
    ----------
    n_configs : int
        Number of configurations to sample (default: 7000 for overnight run)

    Returns
    -------
    list of dict
        List of parameter configurations
    """
    rng = np.random.RandomState(42)

    configurations = []

    for _ in range(n_configs):
        config = {
            'bond_power': float(rng.uniform(1, 10)),
            'history_weight': float(rng.uniform(0.5, 15)),
            'entropy': float(rng.uniform(0.1, 0.5)),
            'bond_alpha': float(rng.uniform(3, 7)),  # Restricted to [3, 7] range
            'bond_offset': float(rng.uniform(0.4, 0.7)),
            'success_threshold': 0.9,
        }
        configurations.append(config)

    return configurations


def main():
    print("=" * 100)
    print("OVERNIGHT PARAMETER SEARCH: bond_weighted_frequency_amplifier")
    print("=" * 100)
    print()

    # Sample configurations
    print("Sampling random configurations...")
    configurations = sample_random_configurations(n_configs=7000)
    print(f"Generated {len(configurations)} configurations")
    print()

    # Parameter ranges
    print("Parameter ranges:")
    print(f"  bond_power:      [1.0, 10.0]")
    print(f"  history_weight:  [0.5, 15.0]")
    print(f"  entropy:         [0.1, 0.5]")
    print(f"  bond_alpha:      [3.0, 7.0]  ← RESTRICTED RANGE")
    print(f"  bond_offset:     [0.4, 0.7]")
    print(f"  success_threshold: 0.9 (fixed)")
    print()

    print(f"Running up to 100 trials per configuration with early stopping...")
    print(f"Early stop: if >40 successes, mark as 'too_successful' and move on")
    print(f"Maximum possible simulations: {len(configurations) * 100:,}")
    print(f"Expected: ~{len(configurations) * 70:,} (with early stopping)")
    print(f"Estimated runtime: ~8 hours")
    print()

    # Run parameter search
    all_results = []

    for i, config in enumerate(tqdm(configurations, desc="Configurations")):
        result = run_configuration(config, n_trials=100, early_stop_threshold=40)
        all_results.append(result)

        # Print progress every 500 configs
        if (i + 1) % 500 == 0:
            avg_success = np.mean([r['success_rate'] for r in all_results])
            n_early_stopped = sum(1 for r in all_results if r.get('early_stopped', False))
            avg_trials = np.mean([r['n_trials'] for r in all_results])
            print(f"\n  Progress: {i+1}/{len(configurations)} configs")
            print(f"    Avg success rate: {avg_success:.1%}")
            print(f"    Early stopped: {n_early_stopped} ({n_early_stopped/(i+1)*100:.1f}%)")
            print(f"    Avg trials/config: {avg_trials:.1f}")

    print()
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()

    # Overall statistics
    success_rates = [r['success_rate'] for r in all_results]
    print(f"Overall success rate statistics:")
    print(f"  Mean:   {np.mean(success_rates):.1%}")
    print(f"  Median: {np.median(success_rates):.1%}")
    print(f"  Min:    {np.min(success_rates):.1%}")
    print(f"  Max:    {np.max(success_rates):.1%}")
    print(f"  Std:    {np.std(success_rates):.1%}")
    print()

    # Early stopping stats
    n_early_stopped = sum(1 for r in all_results if r.get('early_stopped', False))
    avg_trials = np.mean([r['n_trials'] for r in all_results])
    total_sims = sum(r['n_trials'] for r in all_results)
    print(f"Early stopping statistics:")
    print(f"  Configs early stopped: {n_early_stopped} / {len(all_results)} ({n_early_stopped/len(all_results)*100:.1f}%)")
    print(f"  Average trials per config: {avg_trials:.1f}")
    print(f"  Total simulations run: {total_sims:,}")
    print()

    # Interesting configurations
    interesting = [r for r in all_results if 0.2 <= r['success_rate'] <= 0.8]
    print(f"Interesting configurations (20-80% success): {len(interesting)} / {len(all_results)}")
    print()

    # Failure configurations
    failures = [r for r in all_results if r['success_rate'] < 0.5]
    extreme_failures = [r for r in all_results if r['success_rate'] < 0.05]
    print(f"Failure configurations (<50% success): {len(failures)} / {len(all_results)}")
    print(f"Extreme failures (<5% success): {len(extreme_failures)} / {len(all_results)}")
    print()

    if len(failures) > 0:
        print("Top 10 failure configurations (lowest success rate):")
        for r in sorted(failures, key=lambda x: x['success_rate'])[:10]:
            print(f"  Success rate: {r['success_rate']:5.1%}  ({r['n_trials']} trials)")
            print(f"    bond_power={r['config']['bond_power']:.2f}, "
                  f"history_weight={r['config']['history_weight']:.2f}, "
                  f"entropy={r['config']['entropy']:.2f}")
            print(f"    bond_alpha={r['config']['bond_alpha']:.2f}, "
                  f"bond_offset={r['config']['bond_offset']:.3f}")
            print()

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).parent / f"bond_weighted_freq_amp_search_overnight_{timestamp}.json"

    # Convert to JSON-serializable format
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_configurations': len(configurations),
            'max_trials_per_config': 100,
            'early_stop_threshold': 40,
            'total_simulations': total_sims,
            'parameter_ranges': {
                'bond_power': [1.0, 10.0],
                'history_weight': [0.5, 15.0],
                'entropy': [0.1, 0.5],
                'bond_alpha': [3.0, 7.0],  # Restricted range
                'bond_offset': [0.4, 0.7],
                'success_threshold': 0.9,
            },
        },
        'results': all_results,
        'summary': {
            'mean_success_rate': float(np.mean(success_rates)),
            'median_success_rate': float(np.median(success_rates)),
            'std_success_rate': float(np.std(success_rates)),
            'n_early_stopped': n_early_stopped,
            'avg_trials_per_config': float(avg_trials),
            'n_interesting': len(interesting),
            'n_failures': len(failures),
            'n_extreme_failures': len(extreme_failures),
        },
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
