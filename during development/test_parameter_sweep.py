"""Comprehensive parameter sweep for always-complementary therapist strategy.

Tests success rates across systematic variations of:
- Client parameters: entropy, history_weight, smoothing_alpha, bond_power
- System parameters: BOND_ALPHA, MEMORY_SIZE, success threshold
- Initial conditions: memory patterns, pattern_types
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass, asdict
from src.agents.client_agents import create_client
from src.config import (
    sample_u_matrix,
    MAX_SESSIONS,
    OCTANTS,
    BOND_ALPHA,
    MEMORY_SIZE,
    HISTORY_WEIGHT,
    calculate_success_threshold,
)


@dataclass
class ParameterConfig:
    """Configuration for a single parameter sweep run."""
    # Client parameters
    mechanism: str
    entropy: float
    history_weight: float
    smoothing_alpha: float
    bond_power: float

    # System parameters
    bond_alpha: float
    memory_size: int
    success_threshold: float

    # Initial conditions
    initial_memory_pattern: str
    pattern_type: str

    # Results
    success_rate: float = 0.0
    avg_sessions_completed: float = 0.0
    avg_final_bond: float = 0.0
    dropout_rate: float = 0.0


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


def generate_memory_pattern(pattern_name: str, size: int, random_state=None) -> List[Tuple[int, int]]:
    """Generate different initial memory patterns."""
    rng = np.random.RandomState(random_state)

    if pattern_name == "cw_50_50":
        # 50/50 C→W anticomplementary
        return [(6, 2)] * size
    elif pattern_name == "complementary_perfect":
        # Perfect complementarity: D→S
        return [(0, 4)] * size
    elif pattern_name == "conflictual":
        # Conflictual: D→D
        return [(0, 0)] * size
    elif pattern_name == "mixed_random":
        # Random interactions
        return [(rng.randint(0, 8), rng.randint(0, 8)) for _ in range(size)]
    elif pattern_name == "cold_dominant":
        # Client stuck in CD, therapist varies
        client_oct = 7  # CD
        return [(client_oct, rng.randint(0, 8)) for _ in range(size)]
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")


def simulate_therapy_episode(
    client,
    therapist_strategy,
    max_sessions: int = MAX_SESSIONS,
    success_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Simulate a therapy episode and return outcome metrics.

    Parameters
    ----------
    client : BaseClientAgent
        Client agent to simulate
    therapist_strategy : callable
        Function that takes client_action and returns therapist_action
    max_sessions : int
        Maximum number of therapy sessions
    success_threshold : float (0-1)
        Percentile of client's achievable RS range that must be reached
        (0.5 = must reach 50th percentile, 0.8 = must reach 80th percentile)

    Returns
    -------
    dict
        Simulation results including success status and metrics
    """

    initial_bond = client.bond
    initial_rs = client.relationship_satisfaction

    completed_sessions = 0
    dropped_out = False

    for session in range(max_sessions):
        # Client selects action
        client_action = client.select_action()

        # Therapist responds
        therapist_action = therapist_strategy(client_action)

        # Update client memory
        client.update_memory(client_action, therapist_action)
        completed_sessions += 1

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            break

    final_bond = client.bond
    final_rs = client.relationship_satisfaction

    # Calculate client-specific RS threshold based on their u_matrix
    rs_threshold = calculate_success_threshold(client.u_matrix, success_threshold)

    # Success criteria: completed without dropout AND RS above client-specific threshold
    success = (not dropped_out) and (final_rs >= rs_threshold)

    return {
        'success': success,
        'dropped_out': dropped_out,
        'completed_sessions': completed_sessions,
        'initial_bond': initial_bond,
        'final_bond': final_bond,
        'initial_rs': initial_rs,
        'final_rs': final_rs,
        'rs_threshold': rs_threshold,
        'bond_change': final_bond - initial_bond,
        'rs_change': final_rs - initial_rs,
    }


def run_parameter_configuration(
    config: ParameterConfig,
    n_trials: int = 50,
    random_seed: int = 42,
) -> ParameterConfig:
    """Run multiple trials for a single parameter configuration."""

    results = []

    for trial in range(n_trials):
        trial_seed = random_seed + trial
        rng = np.random.RandomState(trial_seed)

        # Generate initial memory
        initial_memory = generate_memory_pattern(
            config.initial_memory_pattern,
            config.memory_size,
            random_state=trial_seed
        )

        # Sample u_matrix for this client
        u_matrix = sample_u_matrix(random_state=trial_seed)

        # Create client with specified parameters
        client_kwargs = {
            'mechanism': config.mechanism,
            'u_matrix': u_matrix,
            'entropy': config.entropy,
            'initial_memory': initial_memory,
            'random_state': trial_seed,
        }

        # Add mechanism-specific parameters
        if 'amplifier' in config.mechanism:
            client_kwargs['history_weight'] = config.history_weight

        if 'conditional' in config.mechanism:
            client_kwargs['smoothing_alpha'] = config.smoothing_alpha

        if 'bond_weighted' in config.mechanism:
            client_kwargs['bond_power'] = config.bond_power

        # Create client
        client = create_client(**client_kwargs)

        # Override BOND_ALPHA if needed by recalculating bond
        if config.bond_alpha != BOND_ALPHA:
            from src.config import rs_to_bond
            # Recalculate bond with new alpha using client's u_matrix bounds
            rs_min = np.min(client.u_matrix)
            rs_max = np.max(client.u_matrix)
            client.bond = rs_to_bond(
                client.relationship_satisfaction,
                rs_min,
                rs_max,
                alpha=config.bond_alpha
            )

        # Run simulation
        result = simulate_therapy_episode(
            client,
            always_complement,
            max_sessions=MAX_SESSIONS,
            success_threshold=config.success_threshold,
        )

        results.append(result)

    # Aggregate results
    config.success_rate = sum(r['success'] for r in results) / n_trials
    config.dropout_rate = sum(r['dropped_out'] for r in results) / n_trials
    config.avg_sessions_completed = float(np.mean([r['completed_sessions'] for r in results]))
    config.avg_final_bond = float(np.mean([r['final_bond'] for r in results]))

    return config


def run_parameter_sweep(
    mechanisms: List[str],
    entropies: List[float],
    history_weights: List[float],
    smoothing_alphas: List[float],
    bond_powers: List[float],
    bond_alphas: List[float],
    memory_sizes: List[int],
    success_thresholds: List[float],
    initial_memory_patterns: List[str],
    pattern_types: List[str],
    n_trials: int = 50,
    random_seed: int = 42,
) -> List[ParameterConfig]:
    """Run comprehensive parameter sweep."""

    results = []

    # Generate all combinations
    all_combinations = list(itertools.product(
        mechanisms,
        entropies,
        history_weights,
        smoothing_alphas,
        bond_powers,
        bond_alphas,
        memory_sizes,
        success_thresholds,
        initial_memory_patterns,
        pattern_types,
    ))

    total_configs = len(all_combinations)
    print(f"Testing {total_configs} parameter configurations...")
    print(f"Each configuration runs {n_trials} trials")
    print(f"Total simulations: {total_configs * n_trials}")
    print()

    for idx, (mechanism, entropy, history_weight, smoothing_alpha, bond_power,
              bond_alpha, memory_size, success_threshold, memory_pattern, pattern_type) in enumerate(all_combinations):

        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Progress: {idx + 1}/{total_configs} ({(idx+1)/total_configs*100:.1f}%)")

        # Create configuration
        config = ParameterConfig(
            mechanism=mechanism,
            entropy=entropy,
            history_weight=history_weight,
            smoothing_alpha=smoothing_alpha,
            bond_power=bond_power,
            bond_alpha=bond_alpha,
            memory_size=memory_size,
            success_threshold=success_threshold,
            initial_memory_pattern=memory_pattern,
            pattern_type=pattern_type,
        )

        # Run trials
        config = run_parameter_configuration(config, n_trials=n_trials, random_seed=random_seed)

        results.append(config)

    return results


def print_results_summary(results: List[ParameterConfig], top_n: int = 20):
    """Print summary of results, highlighting interesting cases."""

    print("\n" + "=" * 100)
    print("PARAMETER SWEEP RESULTS SUMMARY")
    print("=" * 100)

    # Sort by success rate
    results_sorted = sorted(results, key=lambda x: x.success_rate)

    # Lowest success rates (most interesting failures)
    print(f"\n{'LOWEST SUCCESS RATES (Top {top_n})':^100}")
    print("-" * 100)
    print(f"{'Mech':<10} {'Ent':>5} {'HW':>5} {'SA':>5} {'BP':>5} {'BA':>5} {'MS':>4} {'Thr':>5} {'Memory':<20} {'Succ%':>6} {'Drop%':>6} {'Bond':>6}")
    print("-" * 100)

    for config in results_sorted[:top_n]:
        print(f"{config.mechanism[:10]:<10} "
              f"{config.entropy:5.2f} "
              f"{config.history_weight:5.2f} "
              f"{config.smoothing_alpha:5.2f} "
              f"{config.bond_power:5.2f} "
              f"{config.bond_alpha:5.2f} "
              f"{config.memory_size:4d} "
              f"{config.success_threshold:5.2f} "
              f"{config.initial_memory_pattern[:20]:<20} "
              f"{config.success_rate*100:6.1f} "
              f"{config.dropout_rate*100:6.1f} "
              f"{config.avg_final_bond:6.3f}")

    # Highest success rates
    print(f"\n{'HIGHEST SUCCESS RATES (Top {top_n})':^100}")
    print("-" * 100)
    print(f"{'Mech':<10} {'Ent':>5} {'HW':>5} {'SA':>5} {'BP':>5} {'BA':>5} {'MS':>4} {'Thr':>5} {'Memory':<20} {'Succ%':>6} {'Drop%':>6} {'Bond':>6}")
    print("-" * 100)

    for config in results_sorted[-top_n:]:
        print(f"{config.mechanism[:10]:<10} "
              f"{config.entropy:5.2f} "
              f"{config.history_weight:5.2f} "
              f"{config.smoothing_alpha:5.2f} "
              f"{config.bond_power:5.2f} "
              f"{config.bond_alpha:5.2f} "
              f"{config.memory_size:4d} "
              f"{config.success_threshold:5.2f} "
              f"{config.initial_memory_pattern[:20]:<20} "
              f"{config.success_rate*100:6.1f} "
              f"{config.dropout_rate*100:6.1f} "
              f"{config.avg_final_bond:6.3f}")

    # Overall statistics
    print(f"\n{'OVERALL STATISTICS':^100}")
    print("-" * 100)
    success_rates = [r.success_rate for r in results]
    dropout_rates = [r.dropout_rate for r in results]

    print(f"Total configurations tested: {len(results)}")
    print(f"Mean success rate: {np.mean(success_rates)*100:.1f}%")
    print(f"Median success rate: {np.median(success_rates)*100:.1f}%")
    print(f"Std success rate: {np.std(success_rates)*100:.1f}%")
    print(f"Min success rate: {np.min(success_rates)*100:.1f}%")
    print(f"Max success rate: {np.max(success_rates)*100:.1f}%")
    print()
    print(f"Mean dropout rate: {np.mean(dropout_rates)*100:.1f}%")
    print(f"Configurations with 0% success: {sum(1 for r in results if r.success_rate == 0)}")
    print(f"Configurations with 100% success: {sum(1 for r in results if r.success_rate == 1.0)}")


def save_results(results: List[ParameterConfig], filename: str):
    """Save results to JSON file."""
    results_dict = [asdict(r) for r in results]

    script_dir = Path(__file__).parent
    filepath = script_dir / filename

    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✓ Results saved to '{filepath}'")


def main():
    """Run parameter sweep with predefined ranges."""

    # Define parameter ranges
    # Start with a manageable subset, can expand later

    mechanisms = [
        'conditional_amplifier',
        'bond_weighted_conditional_amplifier',
        'frequency_amplifier',
    ]

    entropies = [0.5, 1.0, 2.0]  # Low, medium, high exploration

    history_weights = [0.5, 1.0, 2.0]  # Weak, normal, strong history influence

    smoothing_alphas = [0.01, 0.1, 0.5]  # Strong data-driven, normal, more uniform

    bond_powers = [1.0, 2.0]  # Linear, quadratic bond scaling (only for bond_weighted)

    bond_alphas = [2.0, 5.0, 8.0]  # How RS maps to bond (default is 5.0)

    memory_sizes = [50]  # Fixed at 50 (base_client requirement)

    success_thresholds = [0.3, 0.5, 0.7]  # Lenient, normal, strict

    initial_memory_patterns = [
        'cw_50_50',  # The problematic C→W pattern
        'complementary_perfect',  # Ideal starting point
        'conflictual',  # Worst starting point
    ]

    # We'll ignore pattern_types for now since we're using explicit memory patterns
    pattern_types = ['none']  # Placeholder

    print("=" * 100)
    print("COMPREHENSIVE PARAMETER SWEEP")
    print("=" * 100)
    print("\nParameter ranges:")
    print(f"  Mechanisms: {mechanisms}")
    print(f"  Entropies: {entropies}")
    print(f"  History weights: {history_weights}")
    print(f"  Smoothing alphas: {smoothing_alphas}")
    print(f"  Bond powers: {bond_powers}")
    print(f"  Bond alphas: {bond_alphas}")
    print(f"  Memory sizes: {memory_sizes}")
    print(f"  Success thresholds: {success_thresholds}")
    print(f"  Initial memory patterns: {initial_memory_patterns}")
    print()

    # Calculate total combinations
    total = (len(mechanisms) * len(entropies) * len(history_weights) *
             len(smoothing_alphas) * len(bond_powers) * len(bond_alphas) *
             len(memory_sizes) * len(success_thresholds) *
             len(initial_memory_patterns) * len(pattern_types))

    print(f"Total parameter combinations: {total}")

    # Ask for confirmation if too many
    if total > 500:
        response = input(f"\nThis will run {total} configurations. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run sweep
    results = run_parameter_sweep(
        mechanisms=mechanisms,
        entropies=entropies,
        history_weights=history_weights,
        smoothing_alphas=smoothing_alphas,
        bond_powers=bond_powers,
        bond_alphas=bond_alphas,
        memory_sizes=memory_sizes,
        success_thresholds=success_thresholds,
        initial_memory_patterns=initial_memory_patterns,
        pattern_types=pattern_types,
        n_trials=50,  # 50 trials per configuration for statistical reliability
        random_seed=42,
    )

    # Print results
    print_results_summary(results, top_n=20)

    # Save to file
    save_results(results, 'parameter_sweep_results.json')

    print("\n" + "=" * 100)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
