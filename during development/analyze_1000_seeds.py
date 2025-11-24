#!/usr/bin/env python3
"""
Analyze therapy outcomes across 1000 different random seeds.

Tests whether therapy stabilizes above threshold (success), below threshold (failure),
or shows cyclical patterns (oscillating around threshold).
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.agents.client_agents import create_client
from src import config
from src.config import sample_u_matrix, calculate_success_threshold
import json
from collections import Counter


def always_complement(client_action: int) -> int:
    """Simple complementary strategy."""
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


def generate_memory_pattern(pattern_name: str, size: int = 50, random_state=None):
    """Generate initial memory pattern."""
    rng = np.random.RandomState(random_state)

    if pattern_name == "cw_50_50":
        return [(6, 2)] * size
    elif pattern_name == "complementary_perfect":
        return [(0, 4)] * size
    elif pattern_name == "conflictual":
        return [(0, 0)] * size
    elif pattern_name == "mixed_random":
        return [(rng.randint(0, 8), rng.randint(0, 8)) for _ in range(size)]
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")


def run_single_seed(
    seed: int,
    mechanism: str,
    initial_memory_pattern: str,
    success_threshold_percentile: float,
    max_sessions: int,
    entropy: float,
    history_weight: float,
    bond_power: float,
    bond_alpha: float,
    bond_offset: float,
):
    """Run therapy simulation for a single seed and return trajectory."""

    # Setup
    rng = np.random.RandomState(seed)
    u_matrix = sample_u_matrix(random_state=seed)
    initial_memory = generate_memory_pattern(initial_memory_pattern, size=50, random_state=seed)

    # Set global bond parameters
    config.BOND_ALPHA = bond_alpha
    config.BOND_OFFSET = bond_offset

    # Create client
    client_kwargs = {
        'mechanism': mechanism,
        'u_matrix': u_matrix,
        'entropy': entropy,
        'initial_memory': initial_memory,
        'random_state': seed,
    }

    if 'amplifier' in mechanism:
        client_kwargs['history_weight'] = history_weight

    if 'bond_weighted' in mechanism:
        client_kwargs['bond_power'] = bond_power

    client = create_client(**client_kwargs)

    # Calculate RS threshold
    rs_threshold = calculate_success_threshold(u_matrix, success_threshold_percentile)

    # Track trajectory
    rs_trajectory = [client.relationship_satisfaction]
    bond_trajectory = [client.bond]

    # Run sessions
    dropped_out = False
    for session in range(1, max_sessions + 1):
        # Select action and get therapist response
        client_action = client.select_action()
        therapist_action = always_complement(client_action)

        # Update memory
        client.update_memory(client_action, therapist_action)

        # Record new state
        rs_trajectory.append(client.relationship_satisfaction)
        bond_trajectory.append(client.bond)

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            break

    return {
        'seed': seed,
        'rs_trajectory': rs_trajectory,
        'bond_trajectory': bond_trajectory,
        'rs_threshold': rs_threshold,
        'final_rs': rs_trajectory[-1],
        'sessions_completed': len(rs_trajectory) - 1,
        'dropped_out': dropped_out,
        'u_matrix_min': float(u_matrix.min()),
        'u_matrix_max': float(u_matrix.max()),
    }


def categorize_outcome(result: dict, window: int = 50) -> str:
    """
    Categorize outcome based on final trajectory window.

    Returns:
    - 'success': Stabilized above threshold
    - 'failure': Stabilized below threshold
    - 'cyclical': Oscillating around threshold
    """
    rs_trajectory = result['rs_trajectory']
    threshold = result['rs_threshold']

    # Look at last 'window' sessions (or all if fewer)
    final_window = rs_trajectory[-min(window, len(rs_trajectory)):]

    # Calculate statistics
    mean_rs = np.mean(final_window)
    std_rs = np.std(final_window)

    # Count threshold crossings in final window
    above_threshold = [rs > threshold for rs in final_window]
    crossings = sum(1 for i in range(1, len(above_threshold))
                   if above_threshold[i] != above_threshold[i-1])

    # Categorization logic
    # Cyclical: crosses threshold multiple times (3+) in final window
    if crossings >= 3:
        return 'cyclical'

    # Success: mean above threshold and stays relatively stable
    if mean_rs > threshold:
        return 'success'

    # Failure: mean below threshold
    return 'failure'


def main():
    """Run analysis across 1000 seeds."""

    print("=" * 100)
    print("ANALYZING THERAPY OUTCOMES ACROSS 1000 SEEDS")
    print("=" * 100)
    print()

    # Parameters
    params = {
        'mechanism': 'conditional_amplifier',
        'initial_memory_pattern': 'cw_50_50',
        'success_threshold_percentile': 0.9,
        'max_sessions': 500,
        'entropy': 3.0,
        'history_weight': 2.0,
        'bond_power': 2.0,
        'bond_alpha': 3.0,
        'bond_offset': 0.7,
    }

    print("PARAMETERS:")
    print("-" * 100)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    print("Running simulations...")
    print()

    # Run simulations
    results = []
    n_seeds = 1000

    for seed in range(n_seeds):
        if (seed + 1) % 100 == 0:
            print(f"  Completed {seed + 1}/{n_seeds} seeds...")

        result = run_single_seed(seed=seed, **params)
        results.append(result)

    print()
    print("Simulations complete. Categorizing outcomes...")
    print()

    # Categorize outcomes
    for result in results:
        result['outcome'] = categorize_outcome(result)

    # Count outcomes
    outcome_counts = Counter(r['outcome'] for r in results)

    # Organize by category
    success_seeds = [r for r in results if r['outcome'] == 'success']
    failure_seeds = [r for r in results if r['outcome'] == 'failure']
    cyclical_seeds = [r for r in results if r['outcome'] == 'cyclical']

    # Print summary
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()

    print(f"Total seeds tested: {n_seeds}")
    print()

    print("OUTCOME DISTRIBUTION:")
    print("-" * 100)
    print(f"  Success (stabilized above threshold):  {outcome_counts['success']:4d} ({outcome_counts['success']/n_seeds*100:5.1f}%)")
    print(f"  Failure (stabilized below threshold):  {outcome_counts['failure']:4d} ({outcome_counts['failure']/n_seeds*100:5.1f}%)")
    print(f"  Cyclical (oscillating around threshold): {outcome_counts['cyclical']:4d} ({outcome_counts['cyclical']/n_seeds*100:5.1f}%)")
    print()

    # Example seeds for each category
    print("EXAMPLE SEEDS FOR EACH CATEGORY:")
    print("-" * 100)

    if success_seeds:
        print(f"Success examples: {[r['seed'] for r in success_seeds[:10]]}")
        avg_final_rs = np.mean([r['final_rs'] for r in success_seeds])
        print(f"  Average final RS: {avg_final_rs:.2f}")
        print()

    if failure_seeds:
        print(f"Failure examples: {[r['seed'] for r in failure_seeds[:10]]}")
        avg_final_rs = np.mean([r['final_rs'] for r in failure_seeds])
        print(f"  Average final RS: {avg_final_rs:.2f}")
        print()

    if cyclical_seeds:
        print(f"Cyclical examples: {[r['seed'] for r in cyclical_seeds[:10]]}")
        avg_final_rs = np.mean([r['final_rs'] for r in cyclical_seeds])
        print(f"  Average final RS: {avg_final_rs:.2f}")
        print()

    # Statistics
    print("FINAL RS STATISTICS BY CATEGORY:")
    print("-" * 100)

    for category in ['success', 'failure', 'cyclical']:
        category_results = [r for r in results if r['outcome'] == category]
        if category_results:
            final_rs_values = [r['final_rs'] for r in category_results]
            print(f"{category.capitalize()}:")
            print(f"  Mean final RS: {np.mean(final_rs_values):7.2f}")
            print(f"  Median final RS: {np.median(final_rs_values):7.2f}")
            print(f"  Std dev: {np.std(final_rs_values):7.2f}")
            print(f"  Min: {np.min(final_rs_values):7.2f}")
            print(f"  Max: {np.max(final_rs_values):7.2f}")
            print()

    # Check seed 42 specifically
    seed_42_result = next(r for r in results if r['seed'] == 42)
    print("SEED 42 (your example):")
    print("-" * 100)
    print(f"  Outcome: {seed_42_result['outcome']}")
    print(f"  Final RS: {seed_42_result['final_rs']:.2f}")
    print(f"  RS threshold: {seed_42_result['rs_threshold']:.2f}")
    print(f"  Sessions completed: {seed_42_result['sessions_completed']}")
    print()

    # Save results to JSON
    output_file = Path(__file__).parent / "seed_analysis_results.json"

    # Prepare data for JSON (convert numpy types)
    export_data = {
        'parameters': params,
        'summary': {
            'total_seeds': n_seeds,
            'success_count': outcome_counts['success'],
            'failure_count': outcome_counts['failure'],
            'cyclical_count': outcome_counts['cyclical'],
        },
        'results': [
            {
                'seed': r['seed'],
                'outcome': r['outcome'],
                'final_rs': float(r['final_rs']),
                'rs_threshold': float(r['rs_threshold']),
                'sessions_completed': r['sessions_completed'],
                'dropped_out': r['dropped_out'],
                # Store only last 100 sessions of trajectory to save space
                'rs_trajectory_final': [float(x) for x in r['rs_trajectory'][-100:]],
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print()

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
