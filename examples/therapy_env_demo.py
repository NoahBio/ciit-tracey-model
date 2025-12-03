"""
Demonstration of TherapyEnv usage.

This script shows how to use the TherapyEnv for training therapist agents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.environment import TherapyEnv


def complementary_policy(client_action: int) -> int:
    """
    Simple complementary therapist policy.

    Returns the complementary octant to the client's action,
    following interpersonal theory.
    """
    complement_map = {
        0: 4,  # D -> S
        1: 3,  # WD -> WS
        2: 2,  # W -> W
        3: 1,  # WS -> WD
        4: 0,  # S -> D
        5: 7,  # CS -> CD
        6: 6,  # C -> C
        7: 5,  # CD -> CS
    }
    return complement_map[client_action]


def run_episode(env, policy_fn, seed=None, verbose=False):
    """
    Run a single episode with the given policy.

    Parameters
    ----------
    env : TherapyEnv
        The therapy environment
    policy_fn : callable
        Function that takes client_action and returns therapist_action
    seed : int or None
        Random seed for reproducibility
    verbose : bool
        Whether to print step-by-step information

    Returns
    -------
    dict
        Episode statistics (success, sessions, reward, RS trajectory)
    """
    obs, info = env.reset(seed=seed)

    if verbose:
        print(f"\nNew Episode (seed={seed})")
        print(f"  Pattern: {info['pattern']}")
        print(f"  Initial RS: {info['initial_rs']:.2f}")
        print(f"  Success threshold: {info['rs_threshold']:.2f}")
        print(f"  Initial bond: {info['initial_bond']:.2f}\n")

    rs_trajectory = [info['initial_rs']]
    bond_trajectory = [info['initial_bond']]
    actions_history = []

    total_reward = 0
    session = 0

    for session in range(100):
        # Policy selects action
        client_action = obs['client_action']
        therapist_action = policy_fn(client_action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(therapist_action)

        # Track
        rs_trajectory.append(info['rs'])
        bond_trajectory.append(info['bond'])
        actions_history.append((client_action, therapist_action))
        total_reward += reward

        if verbose and (session < 5 or session % 10 == 9):
            print(f"  Session {session+1:3d}: C={client_action} T={therapist_action} "
                  f"RS={info['rs']:6.2f} Bond={info['bond']:.3f}")

        # Check termination
        if terminated or truncated:
            if verbose:
                print(f"\n  Episode ended at session {session+1}")
                print(f"    Success: {info['success']}")
                print(f"    Dropout: {info['dropped_out']}")
                print(f"    Max reached: {info['max_reached']}")
                print(f"    Total reward: {total_reward:.1f}")
                print(f"    Final RS: {info['rs']:.2f}")
                print(f"    Final bond: {info['bond']:.3f}")
            break

    return {
        'success': info['success'],
        'dropped_out': info['dropped_out'],
        'sessions': session + 1,
        'total_reward': total_reward,
        'rs_trajectory': rs_trajectory,
        'bond_trajectory': bond_trajectory,
        'actions': actions_history,
        'pattern': info.get('pattern', 'unknown'),
    }


def main():
    """Run demonstrations."""

    print("=" * 70)
    print("TherapyEnv Demonstration")
    print("=" * 70)

    # Example 1: Single episode with complementary policy
    print("\n1. Single episode with complementary therapist")
    print("-" * 70)
    env = TherapyEnv(pattern="cold_stuck", entropy=0.5, threshold=0.8, max_sessions=50)
    result = run_episode(env, complementary_policy, seed=42, verbose=True)

    # Example 2: Multiple episodes to measure success rate
    print("\n\n2. Success rate across 20 episodes")
    print("-" * 70)
    env = TherapyEnv(
        pattern=["cold_stuck", "dominant_stuck", "submissive_stuck"],
        entropy=0.5,
        threshold=0.8,
        max_sessions=100
    )

    results = []
    for seed in range(20):
        result = run_episode(env, complementary_policy, seed=seed, verbose=False)
        results.append(result)

    # Calculate statistics
    successes = sum(1 for r in results if r['success'])
    dropouts = sum(1 for r in results if r['dropped_out'])
    success_sessions = [r['sessions'] for r in results if r['success']]

    print(f"  Episodes: {len(results)}")
    print(f"  Successes: {successes} ({100*successes/len(results):.1f}%)")
    print(f"  Dropouts: {dropouts} ({100*dropouts/len(results):.1f}%)")
    if success_sessions:
        print(f"  Avg sessions to success: {np.mean(success_sessions):.1f} +/- {np.std(success_sessions):.1f}")
    print(f"  Avg total reward: {np.mean([r['total_reward'] for r in results]):.1f}")

    # Example 3: Different client parameters
    print("\n\n3. Impact of client entropy on success rate")
    print("-" * 70)

    for entropy in [0.3, 0.5, 1.0, 2.0]:
        env = TherapyEnv(pattern="cold_stuck", entropy=entropy, threshold=0.8, max_sessions=100)
        results = [run_episode(env, complementary_policy, seed=i, verbose=False) for i in range(10)]
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_sessions = np.mean([r['sessions'] for r in results if r['success']]) if any(r['success'] for r in results) else 0
        print(f"  Entropy={entropy:.1f}: Success={100*success_rate:.0f}%  Avg sessions={avg_sessions:.1f}")

    # Example 4: Different patterns
    print("\n\n4. Success rate by initial pattern")
    print("-" * 70)

    patterns = ["cold_stuck", "dominant_stuck", "submissive_stuck", "mixed_random", "complementary_perfect"]
    for pattern in patterns:
        env = TherapyEnv(pattern=pattern, entropy=0.5, threshold=0.8, max_sessions=100)
        results = [run_episode(env, complementary_policy, seed=i, verbose=False) for i in range(10)]
        success_rate = sum(1 for r in results if r['success']) / len(results)
        dropout_rate = sum(1 for r in results if r['dropped_out']) / len(results)
        print(f"  {pattern:25s}: Success={100*success_rate:3.0f}%  Dropout={100*dropout_rate:3.0f}%")

    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
