"""Compare trained RL policy against baseline strategies.

This module provides tools to evaluate a trained therapy policy against
several baseline strategies to assess performance improvements.

Baselines:
- random_therapist: Uniform random action selection
- always_complement: Always plays complementary actions (existing strategy)
- optimal_static: Oracle that always plays action maximizing mean utility

Example usage:
    python -m src.evaluation.baseline_comparison \
        --policy-path models/experiment/policy.pth \
        --config configs/experiment.yaml \
        --n-episodes 1000 \
        --output results/comparison.json
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environment import TherapyEnv
from src.training import load_config
from src.training.networks import make_therapy_networks
from src.evaluation.evaluate_policy import load_policy, evaluate_episode


# ============================================================================
# Baseline Strategies
# ============================================================================

def random_therapist(env: TherapyEnv, obs: Dict, rng: np.random.Generator) -> int:
    """
    Random baseline: Select actions uniformly at random.

    Parameters
    ----------
    env : TherapyEnv
        Environment instance (for action space)
    obs : dict
        Current observation (unused)
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    int
        Random action from action space
    """
    return int(rng.integers(0, env.action_space.n))  # type: ignore[attr-defined]


def always_complement(env: TherapyEnv, obs: Dict, rng: np.random.Generator) -> int:
    """
    Complementary baseline: Always play action complementary to client.

    Uses interpersonal complementarity: reciprocate on affiliation dimension,
    respond oppositely on control dimension (D↔S, W↔W, C↔C).

    Parameters
    ----------
    env : TherapyEnv
        Environment instance
    obs : dict
        Current observation containing client_action
    rng : np.random.Generator
        Random number generator (unused, for signature compatibility)

    Returns
    -------
    int
        Complementary action to client's current action
    """
    # Complementary mapping: D↔S, reciprocate on affiliation
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
    client_action = int(obs['client_action'])
    return complement_map[client_action]


def optimal_static(env: TherapyEnv, obs: Dict, rng: np.random.Generator) -> int:
    """
    Oracle baseline: Always play action that maximizes mean utility.

    This is an oracle strategy that knows the client's utility function
    and always plays the action with highest expected utility. This provides
    an upper bound on static (non-adaptive) strategies.

    Parameters
    ----------
    env : TherapyEnv
        Environment instance with client utilities
    obs : dict
        Current observation (unused)
    rng : np.random.Generator
        Random number generator (unused, for signature compatibility)

    Returns
    -------
    int
        Action with highest mean utility for the client
    """
    # Access client's utility function
    if not hasattr(env, 'client') or env.client is None:  # type: ignore[attr-defined]
        warnings.warn("Environment has no client; using random action")
        return random_therapist(env, obs, rng)

    client = env.client  # type: ignore[attr-defined]

    # Compute mean utility for each therapist action
    n_actions = env.action_space.n  # type: ignore[attr-defined]
    mean_utilities = np.zeros(n_actions)

    for therapist_action in range(n_actions):
        # Average utility across all possible client actions
        utilities = []
        for client_action in range(8):  # 8 octants
            utility = client.utility_function.get_utility(
                client_action,
                therapist_action
            )
            utilities.append(utility)
        mean_utilities[therapist_action] = np.mean(utilities)

    # Return action with highest mean utility
    return int(np.argmax(mean_utilities))


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_baseline(
    strategy_fn: Callable,
    env_kwargs: Dict[str, Any],
    n_episodes: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a baseline strategy over multiple episodes.

    Parameters
    ----------
    strategy_fn : callable
        Baseline strategy function: (env, obs, rng) -> action
    env_kwargs : dict
        Environment configuration parameters
    n_episodes : int
        Number of episodes to evaluate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Results dictionary with metrics:
        - episodes: List of episode results
        - success_rate: Fraction of successful episodes
        - dropout_rate: Fraction of dropout episodes
        - mean_sessions: Average sessions to success
        - mean_reward: Average episode reward
    """
    rng = np.random.default_rng(seed)
    env = TherapyEnv(**env_kwargs)

    episodes = []

    for episode_idx in tqdm(range(n_episodes), desc="Evaluating baseline"):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False
        truncated = False

        total_reward = 0.0
        actions = []
        rs_trajectory = []
        complementarity_trajectory = []

        while not (done or truncated):
            # Get action from baseline strategy
            action = strategy_fn(env, obs, rng)
            actions.append(action)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Record metrics
            if hasattr(env, 'client') and env.client is not None:  # type: ignore[attr-defined]
                rs_trajectory.append(float(env.client.perception.rs))  # type: ignore[attr-defined]

                # Compute complementarity
                last_client_action = int(obs['client_action'])
                complementarity = abs((last_client_action - action + 4) % 8 - 4)
                complementarity_trajectory.append(complementarity)

        # Record episode results
        episodes.append({
            'reward': float(total_reward),
            'length': len(actions),
            'success': info.get('success', False),
            'dropout': info.get('dropout', False),
            'rs_trajectory': rs_trajectory,
            'complementarity_trajectory': complementarity_trajectory,
            'actions': actions,
        })

    # Compute summary metrics
    successes = [ep['success'] for ep in episodes]
    dropouts = [ep['dropout'] for ep in episodes]
    sessions = [ep['length'] for ep in episodes if ep['success']]
    rewards = [ep['reward'] for ep in episodes]

    return {
        'episodes': episodes,
        'success_rate': float(np.mean(successes)),
        'dropout_rate': float(np.mean(dropouts)),
        'mean_sessions': float(np.mean(sessions)) if sessions else float('nan'),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
    }


def evaluate_rl_policy(
    policy_path: Path,
    config: Any,
    env_kwargs: Dict[str, Any],
    n_episodes: int,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate the trained RL policy over multiple episodes.

    Parameters
    ----------
    policy_path : Path
        Path to saved policy checkpoint
    config : TrainingConfig
        Training configuration
    env_kwargs : dict
        Environment configuration parameters
    n_episodes : int
        Number of episodes to evaluate
    seed : int
        Random seed for reproducibility
    device : str
        Device to run policy on ('cpu' or 'cuda')

    Returns
    -------
    dict
        Results dictionary with same structure as evaluate_baseline()
    """
    # Load policy
    actor = load_policy(policy_path, config, device)
    actor.eval()

    # Create environment
    rng = np.random.default_rng(seed)
    env = TherapyEnv(**env_kwargs)

    episodes = []

    for episode_idx in tqdm(range(n_episodes), desc="Evaluating RL policy"):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False
        truncated = False

        total_reward = 0.0
        actions = []
        rs_trajectory = []
        complementarity_trajectory = []

        while not (done or truncated):
            # Get action from RL policy
            with torch.no_grad():
                obs_tensor = {
                    'client_action': torch.tensor([obs['client_action']], dtype=torch.long, device=device),
                    'session_number': torch.tensor([obs['session_number']], dtype=torch.float32, device=device),
                    'history': torch.tensor([obs['history']], dtype=torch.long, device=device),
                }
                logits, _ = actor(obs_tensor)
                action = int(torch.argmax(logits, dim=-1).cpu().item())

            actions.append(action)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Record metrics
            if hasattr(env, 'client') and env.client is not None:  # type: ignore[attr-defined]
                rs_trajectory.append(float(env.client.perception.rs))  # type: ignore[attr-defined]

                # Compute complementarity
                last_client_action = int(obs['client_action'])
                complementarity = abs((last_client_action - action + 4) % 8 - 4)
                complementarity_trajectory.append(complementarity)

        # Record episode results
        episodes.append({
            'reward': float(total_reward),
            'length': len(actions),
            'success': info.get('success', False),
            'dropout': info.get('dropout', False),
            'rs_trajectory': rs_trajectory,
            'complementarity_trajectory': complementarity_trajectory,
            'actions': actions,
        })

    # Compute summary metrics
    successes = [ep['success'] for ep in episodes]
    dropouts = [ep['dropout'] for ep in episodes]
    sessions = [ep['length'] for ep in episodes if ep['success']]
    rewards = [ep['reward'] for ep in episodes]

    return {
        'episodes': episodes,
        'success_rate': float(np.mean(successes)),
        'dropout_rate': float(np.mean(dropouts)),
        'mean_sessions': float(np.mean(sessions)) if sessions else float('nan'),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
    }


# ============================================================================
# Statistical Testing
# ============================================================================

def paired_bootstrap_test(
    metric1: List[float],
    metric2: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Paired bootstrap test for comparing two metrics.

    Computes bootstrap confidence interval for the difference between
    two paired metrics (e.g., success rates from same seeds).

    Parameters
    ----------
    metric1 : list of float
        First metric values (one per episode)
    metric2 : list of float
        Second metric values (one per episode)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for CI (e.g., 0.95 for 95% CI)

    Returns
    -------
    dict
        Results with keys:
        - mean_diff: Mean difference (metric1 - metric2)
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        - p_value: Approximate p-value (fraction of bootstrap samples with diff <= 0)
    """
    metric1_arr = np.array(metric1)
    metric2_arr = np.array(metric2)

    assert len(metric1_arr) == len(metric2_arr), "Metrics must have same length"

    n = len(metric1_arr)
    rng = np.random.default_rng(42)

    # Observed difference
    observed_diff = np.mean(metric1_arr) - np.mean(metric2_arr)

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_diff = np.mean(metric1_arr[indices]) - np.mean(metric2_arr[indices])
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Approximate p-value (two-tailed)
    p_value = 2 * min(
        np.mean(bootstrap_diffs <= 0),
        np.mean(bootstrap_diffs >= 0)
    )

    return {
        'mean_diff': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
    }


def compare_strategies(
    results: Dict[str, Dict[str, Any]],
    reference: str = "rl_policy",
) -> Dict[str, Dict[str, Any]]:
    """
    Perform statistical comparisons between strategies.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy names to evaluation results
    reference : str
        Reference strategy to compare against (default: 'rl_policy')

    Returns
    -------
    dict
        Statistical test results for each comparison
    """
    if reference not in results:
        raise ValueError(f"Reference strategy '{reference}' not in results")

    ref_results = results[reference]
    comparisons = {}

    for name, strategy_results in results.items():
        if name == reference:
            continue

        # Extract episode-level metrics
        ref_successes = [float(ep['success']) for ep in ref_results['episodes']]
        strategy_successes = [float(ep['success']) for ep in strategy_results['episodes']]

        ref_rewards = [ep['reward'] for ep in ref_results['episodes']]
        strategy_rewards = [ep['reward'] for ep in strategy_results['episodes']]

        # Perform bootstrap tests
        success_test = paired_bootstrap_test(ref_successes, strategy_successes)
        reward_test = paired_bootstrap_test(ref_rewards, strategy_rewards)

        comparisons[name] = {
            'success_rate': success_test,
            'reward': reward_test,
        }

    return comparisons


# ============================================================================
# Visualization and Output
# ============================================================================

def print_comparison_table(
    results: Dict[str, Dict[str, Any]],
    comparisons: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Print formatted comparison table to console.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy names to evaluation results
    comparisons : dict, optional
        Statistical comparison results from compare_strategies()
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80 + "\n")

    # Table header
    print(f"{'Strategy':<20} {'Success Rate':<15} {'Dropout Rate':<15} {'Avg Sessions':<15} {'Mean Reward':<15}")
    print("-" * 80)

    # Table rows
    for name, strategy_results in results.items():
        success_rate = strategy_results['success_rate']
        dropout_rate = strategy_results['dropout_rate']
        mean_sessions = strategy_results['mean_sessions']
        mean_reward = strategy_results['mean_reward']

        print(f"{name:<20} {success_rate:>6.2%}         {dropout_rate:>6.2%}         "
              f"{mean_sessions:>6.1f}         {mean_reward:>8.2f}")

    print("-" * 80 + "\n")

    # Statistical comparisons
    if comparisons:
        print("\nSTATISTICAL COMPARISONS (vs RL Policy)")
        print("-" * 80)

        for name, comp in comparisons.items():
            print(f"\n{name}:")

            # Success rate
            success_diff = comp['success_rate']['mean_diff']
            success_ci = (comp['success_rate']['ci_lower'], comp['success_rate']['ci_upper'])
            success_p = comp['success_rate']['p_value']

            print(f"  Success Rate: {success_diff:+.2%} [{success_ci[0]:.2%}, {success_ci[1]:.2%}], p={success_p:.4f}")

            # Reward
            reward_diff = comp['reward']['mean_diff']
            reward_ci = (comp['reward']['ci_lower'], comp['reward']['ci_upper'])
            reward_p = comp['reward']['p_value']

            print(f"  Reward:       {reward_diff:+.2f} [{reward_ci[0]:.2f}, {reward_ci[1]:.2f}], p={reward_p:.4f}")

        print()


def plot_comparison(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generate comparison visualizations.

    Creates bar charts comparing strategies on key metrics.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy names to evaluation results
    output_dir : Path
        Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(results.keys())

    # Extract metrics
    success_rates = [results[s]['success_rate'] for s in strategies]
    dropout_rates = [results[s]['dropout_rate'] for s in strategies]
    mean_sessions = [results[s]['mean_sessions'] for s in strategies]
    mean_rewards = [results[s]['mean_reward'] for s in strategies]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strategy Comparison', fontsize=16)

    # Success rate
    axes[0, 0].bar(strategies, success_rates, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate by Strategy')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Dropout rate
    axes[0, 1].bar(strategies, dropout_rates, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[0, 1].set_ylabel('Dropout Rate')
    axes[0, 1].set_title('Dropout Rate by Strategy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Mean sessions
    axes[1, 0].bar(strategies, mean_sessions, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[1, 0].set_ylabel('Mean Sessions')
    axes[1, 0].set_title('Average Sessions to Success')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Mean reward
    axes[1, 1].bar(strategies, mean_rewards, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[1, 1].set_ylabel('Mean Reward')
    axes[1, 1].set_title('Mean Episode Reward')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def save_results(
    results: Dict[str, Dict[str, Any]],
    comparisons: Optional[Dict[str, Dict[str, Any]]],
    output_path: Path,
) -> None:
    """
    Save results to JSON file.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy names to evaluation results
    comparisons : dict, optional
        Statistical comparison results
    output_path : Path
        Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove episode trajectories for more compact output
    compact_results = {}
    for name, strategy_results in results.items():
        compact_results[name] = {
            'success_rate': strategy_results['success_rate'],
            'dropout_rate': strategy_results['dropout_rate'],
            'mean_sessions': strategy_results['mean_sessions'],
            'mean_reward': strategy_results['mean_reward'],
            'std_reward': strategy_results['std_reward'],
            'n_episodes': len(strategy_results['episodes']),
        }

    output_data = {
        'summary': compact_results,
        'comparisons': comparisons or {},
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved results to {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function for baseline comparison."""
    parser = argparse.ArgumentParser(
        description="Compare RL policy against baseline strategies"
    )

    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Path to trained policy checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config file (.yaml)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1000,
        help="Number of episodes per strategy (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_comparison.json"),
        help="Output path for results JSON (default: results/baseline_comparison.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run policy on (default: cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistical testing"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    env_kwargs = config.get_env_kwargs()

    # Check policy exists
    if not args.policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy_path}")

    print(f"\nEvaluating strategies with {args.n_episodes} episodes each...\n")

    # Evaluate RL policy
    print("1/4: Evaluating RL policy...")
    rl_results = evaluate_rl_policy(
        args.policy_path,
        config,
        env_kwargs,
        args.n_episodes,
        seed=args.seed,
        device=args.device,
    )

    # Evaluate random baseline
    print("\n2/4: Evaluating random baseline...")
    random_results = evaluate_baseline(
        random_therapist,
        env_kwargs,
        args.n_episodes,
        seed=args.seed,
    )

    # Evaluate complementary baseline
    print("\n3/4: Evaluating complementary baseline...")
    complement_results = evaluate_baseline(
        always_complement,
        env_kwargs,
        args.n_episodes,
        seed=args.seed,
    )

    # Evaluate optimal static baseline
    print("\n4/4: Evaluating optimal static baseline...")
    optimal_results = evaluate_baseline(
        optimal_static,
        env_kwargs,
        args.n_episodes,
        seed=args.seed,
    )

    # Collect results
    results = {
        'rl_policy': rl_results,
        'random': random_results,
        'complement': complement_results,
        'optimal_static': optimal_results,
    }

    # Statistical comparisons
    comparisons = None
    if not args.no_stats:
        print("\nPerforming statistical comparisons...")
        comparisons = compare_strategies(results, reference='rl_policy')

    # Print results table
    print_comparison_table(results, comparisons)

    # Generate plots
    if not args.no_plot:
        print("\nGenerating comparison plots...")
        plot_comparison(results, args.output.parent / "plots")

    # Save results
    save_results(results, comparisons, args.output)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
