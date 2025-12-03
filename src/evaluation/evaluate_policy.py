"""Evaluate trained therapist RL policy."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.environment import TherapyEnv
from src.training import TrainingConfig, load_config, make_therapy_networks


def load_policy(policy_path: Path, config: TrainingConfig, device: str = "cpu"):
    """
    Load trained policy from checkpoint.

    Parameters
    ----------
    policy_path : Path
        Path to policy checkpoint (.pth file)
    config : TrainingConfig
        Training configuration
    device : str
        Device to load policy on

    Returns
    -------
    actor : nn.Module
        Loaded actor network
    """
    # Create dummy environment to get spaces
    env = TherapyEnv(**config.get_env_kwargs(), random_state=42)
    obs_space = env.observation_space  # type: ignore[assignment]
    act_space = env.action_space  # type: ignore[assignment]

    # Create networks
    actor, _ = make_therapy_networks(
        observation_space=obs_space,  # type: ignore[arg-type]
        action_space=act_space,  # type: ignore[arg-type]
        hidden_sizes=(config.hidden_size, config.hidden_size),
        device=device
    )

    # Load state dict
    if policy_path.suffix == '.pth':
        state_dict = torch.load(policy_path, map_location=device)
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        actor.load_state_dict(state_dict, strict=False)

    actor.eval()
    return actor


def evaluate_episode(
    env: TherapyEnv,
    actor: torch.nn.Module,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run single evaluation episode.

    Parameters
    ----------
    env : TherapyEnv
        Environment instance
    actor : nn.Module
        Policy network
    device : str
        Device for inference

    Returns
    -------
    dict
        Episode results with keys:
        - reward: float
        - length: int
        - success: bool
        - dropout: bool
        - final_rs: float
        - actions: List[int]
        - rs_trajectory: List[float]
        - complementarity_trajectory: List[float]
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    actions = []
    rs_trajectory = [info.get('RS', 0.0)]
    complementarity_trajectory = [info.get('complementarity', 0.0)]

    while not done:
        # Convert observation to format expected by network
        obs_tensor = {
            'client_action': torch.tensor([obs['client_action']], dtype=torch.long, device=device),
            'session_number': torch.tensor([obs['session_number']], dtype=torch.float32, device=device),
            'history': torch.tensor([obs['history']], dtype=torch.long, device=device)
        }

        # Create simple namespace object for network
        class ObsBatch:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        obs_batch = ObsBatch(obs_tensor)

        # Get action from policy
        with torch.no_grad():
            action_probs, _ = actor(obs_batch)
            action = action_probs.argmax(dim=-1).item()

        actions.append(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Track trajectories
        rs_trajectory.append(info.get('RS', rs_trajectory[-1]))
        complementarity_trajectory.append(info.get('complementarity', complementarity_trajectory[-1]))

    return {
        'reward': total_reward,
        'length': len(actions),
        'success': info.get('success', False),
        'dropout': info.get('dropout', False),
        'final_rs': rs_trajectory[-1],
        'actions': actions,
        'rs_trajectory': rs_trajectory,
        'complementarity_trajectory': complementarity_trajectory,
        'pattern': getattr(env, 'client', None).initial_pattern if hasattr(env, 'client') else 'unknown'  # type: ignore[attr-defined]
    }


def evaluate_policy(
    policy_path: Path,
    config: TrainingConfig,
    n_episodes: int = 100,
    device: str = "cpu",
    patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate policy across multiple episodes.

    Parameters
    ----------
    policy_path : Path
        Path to policy checkpoint
    config : TrainingConfig
        Configuration
    n_episodes : int
        Number of episodes per pattern
    device : str
        Device for inference
    patterns : List[str], optional
        Patterns to evaluate. If None, uses config.patterns

    Returns
    -------
    dict
        Evaluation results
    """
    print(f"Loading policy from {policy_path}...")
    actor = load_policy(policy_path, config, device)

    if patterns is None:
        patterns = config.patterns

    print(f"Evaluating on patterns: {patterns}")
    print(f"Episodes per pattern: {n_episodes}")

    results = {
        'overall': defaultdict(list),
        'by_pattern': {pattern: defaultdict(list) for pattern in patterns}
    }

    # Run evaluation episodes
    total_episodes = len(patterns) * n_episodes
    episode_count = 0

    for pattern in patterns:
        print(f"\nEvaluating pattern: {pattern}")

        for ep in range(n_episodes):
            # Create environment for this pattern
            env_kwargs = config.get_env_kwargs()
            env_kwargs['pattern'] = pattern
            env = TherapyEnv(**env_kwargs, random_state=config.seed + episode_count)

            # Run episode
            episode_result = evaluate_episode(env, actor, device)

            # Store results
            for key, value in episode_result.items():
                results['by_pattern'][pattern][key].append(value)
                results['overall'][key].append(value)

            episode_count += 1

            if (ep + 1) % 10 == 0:
                print(f"  Completed {ep + 1}/{n_episodes} episodes")

    print(f"\nCompleted {total_episodes} episodes")

    return results


def compute_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary metrics from evaluation results.

    Parameters
    ----------
    results : dict
        Raw evaluation results

    Returns
    -------
    dict
        Computed metrics
    """
    metrics = {}

    # Overall metrics
    overall = results['overall']
    metrics['overall'] = {
        'success_rate': np.mean(overall['success']),
        'dropout_rate': np.mean(overall['dropout']),
        'mean_reward': np.mean(overall['reward']),
        'std_reward': np.std(overall['reward']),
        'mean_length': np.mean(overall['length']),
        'std_length': np.std(overall['length']),
        'mean_final_rs': np.mean(overall['final_rs']),
        'std_final_rs': np.std(overall['final_rs'])
    }

    # Success-only metrics
    successful_episodes = [i for i, success in enumerate(overall['success']) if success]
    if successful_episodes:
        success_lengths = [overall['length'][i] for i in successful_episodes]
        metrics['overall']['mean_sessions_to_success'] = np.mean(success_lengths)
        metrics['overall']['std_sessions_to_success'] = np.std(success_lengths)
    else:
        metrics['overall']['mean_sessions_to_success'] = None
        metrics['overall']['std_sessions_to_success'] = None

    # Per-pattern metrics
    metrics['by_pattern'] = {}
    for pattern, pattern_results in results['by_pattern'].items():
        pattern_metrics = {
            'success_rate': np.mean(pattern_results['success']),
            'dropout_rate': np.mean(pattern_results['dropout']),
            'mean_reward': np.mean(pattern_results['reward']),
            'std_reward': np.std(pattern_results['reward']),
            'mean_length': np.mean(pattern_results['length']),
            'mean_final_rs': np.mean(pattern_results['final_rs'])
        }

        # Success-only metrics per pattern
        successful = [i for i, s in enumerate(pattern_results['success']) if s]
        if successful:
            success_lengths = [pattern_results['length'][i] for i in successful]
            pattern_metrics['mean_sessions_to_success'] = np.mean(success_lengths)
        else:
            pattern_metrics['mean_sessions_to_success'] = None

        metrics['by_pattern'][pattern] = pattern_metrics

    # Action distribution (overall)
    all_actions = [action for actions in overall['actions'] for action in actions]
    action_counts = np.bincount(all_actions, minlength=8)
    metrics['action_distribution'] = (action_counts / action_counts.sum()).tolist()

    return metrics


def print_summary(metrics: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Overall metrics
    overall = metrics['overall']
    print("\nOverall Performance:")
    print(f"  Success Rate:        {overall['success_rate']:.2%}")
    print(f"  Dropout Rate:        {overall['dropout_rate']:.2%}")
    print(f"  Mean Reward:         {overall['mean_reward']:.2f} ± {overall['std_reward']:.2f}")
    print(f"  Mean Episode Length: {overall['mean_length']:.1f} ± {overall['std_length']:.1f} sessions")
    print(f"  Mean Final RS:       {overall['mean_final_rs']:.3f} ± {overall['std_final_rs']:.3f}")

    if overall['mean_sessions_to_success'] is not None:
        print(f"  Sessions to Success: {overall['mean_sessions_to_success']:.1f} ± {overall['std_sessions_to_success']:.1f}")
    else:
        print(f"  Sessions to Success: N/A (no successful episodes)")

    # Per-pattern metrics
    print("\nPer-Pattern Performance:")
    print(f"{'Pattern':<25} {'Success':>10} {'Dropout':>10} {'Reward':>12} {'Length':>10}")
    print("-" * 80)

    for pattern, pm in metrics['by_pattern'].items():
        print(f"{pattern:<25} {pm['success_rate']:>9.1%} {pm['dropout_rate']:>9.1%} "
              f"{pm['mean_reward']:>11.1f} {pm['mean_length']:>9.1f}")

    # Action distribution
    print("\nTherapist Action Distribution:")
    action_names = ['D', 'WD', 'W', 'WS', 'S', 'CS', 'C', 'CD']
    for i, (name, prob) in enumerate(zip(action_names, metrics['action_distribution'])):
        bar = '█' * int(prob * 40)
        print(f"  {name:>3}: {bar:<40} {prob:.1%}")

    print("=" * 80)


def plot_results(results: Dict[str, Any], metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate visualization plots.

    Parameters
    ----------
    results : dict
        Raw evaluation results
    metrics : dict
        Computed metrics
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Success rate by pattern (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    patterns = list(metrics['by_pattern'].keys())
    success_rates = [metrics['by_pattern'][p]['success_rate'] * 100 for p in patterns]

    bars = ax.bar(range(len(patterns)), success_rates, color='steelblue', alpha=0.7)
    ax.set_xlabel('Pattern', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Client Pattern', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, rotation=45, ha='right')
    ax.set_ylim(0, 100)  # Fixed: was [0, 100] which is a list
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_pattern.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'success_by_pattern.png'}")

    # 2. RS trajectory (line plot with confidence bands)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get all RS trajectories and pad to same length
    all_rs_trajectories = results['overall']['rs_trajectory']
    max_length = max(len(traj) for traj in all_rs_trajectories)

    # Pad trajectories and compute mean/std
    padded = np.full((len(all_rs_trajectories), max_length), np.nan)
    for i, traj in enumerate(all_rs_trajectories):
        padded[i, :len(traj)] = traj

    mean_rs = np.nanmean(padded, axis=0)
    std_rs = np.nanstd(padded, axis=0)
    sessions = np.arange(max_length)

    ax.plot(sessions, mean_rs, 'b-', linewidth=2, label='Mean RS')
    ax.fill_between(sessions, mean_rs - std_rs, mean_rs + std_rs,
                     alpha=0.3, color='blue', label='±1 SD')

    ax.set_xlabel('Session Number', fontsize=12)
    ax.set_ylabel('Relationship Satisfaction (RS)', fontsize=12)
    ax.set_title('RS Trajectory Over Sessions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rs_trajectory.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'rs_trajectory.png'}")

    # 3. Therapist action distribution (histogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    action_names = ['D', 'WD', 'W', 'WS', 'S', 'CS', 'C', 'CD']
    action_probs = [p * 100 for p in metrics['action_distribution']]

    bars = ax.bar(action_names, action_probs, color='coral', alpha=0.7)
    ax.set_xlabel('Therapist Action (Octant)', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Therapist Action Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_distribution.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'action_distribution.png'}")

    # 4. Complementarity trajectory
    fig, ax = plt.subplots(figsize=(12, 6))

    all_comp_trajectories = results['overall']['complementarity_trajectory']
    max_length = max(len(traj) for traj in all_comp_trajectories)

    padded = np.full((len(all_comp_trajectories), max_length), np.nan)
    for i, traj in enumerate(all_comp_trajectories):
        padded[i, :len(traj)] = traj

    mean_comp = np.nanmean(padded, axis=0)
    std_comp = np.nanstd(padded, axis=0)
    sessions = np.arange(max_length)

    ax.plot(sessions, mean_comp, 'g-', linewidth=2, label='Mean Complementarity')
    ax.fill_between(sessions, mean_comp - std_comp, mean_comp + std_comp,
                     alpha=0.3, color='green', label='±1 SD')

    ax.set_xlabel('Session Number', fontsize=12)
    ax.set_ylabel('Complementarity Score', fontsize=12)
    ax.set_title('Complementarity Trajectory Over Sessions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'complementarity_trajectory.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'complementarity_trajectory.png'}")


def save_results(results: Dict[str, Any], metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Save evaluation results to JSON.

    Parameters
    ----------
    results : dict
        Raw evaluation results
    metrics : dict
        Computed metrics
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, defaultdict):
            return {k: convert_to_serializable(v) for k, v in dict(obj).items()}
        return obj

    # Save metrics summary
    metrics_serializable = convert_to_serializable(metrics)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\nSaved: {output_dir / 'metrics.json'}")

    # Save full results (without trajectories to keep file size reasonable)
    results_summary = {
        'overall': {
            'rewards': results['overall']['reward'],
            'lengths': results['overall']['length'],
            'successes': results['overall']['success'],
            'dropouts': results['overall']['dropout'],
            'final_rs': results['overall']['final_rs']
        },
        'by_pattern': {}
    }

    for pattern, pattern_results in results['by_pattern'].items():
        results_summary['by_pattern'][pattern] = {
            'rewards': pattern_results['reward'],
            'lengths': pattern_results['length'],
            'successes': pattern_results['success'],
            'dropouts': pattern_results['dropout'],
            'final_rs': pattern_results['final_rs']
        }

    results_serializable = convert_to_serializable(results_summary)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Saved: {output_dir / 'results.json'}")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained therapist RL policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Policy and config
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML. If not provided, must specify environment parameters."
    )

    # Evaluation settings
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes per pattern"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to results/{policy_name}/"
    )

    # Environment parameters (override config if provided)
    parser.add_argument(
        "--mechanism",
        type=str,
        default=None,
        choices=[
            "bond_only",
            "frequency_amplifier",
            "conditional_amplifier",
            "bond_weighted_frequency_amplifier",
            "bond_weighted_conditional_amplifier"
        ],
        help="Client mechanism (overrides config)"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Client patterns to evaluate (overrides config)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Success threshold (overrides config)"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        if args.mechanism is None or args.patterns is None:
            print("Error: Must provide either --config or both --mechanism and --patterns")
            sys.exit(1)

        config = TrainingConfig(
            mechanism=args.mechanism,
            patterns=args.patterns,
            threshold=args.threshold if args.threshold else 0.9,
            seed=args.seed
        )

    # Override config with command line args if provided
    if args.mechanism:
        config.mechanism = args.mechanism
    if args.patterns:
        config.patterns = args.patterns
    if args.threshold:
        config.threshold = args.threshold

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        policy_name = Path(args.policy_path).parent.name
        output_dir = Path("results") / policy_name

    print("\n" + "=" * 80)
    print("POLICY EVALUATION")
    print("=" * 80)
    print(f"Policy: {args.policy_path}")
    print(f"Mechanism: {config.mechanism}")
    print(f"Patterns: {config.patterns}")
    print(f"Episodes per pattern: {args.n_episodes}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    # Run evaluation
    try:
        results = evaluate_policy(
            policy_path=Path(args.policy_path),
            config=config,
            n_episodes=args.n_episodes,
            device=args.device,
            patterns=config.patterns
        )

        # Compute metrics
        metrics = compute_metrics(results)

        # Print summary
        print_summary(metrics)

        # Save results
        save_results(results, metrics, output_dir)

        # Generate plots
        print("\nGenerating plots...")
        plot_results(results, metrics, output_dir)

        print("\n" + "=" * 80)
        print(f"Evaluation complete! Results saved to {output_dir}/")
        print("=" * 80)

    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
