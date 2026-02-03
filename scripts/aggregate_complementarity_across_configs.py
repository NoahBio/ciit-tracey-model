"""Aggregate complementarity visualization across multiple Optuna configs.

This script samples configs from an Optuna database uniformly across the v2_advantage
spectrum and aggregates complementarity dynamics across all configs.

Usage:
    python scripts/aggregate_complementarity_across_configs.py \
        --db-path optuna_studies/freq_amp_v2_optimization.db \
        --study-name freq_amp_v2_optimization \
        --n-configs 1000 \
        --n-seeds 10000 \
        --max-sessions 300
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import optuna
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import gc
import psutil
import os
import pickle

from visualize_complementarity import (
    run_simulation_with_complementarity_tracking,
    run_baseline_complementary_simulation,
    ComplementarityVisualizer,
)
from src.config import get_u_matrix_by_name, list_available_u_matrices


def log_memory_usage(label: str = ""):
    """Log current memory usage for monitoring."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / (1024 ** 2)
        print(f"[MEMORY] {label}: {mem_mb:.1f} MB")
    except NameError:
        # psutil not available
        pass


class IncrementalTrajectoryStats:
    """Online computation of mean and std for trajectories using Welford's algorithm.

    This allows computing statistics incrementally without storing all trajectories in memory.
    Memory usage: O(max_length) instead of O(n_results * max_length).

    Example:
        stats = IncrementalTrajectoryStats(max_length=300)
        for trajectory in trajectories:
            stats.update(trajectory)
        mean, std = stats.get_stats()
    """

    def __init__(self, max_length: int = 500):
        """Initialize with maximum expected trajectory length.

        Args:
            max_length: Initial capacity for trajectory length (will expand if needed)
        """
        self.max_length = max_length
        self.count = np.zeros(max_length, dtype=np.int64)
        self.mean = np.zeros(max_length, dtype=np.float64)
        self.m2 = np.zeros(max_length, dtype=np.float64)  # Sum of squared differences

    def update(self, trajectory: List[float]) -> None:
        """Add a single trajectory to the running statistics.

        Uses Welford's online algorithm for numerical stability.

        Args:
            trajectory: List of float values (variable length OK)
        """
        n = len(trajectory)

        # Dynamically expand arrays if needed
        if n > self.max_length:
            new_size = n
            self.count = np.pad(self.count, (0, new_size - self.max_length))
            self.mean = np.pad(self.mean, (0, new_size - self.max_length))
            self.m2 = np.pad(self.m2, (0, new_size - self.max_length))
            self.max_length = new_size

        # Welford's algorithm
        for i in range(n):
            if not np.isnan(trajectory[i]):
                self.count[i] += 1
                delta = trajectory[i] - self.mean[i]
                self.mean[i] += delta / self.count[i]
                delta2 = trajectory[i] - self.mean[i]
                self.m2[i] += delta * delta2

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current mean and std arrays.

        Returns:
            Tuple of (mean_array, std_array)
        """
        # Find actual max length used
        actual_length = int(np.max(np.where(self.count > 0)[0]) + 1) if np.any(self.count > 0) else 0

        if actual_length == 0:
            return np.array([]), np.array([])

        mean = self.mean[:actual_length].copy()

        # Calculate std with numerical stability check
        variance = np.zeros(actual_length)
        mask = self.count[:actual_length] > 0
        variance[mask] = self.m2[:actual_length][mask] / self.count[:actual_length][mask]
        std = np.sqrt(variance)

        # Set to NaN where no data
        mean[~mask] = np.nan
        std[~mask] = np.nan

        return mean, std

    def get_count(self) -> int:
        """Return total number of trajectories added."""
        return int(self.count[0]) if len(self.count) > 0 and self.count[0] > 0 else 0


@dataclass
class ConfigResult:
    """Results for a single config across all seeds."""
    config_idx: int
    trial_number: int
    params: Dict

    # Categorized results
    v2_advantage_results: List = field(default_factory=list)
    baseline_advantage_results: List = field(default_factory=list)
    both_success_results: List = field(default_factory=list)
    both_fail_results: List = field(default_factory=list)

    # Success tracking
    baseline_successes: List[bool] = field(default_factory=list)


def load_and_sample_configs(
    db_path: str,
    study_name: str,
    n_configs: int = 1000
) -> List[Dict]:
    """Load trials from Optuna DB and sample uniformly across v2_advantage spectrum.

    Args:
        db_path: Path to SQLite database
        study_name: Name of the study
        n_configs: Number of configs to sample

    Returns:
        List of config dicts with trial info and params
    """
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Get completed trials with valid advantage values
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None
              and not np.isnan(t.value)]

    if len(trials) == 0:
        raise ValueError("No completed trials found in database")

    print(f"Found {len(trials)} completed trials")

    # Sort by advantage value
    trials_sorted = sorted(trials, key=lambda t: t.value)
    advantages = [t.value for t in trials_sorted]

    print(f"Advantage range: {min(advantages):.2f} to {max(advantages):.2f}")

    # Sample uniformly across the advantage spectrum
    n_available = len(trials_sorted)
    if n_configs >= n_available:
        print(f"Requested {n_configs} configs, but only {n_available} available. Using all.")
        sampled_trials = trials_sorted
    else:
        # Sample uniformly by index
        indices = np.linspace(0, n_available - 1, n_configs, dtype=int)
        sampled_trials = [trials_sorted[i] for i in indices]

    # Convert to config dicts
    configs = []
    for i, trial in enumerate(sampled_trials):
        config = {
            'config_idx': i,
            'trial_number': trial.number,
            'advantage': trial.value,
            'params': trial.params.copy(),
        }
        configs.append(config)

    print(f"Sampled {len(configs)} configs")
    print(f"Sampled advantage range: {configs[0]['advantage']:.2f} to {configs[-1]['advantage']:.2f}")

    return configs


def run_config_simulations(
    config: Dict,
    n_seeds: int,
    max_sessions: int,
    mechanism: str = 'frequency_amplifier',
    pattern: str = 'cold_stuck',
    window_size: int = 10,
    u_matrix_name: Optional[str] = None,
) -> ConfigResult:
    """Run simulations for a single config across all seeds.

    Args:
        config: Config dict with params
        n_seeds: Number of seeds to simulate
        max_sessions: Maximum sessions per simulation
        mechanism: Client mechanism
        pattern: Initial memory pattern
        window_size: Complementarity tracking window

    Returns:
        ConfigResult with categorized simulation results
    """
    params = config['params']

    # Fixed parameters
    enable_parataxic = True
    entropy = 1.0

    # Build baseline simulation kwargs (no window_size)
    baseline_kwargs = {
        'mechanism': mechanism,
        'pattern': pattern,
        'success_threshold_percentile': params.get('threshold', 0.8),
        'enable_parataxic': enable_parataxic,
        'baseline_accuracy': params.get('baseline_accuracy', 0.5),
        'max_sessions': max_sessions,
        'entropy': entropy,
        'history_weight': 1.0,
        'bond_power': 1.0,
        'bond_alpha': params.get('bond_alpha', 5.0),
        'bond_offset': params.get('bond_offset', 0.8),
        'recency_weighting_factor': params.get('recency_weighting_factor', 2),
    }

    # V2 simulation kwargs (includes window_size)
    v2_sim_kwargs = baseline_kwargs.copy()
    v2_sim_kwargs['window_size'] = window_size

    # V2-specific params
    v2_kwargs = v2_sim_kwargs.copy()
    v2_kwargs.update({
        'therapist_version': 'v2',
        'perception_window': params.get('perception_window', 15),
        'seeding_benefit_scaling': params.get('seeding_benefit_scaling', 0.3),
        'skip_seeding_accuracy_threshold': params.get('skip_seeding_accuracy_threshold', 0.9),
        'quick_seed_actions_threshold': params.get('quick_seed_actions_threshold', 3),
        'abort_consecutive_failures_threshold': params.get('abort_consecutive_failures_threshold', 5),
    })

    # Initialize result storage
    result = ConfigResult(
        config_idx=config['config_idx'],
        trial_number=config['trial_number'],
        params=params,
    )

    # Run simulations for each seed
    for seed in range(n_seeds):
        # Baseline complementary therapist
        baseline_success = run_baseline_complementary_simulation(
            seed=seed,
            u_matrix_name=u_matrix_name,
            **baseline_kwargs
        )
        result.baseline_successes.append(baseline_success)

        # V2 omniscient therapist with complementarity tracking
        v2_result = run_simulation_with_complementarity_tracking(
            seed=seed,
            u_matrix_name=u_matrix_name,
            **v2_kwargs
        )

        # Categorize based on success pattern
        if v2_result.success and not baseline_success:
            result.v2_advantage_results.append(v2_result)
        elif baseline_success and not v2_result.success:
            result.baseline_advantage_results.append(v2_result)
        elif v2_result.success and baseline_success:
            result.both_success_results.append(v2_result)
        else:  # Neither succeeded
            result.both_fail_results.append(v2_result)

    return result


def process_config_wrapper(args):
    """Wrapper for multiprocessing."""
    config, n_seeds, max_sessions, mechanism, pattern, window_size, u_matrix_name = args
    return run_config_simulations(
        config, n_seeds, max_sessions, mechanism, pattern, window_size, u_matrix_name
    )


def save_config_checkpoint(result: ConfigResult, output_dir: Path) -> None:
    """Save a single ConfigResult to disk as a checkpoint.

    Uses pickle for fast serialization. This enables recovery from failures
    and reduces memory pressure.

    Args:
        result: ConfigResult to save
        output_dir: Directory to save checkpoint files
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / f"config_{result.config_idx:04d}_trial_{result.trial_number}.pkl"

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_config_checkpoint(checkpoint_path: Path) -> ConfigResult:
    """Load a ConfigResult from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint pickle file

    Returns:
        Loaded ConfigResult
    """
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def aggregate_trajectories(all_results: List) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate complementarity trajectories across multiple simulation results.

    NOTE: This function is DEPRECATED for large-scale aggregation. Use
    IncrementalTrajectoryStats for memory-efficient online aggregation.
    Kept for backward compatibility with small datasets.

    Args:
        all_results: List of SimulationResult objects

    Returns:
        Tuple of (mean_trajectory, std_trajectory) as numpy arrays
    """
    if not all_results:
        return np.array([]), np.array([])

    # Find max length
    max_length = max(len(r.overall_enacted_distance_trajectory) for r in all_results)

    # Initialize array
    n_results = len(all_results)
    trajectories = np.full((n_results, max_length), np.nan)

    # Fill trajectories
    for i, result in enumerate(all_results):
        length = len(result.overall_enacted_distance_trajectory)
        trajectories[i, :length] = result.overall_enacted_distance_trajectory

    # Calculate mean and std
    mean_traj = np.nanmean(trajectories, axis=0)
    std_traj = np.nanstd(trajectories, axis=0)

    return mean_traj, std_traj


def aggregate_trajectories_online(stats: IncrementalTrajectoryStats) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate trajectories using pre-computed online statistics.

    Memory-efficient replacement for aggregate_trajectories() for large datasets.

    Args:
        stats: IncrementalTrajectoryStats object with accumulated statistics

    Returns:
        Tuple of (mean_trajectory, std_trajectory) as numpy arrays
    """
    return stats.get_stats()


def visualize_aggregated_results(
    v2_adv_mean: np.ndarray,
    v2_adv_std: np.ndarray,
    baseline_adv_mean: np.ndarray,
    baseline_adv_std: np.ndarray,
    remaining_mean: np.ndarray,
    remaining_std: np.ndarray,
    breakdown: Dict[str, int],
    output_path: Path,
    title_suffix: str = "",
):
    """Create visualization of aggregated complementarity results.

    Args:
        v2_adv_mean/std: V2 advantage trajectories
        baseline_adv_mean/std: Baseline advantage trajectories
        remaining_mean/std: Remaining trajectories
        breakdown: Seed count breakdown dict
        output_path: Where to save the plot
        title_suffix: Additional title text
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 0.6]})
    ax_comp = axes[0]
    ax_table = axes[1]

    # Plot V2 advantage (green)
    if len(v2_adv_mean) > 0:
        sessions = np.arange(len(v2_adv_mean)) + 1
        ax_comp.plot(sessions, v2_adv_mean,
                    color='#00AA00', linewidth=3, alpha=1.0,
                    label=f'V2 Advantage (n={breakdown["v2_advantage"]})')
        ax_comp.fill_between(sessions,
                            v2_adv_mean - v2_adv_std,
                            v2_adv_mean + v2_adv_std,
                            color='#00AA00', alpha=0.2)

    # Plot baseline advantage (red)
    if len(baseline_adv_mean) > 0:
        sessions = np.arange(len(baseline_adv_mean)) + 1
        ax_comp.plot(sessions, baseline_adv_mean,
                    color='#CC0000', linewidth=3, alpha=1.0,
                    label=f'Baseline Advantage (n={breakdown["baseline_advantage"]})')
        ax_comp.fill_between(sessions,
                            baseline_adv_mean - baseline_adv_std,
                            baseline_adv_mean + baseline_adv_std,
                            color='#CC0000', alpha=0.2)

    # Plot remaining (black)
    if len(remaining_mean) > 0:
        sessions = np.arange(len(remaining_mean)) + 1
        ax_comp.plot(sessions, remaining_mean,
                    color='#333333', linewidth=2, alpha=0.8,
                    label=f'Remaining (n={breakdown["both_success"] + breakdown["both_fail"]})')
        ax_comp.fill_between(sessions,
                            remaining_mean - remaining_std,
                            remaining_mean + remaining_std,
                            color='#333333', alpha=0.2)

    # Formatting
    ax_comp.set_xlabel('Session Number', fontsize=14, fontweight='bold')
    ax_comp.set_ylabel('Mean Octant Distance', fontsize=14, fontweight='bold')
    ax_comp.set_ylim(-0.2, 4.2)
    ax_comp.set_title(f'Aggregated Octant Distance Over Time{title_suffix}',
                     fontsize=16, fontweight='bold', pad=15)
    ax_comp.axhline(y=0, color='green', linestyle=':', alpha=0.5, linewidth=2)
    ax_comp.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_comp.grid(True, alpha=0.3, linewidth=0.8)
    ax_comp.tick_params(axis='both', which='major', labelsize=12)

    # Add seed breakdown table
    ax_table.axis('off')
    total_seeds = sum(breakdown.values())

    if total_seeds > 0:
        table_data = [
            ['V2 Advantage', 'Baseline Advantage', 'Both Success', 'Both Fail', 'Total'],
            [
                f"{breakdown['v2_advantage']} ({100*breakdown['v2_advantage']/total_seeds:.1f}%)",
                f"{breakdown['baseline_advantage']} ({100*breakdown['baseline_advantage']/total_seeds:.1f}%)",
                f"{breakdown['both_success']} ({100*breakdown['both_success']/total_seeds:.1f}%)",
                f"{breakdown['both_fail']} ({100*breakdown['both_fail']/total_seeds:.1f}%)",
                f"{total_seeds}",
            ],
        ]
    else:
        table_data = [
            ['V2 Advantage', 'Baseline Advantage', 'Both Success', 'Both Fail', 'Total'],
            [
                f"{breakdown['v2_advantage']} (0.0%)",
                f"{breakdown['baseline_advantage']} (0.0%)",
                f"{breakdown['both_success']} (0.0%)",
                f"{breakdown['both_fail']} (0.0%)",
                "0",
            ],
        ]

    cell_colors = [
        ['#E8E8E8', '#E8E8E8', '#E8E8E8', '#E8E8E8', '#E8E8E8'],
        ['#90EE90', '#FFCCCB', '#ADD8E6', '#D3D3D3', 'white'],
    ]

    table = ax_table.table(
        cellText=[table_data[1]],
        colLabels=table_data[0],
        cellColours=[cell_colors[1]],
        colColours=cell_colors[0],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    ax_table.set_title('Seed Breakdown by Outcome', fontsize=12, fontweight='bold', pad=10)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate complementarity across multiple Optuna configs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Database parameters
    parser.add_argument('--db-path', type=str,
                       default='optuna_studies/freq_amp_v2_optimization.db',
                       help='Path to Optuna database')
    parser.add_argument('--study-name', type=str,
                       default='freq_amp_v2_optimization',
                       help='Name of the Optuna study')

    # Sampling parameters
    parser.add_argument('--n-configs', type=int, default=1000,
                       help='Number of configs to sample from database')
    parser.add_argument('--n-seeds', type=int, default=10000,
                       help='Number of seeds per config')

    # Simulation parameters
    parser.add_argument('--max-sessions', type=int, default=300,
                       help='Maximum sessions per simulation')
    parser.add_argument('--window-size', type=int, default=10,
                       help='Complementarity tracking window size')
    parser.add_argument('--mechanism', type=str, default='frequency_amplifier',
                       help='Client mechanism')
    parser.add_argument('--pattern', type=str, default='cold_stuck',
                       help='Initial memory pattern')

    parser.add_argument('--u-matrix', type=str, default=None,
                       help=f'Named U-matrix to use (default: random sampling). '
                            f'Available: {", ".join(list_available_u_matrices())}')

    # Performance parameters
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of configs to process in each batch')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: timestamped directory)')

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "results" / f"aggregated_complementarity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AGGREGATED COMPLEMENTARITY VISUALIZATION")
    print("=" * 80)
    print(f"Database: {args.db_path}")
    print(f"Study: {args.study_name}")
    print(f"Configs to sample: {args.n_configs}")
    print(f"Seeds per config: {args.n_seeds}")
    print(f"Max sessions: {args.max_sessions}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load and sample configs
    print("\nLoading configs from database...")
    db_path = project_root / args.db_path
    configs = load_and_sample_configs(
        str(db_path),
        args.study_name,
        args.n_configs
    )

    # Save config info
    config_info = {
        'timestamp': datetime.now().isoformat(),
        'n_configs': len(configs),
        'n_seeds_per_config': args.n_seeds,
        'u_matrix_name': args.u_matrix if args.u_matrix else "random_sampled",
        'total_simulations': len(configs) * args.n_seeds * 2,  # x2 for baseline + V2
        'configs': [
            {
                'idx': c['config_idx'],
                'trial': c['trial_number'],
                'advantage': c['advantage'],
            }
            for c in configs
        ],
    }

    with open(output_dir / "config_info.json", "w") as f:
        json.dump(config_info, f, indent=2)

    # Initialize workers
    n_workers = args.n_workers or mp.cpu_count()
    print(f"\nProcessing configs with {n_workers} workers...")
    print(f"Total simulations: {len(configs) * args.n_seeds * 2:,}")
    print(f"(This will take a while...)\n")

    # Initialize online statistics accumulators (memory-efficient approach)
    print("\nInitializing online statistics accumulators...")
    v2_adv_stats = IncrementalTrajectoryStats(max_length=args.max_sessions)
    baseline_adv_stats = IncrementalTrajectoryStats(max_length=args.max_sessions)
    both_success_stats = IncrementalTrajectoryStats(max_length=args.max_sessions)
    both_fail_stats = IncrementalTrajectoryStats(max_length=args.max_sessions)

    # Counters for breakdown
    breakdown_counts = {
        'v2_advantage': 0,
        'baseline_advantage': 0,
        'both_success': 0,
        'both_fail': 0,
    }

    log_memory_usage("After initializing online stats")

    # Process configs in parallel with online aggregation
    print(f"\nProcessing configs with {n_workers} workers (memory-efficient mode)...")
    print(f"Total simulations: {len(configs) * args.n_seeds * 2:,}")
    print(f"Checkpoints will be saved to: {output_dir / 'checkpoints'}")
    print(f"(This will take a while...)\n")

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        # Submit all tasks
        task_args = [
            (config, args.n_seeds, args.max_sessions,
             args.mechanism, args.pattern, args.window_size, args.u_matrix)
            for config in configs
        ]

        futures = {
            executor.submit(process_config_wrapper, task_arg): i
            for i, task_arg in enumerate(task_args)
        }

        # Collect results with progress bar and immediate online aggregation
        with tqdm(total=len(configs), desc="Processing configs") as pbar:
            for future in as_completed(futures):
                try:
                    config_result = future.result(timeout=600)  # 10 minute timeout per config

                    # Save checkpoint to disk immediately
                    save_config_checkpoint(config_result, output_dir)

                    # Update online statistics (memory-efficient aggregation)
                    for sim_result in config_result.v2_advantage_results:
                        v2_adv_stats.update(sim_result.overall_enacted_distance_trajectory)
                        breakdown_counts['v2_advantage'] += 1

                    for sim_result in config_result.baseline_advantage_results:
                        baseline_adv_stats.update(sim_result.overall_enacted_distance_trajectory)
                        breakdown_counts['baseline_advantage'] += 1

                    for sim_result in config_result.both_success_results:
                        both_success_stats.update(sim_result.overall_enacted_distance_trajectory)
                        breakdown_counts['both_success'] += 1

                    for sim_result in config_result.both_fail_results:
                        both_fail_stats.update(sim_result.overall_enacted_distance_trajectory)
                        breakdown_counts['both_fail'] += 1

                    # Free memory immediately - don't keep ConfigResult in RAM!
                    del config_result

                    # Remove future from dict to allow garbage collection
                    del futures[future]

                    # Periodic garbage collection and memory logging
                    if pbar.n % 100 == 0 and pbar.n > 0:
                        gc.collect()
                        log_memory_usage(f"After {pbar.n} configs")

                    pbar.update(1)

                except TimeoutError:
                    config_idx = futures[future]
                    print(f"\nTimeout processing config {config_idx}")
                    del futures[future]  # Free future even on timeout
                    pbar.update(1)
                except Exception as e:
                    config_idx = futures[future]
                    print(f"\nError processing config {config_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    del futures[future]  # Free future even on error
                    pbar.update(1)

    print(f"\nSuccessfully processed {sum(breakdown_counts.values()):,} total simulations")
    log_memory_usage("After all configs processed")

    # Calculate breakdown
    breakdown = breakdown_counts

    print(f"\nSeed breakdown across all configs:")
    print(f"  V2 advantage: {breakdown['v2_advantage']:,}")
    print(f"  Baseline advantage: {breakdown['baseline_advantage']:,}")
    print(f"  Both success: {breakdown['both_success']:,}")
    print(f"  Both fail: {breakdown['both_fail']:,}")
    print(f"  Total: {sum(breakdown.values()):,}")

    # Extract mean/std from online stats (NO LARGE ARRAY CREATION!)
    print("\nExtracting aggregated trajectories from online statistics...")
    v2_adv_mean, v2_adv_std = aggregate_trajectories_online(v2_adv_stats)
    baseline_adv_mean, baseline_adv_std = aggregate_trajectories_online(baseline_adv_stats)

    # Combine both_success and both_fail for "remaining" category
    print("Combining both_success and both_fail into 'remaining' category...")
    remaining_stats = IncrementalTrajectoryStats(max_length=args.max_sessions)

    # Re-aggregate from checkpoints for remaining category
    # (This is the only place we reload from disk, but just to combine two categories)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_files = sorted(checkpoint_dir.glob("config_*.pkl"))

    print(f"Re-aggregating {len(checkpoint_files)} configs for 'remaining' category...")
    for checkpoint_path in tqdm(checkpoint_files, desc="Merging remaining"):
        config_result = load_config_checkpoint(checkpoint_path)

        # Add both_success and both_fail to remaining stats
        for sim_result in config_result.both_success_results:
            remaining_stats.update(sim_result.overall_enacted_distance_trajectory)

        for sim_result in config_result.both_fail_results:
            remaining_stats.update(sim_result.overall_enacted_distance_trajectory)

        # Free memory
        del config_result

    remaining_mean, remaining_std = aggregate_trajectories_online(remaining_stats)

    # Free the large stats objects
    del v2_adv_stats, baseline_adv_stats, both_success_stats, both_fail_stats, remaining_stats
    gc.collect()

    log_memory_usage("After aggregation complete")
    print("\nMemory-efficient aggregation complete!")

    # Save aggregated data
    np.savez(
        output_dir / "aggregated_trajectories.npz",
        v2_adv_mean=v2_adv_mean,
        v2_adv_std=v2_adv_std,
        baseline_adv_mean=baseline_adv_mean,
        baseline_adv_std=baseline_adv_std,
        remaining_mean=remaining_mean,
        remaining_std=remaining_std,
    )

    # Create visualization
    print("\nGenerating visualization...")
    title_suffix = f" ({args.n_configs} configs, {args.n_seeds:,} seeds/config)"

    fig = visualize_aggregated_results(
        v2_adv_mean, v2_adv_std,
        baseline_adv_mean, baseline_adv_std,
        remaining_mean, remaining_std,
        breakdown,
        output_dir / "aggregated_complementarity.png",
        title_suffix=title_suffix
    )

    # Save summary statistics
    summary = {
        'breakdown': breakdown,
        'v2_advantage_final_mean_distance': float(v2_adv_mean[-1]) if len(v2_adv_mean) > 0 else None,
        'baseline_advantage_final_mean_distance': float(baseline_adv_mean[-1]) if len(baseline_adv_mean) > 0 else None,
        'remaining_final_mean_distance': float(remaining_mean[-1]) if len(remaining_mean) > 0 else None,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to: {output_dir}")
    print("  - aggregated_complementarity.png")
    print("  - aggregated_trajectories.npz")
    print("  - config_info.json")
    print("  - summary.json")

    plt.show()


if __name__ == "__main__":
    main()
