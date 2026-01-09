"""Visualization script for complementarity dynamics in therapy simulations.

This script runs therapy simulations and visualizes how complementarity evolves
over time, with support for warm/cold filtering and comparison of different
client mechanisms, initial memory patterns, and therapist versions.

Usage:
    python scripts/visualize_complementarity.py \
        --mechanisms frequency_amplifier \
        --patterns cold_stuck \
        --therapist-versions v2 \
        --n-seeds 30
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import argparse
from tqdm import tqdm

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
    OmniscientStrategicTherapistV1,
    OmniscientStrategicTherapistV2,
)
from src import config
from src.config import sample_u_matrix, calculate_success_threshold
from src.analysis.complementarity_tracker import ComplementarityTracker


# Color palette for client mechanisms
MECHANISM_COLORS = {
    'bond_only': '#FF6B6B',
    'frequency_amplifier': '#4ECDC4',
    'conditional_amplifier': '#45B7D1',
    'bond_weighted_conditional_amplifier': '#96CEB4',
    'bond_weighted_frequency_amplifier': '#FFEAA7',
}

# Line styles for memory patterns
PATTERN_LINESTYLES = {
    'cold_stuck': '-',
    'dominant_stuck': '--',
    'submissive_stuck': '-.',
    'cold_warm': ':',
    'complementary_perfect': (0, (3, 1, 1, 1)),
    'conflictual': (0, (5, 1)),
    'mixed_random': (0, (1, 1)),
}

# Marker styles for therapist versions
THERAPIST_MARKERS = {
    'v1': 'o',
    'v2': 's',
}


@dataclass
class SimulationResult:
    """Results from a single simulation run with complementarity tracking."""
    seed: int
    mechanism: str
    pattern: str
    therapist_version: str

    success: bool
    total_sessions: int
    rs_threshold: float

    # Complementarity trajectories
    overall_enacted_trajectory: List[float]
    warm_enacted_trajectory: List[float]
    cold_enacted_trajectory: List[float]
    overall_perceived_trajectory: List[float] = field(default_factory=list)
    warm_perceived_trajectory: List[float] = field(default_factory=list)
    cold_perceived_trajectory: List[float] = field(default_factory=list)

    # Additional tracking
    final_rs: float = 0.0
    final_bond: float = 0.0


@dataclass
class AggregatedResults:
    """Aggregated results across multiple seeds."""
    config_name: str
    mechanism: str
    pattern: str
    therapist_version: str
    n_runs: int
    success_rate: float

    # Overall complementarity
    mean_overall_enacted: np.ndarray
    std_overall_enacted: np.ndarray
    mean_overall_perceived: np.ndarray
    std_overall_perceived: np.ndarray

    # Warm complementarity
    mean_warm_enacted: np.ndarray
    std_warm_enacted: np.ndarray
    mean_warm_perceived: np.ndarray
    std_warm_perceived: np.ndarray

    # Cold complementarity
    mean_cold_enacted: np.ndarray
    std_cold_enacted: np.ndarray
    mean_cold_perceived: np.ndarray
    std_cold_perceived: np.ndarray


def run_simulation_with_complementarity_tracking(
    seed: int,
    mechanism: str,
    pattern: str,
    therapist_version: str,
    window_size: int = 10,
    max_sessions: int = 100,
    success_threshold_percentile: float = 0.8,
    enable_parataxic: bool = False,
    entropy: float = 0.1,
    baseline_accuracy: float = 0.5,
    perception_window: int = 15,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 5.0,
    bond_offset: float = 0.7,
    recency_weighting_factor: int = 2,
    seeding_benefit_scaling: float = 0.3,
    skip_seeding_accuracy_threshold: float = 0.9,
    quick_seed_actions_threshold: int = 3,
    abort_consecutive_failures_threshold: int = 5,
) -> SimulationResult:
    """Run a single simulation with complementarity tracking.

    Args:
        seed: Random seed for reproducibility
        mechanism: Client mechanism type
        pattern: Initial memory pattern
        therapist_version: 'v1' or 'v2'
        window_size: Sliding window size for complementarity calculation
        (remaining args are simulation parameters)

    Returns:
        SimulationResult with complementarity trajectories
    """
    # Setup
    rng = np.random.RandomState(seed)
    u_matrix = sample_u_matrix(random_state=seed)

    # Generate initial memory
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type=pattern,
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
    client.success_threshold = rs_threshold

    # Create therapist
    TherapistClass = OmniscientStrategicTherapistV2 if therapist_version == 'v2' else OmniscientStrategicTherapistV1
    therapist = TherapistClass(
        client_ref=client,
        perception_window=perception_window,
        baseline_accuracy=baseline_accuracy,
        seeding_benefit_scaling=seeding_benefit_scaling,
        skip_seeding_accuracy_threshold=skip_seeding_accuracy_threshold,
        quick_seed_actions_threshold=quick_seed_actions_threshold,
        abort_consecutive_failures_threshold=abort_consecutive_failures_threshold,
    )

    # Initialize complementarity tracker
    comp_tracker = ComplementarityTracker(window_size=window_size)

    # Track initial state
    initial_rs = client.relationship_satisfaction
    initial_bond = client.bond

    # Complementarity trajectories
    overall_enacted_traj = []
    warm_enacted_traj = []
    cold_enacted_traj = []
    overall_perceived_traj = []
    warm_perceived_traj = []
    cold_perceived_traj = []

    # Run sessions
    success = False
    session = 0

    for session in range(1, max_sessions + 1):
        # Client selects action
        client_action = client.select_action()

        # Therapist responds
        therapist_action, _ = therapist.decide_action(client_action, session)

        # Get enacted and perceived actions (for parataxic distortion tracking)
        if enable_parataxic and hasattr(client, 'parataxic_history'):
            # Update memory first to capture parataxic distortion
            client.update_memory(client_action, therapist_action)

            # Get the actual and perceived actions from parataxic history
            if len(client.parataxic_history) > 0:
                last_record = client.parataxic_history[-1]
                enacted_action = last_record.actual_therapist_action
                perceived_action = last_record.perceived_therapist_action
            else:
                enacted_action = therapist_action
                perceived_action = therapist_action

            # Track both perspectives
            comp_tracker.add_interaction(
                client_action,
                therapist_action,
                enacted_action=enacted_action,
                perceived_action=perceived_action
            )
        else:
            # No parataxic distortion, single perspective
            client.update_memory(client_action, therapist_action)
            comp_tracker.add_interaction(client_action, therapist_action)

        # Record complementarity rates
        overall_enacted, warm_enacted, cold_enacted = comp_tracker.get_all_rates('enacted')
        overall_enacted_traj.append(overall_enacted)
        warm_enacted_traj.append(warm_enacted)
        cold_enacted_traj.append(cold_enacted)

        if enable_parataxic:
            overall_perceived, warm_perceived, cold_perceived = comp_tracker.get_all_rates('perceived')
            overall_perceived_traj.append(overall_perceived)
            warm_perceived_traj.append(warm_perceived)
            cold_perceived_traj.append(cold_perceived)

        # Process feedback (v2 only)
        if hasattr(therapist, 'process_feedback_after_memory_update'):
            therapist.process_feedback_after_memory_update(session, client_action)

        # Get new state
        new_rs = client.relationship_satisfaction
        new_bond = client.bond

        # Check termination
        if new_rs >= rs_threshold:
            success = True
            break

        if client.check_dropout():
            break

    return SimulationResult(
        seed=seed,
        mechanism=mechanism,
        pattern=pattern,
        therapist_version=therapist_version,
        success=success,
        total_sessions=session,
        rs_threshold=rs_threshold,
        overall_enacted_trajectory=overall_enacted_traj,
        warm_enacted_trajectory=warm_enacted_traj,
        cold_enacted_trajectory=cold_enacted_traj,
        overall_perceived_trajectory=overall_perceived_traj,
        warm_perceived_trajectory=warm_perceived_traj,
        cold_perceived_trajectory=cold_perceived_traj,
        final_rs=new_rs,
        final_bond=new_bond,
    )


def aggregate_results(results: List[SimulationResult]) -> AggregatedResults:
    """Aggregate results across multiple seeds.

    Args:
        results: List of SimulationResult from different seeds

    Returns:
        AggregatedResults with mean and std trajectories
    """
    if not results:
        raise ValueError("No results to aggregate")

    n_runs = len(results)
    config_name = f"{results[0].mechanism}_{results[0].pattern}_{results[0].therapist_version}"

    # Calculate success rate
    success_rate = sum(r.success for r in results) / n_runs * 100

    # Pad trajectories to same length
    max_length = max(len(r.overall_enacted_trajectory) for r in results)

    # Initialize arrays with NaN
    overall_enacted_arr = np.full((n_runs, max_length), np.nan)
    warm_enacted_arr = np.full((n_runs, max_length), np.nan)
    cold_enacted_arr = np.full((n_runs, max_length), np.nan)
    overall_perceived_arr = np.full((n_runs, max_length), np.nan)
    warm_perceived_arr = np.full((n_runs, max_length), np.nan)
    cold_perceived_arr = np.full((n_runs, max_length), np.nan)

    # Fill arrays
    for i, result in enumerate(results):
        length = len(result.overall_enacted_trajectory)
        overall_enacted_arr[i, :length] = result.overall_enacted_trajectory
        warm_enacted_arr[i, :length] = result.warm_enacted_trajectory
        cold_enacted_arr[i, :length] = result.cold_enacted_trajectory

        if result.overall_perceived_trajectory:
            overall_perceived_arr[i, :length] = result.overall_perceived_trajectory
            warm_perceived_arr[i, :length] = result.warm_perceived_trajectory
            cold_perceived_arr[i, :length] = result.cold_perceived_trajectory

    # Calculate statistics
    return AggregatedResults(
        config_name=config_name,
        mechanism=results[0].mechanism,
        pattern=results[0].pattern,
        therapist_version=results[0].therapist_version,
        n_runs=n_runs,
        success_rate=success_rate,
        mean_overall_enacted=np.nanmean(overall_enacted_arr, axis=0),
        std_overall_enacted=np.nanstd(overall_enacted_arr, axis=0),
        mean_overall_perceived=np.nanmean(overall_perceived_arr, axis=0),
        std_overall_perceived=np.nanstd(overall_perceived_arr, axis=0),
        mean_warm_enacted=np.nanmean(warm_enacted_arr, axis=0),
        std_warm_enacted=np.nanstd(warm_enacted_arr, axis=0),
        mean_warm_perceived=np.nanmean(warm_perceived_arr, axis=0),
        std_warm_perceived=np.nanstd(warm_perceived_arr, axis=0),
        mean_cold_enacted=np.nanmean(cold_enacted_arr, axis=0),
        std_cold_enacted=np.nanstd(cold_enacted_arr, axis=0),
        mean_cold_perceived=np.nanmean(cold_perceived_arr, axis=0),
        std_cold_perceived=np.nanstd(cold_perceived_arr, axis=0),
    )


class ComplementarityVisualizer:
    """Interactive visualizer for complementarity dynamics."""

    def __init__(self, aggregated_results: List[AggregatedResults],
                 complementarity_type: str = 'both',
                 enable_parataxic: bool = False):
        """Initialize the visualizer.

        Args:
            aggregated_results: List of aggregated results to plot
            complementarity_type: 'enacted', 'perceived', or 'both'
            enable_parataxic: Whether parataxic distortion was enabled
        """
        self.aggregated_results = aggregated_results
        self.complementarity_type = complementarity_type
        self.enable_parataxic = enable_parataxic
        self.filter_mode = 'overall'

        # Create figure
        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 10),
                                            gridspec_kw={'height_ratios': [3, 1]})
        self.ax_comp = self.axes[0]
        self.ax_success = self.axes[1]

        # Add filter radio buttons
        self.setup_filter_buttons()

    def setup_filter_buttons(self):
        """Setup radio buttons for warm/cold/overall filtering."""
        rax = plt.axes([0.02, 0.4, 0.15, 0.15])
        self.radio = RadioButtons(rax, ('Overall', 'Warm Only', 'Cold Only'))
        self.radio.on_clicked(self.update_filter)

    def update_filter(self, label):
        """Update plot based on filter selection."""
        filter_map = {'Overall': 'overall', 'Warm Only': 'warm', 'Cold Only': 'cold'}
        self.filter_mode = filter_map[label]
        self.plot()

    def get_line_style(self, mechanism: str, pattern: str, version: str) -> Dict:
        """Generate line style for a configuration."""
        return {
            'color': MECHANISM_COLORS[mechanism],
            'linestyle': PATTERN_LINESTYLES[pattern],
            'marker': THERAPIST_MARKERS[version],
            'markevery': 10,
            'linewidth': 2,
            'markersize': 6,
        }

    def plot(self):
        """Plot complementarity trajectories."""
        self.ax_comp.clear()
        self.ax_success.clear()

        for agg_result in self.aggregated_results:
            # Select data based on filter mode
            if self.filter_mode == 'overall':
                mean_enacted = agg_result.mean_overall_enacted
                std_enacted = agg_result.std_overall_enacted
                mean_perceived = agg_result.mean_overall_perceived
                std_perceived = agg_result.std_overall_perceived
            elif self.filter_mode == 'warm':
                mean_enacted = agg_result.mean_warm_enacted
                std_enacted = agg_result.std_warm_enacted
                mean_perceived = agg_result.mean_warm_perceived
                std_perceived = agg_result.std_warm_perceived
            else:  # cold
                mean_enacted = agg_result.mean_cold_enacted
                std_enacted = agg_result.std_cold_enacted
                mean_perceived = agg_result.mean_cold_perceived
                std_perceived = agg_result.std_cold_perceived

            sessions = np.arange(len(mean_enacted)) + 1
            style = self.get_line_style(
                agg_result.mechanism,
                agg_result.pattern,
                agg_result.therapist_version
            )

            # Plot enacted complementarity
            if self.complementarity_type in ['enacted', 'both']:
                label = f"{agg_result.config_name}"
                if self.complementarity_type == 'both':
                    label += " (enacted)"

                self.ax_comp.plot(sessions, mean_enacted, label=label, **style)
                self.ax_comp.fill_between(
                    sessions,
                    mean_enacted - std_enacted,
                    mean_enacted + std_enacted,
                    alpha=0.2,
                    color=style['color']
                )

            # Plot perceived complementarity if enabled
            if self.complementarity_type in ['perceived', 'both'] and self.enable_parataxic:
                label = f"{agg_result.config_name}"
                if self.complementarity_type == 'both':
                    label += " (perceived)"

                # Use dashed line for perceived
                perceived_style = style.copy()
                perceived_style['linestyle'] = '--'
                perceived_style['alpha'] = 0.7

                self.ax_comp.plot(sessions, mean_perceived, label=label, **perceived_style)
                self.ax_comp.fill_between(
                    sessions,
                    mean_perceived - std_perceived,
                    mean_perceived + std_perceived,
                    alpha=0.1,
                    color=style['color']
                )

        # Finalize complementarity plot
        self.ax_comp.set_ylabel('Complementarity Rate (%)', fontsize=12, fontweight='bold')
        self.ax_comp.set_ylim(0, 105)
        self.ax_comp.set_title(f'Complementarity Over Time ({self.filter_mode.capitalize()})',
                                fontsize=14, fontweight='bold')
        self.ax_comp.legend(loc='best', fontsize=9, ncol=2)
        self.ax_comp.grid(True, alpha=0.3)
        self.ax_comp.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='Perfect complementarity')

        # Plot success rate
        config_names = [r.config_name for r in self.aggregated_results]
        success_rates = [r.success_rate for r in self.aggregated_results]
        colors = [MECHANISM_COLORS[r.mechanism] for r in self.aggregated_results]

        x_pos = np.arange(len(config_names))
        self.ax_success.bar(x_pos, success_rates, color=colors, alpha=0.7)
        self.ax_success.set_xticks(x_pos)
        self.ax_success.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
        self.ax_success.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        self.ax_success.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        self.ax_success.set_ylim(0, 105)
        self.ax_success.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, (x, y) in enumerate(zip(x_pos, success_rates)):
            self.ax_success.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)

        self.fig.tight_layout(rect=[0.18, 0, 1, 1])
        self.fig.canvas.draw()

    def show(self):
        """Display the interactive plot."""
        self.plot()
        plt.show()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize complementarity dynamics in therapy simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Categorical parameters to compare
    parser.add_argument('--mechanisms', nargs='+',
                       default=['frequency_amplifier'],
                       choices=['bond_only', 'frequency_amplifier',
                               'conditional_amplifier',
                               'bond_weighted_conditional_amplifier',
                               'bond_weighted_frequency_amplifier'],
                       help='Client mechanisms to compare')

    parser.add_argument('--patterns', nargs='+',
                       default=['cold_stuck'],
                       choices=['cold_stuck', 'dominant_stuck', 'submissive_stuck',
                               'cold_warm', 'complementary_perfect', 'conflictual',
                               'mixed_random'],
                       help='Initial memory patterns to compare')

    parser.add_argument('--therapist-versions', nargs='+',
                       default=['v2'],
                       choices=['v1', 'v2'],
                       help='Therapist versions to compare')

    # Window and filtering
    parser.add_argument('--window-size', type=int, default=10,
                       help='Sliding window size for complementarity calculation')

    parser.add_argument('--complementarity-type', type=str, default='both',
                       choices=['enacted', 'perceived', 'both'],
                       help='Display enacted, perceived, or both complementarity')

    # Statistical parameters
    parser.add_argument('--n-seeds', type=int, default=30,
                       help='Number of random seeds per configuration')

    # Simulation parameters
    parser.add_argument('--max-sessions', type=int, default=100,
                       help='Maximum therapy sessions')

    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Success threshold percentile')

    parser.add_argument('--entropy', type=float, default=0.1,
                       help='Client entropy parameter')

    parser.add_argument('--enable-parataxic', action='store_true',
                       help='Enable parataxic distortion')

    parser.add_argument('--baseline-accuracy', type=float, default=0.5,
                       help='Baseline perception accuracy for parataxic distortion')

    parser.add_argument('--bond-offset', type=float, default=0.7,
                       help='Bond offset parameter')

    parser.add_argument('--bond-alpha', type=float, default=5.0,
                       help='Bond alpha parameter')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Save plot to file (optional)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print("=" * 70)
    print("Complementarity Visualization Script")
    print("=" * 70)
    print(f"Mechanisms: {args.mechanisms}")
    print(f"Patterns: {args.patterns}")
    print(f"Therapist versions: {args.therapist_versions}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Window size: {args.window_size}")
    print(f"Complementarity type: {args.complementarity_type}")
    print(f"Parataxic distortion: {args.enable_parataxic}")
    print("=" * 70)

    all_aggregated_results = []

    # Generate all configuration combinations
    configs = [
        (mech, pat, ver)
        for mech in args.mechanisms
        for pat in args.patterns
        for ver in args.therapist_versions
    ]

    print(f"\nRunning simulations for {len(configs)} configuration(s)...\n")

    for mechanism, pattern, version in configs:
        print(f"\nConfiguration: {mechanism} + {pattern} + {version}")
        print("-" * 70)

        results = []
        for seed in tqdm(range(args.n_seeds), desc=f"Seeds"):
            result = run_simulation_with_complementarity_tracking(
                seed=seed,
                mechanism=mechanism,
                pattern=pattern,
                therapist_version=version,
                window_size=args.window_size,
                max_sessions=args.max_sessions,
                success_threshold_percentile=args.threshold,
                enable_parataxic=args.enable_parataxic,
                entropy=args.entropy,
                baseline_accuracy=args.baseline_accuracy,
                bond_offset=args.bond_offset,
                bond_alpha=args.bond_alpha,
            )
            results.append(result)

        # Aggregate results
        agg_result = aggregate_results(results)
        all_aggregated_results.append(agg_result)

        print(f"Success rate: {agg_result.success_rate:.1f}%")
        print(f"Mean sessions: {np.mean([r.total_sessions for r in results]):.1f}")

    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    # Create and display visualization
    viz = ComplementarityVisualizer(
        all_aggregated_results,
        complementarity_type=args.complementarity_type,
        enable_parataxic=args.enable_parataxic
    )

    if args.output:
        viz.plot()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {args.output}")

    viz.show()


if __name__ == "__main__":
    main()
