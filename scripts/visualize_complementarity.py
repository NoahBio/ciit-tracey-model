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
from datetime import datetime
import json

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
from src.config import get_u_matrix_by_name, list_available_u_matrices
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

    # Complementarity rate trajectories (0-100%)
    overall_enacted_trajectory: List[float]
    warm_enacted_trajectory: List[float]
    cold_enacted_trajectory: List[float]
    overall_perceived_trajectory: List[float] = field(default_factory=list)
    warm_perceived_trajectory: List[float] = field(default_factory=list)
    cold_perceived_trajectory: List[float] = field(default_factory=list)

    # Distance trajectories (0-4 scale, where 0 = perfectly complementary)
    overall_enacted_distance_trajectory: List[float] = field(default_factory=list)
    warm_enacted_distance_trajectory: List[float] = field(default_factory=list)
    cold_enacted_distance_trajectory: List[float] = field(default_factory=list)
    overall_perceived_distance_trajectory: List[float] = field(default_factory=list)
    warm_perceived_distance_trajectory: List[float] = field(default_factory=list)
    cold_perceived_distance_trajectory: List[float] = field(default_factory=list)

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

    # Overall complementarity rate (0-100%)
    mean_overall_enacted: np.ndarray
    std_overall_enacted: np.ndarray
    mean_overall_perceived: np.ndarray
    std_overall_perceived: np.ndarray

    # Warm complementarity rate
    mean_warm_enacted: np.ndarray
    std_warm_enacted: np.ndarray
    mean_warm_perceived: np.ndarray
    std_warm_perceived: np.ndarray

    # Cold complementarity rate
    mean_cold_enacted: np.ndarray
    std_cold_enacted: np.ndarray
    mean_cold_perceived: np.ndarray
    std_cold_perceived: np.ndarray

    # Overall distance (0-4 scale)
    mean_overall_enacted_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    std_overall_enacted_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_overall_perceived_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    std_overall_perceived_distance: np.ndarray = field(default_factory=lambda: np.array([]))

    # Warm distance
    mean_warm_enacted_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    std_warm_enacted_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_warm_perceived_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    std_warm_perceived_distance: np.ndarray = field(default_factory=lambda: np.array([]))

    # Cold distance
    mean_cold_enacted_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    std_cold_enacted_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_cold_perceived_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    std_cold_perceived_distance: np.ndarray = field(default_factory=lambda: np.array([]))

    # Additional statistics
    baseline_success_rate: float = 0.0  # Success rate of always-complementary therapist
    overall_noncomplementarity_pct: float = 0.0  # % of non-complementary actions
    overall_mean_distance: float = 0.0  # Mean distance across all sessions and seeds


# Complementary action mapping (same as in therapist agents)
COMPLEMENT_MAP = {
    0: 4,  # D → S
    1: 3,  # WD → WS
    2: 2,  # W → W
    3: 1,  # WS → WD
    4: 0,  # S → D
    5: 7,  # CS → CD
    6: 6,  # C → C
    7: 5,  # CD → CS
}


def always_complement(client_action: int) -> int:
    """Simple complementary therapist strategy."""
    return COMPLEMENT_MAP[client_action]


def run_baseline_complementary_simulation(
    seed: int,
    mechanism: str,
    pattern: str,
    max_sessions: int = 100,
    success_threshold_percentile: float = 0.8,
    enable_parataxic: bool = False,
    entropy: float = 0.1,
    baseline_accuracy: float = 0.5,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 5.0,
    bond_offset: float = 0.7,
    recency_weighting_factor: int = 2,
    u_matrix_name: Optional[str] = None,
) -> bool:
    """Run a single simulation with always-complementary therapist.

    Returns:
        bool: True if successful, False otherwise
    """
    # Setup
    rng = np.random.RandomState(seed)

    # Get U-matrix: use named matrix if specified, otherwise sample
    if u_matrix_name is not None:
        u_matrix = get_u_matrix_by_name(u_matrix_name)
        if u_matrix is None:
            raise ValueError(
                f"Unknown U-matrix name: '{u_matrix_name}'. "
                f"Available names: {list_available_u_matrices()}"
            )
    else:
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

    # Run sessions with always-complementary therapist
    success = False
    for session in range(1, max_sessions + 1):
        # Client selects action
        client_action = client.select_action()

        # Therapist responds with complementary action
        therapist_action = always_complement(client_action)

        # Update client memory
        client.update_memory(client_action, therapist_action)

        # Get new state
        new_rs = client.relationship_satisfaction

        # Check termination
        if new_rs >= rs_threshold:
            success = True
            break

        if client.check_dropout():
            break

    return success


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
    u_matrix_name: Optional[str] = None,
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

    # Get U-matrix: use named matrix if specified, otherwise sample
    if u_matrix_name is not None:
        u_matrix = get_u_matrix_by_name(u_matrix_name)
        if u_matrix is None:
            raise ValueError(
                f"Unknown U-matrix name: '{u_matrix_name}'. "
                f"Available names: {list_available_u_matrices()}"
            )
    else:
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
    
    # Base parameters for both V1 and V2
    therapist_kwargs = {
        'client_ref': client,
        'perception_window': perception_window,
        'baseline_accuracy': baseline_accuracy,
    }
    
    # V2-specific parameters
    if therapist_version == 'v2':
        therapist_kwargs.update({
            'seeding_benefit_scaling': seeding_benefit_scaling,
            'skip_seeding_accuracy_threshold': skip_seeding_accuracy_threshold,
            'quick_seed_actions_threshold': quick_seed_actions_threshold,
            'abort_consecutive_failures_threshold': abort_consecutive_failures_threshold,
        })
    
    therapist = TherapistClass(**therapist_kwargs)

    # Initialize complementarity tracker
    comp_tracker = ComplementarityTracker(window_size=window_size)

    # Track initial state
    initial_rs = client.relationship_satisfaction
    initial_bond = client.bond

    # Complementarity rate trajectories
    overall_enacted_traj = []
    warm_enacted_traj = []
    cold_enacted_traj = []
    overall_perceived_traj = []
    warm_perceived_traj = []
    cold_perceived_traj = []

    # Distance trajectories
    overall_enacted_dist_traj = []
    warm_enacted_dist_traj = []
    cold_enacted_dist_traj = []
    overall_perceived_dist_traj = []
    warm_perceived_dist_traj = []
    cold_perceived_dist_traj = []

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

        # Record distances
        overall_dist, warm_dist, cold_dist = comp_tracker.get_all_distances('enacted')
        overall_enacted_dist_traj.append(overall_dist)
        warm_enacted_dist_traj.append(warm_dist)
        cold_enacted_dist_traj.append(cold_dist)

        if enable_parataxic:
            overall_perceived, warm_perceived, cold_perceived = comp_tracker.get_all_rates('perceived')
            overall_perceived_traj.append(overall_perceived)
            warm_perceived_traj.append(warm_perceived)
            cold_perceived_traj.append(cold_perceived)

            # Also record perceived distances
            overall_perc_dist, warm_perc_dist, cold_perc_dist = comp_tracker.get_all_distances('perceived')
            overall_perceived_dist_traj.append(overall_perc_dist)
            warm_perceived_dist_traj.append(warm_perc_dist)
            cold_perceived_dist_traj.append(cold_perc_dist)

        # Process feedback (v2 only)
        if hasattr(therapist, 'process_feedback_after_memory_update'):
            therapist.process_feedback_after_memory_update(session, client_action)  # type: ignore[attr-defined]

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
        overall_enacted_distance_trajectory=overall_enacted_dist_traj,
        warm_enacted_distance_trajectory=warm_enacted_dist_traj,
        cold_enacted_distance_trajectory=cold_enacted_dist_traj,
        overall_perceived_distance_trajectory=overall_perceived_dist_traj,
        warm_perceived_distance_trajectory=warm_perceived_dist_traj,
        cold_perceived_distance_trajectory=cold_perceived_dist_traj,
        final_rs=new_rs,
        final_bond=new_bond,
    )


def aggregate_results(results: List[SimulationResult], baseline_success_rate: float = 0.0) -> AggregatedResults:
    """Aggregate results across multiple seeds.

    Args:
        results: List of SimulationResult from different seeds
        baseline_success_rate: Success rate of baseline complementary therapist

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

    # Initialize rate arrays with NaN
    overall_enacted_arr = np.full((n_runs, max_length), np.nan)
    warm_enacted_arr = np.full((n_runs, max_length), np.nan)
    cold_enacted_arr = np.full((n_runs, max_length), np.nan)
    overall_perceived_arr = np.full((n_runs, max_length), np.nan)
    warm_perceived_arr = np.full((n_runs, max_length), np.nan)
    cold_perceived_arr = np.full((n_runs, max_length), np.nan)

    # Initialize distance arrays with NaN
    overall_enacted_dist_arr = np.full((n_runs, max_length), np.nan)
    warm_enacted_dist_arr = np.full((n_runs, max_length), np.nan)
    cold_enacted_dist_arr = np.full((n_runs, max_length), np.nan)
    overall_perceived_dist_arr = np.full((n_runs, max_length), np.nan)
    warm_perceived_dist_arr = np.full((n_runs, max_length), np.nan)
    cold_perceived_dist_arr = np.full((n_runs, max_length), np.nan)

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

        # Fill distance arrays
        if result.overall_enacted_distance_trajectory:
            overall_enacted_dist_arr[i, :length] = result.overall_enacted_distance_trajectory
            warm_enacted_dist_arr[i, :length] = result.warm_enacted_distance_trajectory
            cold_enacted_dist_arr[i, :length] = result.cold_enacted_distance_trajectory

        if result.overall_perceived_distance_trajectory:
            overall_perceived_dist_arr[i, :length] = result.overall_perceived_distance_trajectory
            warm_perceived_dist_arr[i, :length] = result.warm_perceived_distance_trajectory
            cold_perceived_dist_arr[i, :length] = result.cold_perceived_distance_trajectory

    # Calculate overall non-complementarity percentage
    # Average complementarity across all sessions and seeds
    overall_mean_comp = np.nanmean(overall_enacted_arr)
    overall_noncomplementarity_pct = 100.0 - overall_mean_comp

    # Calculate overall mean distance
    overall_mean_dist = np.nanmean(overall_enacted_dist_arr)

    # Calculate statistics
    return AggregatedResults(
        config_name=config_name,
        mechanism=results[0].mechanism,
        pattern=results[0].pattern,
        therapist_version=results[0].therapist_version,
        n_runs=n_runs,
        success_rate=success_rate,
        # Rate statistics
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
        # Distance statistics
        mean_overall_enacted_distance=np.nanmean(overall_enacted_dist_arr, axis=0),
        std_overall_enacted_distance=np.nanstd(overall_enacted_dist_arr, axis=0),
        mean_overall_perceived_distance=np.nanmean(overall_perceived_dist_arr, axis=0),
        std_overall_perceived_distance=np.nanstd(overall_perceived_dist_arr, axis=0),
        mean_warm_enacted_distance=np.nanmean(warm_enacted_dist_arr, axis=0),
        std_warm_enacted_distance=np.nanstd(warm_enacted_dist_arr, axis=0),
        mean_warm_perceived_distance=np.nanmean(warm_perceived_dist_arr, axis=0),
        std_warm_perceived_distance=np.nanstd(warm_perceived_dist_arr, axis=0),
        mean_cold_enacted_distance=np.nanmean(cold_enacted_dist_arr, axis=0),
        std_cold_enacted_distance=np.nanstd(cold_enacted_dist_arr, axis=0),
        mean_cold_perceived_distance=np.nanmean(cold_perceived_dist_arr, axis=0),
        std_cold_perceived_distance=np.nanstd(cold_perceived_dist_arr, axis=0),
        # Additional statistics
        baseline_success_rate=baseline_success_rate,
        overall_noncomplementarity_pct=overall_noncomplementarity_pct,  # type: ignore[arg-type]
        overall_mean_distance=overall_mean_dist,
    )


class ComplementarityVisualizer:
    """Interactive visualizer for complementarity dynamics."""

    def __init__(self, aggregated_results: List[AggregatedResults],
                 complementarity_type: str = 'both',
                 enable_parataxic: bool = False,
                 metric: str = 'complementarity_rate',
                 seed_breakdown: Optional[Dict[str, int]] = None):
        """Initialize the visualizer.

        Args:
            aggregated_results: List of aggregated results to plot
            complementarity_type: 'enacted', 'perceived', or 'both'
            enable_parataxic: Whether parataxic distortion was enabled
            metric: 'complementarity_rate' (0-100%) or 'octant_distance' (0-4)
            seed_breakdown: Optional dict with keys 'v2_advantage', 'baseline_advantage',
                           'both_success', 'both_fail' containing counts
        """
        self.aggregated_results = aggregated_results
        self.complementarity_type = complementarity_type
        self.enable_parataxic = enable_parataxic
        self.metric = metric
        self.filter_mode = 'overall'
        self.seed_breakdown = seed_breakdown

        # Create figure - add extra row for breakdown table if provided
        if seed_breakdown:
            self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 12),
                                                gridspec_kw={'height_ratios': [3, 1, 0.6]})
            self.ax_comp = self.axes[0]
            self.ax_success = self.axes[1]
            self.ax_table = self.axes[2]
        else:
            self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 10),
                                                gridspec_kw={'height_ratios': [3, 1]})
            self.ax_comp = self.axes[0]
            self.ax_success = self.axes[1]
            self.ax_table = None


    def get_line_style(self, mechanism: str, pattern: str, version: str, config_name: str = '') -> Dict:
        """Generate line style for a configuration."""
        # Check if this is a V2 advantage or remaining group
        # Use very distinct colors to make groups visually obvious
        if config_name.endswith('_v2_advantage'):
            color = '#00AA00'  # Bright green - clearly distinct from teal
            linewidth = 3.0
            alpha = 1.0
        elif config_name.endswith('_v2_disadvantage'):
            color = '#CC0000'  # Red - V2 performed worse than baseline
            linewidth = 3.0
            alpha = 1.0
        elif config_name.endswith('_remaining'):
            color = '#333333'  # Dark gray (nearly black)
            linewidth = 2.0
            alpha = 0.8
        else:
            # Original behavior - use mechanism colors
            color = MECHANISM_COLORS[mechanism]
            linewidth = 2
            alpha = 1.0

        return {
            'color': color,
            'linestyle': PATTERN_LINESTYLES[pattern],
            'marker': THERAPIST_MARKERS[version],
            'markevery': 10,
            'linewidth': linewidth,
            'markersize': 6,
            'alpha': alpha,
        }

    def plot(self):
        """Plot complementarity trajectories."""
        self.ax_comp.clear()
        self.ax_success.clear()

        for agg_result in self.aggregated_results:
            # Select data based on filter mode and metric
            if self.metric == 'octant_distance':
                # Use distance data (0-4 scale)
                if self.filter_mode == 'overall':
                    mean_enacted = agg_result.mean_overall_enacted_distance
                    std_enacted = agg_result.std_overall_enacted_distance
                    mean_perceived = agg_result.mean_overall_perceived_distance
                    std_perceived = agg_result.std_overall_perceived_distance
                elif self.filter_mode == 'warm':
                    mean_enacted = agg_result.mean_warm_enacted_distance
                    std_enacted = agg_result.std_warm_enacted_distance
                    mean_perceived = agg_result.mean_warm_perceived_distance
                    std_perceived = agg_result.std_warm_perceived_distance
                else:  # cold
                    mean_enacted = agg_result.mean_cold_enacted_distance
                    std_enacted = agg_result.std_cold_enacted_distance
                    mean_perceived = agg_result.mean_cold_perceived_distance
                    std_perceived = agg_result.std_cold_perceived_distance
            else:
                # Use rate data (0-100%)
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
                agg_result.therapist_version,
                agg_result.config_name
            )

            # Determine display name for legend
            if agg_result.config_name.endswith('_v2_advantage'):
                display_name = f"V2 Advantage (n={agg_result.n_runs})"
            elif agg_result.config_name.endswith('_remaining'):
                display_name = f"Remaining (n={agg_result.n_runs})"
            else:
                display_name = agg_result.config_name

            # Plot enacted complementarity
            if self.complementarity_type in ['enacted', 'both']:
                label = display_name
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
                label = display_name
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

        # Finalize complementarity plot - adjust labels and limits based on metric
        self.ax_comp.set_xlabel('Session Number', fontsize=14, fontweight='bold')
        if self.metric == 'octant_distance':
            self.ax_comp.set_ylabel('Mean Octant Distance', fontsize=14, fontweight='bold')
            self.ax_comp.set_ylim(-0.2, 4.2)
            self.ax_comp.set_title(f'Octant Distance Over Time ({self.filter_mode.capitalize()})',
                                    fontsize=16, fontweight='bold', pad=15)
            # Reference line at 0 (perfect complementarity)
            self.ax_comp.axhline(y=0, color='green', linestyle=':', alpha=0.5, linewidth=2)
        else:
            self.ax_comp.set_ylabel('Complementarity Rate (%)', fontsize=14, fontweight='bold')
            self.ax_comp.set_ylim(0, 105)
            self.ax_comp.set_title(f'Complementarity Over Time ({self.filter_mode.capitalize()})',
                                    fontsize=16, fontweight='bold', pad=15)
            # Reference line at 100 (perfect complementarity)
            self.ax_comp.axhline(y=100, color='green', linestyle=':', alpha=0.5, linewidth=2)
        self.ax_comp.legend(loc='best', fontsize=11, ncol=1, framealpha=0.9)
        self.ax_comp.grid(True, alpha=0.3, linewidth=0.8)
        self.ax_comp.tick_params(axis='both', which='major', labelsize=12)

        # Add descriptive statistics text box
        if self.aggregated_results:
            stats_text_lines = []
            for agg_result in self.aggregated_results:
                # Use friendly group name if available, otherwise use config_name
                if agg_result.config_name.endswith('_v2_advantage'):
                    display_name = f"V2 Advantage (n={agg_result.n_runs})"
                elif agg_result.config_name.endswith('_remaining'):
                    display_name = f"Remaining (n={agg_result.n_runs})"
                else:
                    display_name = f"{agg_result.mechanism[:15]}_{agg_result.pattern[:10]}_{agg_result.therapist_version}"

                if self.metric == 'octant_distance':
                    stats_text_lines.append(
                        f"{display_name}:\n"
                        f"  Baseline: {agg_result.baseline_success_rate:.1f}%  "
                        f"Mean dist: {agg_result.overall_mean_distance:.2f}"
                    )
                else:
                    stats_text_lines.append(
                        f"{display_name}:\n"
                        f"  Baseline: {agg_result.baseline_success_rate:.1f}%  "
                        f"Non-comp: {agg_result.overall_noncomplementarity_pct:.2f}%"
                    )

            stats_text = "\n".join(stats_text_lines)
            # Place text box in upper right corner
            self.ax_comp.text(
                0.98, 0.98, stats_text,
                transform=self.ax_comp.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
                family='monospace'
            )

        # Plot success rate - use green/black if highlight mode, otherwise mechanism colors
        success_rates = [r.success_rate for r in self.aggregated_results]

        # Determine bar colors and labels based on config_name suffix
        colors = []
        bar_labels = []
        for r in self.aggregated_results:
            if r.config_name.endswith('_v2_advantage'):
                colors.append('green')
                bar_labels.append(f'V2 Advantage (n={r.n_runs})')
            elif r.config_name.endswith('_v2_disadvantage'):
                colors.append('red')
                bar_labels.append(f'V2 Disadvantage (n={r.n_runs})')
            elif r.config_name.endswith('_remaining'):
                colors.append('black')
                bar_labels.append(f'Remaining (n={r.n_runs})')
            else:
                colors.append(MECHANISM_COLORS[r.mechanism])
                bar_labels.append(r.config_name)

        x_pos = np.arange(len(bar_labels))
        self.ax_success.bar(x_pos, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        self.ax_success.set_xticks(x_pos)
        self.ax_success.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=11)
        self.ax_success.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        self.ax_success.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
        self.ax_success.set_ylim(0, 105)
        self.ax_success.grid(True, alpha=0.3, axis='y', linewidth=0.8)
        self.ax_success.tick_params(axis='both', which='major', labelsize=12)

        # Add values on bars
        for i, (x, y) in enumerate(zip(x_pos, success_rates)):
            self.ax_success.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)

        # Render seed breakdown table if provided
        if self.ax_table is not None and self.seed_breakdown:
            self.ax_table.clear()
            self.ax_table.axis('off')

            total_seeds = sum(self.seed_breakdown.values())

            # Build table data
            table_data = [
                ['V2 Advantage', 'Baseline Advantage', 'Both Success', 'Both Fail', 'Total'],
                [
                    f"{self.seed_breakdown['v2_advantage']} ({100*self.seed_breakdown['v2_advantage']/total_seeds:.1f}%)",
                    f"{self.seed_breakdown['baseline_advantage']} ({100*self.seed_breakdown['baseline_advantage']/total_seeds:.1f}%)",
                    f"{self.seed_breakdown['both_success']} ({100*self.seed_breakdown['both_success']/total_seeds:.1f}%)",
                    f"{self.seed_breakdown['both_fail']} ({100*self.seed_breakdown['both_fail']/total_seeds:.1f}%)",
                    f"{total_seeds}",
                ],
            ]

            # Create table with colored cells
            cell_colors = [
                ['#E8E8E8', '#E8E8E8', '#E8E8E8', '#E8E8E8', '#E8E8E8'],  # Header row
                ['#90EE90', '#FFCCCB', '#ADD8E6', '#D3D3D3', 'white'],     # Data row: green, red, blue, gray
            ]

            table = self.ax_table.table(
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

            self.ax_table.set_title('Seed Breakdown by Outcome', fontsize=12, fontweight='bold', pad=10)

        self.fig.tight_layout()  # type: ignore[arg-type]
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

    parser.add_argument('--metric', type=str, default='complementarity_rate',
                       choices=['complementarity_rate', 'octant_distance'],
                       help='Metric to visualize: complementarity_rate (0-100%%) or octant_distance (0-4 scale)')

    # Statistical parameters
    parser.add_argument('--n-seeds', type=int, default=30,
                       help='Number of random seeds per configuration')

    # Simulation parameters - defaults from best Optuna trial (rank 1, trial 2643)
    # Achieved 88% omniscient success vs 73.3% baseline (14.7% advantage)
    parser.add_argument('--max-sessions', type=int, default=1940,
                       help='Maximum therapy sessions (default: 1940 from best Optuna trial)')

    parser.add_argument('--threshold', type=float, default=0.9358603798762596,
                       help='Success threshold percentile (default: 0.936 from best Optuna trial)')

    parser.add_argument('--entropy', type=float, default=0.1,
                       help='Client entropy parameter')

    parser.add_argument('--enable-parataxic', action='store_true', default=True,
                       help='Enable parataxic distortion (default: True)')

    parser.add_argument('--disable-parataxic', dest='enable_parataxic', action='store_false',
                       help='Disable parataxic distortion')

    parser.add_argument('--baseline-accuracy', type=float, default=0.5549619551286054,
                       help='Baseline perception accuracy (default: 0.555 from best Optuna trial)')

    parser.add_argument('--bond-offset', type=float, default=0.624462461360537,
                       help='Bond offset parameter (default: 0.624 from best Optuna trial)')

    parser.add_argument('--bond-alpha', type=float, default=11.847676335038303,
                       help='Bond alpha parameter (default: 11.85 from best Optuna trial)')

    parser.add_argument('--recency-weighting-factor', type=int, default=2,
                       help='Recency weighting factor for client memory')

    parser.add_argument('--perception-window', type=int, default=10,
                       help='Perception window size for therapist (default: 10 from best Optuna trial)')

    parser.add_argument('--seeding-benefit-scaling', type=float, default=1.8658722646107764,
                       help='Seeding benefit scaling (default: 1.87 from best Optuna trial)')

    parser.add_argument('--skip-seeding-accuracy-threshold', type=float, default=0.814677493978211,
                       help='Skip seeding accuracy threshold (default: 0.815 from best Optuna trial)')

    parser.add_argument('--quick-seed-actions-threshold', type=int, default=1,
                       help='Quick seed actions threshold (default: 1 from best Optuna trial)')

    parser.add_argument('--abort-consecutive-failures-threshold', type=int, default=4,
                       help='Abort consecutive failures threshold (default: 4 from best Optuna trial)')

    # Highlighting
    parser.add_argument('--highlight-v2-advantage', action='store_true',
                       help='Highlight seeds where V2 succeeded but baseline failed (green) vs all other seeds (black)')

    parser.add_argument('--highlight-v2-disadvantage', action='store_true',
                       help='Highlight seeds where baseline succeeded but V2 failed (red) vs remaining seeds (black)')

    parser.add_argument('--u-matrix', type=str, default=None,
                       help=f'Named U-matrix to use (default: random sampling). '
                            f'Available: {", ".join(list_available_u_matrices())}')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Save plot to file (optional)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / f"complementarity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command that generated this run
    command_info = {
        "timestamp": timestamp,
        "command": " ".join(sys.argv),
        "parameters": {
            "mechanisms": args.mechanisms,
            "patterns": args.patterns,
            "therapist_versions": args.therapist_versions,
            "n_seeds": args.n_seeds,
            "u_matrix_name": args.u_matrix if args.u_matrix else "random_sampled",
            "window_size": args.window_size,
            "complementarity_type": args.complementarity_type,
            "enable_parataxic": args.enable_parataxic,
            "max_sessions": args.max_sessions,
            "threshold": args.threshold,
            "entropy": args.entropy,
            "baseline_accuracy": args.baseline_accuracy,
            "bond_offset": args.bond_offset,
            "bond_alpha": args.bond_alpha,
            "recency_weighting_factor": args.recency_weighting_factor,
            "perception_window": args.perception_window,
            "seeding_benefit_scaling": args.seeding_benefit_scaling,
            "skip_seeding_accuracy_threshold": args.skip_seeding_accuracy_threshold,
            "quick_seed_actions_threshold": args.quick_seed_actions_threshold,
            "abort_consecutive_failures_threshold": args.abort_consecutive_failures_threshold,
            "highlight_v2_advantage": args.highlight_v2_advantage,
            "highlight_v2_disadvantage": args.highlight_v2_disadvantage,
            "metric": args.metric,
        }
    }

    with open(output_dir / "command.json", "w") as f:
        json.dump(command_info, f, indent=2)

    print("=" * 70)
    print("Complementarity Visualization Script")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Mechanisms: {args.mechanisms}")
    print(f"Patterns: {args.patterns}")
    print(f"Therapist versions: {args.therapist_versions}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Window size: {args.window_size}")
    print(f"Complementarity type: {args.complementarity_type}")
    print(f"Metric: {args.metric}")
    print(f"Parataxic distortion: {args.enable_parataxic}")
    print(f"Highlight V2 advantage: {args.highlight_v2_advantage}")
    print(f"Highlight V2 disadvantage: {args.highlight_v2_disadvantage}")
    print("=" * 70)

    all_aggregated_results = []

    # Initialize seed breakdown counters (used when --highlight-v2-advantage is set)
    v2_advantage_results = []
    baseline_advantage_results = []
    both_success_results = []
    both_fail_results = []

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

        # Run baseline always-complementary therapist simulations
        print("Running baseline always-complementary simulations...")
        baseline_successes = []
        for seed in tqdm(range(args.n_seeds), desc="Baseline"):
            success = run_baseline_complementary_simulation(
                seed=seed,
                mechanism=mechanism,
                pattern=pattern,
                max_sessions=args.max_sessions,
                success_threshold_percentile=args.threshold,
                enable_parataxic=args.enable_parataxic,
                entropy=args.entropy,
                baseline_accuracy=args.baseline_accuracy,
                bond_offset=args.bond_offset,
                bond_alpha=args.bond_alpha,
                recency_weighting_factor=args.recency_weighting_factor,
                u_matrix_name=args.u_matrix,
            )
            baseline_successes.append(success)

        baseline_success_rate = sum(baseline_successes) / len(baseline_successes) * 100
        print(f"Baseline success rate: {baseline_success_rate:.1f}%")

        # Run omniscient therapist simulations
        print("Running omniscient therapist simulations...")
        results = []
        for seed in tqdm(range(args.n_seeds), desc="Omniscient"):
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
                recency_weighting_factor=args.recency_weighting_factor,
                perception_window=args.perception_window,
                seeding_benefit_scaling=args.seeding_benefit_scaling,
                skip_seeding_accuracy_threshold=args.skip_seeding_accuracy_threshold,
                quick_seed_actions_threshold=args.quick_seed_actions_threshold,
                abort_consecutive_failures_threshold=args.abort_consecutive_failures_threshold,
                u_matrix_name=args.u_matrix,
            )
            results.append(result)

        # Handle highlight modes - can enable both simultaneously
        if args.highlight_v2_advantage or args.highlight_v2_disadvantage:
            # Categorize all seeds into 4 groups
            config_v2_adv = []
            config_baseline_adv = []
            config_both_success = []
            config_both_fail = []

            for r in results:
                v2_success = r.success
                baseline_success = baseline_successes[r.seed]

                if v2_success and not baseline_success:
                    config_v2_adv.append(r)
                elif baseline_success and not v2_success:
                    config_baseline_adv.append(r)
                elif v2_success and baseline_success:
                    config_both_success.append(r)
                else:  # neither succeeded
                    config_both_fail.append(r)

            # Accumulate into global lists for the breakdown table
            v2_advantage_results.extend(config_v2_adv)
            baseline_advantage_results.extend(config_baseline_adv)
            both_success_results.extend(config_both_success)
            both_fail_results.extend(config_both_fail)

            # Remaining = both succeeded OR both failed
            remaining_results = config_both_success + config_both_fail

            print(f"\nSeed breakdown:")
            print(f"  V2 advantage: {len(config_v2_adv)}/{len(results)}")
            print(f"  V2 disadvantage (baseline adv): {len(config_baseline_adv)}/{len(results)}")
            print(f"  Both success: {len(config_both_success)}/{len(results)}")
            print(f"  Both fail: {len(config_both_fail)}/{len(results)}")

            # Aggregate groups based on which flags are enabled
            if args.highlight_v2_advantage and len(config_v2_adv) > 0:
                agg_v2_adv = aggregate_results(config_v2_adv, baseline_success_rate=baseline_success_rate)
                agg_v2_adv.config_name = f"{agg_v2_adv.config_name}_v2_advantage"
                all_aggregated_results.append(agg_v2_adv)

            if args.highlight_v2_disadvantage and len(config_baseline_adv) > 0:
                agg_v2_disadv = aggregate_results(config_baseline_adv, baseline_success_rate=baseline_success_rate)
                agg_v2_disadv.config_name = f"{agg_v2_disadv.config_name}_v2_disadvantage"
                all_aggregated_results.append(agg_v2_disadv)

            if len(remaining_results) > 0:
                agg_remaining = aggregate_results(remaining_results, baseline_success_rate=baseline_success_rate)
                agg_remaining.config_name = f"{agg_remaining.config_name}_remaining"
                all_aggregated_results.append(agg_remaining)

            # Overall stats (for display)
            agg_overall = aggregate_results(results, baseline_success_rate=baseline_success_rate)
            print(f"Overall omniscient success rate: {agg_overall.success_rate:.1f}%")
            print(f"Mean sessions: {np.mean([r.total_sessions for r in results]):.1f}")
            print(f"Overall non-complementarity: {agg_overall.overall_noncomplementarity_pct:.2f}%")

        else:
            # Neither flag enabled - aggregate all together (original behavior)
            agg_result = aggregate_results(results, baseline_success_rate=baseline_success_rate)
            all_aggregated_results.append(agg_result)

            print(f"Omniscient success rate: {agg_result.success_rate:.1f}%")
            print(f"Mean sessions: {np.mean([r.total_sessions for r in results]):.1f}")
            print(f"Overall non-complementarity: {agg_result.overall_noncomplementarity_pct:.2f}%")

    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    # Add results summary to command info
    command_info["results"] = []
    for agg_result in all_aggregated_results:
        result_dict = {
            "config_name": agg_result.config_name,
            "mechanism": agg_result.mechanism,
            "pattern": agg_result.pattern,
            "therapist_version": agg_result.therapist_version,
            "n_runs": agg_result.n_runs,
            "omniscient_success_rate": agg_result.success_rate,
            "baseline_success_rate": agg_result.baseline_success_rate,
            "overall_noncomplementarity_pct": agg_result.overall_noncomplementarity_pct,
            "overall_mean_distance": agg_result.overall_mean_distance,
        }
        if agg_result.config_name.endswith('_v2_advantage'):
            result_dict["group"] = "V2 advantage (green)"
        elif agg_result.config_name.endswith('_v2_disadvantage'):
            result_dict["group"] = "V2 disadvantage (red)"
        elif agg_result.config_name.endswith('_remaining'):
            result_dict["group"] = "Remaining trials (black)"
        command_info["results"].append(result_dict)

    # Re-save command.json with results
    with open(output_dir / "command.json", "w") as f:
        json.dump(command_info, f, indent=2)

    # Prepare seed breakdown if highlight mode was used
    seed_breakdown = None
    if args.highlight_v2_advantage:
        seed_breakdown = {
            'v2_advantage': len(v2_advantage_results),
            'baseline_advantage': len(baseline_advantage_results),
            'both_success': len(both_success_results),
            'both_fail': len(both_fail_results),
        }

    # Create and display visualization
    viz = ComplementarityVisualizer(
        all_aggregated_results,
        complementarity_type=args.complementarity_type,
        enable_parataxic=args.enable_parataxic,
        metric=args.metric,
        seed_breakdown=seed_breakdown
    )

    # Save plot to timestamped directory
    viz.plot()
    plot_path = output_dir / "complementarity_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Also save to custom path if specified
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Also saved to: {args.output}")

    print(f"\nAll results saved to: {output_dir}")
    print("  - complementarity_plot.png")
    print("  - command.json (with results summary)")

    viz.show()


if __name__ == "__main__":
    main()
