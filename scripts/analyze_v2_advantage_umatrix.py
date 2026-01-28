"""Comprehensive analysis of v2_advantage seeds vs other categories.

This script:
1. Collects seeds until finding target number of v2_advantage seeds
2. Analyzes u_matrix properties across categories
3. Runs statistical tests comparing groups
4. Analyzes stuck patterns and non-complementary interaction distances

Usage:
    python scripts/analyze_v2_advantage_umatrix.py --target-v2-advantage 75
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Any
from collections import Counter
from dataclasses import dataclass, asdict
from tqdm import tqdm
import optuna
from scipy import stats

from src.config import sample_u_matrix, calculate_success_threshold, OCTANT_NAMES
from evaluate_omniscient_therapist import run_single_simulation


# Complementary mapping
COMPLEMENT = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}


@dataclass
class SeedAnalysis:
    """Analysis results for a single seed."""
    seed: int
    category: str  # 'v2_advantage', 'both_success', 'both_fail', 'comp_advantage'

    # U_matrix properties
    u_min: float
    u_max: float
    u_range: float
    total_diff: float  # Sum of (optimal - complementary) across all client actions
    max_diff: float    # Maximum single diff
    n_suboptimal_octants: int  # How many client actions have non-complementary optimal

    # Simulation results
    v2_success: bool
    comp_success: bool
    v2_sessions: int
    comp_sessions: int

    # Stuck analysis (complementary therapist)
    comp_dominant_octant: int
    comp_dominant_pct: float
    comp_is_stuck: bool  # >80% in one octant
    comp_stuck_is_suboptimal: bool  # Stuck octant has suboptimal complementary response

    # V2 therapist analysis
    v2_dominant_octant: int
    v2_dominant_pct: float
    v2_is_stuck: bool

    # Non-complementary analysis (for v2_advantage only)
    n_noncomp_actions: int
    pct_noncomp_actions: float
    n_noncomp_gt2_octants: int  # Non-comp actions >2 octants from complementary
    pct_noncomp_gt2_octants: float


def octant_distance(a: int, b: int) -> int:
    """Calculate circular distance between two octants (0-4)."""
    diff = abs(a - b)
    return min(diff, 8 - diff)


def analyze_umatrix(seed: int) -> Dict[str, Any]:
    """Analyze u_matrix properties for a seed."""
    u = sample_u_matrix(random_state=seed)

    total_diff = 0.0
    max_diff = 0.0
    n_suboptimal = 0

    for client in range(8):
        comp_therapist = COMPLEMENT[client]
        comp_utility = u[client, comp_therapist]
        optimal_utility = u[client, :].max()
        optimal_therapist = u[client, :].argmax()

        diff = optimal_utility - comp_utility
        total_diff += diff
        max_diff = max(max_diff, diff)

        if optimal_therapist != comp_therapist:
            n_suboptimal += 1

    return {
        'u_min': float(u.min()),
        'u_max': float(u.max()),
        'u_range': float(u.max() - u.min()),
        'total_diff': total_diff,
        'max_diff': max_diff,
        'n_suboptimal_octants': n_suboptimal,
    }


def analyze_stuck_pattern(client_actions: List[int]) -> Tuple[int, float, bool]:
    """Analyze if client is stuck in one octant."""
    if not client_actions:
        return 0, 0.0, False

    counts = Counter(client_actions)
    dominant_octant, dominant_count = counts.most_common(1)[0]
    dominant_pct = dominant_count / len(client_actions)
    is_stuck = dominant_pct > 0.80

    return dominant_octant, dominant_pct, is_stuck


def is_octant_suboptimal(seed: int, octant: int) -> bool:
    """Check if complementary response for octant is suboptimal."""
    u = sample_u_matrix(random_state=seed)
    comp_therapist = COMPLEMENT[octant]
    optimal_therapist = u[octant, :].argmax()
    return optimal_therapist != comp_therapist


def analyze_noncomp_actions(
    client_actions: List[int],
    therapist_actions: List[int]
) -> Tuple[int, float, int, float]:
    """Analyze non-complementary actions and their distances."""
    if not client_actions:
        return 0, 0.0, 0, 0.0

    n_noncomp = 0
    n_gt2_octants = 0

    for client, therapist in zip(client_actions, therapist_actions):
        comp_expected = COMPLEMENT[client]
        if therapist != comp_expected:
            n_noncomp += 1
            dist = octant_distance(therapist, comp_expected)
            if dist > 2:
                n_gt2_octants += 1

    total = len(client_actions)
    pct_noncomp = n_noncomp / total * 100 if total > 0 else 0.0
    pct_gt2 = n_gt2_octants / n_noncomp * 100 if n_noncomp > 0 else 0.0

    return n_noncomp, pct_noncomp, n_gt2_octants, pct_gt2


def collect_and_analyze_seeds(
    target_v2_advantage: int,
    sim_kwargs: Dict[str, Any],
    max_seeds: int = 2000,
) -> List[SeedAnalysis]:
    """Collect seeds until target v2_advantage count, analyzing all."""

    results = []
    n_v2_advantage = 0

    pbar = tqdm(total=target_v2_advantage, desc="Collecting v2_advantage seeds")

    seed = 0
    while n_v2_advantage < target_v2_advantage and seed < max_seeds:
        # Run both therapists
        v2_result = run_single_simulation(
            seed=seed,
            therapist_type='omniscient',
            therapist_version='v2',
            **sim_kwargs
        )
        comp_result = run_single_simulation(
            seed=seed,
            therapist_type='complementary',
            therapist_version='v2',
            **sim_kwargs
        )

        # Determine category
        if v2_result.success and not comp_result.success:
            category = 'v2_advantage'
            n_v2_advantage += 1
            pbar.update(1)
        elif v2_result.success and comp_result.success:
            category = 'both_success'
        elif not v2_result.success and not comp_result.success:
            category = 'both_fail'
        else:
            category = 'comp_advantage'

        # Analyze u_matrix
        u_analysis = analyze_umatrix(seed)

        # Analyze stuck patterns
        comp_dominant, comp_pct, comp_stuck = analyze_stuck_pattern(comp_result.client_actions)
        v2_dominant, v2_pct, v2_stuck = analyze_stuck_pattern(v2_result.client_actions)

        # Check if stuck octant is suboptimal
        comp_stuck_suboptimal = is_octant_suboptimal(seed, comp_dominant) if comp_stuck else False

        # Analyze non-complementary actions
        n_noncomp, pct_noncomp, n_gt2, pct_gt2 = analyze_noncomp_actions(
            v2_result.client_actions, v2_result.therapist_actions
        )

        analysis = SeedAnalysis(
            seed=seed,
            category=category,
            u_min=u_analysis['u_min'],
            u_max=u_analysis['u_max'],
            u_range=u_analysis['u_range'],
            total_diff=u_analysis['total_diff'],
            max_diff=u_analysis['max_diff'],
            n_suboptimal_octants=u_analysis['n_suboptimal_octants'],
            v2_success=v2_result.success,
            comp_success=comp_result.success,
            v2_sessions=v2_result.total_sessions,
            comp_sessions=comp_result.total_sessions,
            comp_dominant_octant=comp_dominant,
            comp_dominant_pct=comp_pct,
            comp_is_stuck=comp_stuck,
            comp_stuck_is_suboptimal=comp_stuck_suboptimal,
            v2_dominant_octant=v2_dominant,
            v2_dominant_pct=v2_pct,
            v2_is_stuck=v2_stuck,
            n_noncomp_actions=n_noncomp,
            pct_noncomp_actions=pct_noncomp,
            n_noncomp_gt2_octants=n_gt2,
            pct_noncomp_gt2_octants=pct_gt2,
        )

        results.append(analysis)
        seed += 1

    pbar.close()

    return results


def run_statistical_tests(results: List[SeedAnalysis]) -> Dict[str, Any]:
    """Run statistical tests comparing groups."""

    # Split by category
    v2_adv = [r for r in results if r.category == 'v2_advantage']
    both_success = [r for r in results if r.category == 'both_success']
    both_fail = [r for r in results if r.category == 'both_fail']

    stats_results = {
        'group_sizes': {
            'v2_advantage': len(v2_adv),
            'both_success': len(both_success),
            'both_fail': len(both_fail),
        }
    }

    # Test 1: Total DIFF comparison
    if len(v2_adv) > 0 and len(both_success) > 0 and len(both_fail) > 0:
        v2_diffs = [r.total_diff for r in v2_adv]
        success_diffs = [r.total_diff for r in both_success]
        fail_diffs = [r.total_diff for r in both_fail]

        # Kruskal-Wallis test (non-parametric)
        h_stat, p_kruskal = stats.kruskal(v2_diffs, success_diffs, fail_diffs)

        # Pairwise Mann-Whitney tests
        u1, p1 = stats.mannwhitneyu(v2_diffs, success_diffs, alternative='two-sided')
        u2, p2 = stats.mannwhitneyu(v2_diffs, fail_diffs, alternative='two-sided')
        u3, p3 = stats.mannwhitneyu(success_diffs, fail_diffs, alternative='two-sided')

        stats_results['total_diff'] = {
            'v2_adv_mean': float(np.mean(v2_diffs)),
            'v2_adv_std': float(np.std(v2_diffs)),
            'both_success_mean': float(np.mean(success_diffs)),
            'both_success_std': float(np.std(success_diffs)),
            'both_fail_mean': float(np.mean(fail_diffs)),
            'both_fail_std': float(np.std(fail_diffs)),
            'kruskal_h': float(h_stat),
            'kruskal_p': float(p_kruskal),
            'mannwhitney_v2_vs_success_p': float(p1),
            'mannwhitney_v2_vs_fail_p': float(p2),
            'mannwhitney_success_vs_fail_p': float(p3),
        }

    # Test 2: n_suboptimal_octants comparison
    if len(v2_adv) > 0 and len(both_success) > 0 and len(both_fail) > 0:
        v2_subopt = [r.n_suboptimal_octants for r in v2_adv]
        success_subopt = [r.n_suboptimal_octants for r in both_success]
        fail_subopt = [r.n_suboptimal_octants for r in both_fail]

        h_stat, p_kruskal = stats.kruskal(v2_subopt, success_subopt, fail_subopt)

        stats_results['n_suboptimal_octants'] = {
            'v2_adv_mean': float(np.mean(v2_subopt)),
            'both_success_mean': float(np.mean(success_subopt)),
            'both_fail_mean': float(np.mean(fail_subopt)),
            'kruskal_h': float(h_stat),
            'kruskal_p': float(p_kruskal),
        }

    return stats_results


def generate_report(results: List[SeedAnalysis], stats_results: Dict[str, Any]) -> str:
    """Generate comprehensive text report."""

    lines = []
    lines.append("=" * 80)
    lines.append("V2 ADVANTAGE U-MATRIX ANALYSIS REPORT")
    lines.append("=" * 80)

    # Split by category
    v2_adv = [r for r in results if r.category == 'v2_advantage']
    both_success = [r for r in results if r.category == 'both_success']
    both_fail = [r for r in results if r.category == 'both_fail']
    comp_adv = [r for r in results if r.category == 'comp_advantage']

    lines.append(f"\n## DATA COLLECTION SUMMARY")
    lines.append(f"Total seeds analyzed: {len(results)}")
    lines.append(f"  - v2_advantage: {len(v2_adv)} ({len(v2_adv)/len(results)*100:.1f}%)")
    lines.append(f"  - both_success: {len(both_success)} ({len(both_success)/len(results)*100:.1f}%)")
    lines.append(f"  - both_fail: {len(both_fail)} ({len(both_fail)/len(results)*100:.1f}%)")
    lines.append(f"  - comp_advantage: {len(comp_adv)} ({len(comp_adv)/len(results)*100:.1f}%)")

    # Analysis 1: Total DIFF
    lines.append(f"\n## ANALYSIS 1: Sum of DIFF (Optimal vs Complementary)")
    lines.append("-" * 80)
    if 'total_diff' in stats_results:
        td = stats_results['total_diff']
        lines.append(f"v2_advantage:   mean={td['v2_adv_mean']:.2f} (std={td['v2_adv_std']:.2f})")
        lines.append(f"both_success:   mean={td['both_success_mean']:.2f} (std={td['both_success_std']:.2f})")
        lines.append(f"both_fail:      mean={td['both_fail_mean']:.2f} (std={td['both_fail_std']:.2f})")
        lines.append(f"\nKruskal-Wallis H={td['kruskal_h']:.2f}, p={td['kruskal_p']:.4f}")
        lines.append(f"Mann-Whitney v2_adv vs both_success: p={td['mannwhitney_v2_vs_success_p']:.4f}")
        lines.append(f"Mann-Whitney v2_adv vs both_fail: p={td['mannwhitney_v2_vs_fail_p']:.4f}")

    # Analysis 2: % stuck in suboptimal octant
    lines.append(f"\n## ANALYSIS 2: % Stuck in Suboptimal Octant")
    lines.append("-" * 80)

    # For v2_advantage seeds
    v2_stuck = [r for r in v2_adv if r.comp_is_stuck]
    v2_stuck_subopt = [r for r in v2_adv if r.comp_is_stuck and r.comp_stuck_is_suboptimal]

    lines.append(f"v2_advantage seeds:")
    lines.append(f"  - Comp therapist stuck (>80% one octant): {len(v2_stuck)}/{len(v2_adv)} ({len(v2_stuck)/len(v2_adv)*100:.1f}%)")
    if len(v2_stuck) > 0:
        lines.append(f"  - Of stuck, in suboptimal octant: {len(v2_stuck_subopt)}/{len(v2_stuck)} ({len(v2_stuck_subopt)/len(v2_stuck)*100:.1f}%)")
    lines.append(f"  - Overall stuck+suboptimal: {len(v2_stuck_subopt)}/{len(v2_adv)} ({len(v2_stuck_subopt)/len(v2_adv)*100:.1f}%)")

    # Compare with other groups
    for group_name, group in [('both_success', both_success), ('both_fail', both_fail)]:
        if group:
            g_stuck = [r for r in group if r.comp_is_stuck]
            g_stuck_subopt = [r for r in group if r.comp_is_stuck and r.comp_stuck_is_suboptimal]
            lines.append(f"\n{group_name} seeds:")
            lines.append(f"  - Comp therapist stuck: {len(g_stuck)}/{len(group)} ({len(g_stuck)/len(group)*100:.1f}%)")
            if len(g_stuck) > 0:
                lines.append(f"  - Of stuck, in suboptimal octant: {len(g_stuck_subopt)}/{len(g_stuck)} ({len(g_stuck_subopt)/len(g_stuck)*100:.1f}%)")

    # Analysis 3: Client stuck patterns comparison
    lines.append(f"\n## ANALYSIS 3: Client 'Stuck' Patterns")
    lines.append("-" * 80)

    for group_name, group in [('v2_advantage', v2_adv), ('both_success', both_success), ('both_fail', both_fail)]:
        if group:
            stuck_comp = sum(1 for r in group if r.comp_is_stuck)
            stuck_v2 = sum(1 for r in group if r.v2_is_stuck)
            lines.append(f"{group_name}:")
            lines.append(f"  - Stuck under comp therapy: {stuck_comp}/{len(group)} ({stuck_comp/len(group)*100:.1f}%)")
            lines.append(f"  - Stuck under v2 therapy: {stuck_v2}/{len(group)} ({stuck_v2/len(group)*100:.1f}%)")

    # Analysis 4: Non-complementary interaction distances
    lines.append(f"\n## ANALYSIS 4: Non-Complementary Interaction Distances (V2 therapist)")
    lines.append("-" * 80)

    if v2_adv:
        mean_pct_noncomp = np.mean([r.pct_noncomp_actions for r in v2_adv])
        mean_pct_gt2 = np.mean([r.pct_noncomp_gt2_octants for r in v2_adv if r.n_noncomp_actions > 0])

        lines.append(f"For v2_advantage seeds:")
        lines.append(f"  - Mean % non-complementary actions: {mean_pct_noncomp:.2f}%")
        lines.append(f"  - Mean % of non-comp actions >2 octants from complementary: {mean_pct_gt2:.2f}%")

        # Distribution of non-comp distances
        all_gt2_counts = [r.n_noncomp_gt2_octants for r in v2_adv]
        all_noncomp_counts = [r.n_noncomp_actions for r in v2_adv]
        lines.append(f"  - Total non-comp actions: {sum(all_noncomp_counts)}")
        lines.append(f"  - Total non-comp >2 octants: {sum(all_gt2_counts)}")
        lines.append(f"  - Overall % >2 octants: {sum(all_gt2_counts)/sum(all_noncomp_counts)*100:.2f}%" if sum(all_noncomp_counts) > 0 else "  - N/A")

    # Key finding summary
    lines.append(f"\n## KEY FINDINGS SUMMARY")
    lines.append("=" * 80)

    if v2_adv and len(v2_stuck_subopt) > 0:
        lines.append(f"1. {len(v2_stuck_subopt)/len(v2_adv)*100:.1f}% of v2_advantage is due to complementary therapy")
        lines.append(f"   getting stuck in an octant where its response is suboptimal.")

    if v2_adv:
        lines.append(f"2. {len(v2_stuck)/len(v2_adv)*100:.1f}% of v2_advantage happens with clients 'stuck'")
        lines.append(f"   in one octant under complementary therapy.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze v2_advantage u_matrix properties")
    parser.add_argument('--target-v2-advantage', type=int, default=75,
                        help='Target number of v2_advantage seeds to collect')
    parser.add_argument('--max-seeds', type=int, default=2000,
                        help='Maximum seeds to try')
    args = parser.parse_args()

    # Load optimal config (Trial 2643)
    db_path = project_root / "optuna_studies" / "freq_amp_v2_optimization.db"
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name="freq_amp_v2_optimization", storage=storage_url)

    trial = study.trials[2643]
    params = trial.params

    sim_kwargs = {
        'mechanism': 'frequency_amplifier',
        'initial_memory_pattern': 'cold_stuck',
        'success_threshold_percentile': params.get('threshold', 0.936),
        'enable_parataxic': True,
        'baseline_accuracy': params.get('baseline_accuracy', 0.555),
        'perception_window': params.get('perception_window', 10),
        'max_sessions': 2000,  # Extended as requested
        'entropy': 1.0,
        'history_weight': 1.0,
        'bond_power': 1.0,
        'bond_alpha': params.get('bond_alpha', 11.85),
        'bond_offset': params.get('bond_offset', 0.624),
        'recency_weighting_factor': params.get('recency_weighting_factor', 2),
        'seeding_benefit_scaling': params.get('seeding_benefit_scaling', 1.87),
        'skip_seeding_accuracy_threshold': params.get('skip_seeding_accuracy_threshold', 0.815),
        'quick_seed_actions_threshold': params.get('quick_seed_actions_threshold', 1),
        'abort_consecutive_failures_threshold': params.get('abort_consecutive_failures_threshold', 4),
    }

    print("=" * 80)
    print("V2 ADVANTAGE U-MATRIX ANALYSIS")
    print("=" * 80)
    print(f"Target v2_advantage seeds: {args.target_v2_advantage}")
    print(f"Max seeds to try: {args.max_seeds}")
    print(f"Max sessions per sim: {sim_kwargs['max_sessions']}")
    print("=" * 80)

    # Collect data
    results = collect_and_analyze_seeds(
        target_v2_advantage=args.target_v2_advantage,
        sim_kwargs=sim_kwargs,
        max_seeds=args.max_seeds,
    )

    # Run statistical tests
    stats_results = run_statistical_tests(results)

    # Generate report
    report = generate_report(results, stats_results)
    print(report)

    # Save results
    output_path = project_root / "results" / "v2_advantage_umatrix_analysis.json"
    output_path.parent.mkdir(exist_ok=True)

    output_data = {
        'config': sim_kwargs,
        'results': [asdict(r) for r in results],
        'statistics': stats_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
