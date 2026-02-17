"""Comprehensive statistical analysis of an Optuna hyperparameter optimization study.

Produces 8 rigorous analyses with publication-quality visualizations:
1. Hyperparameter Importance (fANOVA)
2. Deep Dive: Top 15 Trials
3. Statistical Significance Testing (Fisher's exact, z-test, Bonferroni)
4. Mechanism Analysis
5. Top 10 Configuration Comparison
6. Top vs Bottom Percentile Contrast
7. Statistical Robustness / Power Projection
8. Negative Results Analysis

Usage:
    python scripts/analyze_optuna_study.py \
        --db-path optuna_studies/v2_updated_advantage_optimization.db \
        --study-name v2_updated_advantage_optimization \
        --n-seeds 80
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import optuna
import argparse
import json
from datetime import datetime
from scipy import stats
from typing import List, Dict, Tuple


# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_study(db_path: str, study_name: str):
    """Load Optuna study from database."""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None
              and not np.isnan(t.value)]
    return study, trials


def z_test_proportions(p1, p2, n1, n2):
    """Two-proportion z-test. Returns z-stat and two-sided p-value."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val


def proportion_ci(p, n, alpha=0.05):
    """Wilson score confidence interval for a proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def diff_ci(p1, p2, n1, n2, alpha=0.05):
    """Confidence interval for difference of two proportions."""
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return diff - z * se, diff + z * se


# ============================================================================
# ANALYSIS 1: Hyperparameter Importance (fANOVA)
# ============================================================================
def analysis_1_fanova(study, output_dir: Path, report: dict, max_trials: int = 5000):
    """Compute and visualize fANOVA hyperparameter importances.

    For large studies, fANOVA is computed on a stratified sample of trials
    to keep computation tractable (standard practice).
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_total = len(completed)

    if n_total > max_trials:
        print(f"  [1/8] Computing fANOVA importances (sampling {max_trials}/{n_total} trials)...")
        # Create a temporary in-memory study with sampled trials
        rng = np.random.RandomState(42)
        sampled_indices = rng.choice(n_total, size=max_trials, replace=False)
        sampled_trials = [completed[i] for i in sorted(sampled_indices)]

        sampled_study = optuna.create_study(direction='maximize')
        for t in sampled_trials:
            sampled_study.add_trial(t)

        importances = optuna.importance.get_param_importances(
            sampled_study,
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42, n_trees=32),
        )
    else:
        print(f"  [1/8] Computing fANOVA importances ({n_total} trials)...")
        importances = optuna.importance.get_param_importances(
            study,
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42),
        )

    report['fanova_importances'] = importances

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    params = list(importances.keys())
    values = [importances[p] * 100 for p in params]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
    bars = ax.barh(range(len(params)), values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([p.replace('_', ' ').title() for p in params], fontsize=11)
    ax.set_xlabel('Importance (%)', fontsize=13, fontweight='bold')
    ax.set_title('Hyperparameter Importance (fANOVA)', fontsize=15, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / '1_hyperparameter_importance.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Top 3: {params[0]}={values[0]:.1f}%, {params[1]}={values[1]:.1f}%, {params[2]}={values[2]:.1f}%")


# ============================================================================
# ANALYSIS 2: Deep Dive - Top 15 Trials
# ============================================================================
def analysis_2_top15(trials: list, output_dir: Path, report: dict):
    """Detailed analysis and parallel coordinates of top 15 trials."""
    print("  [2/8] Deep dive into top 15 trials...")

    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
    top15 = sorted_trials[:15]

    param_names = sorted(top15[0].params.keys())

    # Collect data for report
    top15_data = []
    for t in top15:
        entry = {
            'trial': t.number,
            'advantage': round(t.value * 100, 2),
            'v2_success': round(t.user_attrs.get('omniscient_success', 0) * 100, 1),
            'bl_success': round(t.user_attrs.get('complementary_success', 0) * 100, 1),
            'v2_dropout': round(t.user_attrs.get('omniscient_dropout', 0) * 100, 2),
            'params': {k: round(v, 4) for k, v in t.params.items()},
        }
        top15_data.append(entry)
    report['top15_trials'] = top15_data

    # Parallel coordinates plot
    fig, ax = plt.subplots(figsize=(16, 7))

    # Normalize each parameter to [0, 1] for parallel coords
    all_param_values = {p: [] for p in param_names}
    for t in sorted_trials:
        for p in param_names:
            all_param_values[p].append(t.params[p])

    param_mins = {p: min(vals) for p, vals in all_param_values.items()}
    param_maxs = {p: max(vals) for p, vals in all_param_values.items()}

    # Plot background (random sample of 500 trials in gray)
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(sorted_trials), size=min(500, len(sorted_trials)), replace=False)
    for idx in bg_indices:
        t = sorted_trials[idx]
        normed = []
        for p in param_names:
            r = param_maxs[p] - param_mins[p]
            normed.append((t.params[p] - param_mins[p]) / r if r > 0 else 0.5)
        ax.plot(range(len(param_names)), normed, color='lightgray', alpha=0.1, linewidth=0.5)

    # Plot top 15 in color
    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=top15[-1].value * 100, vmax=top15[0].value * 100)
    for t in reversed(top15):
        normed = []
        for p in param_names:
            r = param_maxs[p] - param_mins[p]
            normed.append((t.params[p] - param_mins[p]) / r if r > 0 else 0.5)
        color = cmap(norm(t.value * 100))
        ax.plot(range(len(param_names)), normed, color=color, alpha=0.9,
                linewidth=2.5, marker='o', markersize=5)

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels([p.replace('_', '\n') for p in param_names], fontsize=9, ha='center')
    ax.set_ylabel('Normalized Value (0=min, 1=max)', fontsize=12)
    ax.set_title('Top 15 Trials — Parallel Coordinates\n(gray = 500 random trials for context)',
                 fontsize=14, fontweight='bold')

    # Add parameter ranges as text
    for i, p in enumerate(param_names):
        ax.text(i, -0.08, f'{param_mins[p]:.2f}', ha='center', fontsize=7, color='gray',
                transform=ax.get_xaxis_transform())
        ax.text(i, 1.06, f'{param_maxs[p]:.2f}', ha='center', fontsize=7, color='gray',
                transform=ax.get_xaxis_transform())

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Advantage (%)', fontsize=11)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / '2_top15_deep_dive.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Print converged ranges
    print("    Converged parameter ranges in top 15:")
    for p in param_names:
        vals = [t.params[p] for t in top15]
        full_range = param_maxs[p] - param_mins[p]
        top_range = max(vals) - min(vals)
        convergence = (1 - top_range / full_range) * 100 if full_range > 0 else 100
        if convergence > 50:
            print(f"      {p}: [{min(vals):.3f}, {max(vals):.3f}] ({convergence:.0f}% converged)")


# ============================================================================
# ANALYSIS 3: Statistical Significance Testing
# ============================================================================
def analysis_3_significance(trials: list, n_seeds: int, output_dir: Path, report: dict):
    """Fisher's exact test, z-test, Bonferroni correction."""
    print("  [3/8] Statistical significance testing...")

    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)

    # For each trial, compute Fisher's exact test and z-test
    results = []
    for t in sorted_trials:
        v2_succ = t.user_attrs.get('omniscient_success', 0)
        bl_succ = t.user_attrs.get('complementary_success', 0)

        v2_wins = int(round(v2_succ * n_seeds))
        bl_wins = int(round(bl_succ * n_seeds))
        v2_fails = n_seeds - v2_wins
        bl_fails = n_seeds - bl_wins

        # Fisher's exact test (2x2 table)
        table = [[v2_wins, v2_fails], [bl_wins, bl_fails]]
        _, fisher_p = stats.fisher_exact(table, alternative='greater')

        # Z-test
        z_stat, z_p = z_test_proportions(v2_succ, bl_succ, n_seeds, n_seeds)

        # CI for advantage
        ci_low, ci_high = diff_ci(v2_succ, bl_succ, n_seeds, n_seeds)

        results.append({
            'trial': t.number,
            'advantage': t.value,
            'v2_success': v2_succ,
            'bl_success': bl_succ,
            'fisher_p': fisher_p,
            'z_stat': z_stat,
            'z_p': z_p,
            'ci_low': ci_low,
            'ci_high': ci_high,
        })

    n_trials = len(results)
    bonferroni_alpha = 0.05 / n_trials

    # Count significant results
    n_sig_raw = sum(1 for r in results if r['fisher_p'] < 0.05)
    n_sig_bonferroni = sum(1 for r in results if r['fisher_p'] < bonferroni_alpha)

    report['significance'] = {
        'n_trials_tested': n_trials,
        'bonferroni_alpha': bonferroni_alpha,
        'n_significant_raw_p05': n_sig_raw,
        'n_significant_bonferroni': n_sig_bonferroni,
        'best_trial': results[0],
        'top10': results[:10],
    }

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: P-value distribution for top 200 trials
    top_n = min(200, len(results))
    advantages_top = [r['advantage'] * 100 for r in results[:top_n]]
    fisher_ps = [r['fisher_p'] for r in results[:top_n]]
    log_ps = [-np.log10(max(p, 1e-20)) for p in fisher_ps]

    ax = axes[0]
    scatter = ax.scatter(advantages_top, log_ps, c=advantages_top,
                        cmap='RdYlGn', s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
    ax.axhline(-np.log10(0.05), color='orange', linestyle='--', linewidth=1.5,
               label=f'p=0.05')
    ax.axhline(-np.log10(bonferroni_alpha), color='red', linestyle='--', linewidth=1.5,
               label=f'Bonferroni (p={bonferroni_alpha:.2e})')
    ax.set_xlabel('Advantage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax.set_title(f'Statistical Significance (Top {top_n} Trials)\n'
                 f'Fisher\'s Exact Test, one-sided', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Right: CI forest plot for top 15
    ax = axes[1]
    top15_results = results[:15]
    y_positions = range(len(top15_results))

    for i, r in enumerate(top15_results):
        ci_low_pct = r['ci_low'] * 100
        ci_high_pct = r['ci_high'] * 100
        adv_pct = r['advantage'] * 100

        color = 'green' if r['fisher_p'] < 0.05 else 'gray'
        marker = '*' if r['fisher_p'] < bonferroni_alpha else 'o'

        ax.errorbar(adv_pct, i, xerr=[[adv_pct - ci_low_pct], [ci_high_pct - adv_pct]],
                    fmt=marker, color=color, capsize=4, capthick=1.5, linewidth=1.5,
                    markersize=8)
        sig_label = ''
        if r['fisher_p'] < bonferroni_alpha:
            sig_label = ' ***'
        elif r['fisher_p'] < 0.05:
            sig_label = ' *'
        ax.text(ci_high_pct + 1, i, f"p={r['fisher_p']:.4f}{sig_label}", va='center', fontsize=9)

    ax.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Trial {r['trial']}" for r in top15_results], fontsize=10)
    ax.set_xlabel('Advantage (%) with 95% CI', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Trials — Confidence Intervals\n'
                 f'(* p<0.05, *** survives Bonferroni)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / '3_statistical_significance.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"    Significant at p<0.05: {n_sig_raw}/{n_trials}")
    print(f"    Survives Bonferroni (alpha={bonferroni_alpha:.2e}): {n_sig_bonferroni}/{n_trials}")
    print(f"    Best trial p-value: {results[0]['fisher_p']:.6f}")


# ============================================================================
# ANALYSIS 4: Mechanism Analysis
# ============================================================================
def analysis_4_mechanism(trials: list, n_seeds: int, output_dir: Path, report: dict):
    """Analyze whether advantage comes from rescuing failures vs marginal outperformance."""
    print("  [4/8] Mechanism analysis...")

    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)

    v2_rates = np.array([t.user_attrs.get('omniscient_success', 0) for t in sorted_trials])
    bl_rates = np.array([t.user_attrs.get('complementary_success', 0) for t in sorted_trials])
    advantages = np.array([t.value for t in sorted_trials])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Scatter plot - baseline vs V2 success
    ax = axes[0]
    scatter = ax.scatter(bl_rates * 100, v2_rates * 100,
                        c=advantages * 100, cmap='RdYlGn',
                        s=8, alpha=0.4, edgecolors='none')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='No advantage')
    ax.set_xlabel('Baseline Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('V2 Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('V2 vs Baseline Success Rates\n(color = advantage)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Advantage (%)', fontsize=10)

    # Right: Advantage by baseline success bins
    ax = axes[1]
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    bin_means = []
    bin_v2_means = []
    bin_bl_means = []
    bin_counts = []

    for low, high in bins:
        mask = (bl_rates >= low) & (bl_rates < high)
        if mask.sum() > 0:
            bin_means.append(advantages[mask].mean() * 100)
            bin_v2_means.append(v2_rates[mask].mean() * 100)
            bin_bl_means.append(bl_rates[mask].mean() * 100)
            bin_counts.append(mask.sum())
        else:
            bin_means.append(0)
            bin_v2_means.append(0)
            bin_bl_means.append(0)
            bin_counts.append(0)

    x = np.arange(len(bin_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, bin_v2_means, width, label='V2 Success', color='#2ecc71',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, bin_bl_means, width, label='Baseline Success', color='#e74c3c',
                   edgecolor='black', linewidth=0.5)

    # Add advantage labels
    for i, (v2m, blm, advm, n) in enumerate(zip(bin_v2_means, bin_bl_means, bin_means, bin_counts)):
        ax.annotate(f'+{advm:.1f}%\n(n={n:,})',
                   xy=(i, max(v2m, blm) + 1), fontsize=9, ha='center', fontweight='bold',
                   color='darkgreen')

    ax.set_xlabel('Baseline Success Rate Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Advantage by Baseline Difficulty Level',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(bin_v2_means + bin_bl_means) * 1.2)

    report['mechanism_analysis'] = {
        'bins': bin_labels,
        'mean_advantage_by_bin': bin_means,
        'v2_success_by_bin': bin_v2_means,
        'bl_success_by_bin': bin_bl_means,
        'counts_by_bin': bin_counts,
    }

    fig.tight_layout()
    fig.savefig(output_dir / '4_mechanism_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# ANALYSIS 5: Top 10 Configuration Comparison
# ============================================================================
def analysis_5_top10_table(trials: list, n_seeds: int, output_dir: Path, report: dict):
    """Table visualization of top 10 configurations with p-values."""
    print("  [5/8] Top 10 configuration comparison...")

    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    headers = ['Rank', 'Trial', 'Advantage', 'V2 Success', 'BL Success',
               'V2 Dropout', 'z-stat', 'p-value', 'Sig?']

    table_data = []
    top10_report = []
    for i, t in enumerate(sorted_trials):
        v2_succ = t.user_attrs.get('omniscient_success', 0)
        bl_succ = t.user_attrs.get('complementary_success', 0)
        v2_drop = t.user_attrs.get('omniscient_dropout', 0)
        z_stat, z_p = z_test_proportions(v2_succ, bl_succ, n_seeds, n_seeds)

        sig = ''
        if z_p < 0.001:
            sig = '***'
        elif z_p < 0.01:
            sig = '**'
        elif z_p < 0.05:
            sig = '*'

        table_data.append([
            f'{i+1}',
            f'{t.number}',
            f'{t.value*100:.2f}%',
            f'{v2_succ*100:.1f}%',
            f'{bl_succ*100:.1f}%',
            f'{v2_drop*100:.1f}%',
            f'{z_stat:.2f}',
            f'{z_p:.4f}',
            sig,
        ])
        top10_report.append({
            'rank': i+1, 'trial': t.number, 'advantage': round(t.value*100, 2),
            'v2_success': round(v2_succ*100, 1), 'bl_success': round(bl_succ*100, 1),
            'z_stat': round(z_stat, 3), 'p_value': round(z_p, 6), 'significant': sig,
        })

    report['top10_comparison'] = top10_report

    # Color cells based on significance
    cell_colors = []
    for row in table_data:
        row_colors = ['white'] * len(headers)
        sig = row[-1]
        if '***' in sig:
            row_colors[-1] = '#90EE90'
            row_colors[2] = '#90EE90'
        elif '**' in sig:
            row_colors[-1] = '#AAFFAA'
            row_colors[2] = '#AAFFAA'
        elif '*' in sig:
            row_colors[-1] = '#CCFFCC'
            row_colors[2] = '#CCFFCC'
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellColours=cell_colors,
                     colColours=['#D4E6F1'] * len(headers),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    ax.set_title(f'Top 10 Configurations by Advantage (n={n_seeds} seeds per trial)\n'
                 f'* p<0.05  ** p<0.01  *** p<0.001 (z-test for proportions)',
                 fontsize=13, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(output_dir / '5_top10_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# ANALYSIS 6: Top vs Bottom Percentile Contrast
# ============================================================================
def analysis_6_percentile_contrast(trials: list, output_dir: Path, report: dict):
    """Compare parameter distributions between top 10% and bottom 10%."""
    print("  [6/8] Top vs bottom percentile contrast...")

    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
    n = len(sorted_trials)
    top_10pct = sorted_trials[:int(n * 0.1)]
    bottom_10pct = sorted_trials[int(n * 0.9):]

    param_names = sorted(top_10pct[0].params.keys())

    contrast_data = {}
    for p in param_names:
        top_vals = np.array([t.params[p] for t in top_10pct])
        bot_vals = np.array([t.params[p] for t in bottom_10pct])

        top_mean = np.mean(top_vals)
        bot_mean = np.mean(bot_vals)

        # Cohen's d
        pooled_std = np.sqrt((np.std(top_vals)**2 + np.std(bot_vals)**2) / 2)
        cohens_d = (top_mean - bot_mean) / pooled_std if pooled_std > 0 else 0

        # Percent difference
        pct_diff = ((top_mean - bot_mean) / abs(bot_mean) * 100) if bot_mean != 0 else 0

        contrast_data[p] = {
            'top_mean': round(top_mean, 4),
            'bottom_mean': round(bot_mean, 4),
            'cohens_d': round(cohens_d, 3),
            'pct_diff': round(pct_diff, 1),
        }

    report['percentile_contrast'] = contrast_data

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Grouped bar chart
    ax = axes[0]
    # Sort by absolute Cohen's d
    sorted_params = sorted(param_names, key=lambda p: abs(contrast_data[p]['cohens_d']), reverse=True)

    x = np.arange(len(sorted_params))
    width = 0.35

    top_means = [contrast_data[p]['top_mean'] for p in sorted_params]
    bot_means = [contrast_data[p]['bottom_mean'] for p in sorted_params]

    # Normalize for display (different scales)
    # Use percent difference instead
    pct_diffs = [contrast_data[p]['pct_diff'] for p in sorted_params]
    cohens_ds = [contrast_data[p]['cohens_d'] for p in sorted_params]

    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in pct_diffs]
    bars = ax.barh(x, pct_diffs, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels([p.replace('_', ' ').title() for p in sorted_params], fontsize=10)
    ax.set_xlabel('% Difference (Top 10% vs Bottom 10%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Difference:\nTop 10% vs Bottom 10% by Advantage',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, pct_diffs):
        offset = 1 if val >= 0 else -1
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', ha=ha, fontsize=9, fontweight='bold')

    # Right: Cohen's d effect sizes
    ax = axes[1]
    colors_d = []
    for d in cohens_ds:
        if abs(d) >= 0.8:
            colors_d.append('#c0392b')
        elif abs(d) >= 0.5:
            colors_d.append('#e67e22')
        elif abs(d) >= 0.2:
            colors_d.append('#f1c40f')
        else:
            colors_d.append('#95a5a6')

    bars = ax.barh(x, cohens_ds, color=colors_d, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([p.replace('_', ' ').title() for p in sorted_params], fontsize=10)
    ax.set_xlabel("Cohen's d", fontsize=12, fontweight='bold')
    ax.set_title("Effect Size (Cohen's d)\nRed=large, Orange=medium, Yellow=small, Gray=negligible",
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    for threshold, label in [(0.8, 'Large'), (0.5, 'Medium'), (0.2, 'Small')]:
        ax.axvline(threshold, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(-threshold, color='gray', linestyle=':', alpha=0.4)
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / '6_top_vs_bottom_contrast.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# ANALYSIS 7: Statistical Robustness / Power Projection
# ============================================================================
def analysis_7_power_projection(trials: list, n_seeds: int, output_dir: Path, report: dict):
    """Project confidence intervals and detection probabilities at different sample sizes."""
    print("  [7/8] Statistical robustness / power projection...")

    best_trial = sorted(trials, key=lambda t: t.value, reverse=True)[0]
    observed_advantage = best_trial.value
    v2_rate = best_trial.user_attrs.get('omniscient_success', 0)
    bl_rate = best_trial.user_attrs.get('complementary_success', 0)

    sample_sizes = [80, 150, 300, 500, 1000]
    projections = []

    for n in sample_sizes:
        # CI width for the advantage
        se_diff = np.sqrt(v2_rate * (1 - v2_rate) / n + bl_rate * (1 - bl_rate) / n)
        ci_low = observed_advantage - 1.96 * se_diff
        ci_high = observed_advantage + 1.96 * se_diff

        # P(advantage > 0%) under observed effect
        z_for_zero = observed_advantage / se_diff
        prob_positive = stats.norm.cdf(z_for_zero)

        # P(advantage > 5%) under observed effect
        z_for_5pct = (observed_advantage - 0.05) / se_diff
        prob_above_5 = stats.norm.cdf(z_for_5pct)

        # Power to detect at p<0.05 (z-test)
        z_alpha = 1.645  # one-sided
        z_power = observed_advantage / se_diff - z_alpha
        power = stats.norm.cdf(z_power)

        projections.append({
            'n_seeds': n,
            'se': round(se_diff * 100, 2),
            'ci_low': round(ci_low * 100, 2),
            'ci_high': round(ci_high * 100, 2),
            'ci_width': round((ci_high - ci_low) * 100, 2),
            'prob_positive': round(prob_positive * 100, 2),
            'prob_above_5pct': round(prob_above_5 * 100, 2),
            'power_p05': round(power * 100, 2),
        })

    report['power_projection'] = {
        'observed_advantage': round(observed_advantage * 100, 2),
        'v2_rate': round(v2_rate * 100, 1),
        'bl_rate': round(bl_rate * 100, 1),
        'projections': projections,
    }

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ns = [p['n_seeds'] for p in projections]
    ci_lows = [p['ci_low'] for p in projections]
    ci_highs = [p['ci_high'] for p in projections]
    probs_pos = [p['prob_positive'] for p in projections]
    probs_5 = [p['prob_above_5pct'] for p in projections]
    powers = [p['power_p05'] for p in projections]

    # Left: CI width narrowing
    ax = axes[0]
    ax.fill_between(ns, ci_lows, ci_highs, alpha=0.3, color='blue')
    ax.plot(ns, [observed_advantage * 100] * len(ns), 'b-', linewidth=2, label='Point estimate')
    ax.plot(ns, ci_lows, 'b--', linewidth=1, alpha=0.7)
    ax.plot(ns, ci_highs, 'b--', linewidth=1, alpha=0.7)
    ax.axhline(0, color='red', linestyle='-', alpha=0.5, linewidth=1)
    ax.axhline(5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Seeds per Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Advantage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'95% CI Narrowing\n(observed: {observed_advantage*100:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    for i, n in enumerate(ns):
        width = ci_highs[i] - ci_lows[i]
        ax.annotate(f'CI width:\n{width:.1f}%', xy=(n, ci_highs[i] + 1),
                   fontsize=8, ha='center')

    # Middle: Detection probabilities
    ax = axes[1]
    ax.plot(ns, probs_pos, 'g-o', linewidth=2, markersize=8, label='P(advantage > 0%)')
    ax.plot(ns, probs_5, 'b-s', linewidth=2, markersize=8, label='P(advantage > 5%)')
    ax.axhline(95, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Seeds per Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Detection Probability\n(assuming true effect = observed)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(50, 101)
    ax.grid(alpha=0.3)

    for n, pp, p5 in zip(ns, probs_pos, probs_5):
        ax.annotate(f'{pp:.1f}%', xy=(n, pp + 1), fontsize=8, ha='center', color='green')
        ax.annotate(f'{p5:.1f}%', xy=(n, p5 - 3), fontsize=8, ha='center', color='blue')

    # Right: Power to detect at p<0.05
    ax = axes[2]
    ax.plot(ns, powers, 'r-^', linewidth=2, markersize=8)
    ax.axhline(80, color='gray', linestyle=':', alpha=0.5, label='80% power')
    ax.axhline(95, color='gray', linestyle='--', alpha=0.5, label='95% power')
    ax.set_xlabel('Seeds per Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Power to Detect Effect at p<0.05\n(one-sided z-test)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(50, 101)
    ax.grid(alpha=0.3)

    for n, pw in zip(ns, powers):
        ax.annotate(f'{pw:.1f}%', xy=(n, pw + 1.5), fontsize=9, ha='center', fontweight='bold')

    fig.suptitle(f'Best Trial #{best_trial.number}: Power Analysis for Observed Advantage of {observed_advantage*100:.1f}%',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / '7_power_projection.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# ANALYSIS 8: Negative Results Analysis
# ============================================================================
def analysis_8_negative_results(trials: list, output_dir: Path, report: dict):
    """Analyze trials where V2 performed worse than baseline."""
    print("  [8/8] Negative results analysis...")

    advantages = np.array([t.value for t in trials])
    negative_mask = advantages < 0
    n_negative = negative_mask.sum()
    pct_negative = n_negative / len(trials) * 100

    negative_trials = [t for t in trials if t.value < 0]
    positive_trials = [t for t in trials if t.value > 0]

    report['negative_results'] = {
        'n_negative': n_negative,
        'pct_negative': round(pct_negative, 1),
        'max_disadvantage': round(min(advantages) * 100, 2),
        'mean_negative_advantage': round(np.mean([t.value for t in negative_trials]) * 100, 2) if negative_trials else 0,
    }

    param_names = sorted(trials[0].params.keys())

    # Compare parameter distributions: negative vs positive trials
    param_comparison = {}
    for p in param_names:
        neg_vals = np.array([t.params[p] for t in negative_trials])
        pos_vals = np.array([t.params[p] for t in positive_trials])
        if len(neg_vals) > 0 and len(pos_vals) > 0:
            _, ks_p = stats.ks_2samp(neg_vals, pos_vals)
            param_comparison[p] = {
                'neg_mean': round(np.mean(neg_vals), 4),
                'pos_mean': round(np.mean(pos_vals), 4),
                'ks_p_value': round(ks_p, 6),
            }

    report['negative_results']['param_comparison'] = param_comparison

    # Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: Advantage histogram with negative region highlighted
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(min(advantages), max(advantages), 80)
    ax.hist(advantages * 100, bins=bins * 100, color='#3498db', alpha=0.7,
            edgecolor='black', linewidth=0.3, label='All trials')
    # Highlight negative
    neg_advs = advantages[negative_mask] * 100
    ax.hist(neg_advs, bins=bins * 100, color='#e74c3c', alpha=0.8,
            edgecolor='black', linewidth=0.3, label=f'Negative ({n_negative}, {pct_negative:.1f}%)')
    ax.axvline(0, color='black', linewidth=2, linestyle='-')
    ax.set_xlabel('Advantage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Advantage Distribution\n{n_negative}/{len(trials)} trials negative ({pct_negative:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Top-right: Parameters that differentiate negative from positive
    ax = fig.add_subplot(gs[0, 1])
    sorted_by_ks = sorted(param_comparison.items(), key=lambda x: x[1]['ks_p_value'])
    param_labels = [p.replace('_', '\n') for p, _ in sorted_by_ks]
    ks_pvals = [-np.log10(max(v['ks_p_value'], 1e-50)) for _, v in sorted_by_ks]

    colors = ['#e74c3c' if p < 1e-10 else '#e67e22' if p < 0.001 else '#f1c40f' if p < 0.05 else '#95a5a6'
              for _, v in sorted_by_ks for p in [v['ks_p_value']]]
    ax.barh(range(len(param_labels)), ks_pvals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(param_labels)))
    ax.set_yticklabels(param_labels, fontsize=9)
    ax.set_xlabel('-log10(KS p-value)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Distribution Difference\n(Negative vs Positive Trials, KS test)',
                 fontsize=13, fontweight='bold')
    ax.axvline(-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Bottom-left: Box plots for top 3 differentiating parameters
    ax = fig.add_subplot(gs[1, 0])
    top3_params = [p for p, _ in sorted_by_ks[:3]]
    positions = []
    labels = []
    data_neg = []
    data_pos = []
    for i, p in enumerate(top3_params):
        neg_vals = [t.params[p] for t in negative_trials]
        pos_vals = [t.params[p] for t in positive_trials]
        data_neg.append(neg_vals)
        data_pos.append(pos_vals)

    for i, p in enumerate(top3_params):
        bp_neg = ax.boxplot([data_neg[i]], positions=[i*3], widths=0.8,
                           patch_artist=True, showfliers=False)
        bp_pos = ax.boxplot([data_pos[i]], positions=[i*3+1], widths=0.8,
                           patch_artist=True, showfliers=False)
        bp_neg['boxes'][0].set_facecolor('#e74c3c')
        bp_neg['boxes'][0].set_alpha(0.6)
        bp_pos['boxes'][0].set_facecolor('#2ecc71')
        bp_pos['boxes'][0].set_alpha(0.6)

    ax.set_xticks([i*3+0.5 for i in range(len(top3_params))])
    ax.set_xticklabels([p.replace('_', ' ').title() for p in top3_params], fontsize=10)
    ax.set_title('Top 3 Differentiating Parameters\nRed=Negative trials, Green=Positive trials',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Bottom-right: Worst 10 trials summary
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    worst10 = sorted(negative_trials, key=lambda t: t.value)[:10]

    headers = ['Trial', 'Advantage', 'V2%', 'BL%', 'BL Acc', 'Bond Off', 'MaxSess']
    table_data = []
    for t in worst10:
        table_data.append([
            f'{t.number}',
            f'{t.value*100:.1f}%',
            f'{t.user_attrs.get("omniscient_success", 0)*100:.0f}%',
            f'{t.user_attrs.get("complementary_success", 0)*100:.0f}%',
            f'{t.params.get("baseline_accuracy", 0):.3f}',
            f'{t.params.get("bond_offset", 0):.3f}',
            f'{t.params.get("max_sessions", 0):.0f}',
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     colColours=['#FFCCCB'] * len(headers),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title('10 Worst-Performing Configurations', fontsize=12, fontweight='bold', pad=20)

    fig.savefig(output_dir / '8_negative_results.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"    Negative trials: {n_negative}/{len(trials)} ({pct_negative:.1f}%)")
    print(f"    Max disadvantage: {min(advantages)*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Optuna study analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--db-path', type=str,
                       default='optuna_studies/v2_updated_advantage_optimization.db')
    parser.add_argument('--study-name', type=str,
                       default='v2_updated_advantage_optimization')
    parser.add_argument('--n-seeds', type=int, default=80,
                       help='Number of seeds used per trial in the study')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    # Setup
    db_path = project_root / args.db_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / f"optuna_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE OPTUNA STUDY ANALYSIS")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Study: {args.study_name}")
    print(f"Seeds per trial: {args.n_seeds}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load data
    print("\nLoading study...")
    study, trials = load_study(str(db_path), args.study_name)
    print(f"Loaded {len(trials)} completed trials")

    # Initialize report
    report = {
        'timestamp': timestamp,
        'db_path': str(db_path),
        'study_name': args.study_name,
        'n_seeds': args.n_seeds,
        'n_trials': len(trials),
        'advantage_range': [round(min(t.value for t in trials) * 100, 2),
                           round(max(t.value for t in trials) * 100, 2)],
        'mean_advantage': round(np.mean([t.value for t in trials]) * 100, 2),
    }

    print(f"\nRunning 8 analyses...\n")

    # Run all analyses
    analysis_1_fanova(study, output_dir, report)
    analysis_2_top15(trials, output_dir, report)
    analysis_3_significance(trials, args.n_seeds, output_dir, report)
    analysis_4_mechanism(trials, args.n_seeds, output_dir, report)
    analysis_5_top10_table(trials, args.n_seeds, output_dir, report)
    analysis_6_percentile_contrast(trials, output_dir, report)
    analysis_7_power_projection(trials, args.n_seeds, output_dir, report)
    analysis_8_negative_results(trials, output_dir, report)

    # Save report
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nAll outputs saved to: {output_dir}")
    print("Files generated:")
    for f in sorted(output_dir.glob('*')):
        size = f.stat().st_size
        if size > 1024*1024:
            print(f"  {f.name} ({size/1024/1024:.1f} MB)")
        elif size > 1024:
            print(f"  {f.name} ({size/1024:.1f} KB)")
        else:
            print(f"  {f.name} ({size} B)")


if __name__ == "__main__":
    main()
