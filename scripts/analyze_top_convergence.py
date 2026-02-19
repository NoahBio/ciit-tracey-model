#!/usr/bin/env python3
"""
Analyze hyperparameter distributions for all trials with >20% advantage.
Produces convergence visualizations and descriptive statistics.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import optuna
from scipy import stats

optuna.logging.set_verbosity(optuna.logging.WARNING)

DB_PATH = "sqlite:////home/second_partition/Git_Repos/ciit-tracey-model/optuna_studies/v2_updated_advantage_optimization.db"
STUDY_NAME = "v2_updated_advantage_optimization"
ADVANTAGE_THRESHOLD = 0.20


def load_trials():
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = [t for t in all_trials if t.value > ADVANTAGE_THRESHOLD]
    return all_trials, top_trials


def compute_stats(values):
    """Descriptive statistics for a parameter."""
    arr = np.array(values)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    cv = np.std(arr) / np.mean(arr) if np.mean(arr) != 0 else 0
    return {
        'n': len(arr),
        'mean': round(float(np.mean(arr)), 4),
        'std': round(float(np.std(arr)), 4),
        'cv': round(float(cv), 4),
        'min': round(float(np.min(arr)), 4),
        'q1': round(float(q1), 4),
        'median': round(float(median), 4),
        'q3': round(float(q3), 4),
        'max': round(float(np.max(arr)), 4),
        'iqr': round(float(iqr), 4),
        'range': round(float(np.max(arr) - np.min(arr)), 4),
    }


def convergence_score(top_values, all_values):
    """
    Measure how converged the top trials are relative to the full search space.
    Returns ratio of top IQR to full range (lower = more converged).
    """
    arr_top = np.array(top_values)
    arr_all = np.array(all_values)
    full_range = np.max(arr_all) - np.min(arr_all)
    if full_range == 0:
        return 0.0
    top_iqr = np.percentile(arr_top, 75) - np.percentile(arr_top, 25)
    return top_iqr / full_range


def plot_distributions(all_trials, top_trials, output_dir):
    """Main distribution plot: histogram + KDE for each parameter."""
    param_names = sorted(top_trials[0].params.keys())
    n_params = len(param_names)

    fig, axes = plt.subplots(4, 3, figsize=(20, 22))
    axes = axes.flatten()

    # Collect convergence scores for sorting
    conv_scores = {}
    for p in param_names:
        top_vals = [t.params[p] for t in top_trials]
        all_vals = [t.params[p] for t in all_trials]
        conv_scores[p] = convergence_score(top_vals, all_vals)

    # Sort by convergence (most converged first)
    sorted_params = sorted(param_names, key=lambda p: conv_scores[p])

    for i, p in enumerate(sorted_params):
        ax = axes[i]
        top_vals = np.array([t.params[p] for t in top_trials])
        all_vals = np.array([t.params[p] for t in all_trials])

        # Check if discrete
        unique_top = np.unique(top_vals)
        is_discrete = len(unique_top) <= 10 and all(v == int(v) for v in unique_top)

        if is_discrete:
            # Bar chart for discrete params
            unique_all = np.unique(all_vals)
            all_counts = {v: 0 for v in unique_all}
            for v in all_vals:
                all_counts[v] += 1
            top_counts = {v: 0 for v in unique_all}
            for v in top_vals:
                top_counts[v] = top_counts.get(v, 0) + 1

            x_vals = sorted(unique_all)
            # Normalize to proportions
            all_props = [all_counts[v] / len(all_vals) for v in x_vals]
            top_props = [top_counts.get(v, 0) / len(top_vals) for v in x_vals]

            width = 0.35
            x_pos = np.arange(len(x_vals))
            ax.bar(x_pos - width/2, all_props, width, alpha=0.5, color='gray', label=f'All ({len(all_vals):,})')
            ax.bar(x_pos + width/2, top_props, width, alpha=0.8, color='#e74c3c', label=f'>20% ({len(top_vals)})')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v}' for v in x_vals])
            ax.set_ylabel('Proportion', fontsize=10)
        else:
            # Histogram + KDE for continuous params
            full_range = (min(all_vals), max(all_vals))
            bins = 40

            ax.hist(all_vals, bins=bins, range=full_range, density=True, alpha=0.35,
                    color='gray', label=f'All ({len(all_vals):,})')
            ax.hist(top_vals, bins=bins, range=full_range, density=True, alpha=0.7,
                    color='#e74c3c', label=f'>20% ({len(top_vals)})')

            # KDE for top trials
            if len(unique_top) > 2:
                try:
                    kde = stats.gaussian_kde(top_vals)
                    x_kde = np.linspace(full_range[0], full_range[1], 200)
                    ax.plot(x_kde, kde(x_kde), color='darkred', linewidth=2, linestyle='-')
                except np.linalg.LinAlgError:
                    pass

            ax.set_ylabel('Density', fontsize=10)

            # Mark IQR of top trials
            q1, q3 = np.percentile(top_vals, [25, 75])
            ax.axvspan(q1, q3, alpha=0.15, color='red', label=f'Top IQR [{q1:.3f}, {q3:.3f}]')
            ax.axvline(np.median(top_vals), color='darkred', linestyle='--', linewidth=1.5,
                       label=f'Top median: {np.median(top_vals):.3f}')

        score = conv_scores[p]
        convergence_label = "TIGHT" if score < 0.10 else "MODERATE" if score < 0.30 else "WIDE"
        ax.set_title(f'{p}\nConv. score: {score:.3f} ({convergence_label})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)

    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Hyperparameter Distributions: Top 223 Trials (>20% advantage) vs All 46,682 Trials\n'
                 f'Sorted by convergence score (IQR of top / range of all — lower = more converged)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / 'distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved distributions.png")


def plot_convergence_summary(all_trials, top_trials, output_dir):
    """Bar chart of convergence scores for all parameters."""
    param_names = sorted(top_trials[0].params.keys())

    scores = {}
    for p in param_names:
        top_vals = [t.params[p] for t in top_trials]
        all_vals = [t.params[p] for t in all_trials]
        scores[p] = convergence_score(top_vals, all_vals)

    sorted_params = sorted(scores, key=scores.get)
    sorted_scores = [scores[p] for p in sorted_params]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#27ae60' if s < 0.10 else '#f39c12' if s < 0.30 else '#e74c3c' for s in sorted_scores]
    bars = ax.barh(sorted_params, sorted_scores, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, score in zip(bars, sorted_scores):
        label = "TIGHT" if score < 0.10 else "MODERATE" if score < 0.30 else "WIDE"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f} ({label})', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Convergence Score (Top IQR / Full Range)', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Convergence: How Tightly Do Top Trials Cluster?\n'
                 'Green = tight convergence, Yellow = moderate, Red = wide spread',
                 fontsize=13, fontweight='bold')
    ax.axvline(0.10, color='green', linestyle='--', alpha=0.5, label='Tight threshold')
    ax.axvline(0.30, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2, axis='x')
    ax.set_xlim(0, max(sorted_scores) * 1.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'convergence_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved convergence_summary.png")


def plot_pairwise_scatter(top_trials, output_dir):
    """Scatter plots for the top 3 most important parameters (from fANOVA: bond_offset, baseline_accuracy, recency_weighting_factor)."""
    key_params = ['bond_offset', 'baseline_accuracy', 'recency_weighting_factor']
    advantages = np.array([t.value for t in top_trials])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    pairs = [(0, 1), (0, 2), (1, 2)]

    for ax, (i, j) in zip(axes, pairs):
        x = np.array([t.params[key_params[i]] for t in top_trials])
        y = np.array([t.params[key_params[j]] for t in top_trials])
        sc = ax.scatter(x, y, c=advantages * 100, cmap='RdYlGn', s=30, alpha=0.8,
                        edgecolors='black', linewidth=0.3)
        ax.set_xlabel(key_params[i], fontsize=11, fontweight='bold')
        ax.set_ylabel(key_params[j], fontsize=11, fontweight='bold')
        ax.grid(alpha=0.2)
        plt.colorbar(sc, ax=ax, label='Advantage (%)')

    fig.suptitle('Pairwise Scatter: Top 3 Important Parameters (colored by advantage %)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'pairwise_scatter.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved pairwise_scatter.png")


def plot_advantage_vs_param(all_trials, top_trials, output_dir):
    """For each parameter, scatter advantage vs parameter value, highlighting the >20% zone."""
    param_names = sorted(top_trials[0].params.keys())

    fig, axes = plt.subplots(4, 3, figsize=(20, 22))
    axes = axes.flatten()

    # Sample from all trials for background (plotting 46k points is slow)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_trials), size=min(3000, len(all_trials)), replace=False)
    sampled_all = [all_trials[i] for i in sample_idx]

    for i, p in enumerate(sorted(param_names)):
        ax = axes[i]
        # Background: sampled all trials
        x_all = [t.params[p] for t in sampled_all]
        y_all = [t.value * 100 for t in sampled_all]
        ax.scatter(x_all, y_all, alpha=0.08, s=5, color='gray', rasterized=True)

        # Foreground: top trials
        x_top = [t.params[p] for t in top_trials]
        y_top = [t.value * 100 for t in top_trials]
        ax.scatter(x_top, y_top, alpha=0.6, s=15, color='#e74c3c', edgecolors='darkred',
                   linewidth=0.3, zorder=5)

        ax.axhline(20, color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_xlabel(p, fontsize=10, fontweight='bold')
        ax.set_ylabel('Advantage (%)', fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_title(p, fontsize=10, fontweight='bold')

    for i in range(len(param_names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Advantage vs Each Hyperparameter\n'
                 'Gray: random sample of all trials | Red: trials with >20% advantage',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / 'advantage_vs_params.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved advantage_vs_params.png")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/home/second_partition/Git_Repos/ciit-tracey-model/results/top_convergence_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trials...")
    all_trials, top_trials = load_trials()
    print(f"  All completed: {len(all_trials):,}")
    print(f"  >20% advantage: {len(top_trials)}")
    print(f"  Output: {output_dir}\n")

    # 1. Distribution plots
    print("Generating plots...")
    plot_distributions(all_trials, top_trials, output_dir)
    plot_convergence_summary(all_trials, top_trials, output_dir)
    plot_pairwise_scatter(top_trials, output_dir)
    plot_advantage_vs_param(all_trials, top_trials, output_dir)

    # 2. Descriptive statistics
    print("\nComputing descriptive statistics...")
    param_names = sorted(top_trials[0].params.keys())
    report = {
        'n_all': len(all_trials),
        'n_top': len(top_trials),
        'advantage_threshold': ADVANTAGE_THRESHOLD,
        'advantage_range_top': {
            'min': round(min(t.value for t in top_trials) * 100, 2),
            'max': round(max(t.value for t in top_trials) * 100, 2),
        },
        'parameters': {},
    }

    print(f"\n{'Parameter':<42} {'Mean':>8} {'Std':>8} {'CV':>8} {'IQR':>10} {'Conv':>8} {'Verdict'}")
    print("=" * 100)

    for p in param_names:
        top_vals = [t.params[p] for t in top_trials]
        all_vals = [t.params[p] for t in all_trials]
        top_stats = compute_stats(top_vals)
        all_stats = compute_stats(all_vals)
        conv = convergence_score(top_vals, all_vals)
        verdict = "TIGHT" if conv < 0.10 else "MODERATE" if conv < 0.30 else "WIDE"

        report['parameters'][p] = {
            'top_stats': top_stats,
            'all_stats': all_stats,
            'convergence_score': round(conv, 4),
            'verdict': verdict,
        }

        iqr_str = f"[{top_stats['q1']:.3f}, {top_stats['q3']:.3f}]"
        print(f"  {p:<40} {top_stats['mean']:>8.4f} {top_stats['std']:>8.4f} {top_stats['cv']:>8.4f} {iqr_str:>10} {conv:>8.3f} {verdict}")

    # Save report
    with open(output_dir / 'convergence_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved convergence_report.json")

    print(f"\nAll outputs: {output_dir}")


if __name__ == '__main__':
    main()
