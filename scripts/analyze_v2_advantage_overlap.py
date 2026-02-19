"""Analyze v2_advantage seed overlap across top Optuna configs.

This script:
1. Loads top 15 configs from the freq_amp_v2_optimization.db
2. Runs v2 and complementary simulations for 75 seeds per config
3. Identifies v2_advantage seeds (v2 succeeds, complementary fails)
4. Calculates overlap across 90%+ of configs
5. Compares advantage between same seed across two random configs

Usage:
    python scripts/analyze_v2_advantage_overlap.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import optuna
import json
import random
from typing import Dict, List, Set, Tuple
from collections import Counter
from tqdm import tqdm

from evaluate_omniscient_therapist import run_single_simulation


def load_top_configs(db_path: str, study_name: str, top_k: int = 15) -> List[Dict]:
    """Load top K configs from Optuna database."""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Get completed trials sorted by advantage
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)[:top_k]

    configs = []
    for trial in trials_sorted:
        config = {
            'trial_number': trial.number,
            'advantage': trial.value,
            'params': trial.params.copy(),
        }
        configs.append(config)

    return configs


def run_simulations_for_config(
    config: Dict,
    n_seeds: int = 75,
    therapist_version: str = 'v2'
) -> Tuple[List[bool], List[bool]]:
    """Run v2 and complementary simulations for a config.

    Returns:
        Tuple of (v2_successes, comp_successes) as lists of bools per seed
    """
    params = config['params']

    # Fixed params for freq_amp_v2_optimization
    mechanism = 'frequency_amplifier'
    pattern = 'cold_stuck'
    enable_parataxic = True
    entropy = 1.0

    # Build simulation kwargs from config params
    sim_kwargs = {
        'mechanism': mechanism,
        'initial_memory_pattern': pattern,
        'success_threshold_percentile': params.get('threshold', 0.8),
        'enable_parataxic': enable_parataxic,
        'baseline_accuracy': params.get('baseline_accuracy', 0.5),
        'perception_window': params.get('perception_window', 15),
        'max_sessions': params.get('max_sessions', 100),
        'entropy': entropy,
        'history_weight': 1.0,
        'bond_power': 1.0,
        'bond_alpha': params.get('bond_alpha', 5.0),
        'bond_offset': params.get('bond_offset', 0.8),
        'recency_weighting_factor': params.get('recency_weighting_factor', 2.0),
        'seeding_benefit_scaling': params.get('seeding_benefit_scaling', 0.3),
        'skip_seeding_accuracy_threshold': params.get('skip_seeding_accuracy_threshold', 0.9),
        'quick_seed_actions_threshold': params.get('quick_seed_actions_threshold', 3),
        'abort_consecutive_failures_threshold': params.get('abort_consecutive_failures_threshold', 5),
    }

    v2_successes = []
    comp_successes = []

    for seed in range(n_seeds):
        # V2 therapist
        v2_result = run_single_simulation(
            seed=seed,
            therapist_type='omniscient',
            therapist_version=therapist_version,
            **sim_kwargs
        )
        v2_successes.append(v2_result.success)

        # Complementary therapist
        comp_result = run_single_simulation(
            seed=seed,
            therapist_type='complementary',
            therapist_version=therapist_version,
            **sim_kwargs
        )
        comp_successes.append(comp_result.success)

    return v2_successes, comp_successes


def get_v2_advantage_seeds(v2_successes: List[bool], comp_successes: List[bool]) -> Set[int]:
    """Get seeds where v2 succeeded but complementary failed."""
    advantage_seeds = set()
    for seed in range(len(v2_successes)):
        if v2_successes[seed] and not comp_successes[seed]:
            advantage_seeds.add(seed)
    return advantage_seeds


def calculate_overlap_at_threshold(
    all_advantage_seeds: Dict[int, Set[int]],
    threshold_pct: float = 0.9
) -> Tuple[Set[int], float]:
    """Calculate seeds that appear in >= threshold_pct of configs.

    Args:
        all_advantage_seeds: Dict mapping config_idx to set of v2_advantage seeds
        threshold_pct: Fraction of configs (0.9 = 90%)

    Returns:
        Tuple of (set of common seeds, percentage of overlap)
    """
    n_configs = len(all_advantage_seeds)
    min_configs = int(n_configs * threshold_pct)

    # Count seed occurrences across configs
    seed_counts = Counter()
    for config_seeds in all_advantage_seeds.values():
        for seed in config_seeds:
            seed_counts[seed] += 1

    # Find seeds present in >= min_configs
    common_seeds = {seed for seed, count in seed_counts.items() if count >= min_configs}

    # Calculate what percentage of total v2_advantage seeds are common
    all_unique_seeds = set()
    for config_seeds in all_advantage_seeds.values():
        all_unique_seeds.update(config_seeds)

    overlap_pct = len(common_seeds) / len(all_unique_seeds) * 100 if all_unique_seeds else 0

    return common_seeds, overlap_pct


def compare_same_seed_across_configs(
    all_v2_successes: Dict[int, List[bool]],
    all_comp_successes: Dict[int, List[bool]],
    all_advantage_seeds: Dict[int, Set[int]],
) -> Dict:
    """Compare advantage of same v2_advantage seed between two random configs."""
    # Get union of all v2_advantage seeds
    all_adv_seeds = set()
    for seeds in all_advantage_seeds.values():
        all_adv_seeds.update(seeds)

    if len(all_adv_seeds) == 0:
        return {"error": "No v2_advantage seeds found"}

    # Pick a random v2_advantage seed
    config_indices = list(all_advantage_seeds.keys())
    if len(config_indices) < 2:
        return {"error": "Need at least 2 configs"}

    # Pick two random configs
    random.seed(42)  # Reproducible
    config1, config2 = random.sample(config_indices, 2)

    # Find a seed that's v2_advantage in at least one of them
    # Prefer seeds that are advantage in config1
    adv_seeds_config1 = all_advantage_seeds[config1]
    if adv_seeds_config1:
        seed = random.choice(list(adv_seeds_config1))
    else:
        seed = random.choice(list(all_adv_seeds))

    # Get results for this seed in both configs
    result = {
        'seed': seed,
        'config1': {
            'config_idx': config1,
            'v2_success': all_v2_successes[config1][seed],
            'comp_success': all_comp_successes[config1][seed],
            'is_v2_advantage': seed in all_advantage_seeds[config1],
        },
        'config2': {
            'config_idx': config2,
            'v2_success': all_v2_successes[config2][seed],
            'comp_success': all_comp_successes[config2][seed],
            'is_v2_advantage': seed in all_advantage_seeds[config2],
        },
    }

    return result


def main():
    print("=" * 80)
    print("V2 ADVANTAGE SEED OVERLAP ANALYSIS")
    print("=" * 80)

    # Load top 15 configs
    db_path = project_root / "optuna_studies" / "freq_amp_v2_optimization.db"
    configs = load_top_configs(str(db_path), "freq_amp_v2_optimization", top_k=15)

    print(f"\nLoaded {len(configs)} configs from Optuna study")
    for i, cfg in enumerate(configs):
        print(f"  Config {i}: Trial {cfg['trial_number']}, Advantage: {cfg['advantage']:.1%}")

    # Run simulations for each config
    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS (75 seeds × 15 configs × 2 therapists = 2250 runs)")
    print("=" * 80)

    n_seeds = 75
    all_v2_successes = {}
    all_comp_successes = {}
    all_advantage_seeds = {}

    for i, config in enumerate(tqdm(configs, desc="Configs")):
        v2_successes, comp_successes = run_simulations_for_config(config, n_seeds=n_seeds)
        all_v2_successes[i] = v2_successes
        all_comp_successes[i] = comp_successes
        all_advantage_seeds[i] = get_v2_advantage_seeds(v2_successes, comp_successes)

        n_adv = len(all_advantage_seeds[i])
        v2_rate = sum(v2_successes) / len(v2_successes) * 100
        comp_rate = sum(comp_successes) / len(comp_successes) * 100
        print(f"  Config {i} (Trial {config['trial_number']}): "
              f"V2={v2_rate:.1f}%, Comp={comp_rate:.1f}%, V2_advantage seeds={n_adv}")

    # Analysis 1: Overlap at 90% threshold
    print("\n" + "=" * 80)
    print("ANALYSIS 1: SEED OVERLAP ACROSS 90% OF CONFIGS")
    print("=" * 80)

    common_seeds_90, overlap_pct = calculate_overlap_at_threshold(all_advantage_seeds, 0.9)

    # Also calculate for other thresholds for context
    common_seeds_100, _ = calculate_overlap_at_threshold(all_advantage_seeds, 1.0)
    common_seeds_80, _ = calculate_overlap_at_threshold(all_advantage_seeds, 0.8)
    common_seeds_50, _ = calculate_overlap_at_threshold(all_advantage_seeds, 0.5)

    # All unique v2_advantage seeds
    all_unique = set()
    for seeds in all_advantage_seeds.values():
        all_unique.update(seeds)

    print(f"\nTotal unique v2_advantage seeds across all configs: {len(all_unique)}")
    print(f"\nSeeds appearing in >= X% of configs:")
    print(f"  100% (all 15): {len(common_seeds_100)} seeds -> {sorted(common_seeds_100)}")
    print(f"   90% (≥14):    {len(common_seeds_90)} seeds -> {sorted(common_seeds_90)}")
    print(f"   80% (≥12):    {len(common_seeds_80)} seeds -> {sorted(common_seeds_80)}")
    print(f"   50% (≥8):     {len(common_seeds_50)} seeds -> {sorted(common_seeds_50)}")

    print(f"\n=> {len(common_seeds_90)} of {len(all_unique)} unique v2_advantage seeds "
          f"({len(common_seeds_90)/len(all_unique)*100 if all_unique else 0:.1f}%) "
          f"are shared across 90%+ of configs")

    # Show seed frequency distribution
    seed_counts = Counter()
    for config_seeds in all_advantage_seeds.values():
        for seed in config_seeds:
            seed_counts[seed] += 1

    print("\nSeed frequency distribution (how many configs each seed appears in):")
    freq_dist = Counter(seed_counts.values())
    for n_configs in sorted(freq_dist.keys(), reverse=True):
        seeds_at_freq = [s for s, c in seed_counts.items() if c == n_configs]
        print(f"  {n_configs} configs: {len(seeds_at_freq)} seeds {sorted(seeds_at_freq)[:10]}{'...' if len(seeds_at_freq) > 10 else ''}")

    # Analysis 2: Compare same seed across two random configs
    print("\n" + "=" * 80)
    print("ANALYSIS 2: SAME SEED ACROSS TWO RANDOM CONFIGS")
    print("=" * 80)

    comparison = compare_same_seed_across_configs(
        all_v2_successes, all_comp_successes, all_advantage_seeds
    )

    if 'error' not in comparison:
        print(f"\nSeed {comparison['seed']}:")
        c1 = comparison['config1']
        c2 = comparison['config2']
        print(f"\n  Config {c1['config_idx']} (Trial {configs[c1['config_idx']]['trial_number']}):")
        print(f"    V2 success: {c1['v2_success']}, Comp success: {c1['comp_success']}")
        print(f"    Is V2 advantage: {c1['is_v2_advantage']}")

        print(f"\n  Config {c2['config_idx']} (Trial {configs[c2['config_idx']]['trial_number']}):")
        print(f"    V2 success: {c2['v2_success']}, Comp success: {c2['comp_success']}")
        print(f"    Is V2 advantage: {c2['is_v2_advantage']}")

        # Show more examples
        print("\n  Additional random comparisons:")
        random.seed(123)
        for _ in range(5):
            cfg_a, cfg_b = random.sample(list(all_advantage_seeds.keys()), 2)
            # Pick a random seed
            test_seed = random.randint(0, n_seeds - 1)
            v2_a = all_v2_successes[cfg_a][test_seed]
            comp_a = all_comp_successes[cfg_a][test_seed]
            v2_b = all_v2_successes[cfg_b][test_seed]
            comp_b = all_comp_successes[cfg_b][test_seed]
            adv_a = test_seed in all_advantage_seeds[cfg_a]
            adv_b = test_seed in all_advantage_seeds[cfg_b]
            print(f"    Seed {test_seed}: Config{cfg_a}[V2={v2_a},C={comp_a},adv={adv_a}] vs "
                  f"Config{cfg_b}[V2={v2_b},C={comp_b},adv={adv_b}]")
    else:
        print(f"  Error: {comparison['error']}")

    # Save results
    output_path = project_root / "results" / "v2_advantage_overlap_analysis.json"
    output_path.parent.mkdir(exist_ok=True)

    results = {
        'n_configs': len(configs),
        'n_seeds': n_seeds,
        'configs': [{'trial': c['trial_number'], 'advantage': c['advantage']} for c in configs],
        'v2_advantage_seeds_per_config': {
            str(k): sorted(list(v)) for k, v in all_advantage_seeds.items()
        },
        'total_unique_v2_advantage_seeds': len(all_unique),
        'seeds_in_90pct_configs': sorted(list(common_seeds_90)),
        'seeds_in_100pct_configs': sorted(list(common_seeds_100)),
        'overlap_percentage': overlap_pct,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
