"""Focused parameter sweep targeting likely failure conditions.

Instead of full factorial, tests specific combinations that are most likely
to reveal interesting failure modes.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main sweep module
from test_parameter_sweep import run_parameter_sweep, print_results_summary, save_results
import itertools


def main():
    """Run focused parameter sweep on likely failure conditions."""

    print("=" * 100)
    print("FOCUSED PARAMETER SWEEP (Targeting likely failure modes)")
    print("=" * 100)

    # Test specific scenarios known to be challenging
    test_scenarios = []

    # Scenario 1: Test initial memory patterns with standard settings
    print("\nScenario 1: Memory pattern effects (standard settings)")
    for pattern in ['cw_50_50', 'complementary_perfect', 'conflictual', 'mixed_random']:
        for mech in ['conditional_amplifier', 'bond_weighted_conditional_amplifier']:
            test_scenarios.append({
                'mechanism': mech,
                'entropy': 1.0,
                'history_weight': 1.0,
                'smoothing_alpha': 0.1,
                'bond_power': 1.0,
                'bond_alpha': 5.0,
                'memory_size': 50,
                'success_threshold': 0.5,
                'initial_memory_pattern': pattern,
            })

    # Scenario 2: Test entropy effects with challenging memory
    print("Scenario 2: Entropy effects (with conflictual memory)")
    for entropy in [0.3, 1.0, 3.0]:
        for mech in ['conditional_amplifier', 'bond_weighted_conditional_amplifier']:
            test_scenarios.append({
                'mechanism': mech,
                'entropy': entropy,
                'history_weight': 1.0,
                'smoothing_alpha': 0.1,
                'bond_power': 1.0,
                'bond_alpha': 5.0,
                'memory_size': 50,
                'success_threshold': 0.5,
                'initial_memory_pattern': 'conflictual',
            })

    # Scenario 3: Test success threshold effects
    print("Scenario 3: Success threshold effects (with C→W memory)")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        for mech in ['conditional_amplifier', 'bond_weighted_conditional_amplifier']:
            test_scenarios.append({
                'mechanism': mech,
                'entropy': 1.0,
                'history_weight': 1.0,
                'smoothing_alpha': 0.1,
                'bond_power': 1.0,
                'bond_alpha': 5.0,
                'memory_size': 50,
                'success_threshold': threshold,
                'initial_memory_pattern': 'cw_50_50',
            })

    # Scenario 4: Test history weight effects
    print("Scenario 4: History weight effects (with C→W memory)")
    for hw in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for mech in ['conditional_amplifier', 'frequency_amplifier']:
            test_scenarios.append({
                'mechanism': mech,
                'entropy': 1.0,
                'history_weight': hw,
                'smoothing_alpha': 0.1,
                'bond_power': 1.0,
                'bond_alpha': 5.0,
                'memory_size': 50,
                'success_threshold': 0.5,
                'initial_memory_pattern': 'cw_50_50',
            })

    # Scenario 5: Test bond_power effects
    print("Scenario 5: Bond power effects (with C→W memory)")
    for bp in [0.5, 1.0, 2.0, 3.0]:
        test_scenarios.append({
            'mechanism': 'bond_weighted_conditional_amplifier',
            'entropy': 1.0,
            'history_weight': 1.0,
            'smoothing_alpha': 0.1,
            'bond_power': bp,
            'bond_alpha': 5.0,
            'memory_size': 50,
            'success_threshold': 0.5,
            'initial_memory_pattern': 'cw_50_50',
        })

    # Scenario 6: Test bond_alpha effects
    print("Scenario 6: Bond alpha effects (with conflictual memory)")
    for ba in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for mech in ['conditional_amplifier', 'bond_weighted_conditional_amplifier']:
            test_scenarios.append({
                'mechanism': mech,
                'entropy': 1.0,
                'history_weight': 1.0,
                'smoothing_alpha': 0.1,
                'bond_power': 1.0,
                'bond_alpha': ba,
                'memory_size': 50,
                'success_threshold': 0.5,
                'initial_memory_pattern': 'conflictual',
            })

    # Scenario 7: SKIPPED - memory size variation requires modifying base_client
    # (base_client validates against hardcoded MEMORY_SIZE constant)
    # print("Scenario 7: Memory size effects (with C→W memory)")

    # Scenario 8: Extreme combinations (most likely to fail)
    print("Scenario 8: Extreme combinations")
    extreme_combos = [
        # High entropy + strict threshold + conflictual memory
        {'mechanism': 'conditional_amplifier', 'entropy': 3.0, 'history_weight': 1.0,
         'smoothing_alpha': 0.1, 'bond_power': 1.0, 'bond_alpha': 5.0,
         'memory_size': 50, 'success_threshold': 0.8, 'initial_memory_pattern': 'conflictual'},

        # Very low entropy + conflictual memory (might get stuck)
        {'mechanism': 'conditional_amplifier', 'entropy': 0.2, 'history_weight': 1.0,
         'smoothing_alpha': 0.1, 'bond_power': 1.0, 'bond_alpha': 5.0,
         'memory_size': 50, 'success_threshold': 0.5, 'initial_memory_pattern': 'conflictual'},

        # Very strong history weight + C→W memory (over-reliance on history)
        {'mechanism': 'conditional_amplifier', 'entropy': 1.0, 'history_weight': 10.0,
         'smoothing_alpha': 0.1, 'bond_power': 1.0, 'bond_alpha': 5.0,
         'memory_size': 50, 'success_threshold': 0.5, 'initial_memory_pattern': 'cw_50_50'},

        # High bond_power + low bond_alpha (history becomes irrelevant)
        {'mechanism': 'bond_weighted_conditional_amplifier', 'entropy': 1.0, 'history_weight': 1.0,
         'smoothing_alpha': 0.1, 'bond_power': 5.0, 'bond_alpha': 2.0,
         'memory_size': 50, 'success_threshold': 0.5, 'initial_memory_pattern': 'conflictual'},

        # Very strict threshold + conflictual memory
        {'mechanism': 'conditional_amplifier', 'entropy': 1.0, 'history_weight': 1.0,
         'smoothing_alpha': 0.1, 'bond_power': 1.0, 'bond_alpha': 5.0,
         'memory_size': 50, 'success_threshold': 0.9, 'initial_memory_pattern': 'conflictual'},
    ]
    test_scenarios.extend(extreme_combos)

    # Remove duplicates
    unique_scenarios = []
    seen = set()
    for scenario in test_scenarios:
        key = tuple(sorted(scenario.items()))
        if key not in seen:
            seen.add(key)
            unique_scenarios.append(scenario)

    print(f"\nTotal unique test scenarios: {len(unique_scenarios)}")
    print(f"Trials per scenario: 50")
    print(f"Total simulations: {len(unique_scenarios) * 50}")
    print()

    # Convert scenarios to the format expected by run_parameter_sweep
    # We'll run each scenario individually
    from test_parameter_sweep import ParameterConfig, run_parameter_configuration

    results = []
    for idx, scenario in enumerate(unique_scenarios):
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(unique_scenarios)} ({(idx+1)/len(unique_scenarios)*100:.1f}%)")

        config = ParameterConfig(
            mechanism=scenario['mechanism'],
            entropy=scenario['entropy'],
            history_weight=scenario['history_weight'],
            smoothing_alpha=scenario['smoothing_alpha'],
            bond_power=scenario['bond_power'],
            bond_alpha=scenario['bond_alpha'],
            memory_size=scenario['memory_size'],
            success_threshold=scenario['success_threshold'],
            initial_memory_pattern=scenario['initial_memory_pattern'],
            pattern_type='none',
        )

        config = run_parameter_configuration(config, n_trials=50, random_seed=42)
        results.append(config)

    # Print results
    print_results_summary(results, top_n=20)

    # Save to file
    save_results(results, 'parameter_sweep_focused_results.json')

    # Additional analysis
    print("\n" + "=" * 100)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 100)

    # Group by success rate categories
    perfect = [r for r in results if r.success_rate == 1.0]
    high = [r for r in results if 0.8 <= r.success_rate < 1.0]
    moderate = [r for r in results if 0.5 <= r.success_rate < 0.8]
    low = [r for r in results if 0.2 < r.success_rate < 0.5]
    very_low = [r for r in results if r.success_rate <= 0.2]

    print(f"\nSuccess rate distribution:")
    print(f"  100% success: {len(perfect)} ({len(perfect)/len(results)*100:.1f}%)")
    print(f"  80-99% success: {len(high)} ({len(high)/len(results)*100:.1f}%)")
    print(f"  50-79% success: {len(moderate)} ({len(moderate)/len(results)*100:.1f}%)")
    print(f"  20-49% success: {len(low)} ({len(low)/len(results)*100:.1f}%)")
    print(f"  0-20% success: {len(very_low)} ({len(very_low)/len(results)*100:.1f}%)")

    if very_low or low:
        print(f"\n{'Most problematic configurations (<50% success):':^100}")
        print("-" * 100)
        for r in sorted(very_low + low, key=lambda x: x.success_rate):
            print(f"{r.mechanism[:20]:<20} ent={r.entropy:.1f} hw={r.history_weight:.1f} "
                  f"bp={r.bond_power:.1f} ba={r.bond_alpha:.1f} thr={r.success_threshold:.1f} "
                  f"mem={r.initial_memory_pattern[:15]:<15} → {r.success_rate*100:5.1f}% success")

    print("\n" + "=" * 100)
    print("FOCUSED PARAMETER SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
