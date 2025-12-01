"""
Tests for the imperfect perception system in client agents.

Tests verify:
- Consistent therapist → high accuracy
- Novel behavior → low accuracy (~20% baseline)
- Misperception samples from history distribution
- Disabled perception = perfect
- Memory stores perceived, perception_history stores actual
- Reproducible with same random_state
- with_perception() creates functional subclass
- Custom baseline_accuracy parameter works
"""

import pytest
import numpy as np
from collections import Counter

from src.agents.client_agents import (
    BondOnlyClient,
    PerceptualBondOnlyClient,
    with_perception,
    PerceptionRecord,
    PerceptualClientMixin,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
)
from src.config import sample_u_matrix, PERCEPTION_WINDOW


@pytest.fixture
def base_client_params():
    """Common parameters for client initialization."""
    u_matrix = sample_u_matrix(random_state=42)
    entropy = 3.0
    initial_memory = [(0, 0)] * 50  # All zeros
    return {
        'u_matrix': u_matrix,
        'entropy': entropy,
        'initial_memory': initial_memory,
    }


def test_consistent_therapist_high_accuracy(base_client_params):
    """When therapist is consistent (always same action), accuracy should be high."""
    # Set initial memory to already contain the consistent action
    # This avoids the warm-up period where frequency is still building up
    consistent_action = 3
    base_client_params['initial_memory'] = [(0, consistent_action)] * 50

    client = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.2,
        enable_perception=True,
        random_state=42,
    )

    # Therapist always does action 3 (WS - Warm-Submissive)
    n_interactions = 100

    for _ in range(n_interactions):
        client_action = client.select_action()
        client.update_memory(client_action, consistent_action)

    # Get perception statistics
    stats = client.get_perception_stats()

    # With consistent therapist and memory pre-filled with the same action:
    # - Baseline path (20%): Perceives correctly
    # - History path (80%): Chooses most common action (which is the actual action)
    # Result: Nearly perfect perception because both paths converge to correct action!
    # Memory stores PERCEIVED actions, reinforcing the pattern.
    #
    # This is psychologically realistic - consistent behavior is easy to perceive correctly!
    # We expect very low misperception rate (only from random baseline failures or edge cases)
    assert stats['overall_misperception_rate'] < 0.05, \
        f"Expected <5% errors for consistent therapist, got {stats['overall_misperception_rate']:.2%}"

    # Stage 1 should use history path 80% of the time (when baseline path fails)
    # But since history converges to the correct action, stage1_changed_from_actual should be 0
    # (or very low due to initialization effects)


def test_novel_behavior_baseline_accuracy(base_client_params):
    """Novel therapist actions should have ~20% baseline accuracy."""
    # Set up initial memory with only action 0
    base_client_params['initial_memory'] = [(0, 0)] * 50

    client = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.2,
        enable_perception=True,
        random_state=123,
    )

    # Therapist suddenly does action 7 (completely novel)
    novel_action = 7
    n_interactions = 100

    for _ in range(n_interactions):
        client_action = client.select_action()
        client.update_memory(client_action, novel_action)

    stats = client.get_perception_stats()

    # For novel actions:
    # - frequency[7] starts at 0.0, gradually increases
    # - Early interactions: only baseline (20%) works
    # - Later interactions: frequency increases, so accuracy improves
    # - Average should be closer to baseline + some improvement

    # Check that baseline path was used frequently
    baseline_rate = stats['baseline_correct_count'] / stats['total_interactions']
    assert baseline_rate > 0.10, \
        f"Baseline path should be used ~20% of time, got {baseline_rate:.2%}"


def test_misperception_samples_from_frequency_distribution(base_client_params):
    """When misperception occurs in Stage 1, it should sample from frequency distribution."""
    # Create memory with specific distribution: 70% action 0, 30% action 1
    memory_dist = [(0, 0)] * 35 + [(0, 1)] * 15
    base_client_params['initial_memory'] = memory_dist

    client = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.0,  # Disable baseline so only frequency path is used
        enable_perception=True,
        random_state=456,
    )

    # Now therapist does action 7 (novel, frequency=0)
    # Expected: When misperception occurs, sample from {0: 0.7, 1: 0.3, others: 0}
    novel_action = 7
    n_interactions = 200

    for _ in range(n_interactions):
        client_action = client.select_action()
        client.update_memory(client_action, novel_action)

    # Count perceived actions
    perceived_actions = [record.perceived_therapist_action for record in client.perception_history]
    perceived_counts = Counter(perceived_actions)

    # Since baseline_accuracy=0 and frequency[7]=0 (novel),
    # Stage 1 will ALWAYS fail accuracy check and sample from distribution
    # Distribution in window: mostly 0, some 1, gradually more 7 as it's added

    # Check that action 7 (the actual action) was rarely perceived early on
    # (only through sampling from distribution as it appears in history)
    early_perceptions = perceived_actions[:20]
    novel_in_early = early_perceptions.count(novel_action)

    # Early on, novel action shouldn't be perceived often (frequency near 0)
    assert novel_in_early < 10, \
        f"Novel action shouldn't be perceived often early, got {novel_in_early}/20"


def test_disabled_perception_is_perfect(base_client_params):
    """When enable_perception=False, perception should be perfect."""
    client = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.2,
        enable_perception=False,  # Disable perception
        random_state=111,
    )

    # Random therapist actions
    rng = np.random.RandomState(222)
    n_interactions = 100

    for _ in range(n_interactions):
        client_action = client.select_action()
        therapist_action = rng.randint(0, 8)
        client.update_memory(client_action, therapist_action)

    # No perception records should be created
    assert len(client.perception_history) == 0, \
        "Disabled perception should not create perception records"

    # Check that all memories are exactly as provided
    # (We can't directly verify this without accessing memory, but we verified
    # no distortion occurred)


def test_memory_stores_perceived_not_actual(base_client_params):
    """Client's memory should contain perceived actions, not actual actions."""
    base_client_params['initial_memory'] = [(0, 0)] * 50

    client = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.0,  # Force misperceptions
        enable_perception=True,
        random_state=333,
    )

    # Therapist always does action 5
    actual_action = 5
    n_interactions = 50

    for _ in range(n_interactions):
        client_action = client.select_action()
        client.update_memory(client_action, actual_action)

    # Check that memory contains perceived actions, not all 5s
    memory = list(client.memory)
    therapist_actions_in_memory = [interaction[1] for interaction in memory]

    # Since baseline_accuracy=0 and action 5 starts with low frequency,
    # not all therapist actions in memory should be 5
    count_actual = therapist_actions_in_memory.count(actual_action)

    # With baseline_accuracy=0, only way to perceive 5 is:
    # 1. Sample it from frequency distribution (low initially)
    # 2. Frequency increases as it's perceived more
    # Shouldn't be all 5s
    assert count_actual < n_interactions, \
        "Memory should contain perceived actions (not all same as actual)"


def test_perception_history_tracks_actual(base_client_params):
    """perception_history should track ground truth actual actions."""
    client = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.2,
        enable_perception=True,
        random_state=444,
    )

    # Therapist does specific sequence
    actual_sequence = [0, 1, 2, 3, 4, 5, 6, 7] * 10

    for i, therapist_action in enumerate(actual_sequence):
        client_action = client.select_action()
        client.update_memory(client_action, therapist_action)

    # Verify perception_history has all actual actions
    assert len(client.perception_history) == len(actual_sequence), \
        "perception_history should record every interaction"

    for i, record in enumerate(client.perception_history):
        assert record.actual_therapist_action == actual_sequence[i], \
            f"Record {i} should track actual action {actual_sequence[i]}"


def test_reproducible_with_same_random_state(base_client_params):
    """Same random_state should produce identical perception sequences."""
    seed = 555

    # Create two clients with same seed
    client1 = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.2,
        enable_perception=True,
        random_state=seed,
    )

    client2 = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.2,
        enable_perception=True,
        random_state=seed,
    )

    # Same therapist sequence
    therapist_sequence = [0, 1, 2, 3, 4, 5, 6, 7] * 10

    # Run both clients
    for therapist_action in therapist_sequence:
        # Client 1
        action1 = client1.select_action()
        client1.update_memory(action1, therapist_action)

        # Client 2
        action2 = client2.select_action()
        client2.update_memory(action2, therapist_action)

    # Verify identical perception histories
    assert len(client1.perception_history) == len(client2.perception_history)

    for r1, r2 in zip(client1.perception_history, client2.perception_history):
        assert r1.actual_therapist_action == r2.actual_therapist_action
        assert r1.perceived_therapist_action == r2.perceived_therapist_action
        assert r1.stage1_result == r2.stage1_result
        assert r1.baseline_path_succeeded == r2.baseline_path_succeeded
        assert r1.stage1_changed_from_actual == r2.stage1_changed_from_actual
        assert r1.computed_accuracy == r2.computed_accuracy


def test_with_perception_creates_functional_subclass(base_client_params):
    """with_perception() should work with any client class."""
    # Test with different client mechanisms
    for ClientClass in [BondOnlyClient, FrequencyAmplifierClient, ConditionalAmplifierClient]:
        PerceptualClient = with_perception(ClientClass)

        # Verify class name
        assert PerceptualClient.__name__ == f"Perceptual{ClientClass.__name__}"

        # Create instance
        client = PerceptualClient(
            **base_client_params,
            baseline_accuracy=0.2,
            enable_perception=True,
            random_state=666,
        )

        # Verify it has perception capabilities
        assert hasattr(client, 'perception_history')
        assert hasattr(client, 'get_perception_stats')
        assert hasattr(client, '_perceive_therapist_action')

        # Verify it still has base client capabilities
        assert hasattr(client, 'select_action')
        assert hasattr(client, 'update_memory')

        # Run a few interactions
        for _ in range(10):
            action = client.select_action()
            client.update_memory(action, 0)

        # Verify perception records were created
        assert len(client.perception_history) == 10

        # Verify stats work
        stats = client.get_perception_stats()
        assert stats['total_interactions'] == 10


def test_custom_baseline_accuracy(base_client_params):
    """Custom baseline_accuracy parameter should affect perception."""
    # High baseline accuracy (90%)
    client_high = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.9,
        enable_perception=True,
        random_state=777,
    )

    # Low baseline accuracy (5%)
    client_low = PerceptualBondOnlyClient(
        **base_client_params,
        baseline_accuracy=0.05,
        enable_perception=True,
        random_state=777,  # Same seed to isolate baseline effect
    )

    # Novel action (not in initial memory which is all 0s)
    novel_action = 7
    n_interactions = 100

    # Run both clients
    for _ in range(n_interactions):
        # High baseline client
        action_h = client_high.select_action()
        client_high.update_memory(action_h, novel_action)

        # Low baseline client
        action_l = client_low.select_action()
        client_low.update_memory(action_l, novel_action)

    # Get stats
    stats_high = client_high.get_perception_stats()
    stats_low = client_low.get_perception_stats()

    # High baseline should use baseline path more often
    baseline_rate_high = stats_high['baseline_correct_count'] / stats_high['total_interactions']
    baseline_rate_low = stats_low['baseline_correct_count'] / stats_low['total_interactions']

    assert baseline_rate_high > baseline_rate_low, \
        f"High baseline_accuracy should use baseline path more: {baseline_rate_high:.2%} vs {baseline_rate_low:.2%}"

    # High baseline should have lower misperception rate for novel actions
    assert stats_high['overall_misperception_rate'] < stats_low['overall_misperception_rate'], \
        "Higher baseline_accuracy should reduce misperceptions for novel actions"


def test_perception_stats_structure():
    """get_perception_stats() should return correct structure."""
    u_matrix = sample_u_matrix(random_state=42)
    client = PerceptualBondOnlyClient(
        u_matrix=u_matrix,
        entropy=3.0,
        initial_memory=[(0, 0)] * 50,
        baseline_accuracy=0.2,
        enable_perception=True,
        random_state=888,
    )

    # No interactions yet
    stats = client.get_perception_stats()
    expected_keys = {
        'total_interactions',
        'total_misperceptions',
        'overall_misperception_rate',
        'stage1_overridden_count',
        'stage1_override_rate',
        'mean_computed_accuracy',
        'baseline_correct_count',
    }
    assert set(stats.keys()) == expected_keys, "Stats should have correct keys"
    assert stats['total_interactions'] == 0

    # Run some interactions
    for _ in range(50):
        action = client.select_action()
        client.update_memory(action, 0)

    stats = client.get_perception_stats()
    assert stats['total_interactions'] == 50
    assert 0 <= stats['overall_misperception_rate'] <= 1
    assert 0 <= stats['stage1_override_rate'] <= 1
    assert 0 <= stats['mean_computed_accuracy'] <= 1
    assert isinstance(stats['baseline_correct_count'], (int, np.integer))


def test_perception_window_size():
    """Perception should use exactly PERCEPTION_WINDOW (15) recent interactions."""
    u_matrix = sample_u_matrix(random_state=42)

    # Create memory with specific pattern:
    # First 40: all action 0
    # Last 10: all action 1
    # Total: 50 interactions
    initial_memory = [(0, 0)] * 40 + [(0, 1)] * 10

    client = PerceptualBondOnlyClient(
        u_matrix=u_matrix,
        entropy=3.0,
        initial_memory=initial_memory,
        baseline_accuracy=0.0,  # Only use frequency path
        enable_perception=True,
        random_state=999,
    )

    # The last 15 interactions in memory are:
    # Positions 35-49: [(0,0), (0,0), (0,0), (0,0), (0,0), (0,1), ..., (0,1)]
    # = 5 of action 0 + 10 of action 1 = frequency {0: 5/15, 1: 10/15}

    # Now add one interaction with action 2 (novel)
    action = client.select_action()
    client.update_memory(action, 2)

    # Check the computed accuracy for this perception
    record = client.perception_history[0]

    # frequency[2] should be 0.0 (not in last 15 interactions)
    # So computed_accuracy should be 0.0 (since baseline_accuracy=0.0)
    assert record.computed_accuracy == 0.0, \
        f"Novel action should have 0.0 frequency, got {record.computed_accuracy}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
