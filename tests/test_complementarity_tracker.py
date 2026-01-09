"""Unit tests for ComplementarityTracker class."""

import pytest
import numpy as np
from src.analysis.complementarity_tracker import (
    ComplementarityTracker,
    COMPLEMENT_MAP,
    WARM_OCTANTS,
    COLD_OCTANTS,
)


class TestComplementarityTracker:
    """Test suite for ComplementarityTracker."""

    def test_always_complementary_therapist(self):
        """Test that always-complementary therapist shows 100% complementarity.

        This is the verification test from the plan: a simple always-complementary
        therapist should produce a flat line at 100% complementarity.
        """
        tracker = ComplementarityTracker(window_size=10)

        # Simulate 20 sessions of perfect complementarity
        for client_action in range(8):
            therapist_action = COMPLEMENT_MAP[client_action]
            tracker.add_interaction(client_action, therapist_action)

        # After 8 interactions, should be 100%
        assert tracker.get_overall_rate() == 100.0

        # Continue for more interactions
        for client_action in range(8):
            therapist_action = COMPLEMENT_MAP[client_action]
            tracker.add_interaction(client_action, therapist_action)

        # Should still be 100%
        assert tracker.get_overall_rate() == 100.0

    def test_never_complementary_therapist(self):
        """Test that a therapist who never complements shows 0% complementarity."""
        tracker = ComplementarityTracker(window_size=10)

        # Client always does 0 (D), therapist always responds with 2 (W)
        # This is NOT complementary (complement of 0 is 4)
        for _ in range(10):
            tracker.add_interaction(0, 2)

        assert tracker.get_overall_rate() == 0.0

    def test_random_therapist(self):
        """Test that random therapist shows approximately 12.5% complementarity.

        With 8 octants, random chance should give 1/8 = 12.5% complementarity.
        """
        tracker = ComplementarityTracker(window_size=100)

        np.random.seed(42)
        for _ in range(100):
            client_action = np.random.randint(0, 8)
            therapist_action = np.random.randint(0, 8)
            tracker.add_interaction(client_action, therapist_action)

        comp_rate = tracker.get_overall_rate()
        # Should be close to 12.5% (allow some variance)
        assert 5.0 < comp_rate < 20.0

    def test_warm_filtering(self):
        """Test that warm filtering only counts warm interactions."""
        tracker = ComplementarityTracker(window_size=10)

        # Add warm complementary interactions
        tracker.add_interaction(1, 3)  # WD → WS (complementary)
        tracker.add_interaction(2, 2)  # W → W (complementary)
        tracker.add_interaction(3, 1)  # WS → WD (complementary)

        # Add cold complementary interactions
        tracker.add_interaction(5, 7)  # CS → CD (complementary)
        tracker.add_interaction(6, 6)  # C → C (complementary)
        tracker.add_interaction(7, 5)  # CD → CS (complementary)

        # Overall should be 100%
        assert tracker.get_overall_rate() == 100.0

        # Warm should be 100%
        assert tracker.get_warm_rate() == 100.0

        # Cold should be 100%
        assert tracker.get_cold_rate() == 100.0

    def test_warm_filtering_with_no_warm_interactions(self):
        """Test that warm filtering returns NaN when no warm interactions."""
        tracker = ComplementarityTracker(window_size=10)

        # Add only cold interactions
        tracker.add_interaction(5, 7)  # CS → CD (complementary)
        tracker.add_interaction(6, 6)  # C → C (complementary)

        # Warm rate should be NaN
        assert np.isnan(tracker.get_warm_rate())

        # Cold rate should be 100%
        assert tracker.get_cold_rate() == 100.0

        # Overall should be 100%
        assert tracker.get_overall_rate() == 100.0

    def test_cold_filtering_with_no_cold_interactions(self):
        """Test that cold filtering returns NaN when no cold interactions."""
        tracker = ComplementarityTracker(window_size=10)

        # Add only warm interactions
        tracker.add_interaction(1, 3)  # WD → WS (complementary)
        tracker.add_interaction(2, 2)  # W → W (complementary)

        # Cold rate should be NaN
        assert np.isnan(tracker.get_cold_rate())

        # Warm rate should be 100%
        assert tracker.get_warm_rate() == 100.0

    def test_sliding_window(self):
        """Test that sliding window only considers recent interactions."""
        tracker = ComplementarityTracker(window_size=5)

        # Fill window with complementary interactions
        for i in range(5):
            client_action = i % 8
            therapist_action = COMPLEMENT_MAP[client_action]
            tracker.add_interaction(client_action, therapist_action)

        assert tracker.get_overall_rate() == 100.0

        # Add 5 non-complementary interactions
        for i in range(5):
            tracker.add_interaction(0, 2)  # Not complementary

        # Now window contains only non-complementary interactions
        assert tracker.get_overall_rate() == 0.0

    def test_dual_tracking_enacted_vs_perceived(self):
        """Test dual tracking for enacted vs perceived complementarity."""
        tracker = ComplementarityTracker(window_size=10)

        # Scenario: Therapist enacts complementary, but client perceives non-complementary
        for i in range(5):
            client_action = 0  # D
            enacted_action = 4  # S (complementary)
            perceived_action = 2  # W (not complementary)

            tracker.add_interaction(
                client_action,
                therapist_action=None,  # Not used in dual tracking
                enacted_action=enacted_action,
                perceived_action=perceived_action
            )

        # Enacted should be 100%
        assert tracker.get_overall_rate('enacted') == 100.0

        # Perceived should be 0%
        assert tracker.get_overall_rate('perceived') == 0.0

    def test_reset(self):
        """Test that reset clears all tracked interactions."""
        tracker = ComplementarityTracker(window_size=10)

        # Add some interactions
        for i in range(5):
            tracker.add_interaction(i % 8, COMPLEMENT_MAP[i % 8])

        assert tracker.get_overall_rate() == 100.0

        # Reset
        tracker.reset()

        # Should be empty
        assert tracker.get_overall_rate() == 0.0

    def test_get_all_rates(self):
        """Test get_all_rates returns tuple of (overall, warm, cold)."""
        tracker = ComplementarityTracker(window_size=10)

        # Add mix of warm and cold interactions
        tracker.add_interaction(1, 3)  # Warm complementary
        tracker.add_interaction(2, 2)  # Warm complementary
        tracker.add_interaction(5, 7)  # Cold complementary
        tracker.add_interaction(0, 2)  # Neutral non-complementary

        overall, warm, cold = tracker.get_all_rates()

        assert overall == 75.0  # 3 out of 4 complementary
        assert warm == 100.0  # 2 out of 2 warm complementary
        assert cold == 100.0  # 1 out of 1 cold complementary

    def test_window_size_validation(self):
        """Test that window_size must be at least 1."""
        with pytest.raises(ValueError):
            ComplementarityTracker(window_size=0)

        with pytest.raises(ValueError):
            ComplementarityTracker(window_size=-1)

    def test_empty_tracker(self):
        """Test behavior with no interactions added."""
        tracker = ComplementarityTracker(window_size=10)

        assert tracker.get_overall_rate() == 0.0
        assert np.isnan(tracker.get_warm_rate())
        assert np.isnan(tracker.get_cold_rate())
