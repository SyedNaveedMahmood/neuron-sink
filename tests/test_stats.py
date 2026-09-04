from __future__ import annotations

import math
import unittest

import numpy as np

from neuron_sink.stats import (
    StatisticsError,
    paired_bootstrap_mean_difference,
    paired_bootstrap_target_minus_median_random_rsr,
    random_control_percentile,
    relative_sink_reduction,
    spearman_dose_response,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_mean_difference_is_deterministic_with_explicit_seed(self) -> None:
        left = [0.2, 0.4, 0.6, 0.8]
        right = [0.1, 0.2, 0.3, 0.4]
        first = paired_bootstrap_mean_difference(
            left, right, n_resamples=500, seed=17
        )
        second = paired_bootstrap_mean_difference(
            left, right, n_resamples=500, seed=17
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, 0.25)
        self.assertLessEqual(first.lower, first.estimate)
        self.assertGreaterEqual(first.upper, first.estimate)

    def test_target_minus_random_bootstrap_is_paired_and_deterministic(self) -> None:
        baseline = np.asarray([0.8, 0.7, 0.9, 0.6])
        target = baseline * 0.8
        controls = np.stack([baseline * 0.98, baseline * 0.97, baseline * 0.99])
        first = paired_bootstrap_target_minus_median_random_rsr(
            baseline, target, controls, n_resamples=600, seed=23, batch_size=31
        )
        second = paired_bootstrap_target_minus_median_random_rsr(
            baseline, target, controls, n_resamples=600, seed=23, batch_size=31
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, 0.18)
        self.assertGreater(first.lower, 0.0)

    def test_bad_pairing_is_rejected(self) -> None:
        with self.assertRaises(StatisticsError):
            relative_sink_reduction([0.5, 0.6], [0.4])
        with self.assertRaises(StatisticsError):
            paired_bootstrap_target_minus_median_random_rsr(
                [0.5, 0.6], [0.4, 0.5], [[0.4]], n_resamples=10
            )


class ControlAndDoseTests(unittest.TestCase):
    def test_random_percentile_uses_strict_superiority(self) -> None:
        result = random_control_percentile(0.20, [0.01, 0.02, 0.03, 0.04])
        self.assertTrue(result["target_exceeds_percentile"])
        self.assertEqual(result["target_percentile_rank"], 100.0)
        tied = random_control_percentile(0.04, [0.01, 0.02, 0.03, 0.04], percentile=100)
        self.assertFalse(tied["target_exceeds_percentile"])

    def test_spearman_registered_dose_direction(self) -> None:
        self.assertAlmostEqual(
            spearman_dose_response([0.0, 0.25, 0.5, 0.75, 1.0], [0, 1, 2, 3, 4]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_dose_response([0.0, 0.25, 0.5, 0.75, 1.0], [4, 3, 2, 1, 0]),
            -1.0,
        )
        self.assertTrue(math.isnan(spearman_dose_response([0, 1], [1, 1])))


if __name__ == "__main__":
    unittest.main()
