"""Registered paired statistics for the Stage-B phenomenon gate.

All resampling uses an explicit :class:`numpy.random.Generator`; module-global random
state is never consulted.  Sink reduction is recomputed from the sampled paired
per-example baseline/intervention values on every bootstrap draw, matching the
aggregate definition in ``docs/05_METRICS_AND_SCHEMAS.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


class StatisticsError(ValueError):
    """Raised when paired statistical inputs are malformed."""


@dataclass(frozen=True)
class BootstrapInterval:
    """A deterministic percentile-bootstrap interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
        }


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise StatisticsError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise StatisticsError(f"{name} contains non-finite values")
    return array


def _resampling_contract(
    *, n_resamples: int, seed: int, confidence: float
) -> tuple[int, int, float]:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 1:
        raise StatisticsError("n_resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise StatisticsError("seed must be non-negative")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise StatisticsError("confidence must be finite and in (0, 1)")
    return n_resamples, seed, confidence


def relative_sink_reduction(
    baseline_sink: Sequence[float], intervened_sink: Sequence[float]
) -> float:
    """RSR from paired per-example values, using the registered aggregate ratio."""

    baseline = _finite_vector(baseline_sink, "baseline_sink")
    intervened = _finite_vector(intervened_sink, "intervened_sink")
    if baseline.shape != intervened.shape:
        raise StatisticsError(
            f"baseline/intervention shapes differ: {baseline.shape} != {intervened.shape}"
        )
    baseline_mean = float(baseline.mean())
    return float((baseline_mean - intervened.mean()) / max(baseline_mean, 1e-12))


def paired_bootstrap_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Percentile CI for ``mean(left - right)`` over paired examples."""

    n_resamples, seed, confidence = _resampling_contract(
        n_resamples=n_resamples, seed=seed, confidence=confidence
    )
    left_array = _finite_vector(left, "left")
    right_array = _finite_vector(right, "right")
    if left_array.shape != right_array.shape:
        raise StatisticsError(
            f"paired shapes differ: {left_array.shape} != {right_array.shape}"
        )
    difference = left_array - right_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(n_resamples, difference.size))
    samples = difference[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (tail, 1.0 - tail), method="linear")
    return BootstrapInterval(
        estimate=float(difference.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def target_minus_median_random_rsr(
    baseline_sink: Sequence[float],
    target_sink: Sequence[float],
    random_sinks: Sequence[Sequence[float]],
) -> float:
    """Target RSR minus the median aggregate RSR of matched random draws."""

    baseline = _finite_vector(baseline_sink, "baseline_sink")
    target = _finite_vector(target_sink, "target_sink")
    random_array = np.asarray(random_sinks, dtype=np.float64)
    if target.shape != baseline.shape:
        raise StatisticsError("target_sink must be paired with baseline_sink")
    if random_array.ndim != 2 or random_array.shape[1:] != baseline.shape:
        raise StatisticsError(
            "random_sinks must have shape [draws, examples] paired with baseline_sink"
        )
    if random_array.shape[0] < 1 or not np.isfinite(random_array).all():
        raise StatisticsError("random_sinks must contain finite values for at least one draw")
    target_rsr = relative_sink_reduction(baseline, target)
    random_rsr = np.asarray(
        [relative_sink_reduction(baseline, values) for values in random_array],
        dtype=np.float64,
    )
    return float(target_rsr - np.median(random_rsr))


def paired_bootstrap_target_minus_median_random_rsr(
    baseline_sink: Sequence[float],
    target_sink: Sequence[float],
    random_sinks: Sequence[Sequence[float]],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    batch_size: int = 256,
) -> BootstrapInterval:
    """Paired CI for target RSR minus median matched-random RSR.

    Each resample uses the same example indices for the target and every random draw.
    The median is taken across the resampled aggregate RSR values of the random sets.
    Batching bounds temporary memory without changing the registered statistic.
    """

    n_resamples, seed, confidence = _resampling_contract(
        n_resamples=n_resamples, seed=seed, confidence=confidence
    )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise StatisticsError("batch_size must be a positive integer")
    baseline = _finite_vector(baseline_sink, "baseline_sink")
    target = _finite_vector(target_sink, "target_sink")
    random_array = np.asarray(random_sinks, dtype=np.float64)
    if target.shape != baseline.shape:
        raise StatisticsError("target_sink must be paired with baseline_sink")
    if random_array.ndim != 2 or random_array.shape[1:] != baseline.shape:
        raise StatisticsError(
            "random_sinks must have shape [draws, examples] paired with baseline_sink"
        )
    if random_array.shape[0] < 1 or not np.isfinite(random_array).all():
        raise StatisticsError("random_sinks must contain finite values for at least one draw")

    rng = np.random.default_rng(seed)
    sampled = np.empty(n_resamples, dtype=np.float64)
    offset = 0
    while offset < n_resamples:
        count = min(batch_size, n_resamples - offset)
        indices = rng.integers(0, baseline.size, size=(count, baseline.size))
        baseline_mean = baseline[indices].mean(axis=1)
        denominator = np.maximum(baseline_mean, 1e-12)
        target_rsr = (baseline_mean - target[indices].mean(axis=1)) / denominator
        # [draws, batch, examples] -> aggregate RSR per draw and resample.
        random_means = random_array[:, indices].mean(axis=2)
        random_rsr = (baseline_mean[None, :] - random_means) / denominator[None, :]
        sampled[offset:offset + count] = target_rsr - np.median(
            random_rsr, axis=0
        )
        offset += count

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sampled, (tail, 1.0 - tail), method="linear")
    return BootstrapInterval(
        estimate=target_minus_median_random_rsr(baseline, target, random_array),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def random_control_percentile(
    target_value: float,
    random_values: Sequence[float],
    *,
    percentile: float = 95.0,
) -> dict[str, float | bool]:
    """Compare one target with the registered percentile of random controls."""

    target = float(target_value)
    random_array = _finite_vector(random_values, "random_values")
    percentile = float(percentile)
    if not math.isfinite(target):
        raise StatisticsError("target_value must be finite")
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise StatisticsError("percentile must be in [0, 100]")
    threshold = float(np.percentile(random_array, percentile, method="linear"))
    rank = float(100.0 * np.mean(random_array < target))
    return {
        "target": target,
        "percentile": percentile,
        "random_percentile_value": threshold,
        "target_percentile_rank": rank,
        "target_exceeds_percentile": bool(target > threshold),
    }


def spearman_dose_response(
    suppression_dose: Sequence[float], relative_sink_reductions: Sequence[float]
) -> float:
    """Spearman correlation between ``1-alpha`` and targeted RSR."""

    dose = _finite_vector(suppression_dose, "suppression_dose")
    rsr = _finite_vector(relative_sink_reductions, "relative_sink_reductions")
    if dose.shape != rsr.shape or dose.size < 2:
        raise StatisticsError("dose and RSR must be paired sequences with at least 2 values")
    if np.all(dose == dose[0]) or np.all(rsr == rsr[0]):
        return float("nan")
    value = float(spearmanr(dose, rsr).statistic)
    # A constant dose or effect is not evidence of a directional dose response.
    return value if math.isfinite(value) else float("nan")
