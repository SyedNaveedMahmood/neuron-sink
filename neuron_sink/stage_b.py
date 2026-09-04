"""Stage-B experiment grid, validation boundary, and formal phenomenon gate.

This module contains no model loading and never opens a corpus.  It validates frozen
discovery artefacts, selects the registered operating point from validation rows only,
unlocks the test split only after that selection artefact verifies, and evaluates the
formal held-out gate from test rows only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import PHENOMENON_ROW_FIELDS, EvaluationError, validate_phenomenon_row
from .provenance import canonical_sha256, read_json, write_json
from .selection import (
    CONTROL_TYPE_LAYER_RANDOM,
    CONTROL_TYPE_TARGETED,
    FULL_CONTROL_DRAWS,
    FULL_FRACTIONS_PERCENT,
    FrozenNeuronSets,
    exact_k,
    fraction_label,
)
from .stats import (
    paired_bootstrap_target_minus_median_random_rsr,
    random_control_percentile,
    relative_sink_reduction,
    spearman_dose_response,
)
from .suppression import NeuronSet


FULL_ALPHAS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
FULL_SPLITS: tuple[str, ...] = ("discovery", "validation", "test")
FULL_EXAMPLES_PER_SPLIT = 100
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

OPERATING_POINT_SCHEMA = "stage_b_operating_point_v1"
FORMAL_GATE_SCHEMA = "stage_b_formal_gate_v1"
EXPERIMENT_ID = "stage_b_full_phenomenon_v1"

MIN_RSR = 0.10
MIN_TARGET_MINUS_MEDIAN_RANDOM_RSR = 0.10
MAX_DELTA_CE = 0.10
RANDOM_PERCENTILE = 95.0
MIN_SPEARMAN = 0.8


class StageBError(RuntimeError):
    """Raised when a Stage-B grid, artefact, or stage transition is invalid."""


@dataclass(frozen=True)
class FullCondition:
    """One verified targeted or matched-random full-grid condition."""

    condition_id: str
    condition_order: int
    control_type: str
    control_seed: int | None
    fraction_percent: float
    k: int
    neuron_set: NeuronSet


def stage_b_run_root(
    repository_root: Path | str,
    model_alias: str,
    *,
    registered_run: bool,
    stamp: str,
) -> Path:
    """Return a model-separated Stage-B path outside tracked smoke artefacts."""

    if model_alias not in ("gpt2-small", "gpt2-medium"):
        raise StageBError(f"Unregistered Stage-B model alias {model_alias!r}")
    if not stamp.startswith("run_") or not stamp.endswith("Z"):
        raise StageBError(f"Run stamp is not the registered UTC form: {stamp!r}")
    category = "stage_b_full" if registered_run else "stage_b_preflight"
    return Path(repository_root).resolve() / "results" / category / model_alias / stamp


def registered_full_conditions(frozen: FrozenNeuronSets) -> tuple[FullCondition, ...]:
    """Validate and return the exact 6-target/120-control Stage-B grid."""

    document = frozen.document
    fractions = tuple(float(value) for value in document.get("fractions_percent", ()))
    if fractions != FULL_FRACTIONS_PERCENT:
        raise StageBError(
            f"Frozen fractions {fractions} do not match Stage-B grid "
            f"{FULL_FRACTIONS_PERCENT}"
        )
    if int(document.get("control_draws", -1)) != FULL_CONTROL_DRAWS:
        raise StageBError(
            f"Frozen control_draws={document.get('control_draws')} does not match "
            f"Stage-B count {FULL_CONTROL_DRAWS}"
        )
    raw_records = document.get("conditions")
    condition_ids = tuple(str(value) for value in document.get("condition_ids", ()))
    if not isinstance(raw_records, Mapping):
        raise StageBError("Frozen neuron-set document has no conditions mapping")

    expected: list[str] = []
    for fraction in FULL_FRACTIONS_PERCENT:
        label = fraction_label(fraction)
        expected.append(f"targeted_{label}")
        expected.extend(
            f"layer_random_{label}_s{seed}" for seed in range(FULL_CONTROL_DRAWS)
        )
    if condition_ids != tuple(expected):
        raise StageBError("Frozen Stage-B condition order is not the registered order")

    conditions: list[FullCondition] = []
    for order, condition_id in enumerate(condition_ids, start=1):
        record = raw_records[condition_id]
        control_type = str(record["control_type"])
        control_seed = record.get("control_seed")
        conditions.append(FullCondition(
            condition_id=condition_id,
            condition_order=order,
            control_type=control_type,
            control_seed=None if control_seed is None else int(control_seed),
            fraction_percent=float(record["fraction_percent"]),
            k=int(record["k"]),
            neuron_set=frozen.neuron_sets[condition_id],
        ))

    for fraction in FULL_FRACTIONS_PERCENT:
        same = [item for item in conditions if item.fraction_percent == fraction]
        targets = [item for item in same if item.control_type == CONTROL_TYPE_TARGETED]
        controls = [
            item for item in same if item.control_type == CONTROL_TYPE_LAYER_RANDOM
        ]
        if len(targets) != 1 or len(controls) != FULL_CONTROL_DRAWS:
            raise StageBError(
                f"Fraction {fraction}% needs one target and {FULL_CONTROL_DRAWS} controls"
            )
        if {item.control_seed for item in controls} != set(range(FULL_CONTROL_DRAWS)):
            raise StageBError(f"Fraction {fraction}% has incomplete control draw indices")
    return tuple(conditions)


def phenomenon_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Stable hash of ordered validated per-example rows."""

    normalized: list[list[Any]] = []
    for row in rows:
        validate_phenomenon_row(
            row, allowed_stages=FULL_SPLITS, allowed_alphas=FULL_ALPHAS
        )
        normalized.append([row[field] for field in PHENOMENON_ROW_FIELDS])
    return canonical_sha256(normalized)


def _stage_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    conditions: Sequence[FullCondition],
    expected_examples: int,
) -> tuple[tuple[str, ...], dict[tuple[str, float], list[Mapping[str, Any]]]]:
    if stage not in FULL_SPLITS:
        raise StageBError(f"Unknown Stage-B split {stage!r}")
    if not rows:
        raise StageBError(f"No {stage} rows were supplied")
    if any(str(row.get("stage")) != stage for row in rows):
        raise StageBError(f"The {stage} statistic received rows from another split")

    expected_condition_ids = {condition.condition_id for condition in conditions}
    indexed: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        validate_phenomenon_row(
            row, allowed_stages=FULL_SPLITS, allowed_alphas=FULL_ALPHAS
        )
        if row["control_type"] == "baseline":
            continue
        condition_id = str(row["condition_id"])
        if condition_id not in expected_condition_ids:
            raise StageBError(f"Unexpected condition {condition_id!r} in {stage} rows")
        key = (condition_id, float(row["alpha"]))
        indexed.setdefault(key, []).append(row)

    expected_keys = {
        (condition.condition_id, alpha)
        for condition in conditions
        for alpha in FULL_ALPHAS
    }
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise StageBError(f"Incomplete {stage} grid; missing={missing[:3]}, extra={extra[:3]}")

    reference_ids: tuple[str, ...] | None = None
    reference_baseline: dict[str, tuple[float, float]] = {}
    for key in sorted(indexed):
        members = indexed[key]
        ids = tuple(str(row["example_id"]) for row in members)
        if len(ids) != expected_examples or len(set(ids)) != expected_examples:
            raise StageBError(
                f"Cell {key} has {len(ids)} rows; expected {expected_examples} unique examples"
            )
        if reference_ids is None:
            reference_ids = ids
            reference_baseline = {
                str(row["example_id"]): (
                    float(row["sink_baseline"]), float(row["ce_baseline"])
                )
                for row in members
            }
        if ids != reference_ids:
            raise StageBError(f"Cell {key} does not preserve paired example order")
        for row in members:
            expected = reference_baseline[str(row["example_id"])]
            observed = (float(row["sink_baseline"]), float(row["ce_baseline"]))
            if observed != expected:
                raise StageBError(f"Cell {key} changed its paired baseline values")
    assert reference_ids is not None
    return reference_ids, indexed


def _vectors(
    indexed: Mapping[tuple[str, float], Sequence[Mapping[str, Any]]],
    condition_id: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    members = indexed[(condition_id, float(alpha))]
    return (
        np.asarray([float(row["sink_baseline"]) for row in members]),
        np.asarray([float(row["sink_intervened"]) for row in members]),
        np.asarray([float(row["ce_baseline"]) for row in members]),
        np.asarray([float(row["ce_intervened"]) for row in members]),
    )


def _fraction_statistics(
    indexed: Mapping[tuple[str, float], Sequence[Mapping[str, Any]]],
    *,
    fraction: float,
    alpha: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    label = fraction_label(fraction)
    target_id = f"targeted_{label}"
    baseline, target, ce_baseline, ce_target = _vectors(indexed, target_id, alpha)
    random_sinks = []
    random_rsrs = []
    for seed in range(FULL_CONTROL_DRAWS):
        random_id = f"layer_random_{label}_s{seed}"
        random_baseline, random_sink, _ce_b, _ce_i = _vectors(
            indexed, random_id, alpha
        )
        if not np.array_equal(random_baseline, baseline):
            raise StageBError(f"Random draw {random_id} does not share the paired baseline")
        random_sinks.append(random_sink)
        random_rsrs.append(relative_sink_reduction(baseline, random_sink))
    target_rsr = relative_sink_reduction(baseline, target)
    control_comparison = random_control_percentile(
        target_rsr, random_rsrs, percentile=RANDOM_PERCENTILE
    )
    interval = paired_bootstrap_target_minus_median_random_rsr(
        baseline,
        target,
        random_sinks,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "fraction_percent": float(fraction),
        "k": int(indexed[(target_id, float(alpha))][0]["k"]),
        "alpha": float(alpha),
        "target_condition_id": target_id,
        "target_relative_sink_reduction": target_rsr,
        "random_relative_sink_reductions": random_rsrs,
        "median_random_relative_sink_reduction": float(np.median(random_rsrs)),
        "target_minus_median_random_relative_sink_reduction": interval.estimate,
        "target_minus_median_random_bootstrap": interval.to_dict(),
        "target_delta_ce": float(ce_target.mean() - ce_baseline.mean()),
        "random_control_comparison": control_comparison,
    }


def build_operating_point_document(
    validation_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[FullCondition],
    *,
    model_id: str,
    model_revision: str,
    corpus_manifest_sha256: str,
    sink_scope_sha256: str,
    attribution_sha256: str,
    neuron_sets_sha256: str,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_examples: int = FULL_EXAMPLES_PER_SPLIT,
) -> dict[str, Any]:
    """Select ``k*`` from validation only, or freeze ``k_max_effect`` as exploratory."""

    _ids, indexed = _stage_index(
        validation_rows,
        stage="validation",
        conditions=conditions,
        expected_examples=expected_examples,
    )
    summaries = [
        _fraction_statistics(
            indexed,
            fraction=fraction,
            alpha=0.0,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for fraction in FULL_FRACTIONS_PERCENT
    ]
    for summary in summaries:
        summary["criteria"] = {
            "target_rsr_gte_0p10": bool(
                summary["target_relative_sink_reduction"] >= MIN_RSR
            ),
            "target_minus_median_random_gte_0p10": bool(
                summary["target_minus_median_random_relative_sink_reduction"]
                >= MIN_TARGET_MINUS_MEDIAN_RANDOM_RSR
            ),
            "bootstrap_ci_lower_gt_0": bool(
                summary["target_minus_median_random_bootstrap"]["lower"] > 0.0
            ),
            "delta_ce_lte_0p10": bool(summary["target_delta_ce"] <= MAX_DELTA_CE),
        }
        summary["qualifies_k_star"] = bool(all(summary["criteria"].values()))

    qualifying = [summary for summary in summaries if summary["qualifies_k_star"]]
    if qualifying:
        selected = qualifying[0]
        point_type = "k_star"
        exploratory_only = False
    else:
        selected = max(
            summaries,
            key=lambda item: (
                float(item["target_relative_sink_reduction"]),
                -float(item["fraction_percent"]),
            ),
        )
        point_type = "k_max_effect"
        exploratory_only = True

    document: dict[str, Any] = {
        "schema": OPERATING_POINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "stage": "validation",
        "registered_run": expected_examples == FULL_EXAMPLES_PER_SPLIT,
        "model_id": model_id,
        "model_revision": model_revision,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "sink_scope_sha256": sink_scope_sha256,
        "attribution_sha256": attribution_sha256,
        "neuron_sets_sha256": neuron_sets_sha256,
        "validation_rows_sha256": phenomenon_rows_sha256(validation_rows),
        "n_examples": expected_examples,
        "fractions_percent": list(FULL_FRACTIONS_PERCENT),
        "alpha": 0.0,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "selection_rule": (
            "smallest registered fraction with target RSR >= 0.10, target-minus-"
            "median-random RSR >= 0.10, paired-bootstrap 95% CI lower > 0, and "
            "delta CE <= 0.10 nats/token; otherwise k_max_effect exploratory only"
        ),
        "operating_point_type": point_type,
        "selected_fraction_percent": selected["fraction_percent"],
        "selected_k": selected["k"],
        "selected_condition_id": selected["target_condition_id"],
        "exploratory_only": exploratory_only,
        "validation_summaries": summaries,
    }
    document["operating_point_sha256"] = canonical_sha256(document)
    return document


def verify_operating_point_document(
    document: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    corpus_manifest_sha256: str,
    sink_scope_sha256: str,
    attribution_sha256: str,
    neuron_sets_sha256: str,
) -> Mapping[str, Any]:
    """Re-establish every lock required before opening the test split."""

    if document.get("schema") != OPERATING_POINT_SCHEMA:
        raise StageBError("Missing or unsupported frozen validation operating-point schema")
    stored = str(document.get("operating_point_sha256", ""))
    recomputed = canonical_sha256(
        {key: value for key, value in document.items() if key != "operating_point_sha256"}
    )
    if stored != recomputed:
        raise StageBError("Frozen validation operating-point hash does not reproduce")
    expected = {
        "stage": "validation",
        "registered_run": True,
        "model_id": model_id,
        "model_revision": model_revision,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "sink_scope_sha256": sink_scope_sha256,
        "attribution_sha256": attribution_sha256,
        "neuron_sets_sha256": neuron_sets_sha256,
        "n_examples": FULL_EXAMPLES_PER_SPLIT,
        "fractions_percent": list(FULL_FRACTIONS_PERCENT),
        "alpha": 0.0,
    }
    mismatches = {
        key: (document.get(key), value)
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        raise StageBError(f"Operating-point lock mismatch: {mismatches}")
    point_type = document.get("operating_point_type")
    if point_type not in ("k_star", "k_max_effect"):
        raise StageBError(f"Unknown operating_point_type {point_type!r}")
    selected_fraction = float(document.get("selected_fraction_percent"))
    if selected_fraction not in FULL_FRACTIONS_PERCENT:
        raise StageBError("Operating point uses an unregistered neuron fraction")
    summaries = document.get("validation_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(FULL_FRACTIONS_PERCENT):
        raise StageBError("Operating point does not retain all validation summaries")
    selected_summary = next(
        (item for item in summaries if float(item["fraction_percent"]) == selected_fraction),
        None,
    )
    if selected_summary is None or int(document.get("selected_k")) != int(
        selected_summary["k"]
    ):
        raise StageBError("Operating-point selection does not match its validation summary")
    return document


def freeze_operating_point(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Write a validation decision once; existing paths are never overwritten."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Operating-point artefact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(document))
    return path


def unlock_test_split(
    path: Path | str,
    **expected: str,
) -> Mapping[str, Any]:
    """Load and verify the frozen validation artefact required before test access."""

    path = Path(path)
    if not path.is_file():
        raise StageBError(
            "Locked test split refused: a frozen validation operating-point artefact "
            "does not exist"
        )
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise StageBError("Operating-point artefact is not a JSON object")
    return verify_operating_point_document(document, **expected)


def evaluate_formal_gate(
    test_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[FullCondition],
    *,
    all_identity_pass: bool,
    all_validity_pass: bool,
    state_leakage_pass: bool,
    registered_run: bool,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_examples: int = FULL_EXAMPLES_PER_SPLIT,
) -> dict[str, Any]:
    """Evaluate the exact full causal gate on locked test rows only."""

    if not registered_run:
        return {
            "schema": FORMAL_GATE_SCHEMA,
            "status": "NOT_EVALUATED_DRY_RUN",
            "registered_run": False,
            "test_split_accessed": False,
        }
    _ids, indexed = _stage_index(
        test_rows,
        stage="test",
        conditions=conditions,
        expected_examples=expected_examples,
    )
    model_summaries: list[dict[str, Any]] = []
    for fraction in FULL_FRACTIONS_PERCENT:
        summary = _fraction_statistics(
            indexed,
            fraction=fraction,
            alpha=0.0,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        target_id = str(summary["target_condition_id"])
        rsr_by_alpha = {
            alpha: relative_sink_reduction(*_vectors(indexed, target_id, alpha)[:2])
            for alpha in FULL_ALPHAS
        }
        rho = spearman_dose_response(
            [1.0 - alpha for alpha in FULL_ALPHAS],
            [rsr_by_alpha[alpha] for alpha in FULL_ALPHAS],
        )
        criteria = {
            "test_rsr_gte_0p10": bool(
                summary["target_relative_sink_reduction"] >= MIN_RSR
            ),
            "target_exceeds_random_95th_percentile": bool(
                summary["random_control_comparison"]["target_exceeds_percentile"]
            ),
            "bootstrap_ci_lower_gt_0": bool(
                summary["target_minus_median_random_bootstrap"]["lower"] > 0.0
            ),
            "spearman_gte_0p8": bool(math.isfinite(rho) and rho >= MIN_SPEARMAN),
            "all_outputs_finite_nondegenerate": bool(all_validity_pass),
        }
        summary["rsr_by_alpha"] = {str(alpha): value for alpha, value in rsr_by_alpha.items()}
        summary["spearman_suppression_dose_vs_rsr"] = rho
        summary["criteria"] = criteria
        summary["passes_formal_gate"] = bool(all(criteria.values()))
        model_summaries.append(summary)

    gate_pass = bool(
        any(item["passes_formal_gate"] for item in model_summaries)
        and all_identity_pass
        and all_validity_pass
        and state_leakage_pass
    )
    document: dict[str, Any] = {
        "schema": FORMAL_GATE_SCHEMA,
        "status": "PASS" if gate_pass else "NULL_OR_INVALID",
        "registered_run": True,
        "test_split_accessed": True,
        "test_rows_sha256": phenomenon_rows_sha256(test_rows),
        "n_examples": expected_examples,
        "random_control_draws": FULL_CONTROL_DRAWS,
        "random_percentile": RANDOM_PERCENTILE,
        "minimum_spearman": MIN_SPEARMAN,
        "all_identity_pass": bool(all_identity_pass),
        "all_validity_pass": bool(all_validity_pass),
        "state_leakage_pass": bool(state_leakage_pass),
        "formal_gate_pass": gate_pass,
        "fraction_summaries": model_summaries,
        "passing_fractions": [
            item["fraction_percent"]
            for item in model_summaries
            if item["passes_formal_gate"]
        ],
    }
    document["formal_gate_sha256"] = canonical_sha256(document)
    return document
