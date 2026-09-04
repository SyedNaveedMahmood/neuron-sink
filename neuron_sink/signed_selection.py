"""Stage-C2 positive-signed neuron selection (amendment A005).

The completed Stage C remains ranked by ``mean_abs_attr``. Stage C2 is a separate
experiment that consumes the same discovery-only activation-times-gradient row schema but
orders neurons by descending, strictly positive ``mean_signed_attr``. For ``alpha < 1``,
positive ``a * dS/da`` predicts a first-order sink reduction under suppression.

Only target selection changes here. Fractions, layer-count-matched controls, RNG seeds,
and their integrity checks are inherited unchanged from :mod:`neuron_sink.selection`.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import canonical_sha256, read_json
from .selection import (
    CONTROL_TYPE_LAYER_RANDOM,
    CONTROL_TYPE_TARGETED,
    FrozenAttributionRanking,
    FrozenNeuronSets,
    REGISTERED_BASE_SEED,
    SelectionCondition,
    SelectionError,
    build_neuron_sets_document,
    exact_k,
    fraction_label,
    generate_layer_matched_controls,
    load_frozen_neuron_sets,
    per_layer_counts,
    verify_neuron_sets_document,
)
from .suppression import NeuronSet


SCHEMA_VERSION = "neuron_sets_signed_v1"
RANKING_SCORE = "mean_signed_attr"
SELECTION_METHOD = "global_top_k_positive_mean_signed_attr"
SIGN_REQUIREMENT = "strictly_positive"


def _signed_order(ranking: FrozenAttributionRanking) -> list[Mapping[str, Any]]:
    """Return the amendment-A005 total order with deterministic tie-breaking."""

    return sorted(
        ranking.rows,
        key=lambda row: (
            -float(row[RANKING_SCORE]), int(row["layer"]), int(row["neuron"])
        ),
    )


def positive_score_count(ranking: FrozenAttributionRanking) -> int:
    """Number of eligible discovery neurons with a strictly positive signed score."""

    return sum(float(row[RANKING_SCORE]) > 0.0 for row in ranking.rows)


def select_global_top_k_positive_signed(
    ranking: FrozenAttributionRanking, k: int
) -> NeuronSet:
    """Select the top ``k`` positive signed scores from discovery only."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 1 or k > ranking.pool_size:
        raise ValueError(f"k must be in [1, {ranking.pool_size}], got {k}")
    ordered = _signed_order(ranking)
    selected = ordered[:k]
    if len(selected) != k or float(selected[-1][RANKING_SCORE]) <= 0.0:
        raise SelectionError(
            f"Stage C2 requires {k} strictly positive mean_signed_attr scores, but "
            f"the eligible pool contains only {positive_score_count(ranking)}"
        )

    by_layer: dict[int, list[int]] = {}
    for row in selected:
        by_layer.setdefault(int(row["layer"]), []).append(int(row["neuron"]))
    neuron_set = NeuronSet(
        {
            layer: tuple(sorted(neurons))
            for layer, neurons in sorted(by_layer.items())
        },
        source=CONTROL_TYPE_TARGETED,
        selection_seed=None,
    )
    if sum(len(neurons) for neurons in neuron_set.by_layer.values()) != k:
        raise SelectionError("Stage-C2 signed target does not contain exactly k neurons")
    return neuron_set


def build_signed_selection_conditions(
    ranking: FrozenAttributionRanking,
    fractions_percent: Sequence[float | Decimal],
    *,
    control_draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
) -> tuple[SelectionCondition, ...]:
    """Build signed targets and the unchanged deterministic matched controls."""

    maximum_k = max(exact_k(value, ranking.pool_size) for value in fractions_percent)
    if positive_score_count(ranking) < maximum_k:
        raise SelectionError(
            "Stage C2 stopped before validation: discovery has fewer positive signed "
            f"neurons than maximum registered k={maximum_k}"
        )

    conditions: list[SelectionCondition] = []
    seen_labels: set[str] = set()
    for value in fractions_percent:
        label = fraction_label(value)
        if label in seen_labels:
            raise ValueError(f"Duplicate normalized fraction label {label}")
        seen_labels.add(label)
        fraction = float(Decimal(str(value)))
        k = exact_k(value, ranking.pool_size)
        target = select_global_top_k_positive_signed(ranking, k)
        counts = per_layer_counts(target, ranking.eligible_mlp_layers)
        conditions.append(SelectionCondition(
            condition_id=f"targeted_{label}",
            control_type=CONTROL_TYPE_TARGETED,
            fraction_percent=fraction,
            k=k,
            control_seed=None,
            neuron_set=target,
            per_layer_counts=counts,
        ))
        controls = generate_layer_matched_controls(
            target,
            eligible_layers=ranking.eligible_mlp_layers,
            widths=ranking.mlp_width,
            k=k,
            draws=control_draws,
            base_seed=base_seed,
        )
        for draw_index, control in enumerate(controls):
            conditions.append(SelectionCondition(
                condition_id=f"layer_random_{label}_s{draw_index}",
                control_type=CONTROL_TYPE_LAYER_RANDOM,
                fraction_percent=fraction,
                k=k,
                control_seed=draw_index,
                neuron_set=control,
                per_layer_counts=per_layer_counts(
                    control, ranking.eligible_mlp_layers
                ),
            ))
    return tuple(conditions)


def build_signed_neuron_sets_document(
    ranking: FrozenAttributionRanking,
    conditions: Sequence[SelectionCondition],
    *,
    fractions_percent: Sequence[float | Decimal],
    control_draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
    experiment_id: str,
) -> dict[str, Any]:
    """Create the hash-bearing Stage-C2 discovery selection document."""

    document = build_neuron_sets_document(
        ranking,
        conditions,
        fractions_percent=fractions_percent,
        control_draws=control_draws,
        base_seed=base_seed,
        experiment_id=experiment_id,
        schema=SCHEMA_VERSION,
        selection_method=SELECTION_METHOD,
        ranking_score=RANKING_SCORE,
    )
    document["sign_requirement"] = SIGN_REQUIREMENT
    document["positive_score_count"] = positive_score_count(ranking)
    document["first_order_direction"] = (
        "delta_S_future ~= -(1-alpha) * mean_signed_attr; positive scores predict "
        "sink reduction"
    )
    document["amendment"] = "A005"
    document["neuron_sets_sha256"] = canonical_sha256({
        key: value for key, value in document.items()
        if key != "neuron_sets_sha256"
    })
    return document


def verify_signed_neuron_sets_document(
    document: Mapping[str, Any],
) -> FrozenNeuronSets:
    """Verify Stage-C2 schema and the unchanged random-control construction."""

    frozen = verify_neuron_sets_document(
        document,
        expected_schema=SCHEMA_VERSION,
        expected_selection_method=SELECTION_METHOD,
        expected_ranking_score=RANKING_SCORE,
    )
    if document.get("sign_requirement") != SIGN_REQUIREMENT:
        raise SelectionError("Stage-C2 neuron sets lack the positive-sign requirement")
    if int(document.get("positive_score_count", -1)) < 1:
        raise SelectionError("Stage-C2 positive_score_count must be positive")
    if document.get("amendment") != "A005":
        raise SelectionError("Stage-C2 neuron sets do not cite amendment A005")
    return frozen


def verify_signed_targets(
    frozen: FrozenNeuronSets, ranking: FrozenAttributionRanking
) -> None:
    """Recompute every signed target from attribution and compare exact neuron ids."""

    observed_positive = positive_score_count(ranking)
    if int(frozen.document.get("positive_score_count", -1)) != observed_positive:
        raise SelectionError("Saved Stage-C2 positive-score count does not reproduce")
    for condition_id, record in frozen.document["conditions"].items():
        if record["control_type"] != CONTROL_TYPE_TARGETED:
            continue
        expected = select_global_top_k_positive_signed(ranking, int(record["k"]))
        if frozen.neuron_sets[condition_id] != expected:
            raise SelectionError(
                f"Stage-C2 target {condition_id} does not reproduce from signed ranking"
            )


def load_signed_neuron_sets(
    path: Path | str,
    *,
    ranking: FrozenAttributionRanking | None = None,
) -> FrozenNeuronSets:
    """Load a Stage-C2 neuron-set file and optionally reverify target selection."""

    path = Path(path)
    if not path.is_file():
        raise SelectionError(f"Frozen Stage-C2 neuron sets not found: {path}")
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise SelectionError(f"{path} does not contain a JSON object")
    frozen = load_frozen_neuron_sets(
        path,
        expected_schema=SCHEMA_VERSION,
        expected_selection_method=SELECTION_METHOD,
        expected_ranking_score=RANKING_SCORE,
    )
    if document.get("sign_requirement") != SIGN_REQUIREMENT:
        raise SelectionError("Stage-C2 neuron sets lack the positive-sign requirement")
    if document.get("amendment") != "A005":
        raise SelectionError("Stage-C2 neuron sets do not cite amendment A005")
    if ranking is not None:
        verify_signed_targets(frozen, ranking)
    return frozen


__all__ = [
    "RANKING_SCORE",
    "SCHEMA_VERSION",
    "SELECTION_METHOD",
    "SIGN_REQUIREMENT",
    "build_signed_neuron_sets_document",
    "build_signed_selection_conditions",
    "load_signed_neuron_sets",
    "positive_score_count",
    "select_global_top_k_positive_signed",
    "verify_signed_neuron_sets_document",
    "verify_signed_targets",
]
