"""Global top-k neuron selection and layer-count-matched random controls.

Task 6 consumes the frozen Task-5 attribution table and performs bookkeeping only:
no corpus example, model forward, suppression run, or downstream benchmark is involved.
The registered rules are in ``docs/00_MASTER_EXPERIMENT_DESIGN.md`` ("Neuron-set
sizes" and "Matched controls"):

* derive ``k`` from a model-relative fraction of every eligible MLP neuron;
* select the global top-k by ``mean_abs_attr`` (never the signed diagnostic);
* preserve the target's exact per-layer counts in every random control; and
* sample each control from non-target neurons of that same layer.

Attribution remains a ranking heuristic.  This module creates intervention *candidates*;
causal evidence requires Task 7's held-out targeted-vs-random suppression comparison.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .attribution import (
    FUTURE_LAYER_SEPARATOR,
    RANKING_SCORE,
    ROW_FIELDS,
    attribution_sha256,
)
from .provenance import canonical_sha256, read_json
from .sink_metrics import FrozenSinkScope
from .suppression import NeuronSet


SCHEMA_VERSION = "neuron_sets_v1"
SELECTION_METHOD = "global_top_k_mean_abs_attr"
CONTROL_TYPE_TARGETED = "targeted"
CONTROL_TYPE_LAYER_RANDOM = "layer_random"

# Registered in configs/experiment_plan.yaml.  The API accepts either grid; Task 6 freezes
# only SMOKE_FRACTIONS_PERCENT and five controls per target.
FULL_FRACTIONS_PERCENT: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
SMOKE_FRACTIONS_PERCENT: tuple[float, ...] = (0.05, 0.10, 0.25)
SMOKE_CONTROL_DRAWS = 5
FULL_CONTROL_DRAWS = 20
REGISTERED_BASE_SEED = 0

# Decimal makes the tie rule independent of binary floating-point representation.  The
# minimum-one clause is the registered "nearest valid positive integer" requirement.
ROUNDING_RULE = (
    "nearest integer by Decimal ROUND_HALF_UP on "
    "eligible_pool_size*fraction_percent/100; minimum 1"
)
CONTROL_RNG = "numpy.random.default_rng"
CONTROL_SEED_DERIVATION = (
    "np.random.default_rng([registered_base_seed, control_seed_draw_index, k])"
)

INT_ROW_FIELDS = frozenset({
    "layer", "neuron", "n_examples", "n_tokens", "rank_abs", "rank_abs_in_layer"
})
FLOAT_ROW_FIELDS = frozenset({
    "mean_abs_activation", "mean_signed_attr", "mean_abs_attr"
})
STRING_ROW_FIELDS = frozenset({"future_sink_layers"})

CONDITION_ROW_FIELDS: tuple[str, ...] = (
    "condition_id",
    "control_type",
    "fraction_percent",
    "k",
    "control_seed",
    "layer",
    "neuron",
)


class SelectionError(RuntimeError):
    """Raised when frozen inputs or a selected/control set violate the registration."""


@dataclass(frozen=True)
class FrozenAttributionRanking:
    """Typed and integrity-checked Task-5 rows, ready for global selection."""

    rows: tuple[Mapping[str, Any], ...]
    eligible_mlp_layers: tuple[int, ...]
    mlp_width: Mapping[int, int]
    attribution_sha256: str
    corpus_manifest_sha256: str
    sink_scope_sha256: str
    model_id: str
    model_revision: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rows",
            tuple(MappingProxyType(dict(row)) for row in self.rows),
        )
        object.__setattr__(self, "mlp_width", MappingProxyType(dict(self.mlp_width)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def pool_size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class SelectionCondition:
    """One targeted or layer-matched-random intervention candidate."""

    condition_id: str
    control_type: str
    fraction_percent: float
    k: int
    control_seed: int | None
    neuron_set: NeuronSet
    per_layer_counts: Mapping[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "per_layer_counts", MappingProxyType(dict(self.per_layer_counts))
        )


@dataclass(frozen=True)
class FrozenNeuronSets:
    """A verified ``neuron_sets_v1`` document reconstructed as ``NeuronSet`` objects."""

    document: Mapping[str, Any]
    neuron_sets: Mapping[str, NeuronSet]

    def __post_init__(self) -> None:
        object.__setattr__(self, "document", MappingProxyType(dict(self.document)))
        object.__setattr__(
            self, "neuron_sets", MappingProxyType(dict(self.neuron_sets))
        )


def _typed_attribution_row(raw: Mapping[str, str], row_number: int) -> dict[str, Any]:
    """Restore exactly the Python types emitted by ``attribution_rows``."""

    typed: dict[str, Any] = {}
    try:
        for field in ROW_FIELDS:
            value = raw[field]
            if field in INT_ROW_FIELDS:
                typed[field] = int(value)
            elif field in FLOAT_ROW_FIELDS:
                typed[field] = float(value)
            elif field in STRING_ROW_FIELDS:
                typed[field] = str(value)
            else:  # pragma: no cover - ROW_FIELDS and the type sets are fixed together
                raise SelectionError(f"No Task-5 CSV type registered for field {field!r}")
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionError(
            f"Invalid Task-5 attribution row {row_number}: {exc}"
        ) from exc
    return typed


def _future_layers(cell: str, *, layer: int) -> tuple[int, ...]:
    if not cell:
        raise SelectionError(f"Layer {layer} has an empty future_sink_layers cell")
    try:
        targets = tuple(int(part) for part in cell.split(FUTURE_LAYER_SEPARATOR))
    except ValueError as exc:
        raise SelectionError(
            f"Layer {layer} has malformed future_sink_layers {cell!r}"
        ) from exc
    if len(set(targets)) != len(targets) or any(target <= layer for target in targets):
        raise SelectionError(
            f"Layer {layer} has non-causal or duplicate future targets {targets}; every "
            "target must be unique and strictly later"
        )
    return targets


def _validate_attribution_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligible_layers: Sequence[int],
    widths: Mapping[int, int],
    scope: FrozenSinkScope,
    metadata: Mapping[str, Any],
) -> None:
    """Re-establish Task-5 row, rank, range, and causal-order invariants."""

    eligible = tuple(int(layer) for layer in eligible_layers)
    expected_rows = sum(int(widths[layer]) for layer in eligible)
    if len(rows) != expected_rows:
        raise SelectionError(
            f"Attribution table has {len(rows)} rows; expected {expected_rows} from "
            f"the per-layer widths {dict(widths)}"
        )
    observed_layers = tuple(sorted({int(row["layer"]) for row in rows}))
    if observed_layers != tuple(sorted(eligible)):
        raise SelectionError(
            f"Attribution layers {list(observed_layers)} do not equal the frozen eligible "
            f"layers {list(eligible)}"
        )

    expected_examples = int(metadata.get("n_examples", -1))
    expected_tokens = int(metadata.get("n_tokens", -1))
    for layer in eligible:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        width = int(widths[layer])
        neurons = sorted(int(row["neuron"]) for row in layer_rows)
        if neurons != list(range(width)):
            raise SelectionError(
                f"Layer {layer} neuron ids do not cover [0, {width}) exactly once"
            )
        within_ranks = sorted(int(row["rank_abs_in_layer"]) for row in layer_rows)
        if within_ranks != list(range(1, width + 1)):
            raise SelectionError(
                f"Layer {layer} rank_abs_in_layer is not a permutation of 1..{width}"
            )
        expected_targets = scope.targets_for(layer)
        for row in layer_rows:
            targets = _future_layers(str(row["future_sink_layers"]), layer=layer)
            if targets != expected_targets:
                raise SelectionError(
                    f"Layer {layer} row targets {targets}, but the frozen scope requires "
                    f"{expected_targets}"
                )
            if int(row["n_examples"]) != expected_examples:
                raise SelectionError(
                    f"Layer {layer} row n_examples={row['n_examples']} != metadata "
                    f"{expected_examples}"
                )
            if int(row["n_tokens"]) != expected_tokens:
                raise SelectionError(
                    f"Layer {layer} row n_tokens={row['n_tokens']} != metadata "
                    f"{expected_tokens}"
                )
            for field in FLOAT_ROW_FIELDS:
                if not math.isfinite(float(row[field])):
                    raise SelectionError(
                        f"Layer {layer} neuron {row['neuron']} has non-finite {field}"
                    )

    global_ranks = sorted(int(row["rank_abs"]) for row in rows)
    if global_ranks != list(range(1, expected_rows + 1)):
        raise SelectionError(
            f"rank_abs is not a permutation of 1..{expected_rows}"
        )
    independently_ranked = sorted(
        rows,
        key=lambda row: (
            -float(row[RANKING_SCORE]), int(row["layer"]), int(row["neuron"])
        ),
    )
    if any(int(row["rank_abs"]) != index for index, row in enumerate(
        independently_ranked, start=1
    )):
        raise SelectionError(
            "rank_abs does not equal the registered total order "
            "(-mean_abs_attr, layer, neuron)"
        )


def load_frozen_attribution(
    csv_path: Path | str,
    metadata_path: Path | str,
    *,
    scope: FrozenSinkScope,
    expected_corpus_manifest_sha256: str,
) -> FrozenAttributionRanking:
    """Load Task 5's CSV with exact types and reproduce its attribution hash.

    The caller supplies the already-verified frozen sink scope and the corpus manifest's
    top-level hash.  This loader never opens a corpus manifest or a corpus split.
    """

    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    if not csv_path.is_file():
        raise SelectionError(f"Frozen attribution CSV not found: {csv_path}")
    if not metadata_path.is_file():
        raise SelectionError(f"Frozen attribution metadata not found: {metadata_path}")

    metadata = read_json(metadata_path)
    if not isinstance(metadata, Mapping):
        raise SelectionError(f"{metadata_path} does not contain a JSON object")
    if metadata.get("ranking_score") != RANKING_SCORE:
        raise SelectionError(
            f"Frozen ranking score {metadata.get('ranking_score')!r} != {RANKING_SCORE!r}; "
            "mean_signed_attr is diagnostic and must never drive selection"
        )
    if metadata.get("corpus_manifest_sha256") != expected_corpus_manifest_sha256:
        raise SelectionError(
            "Attribution metadata corpus hash does not match the frozen corpus manifest: "
            f"{metadata.get('corpus_manifest_sha256')} != "
            f"{expected_corpus_manifest_sha256}"
        )
    if metadata.get("sink_scope_sha256") != scope.sink_scope_sha256:
        raise SelectionError(
            "Attribution metadata scope hash does not match the frozen sink scope: "
            f"{metadata.get('sink_scope_sha256')} != {scope.sink_scope_sha256}"
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != ROW_FIELDS:
            raise SelectionError(
                f"Task-5 CSV columns {header} != the registered row fields {ROW_FIELDS}"
            )
        rows = tuple(
            _typed_attribution_row(raw, row_number)
            for row_number, raw in enumerate(reader, start=2)
        )

    stored_sha = str(metadata.get("attribution_sha256", ""))
    reproduced_sha = attribution_sha256(rows)
    if reproduced_sha != stored_sha:
        raise SelectionError(
            f"Frozen attribution hash mismatch: stored {stored_sha} != reproduced "
            f"{reproduced_sha}. Fix CSV typing or restore the frozen Task-5 ranking; do "
            "not regenerate it."
        )

    eligible = tuple(int(layer) for layer in metadata.get("eligible_mlp_layers", ()))
    if eligible != scope.eligible_mlp_layers:
        raise SelectionError(
            f"Attribution eligible layers {list(eligible)} != frozen scope "
            f"{list(scope.eligible_mlp_layers)}"
        )
    raw_widths = metadata.get("mlp_width", {})
    if not isinstance(raw_widths, Mapping):
        raise SelectionError("Attribution metadata mlp_width must be a mapping")
    widths = {int(layer): int(width) for layer, width in raw_widths.items()}
    if tuple(sorted(widths)) != tuple(sorted(eligible)):
        raise SelectionError(
            f"MLP width layers {sorted(widths)} != eligible layers {list(eligible)}"
        )
    if any(width <= 0 for width in widths.values()):
        raise SelectionError(f"Every MLP width must be positive, got {widths}")
    if int(metadata.get("n_rows", -1)) != sum(widths.values()):
        raise SelectionError(
            f"Metadata n_rows={metadata.get('n_rows')} != total width {sum(widths.values())}"
        )

    _validate_attribution_rows(
        rows,
        eligible_layers=eligible,
        widths=widths,
        scope=scope,
        metadata=metadata,
    )
    return FrozenAttributionRanking(
        rows=rows,
        eligible_mlp_layers=eligible,
        mlp_width=widths,
        attribution_sha256=stored_sha,
        corpus_manifest_sha256=str(metadata["corpus_manifest_sha256"]),
        sink_scope_sha256=str(metadata["sink_scope_sha256"]),
        model_id=str(metadata.get("model_id", "")),
        model_revision=str(metadata.get("model_revision", "")),
        metadata=metadata,
    )


def _as_decimal(value: float | Decimal) -> Decimal:
    if isinstance(value, bool):
        raise TypeError("fraction_percent must be a real number, not bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid fraction_percent {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"fraction_percent must be finite and positive, got {value!r}")
    return result


def exact_k(fraction_percent: float | Decimal, pool_size: int) -> int:
    """Convert a percentage to the registered nearest positive integer ``k``."""

    if isinstance(pool_size, bool) or not isinstance(pool_size, (int, np.integer)):
        raise TypeError(f"pool_size must be an integer, got {type(pool_size).__name__}")
    pool_size = int(pool_size)
    if pool_size < 1:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    fraction = _as_decimal(fraction_percent)
    raw = Decimal(pool_size) * fraction / Decimal(100)
    k = max(1, int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
    if k > pool_size:
        raise ValueError(
            f"fraction_percent={fraction} yields k={k}, larger than pool_size={pool_size}"
        )
    return k


def fraction_label(fraction_percent: float | Decimal) -> str:
    """Stable filename-safe label, e.g. 0.10 percent -> ``f0p10``."""

    fraction = _as_decimal(fraction_percent)
    hundredth = fraction.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if hundredth != fraction:
        raise ValueError(
            f"fraction_percent={fraction} has more than two decimal places; condition ids "
            "are registered at hundredth-of-a-percent precision"
        )
    return f"f{format(hundredth, '.2f').replace('.', 'p')}"


def _validate_neuron_set(
    neuron_set: NeuronSet,
    *,
    eligible_layers: Sequence[int],
    widths: Mapping[int, int],
    expected_k: int,
) -> None:
    eligible = set(int(layer) for layer in eligible_layers)
    if not neuron_set.by_layer:
        raise SelectionError("A selected neuron set must not be empty")
    if any(layer not in eligible for layer in neuron_set.by_layer):
        raise SelectionError(
            f"Neuron set contains an ineligible layer; allowed {sorted(eligible)}, got "
            f"{list(neuron_set.by_layer)}"
        )
    total = 0
    for layer, neurons in neuron_set.by_layer.items():
        if not neurons:
            raise SelectionError(
                f"Layer {layer} has zero neurons; zero-count layers must be omitted from "
                "NeuronSet.by_layer"
            )
        if len(neurons) != len(set(neurons)):
            raise SelectionError(f"Layer {layer} contains duplicate neuron ids")
        width = int(widths[layer])
        if any(neuron < 0 or neuron >= width for neuron in neurons):
            raise SelectionError(
                f"Layer {layer} contains a neuron outside [0, {width})"
            )
        total += len(neurons)
    if total != expected_k:
        raise SelectionError(f"Neuron set contains {total} neurons; expected k={expected_k}")


def per_layer_counts(
    neuron_set: NeuronSet, eligible_layers: Sequence[int]
) -> dict[int, int]:
    """Counts for every eligible layer, including explicit zeros for diagnostics."""

    return {
        int(layer): len(neuron_set.by_layer.get(int(layer), ()))
        for layer in eligible_layers
    }


def select_global_top_k(
    ranking: FrozenAttributionRanking, k: int
) -> NeuronSet:
    """Cut the Task-5 global rank at ``k`` and return the existing ``NeuronSet`` type."""

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    k = int(k)
    if k < 1 or k > ranking.pool_size:
        raise ValueError(f"k must be in [1, {ranking.pool_size}], got {k}")
    selected = [row for row in ranking.rows if int(row["rank_abs"]) <= k]
    if len(selected) != k:
        raise SelectionError(f"rank_abs <= {k} selected {len(selected)} rows")
    by_layer_lists: dict[int, list[int]] = {}
    for row in selected:
        by_layer_lists.setdefault(int(row["layer"]), []).append(int(row["neuron"]))
    neuron_set = NeuronSet(
        {
            layer: tuple(sorted(neurons))
            for layer, neurons in sorted(by_layer_lists.items())
            if neurons
        },
        source=CONTROL_TYPE_TARGETED,
        selection_seed=None,
    )
    _validate_neuron_set(
        neuron_set,
        eligible_layers=ranking.eligible_mlp_layers,
        widths=ranking.mlp_width,
        expected_k=k,
    )
    return neuron_set


def generate_layer_matched_controls(
    target: NeuronSet,
    *,
    eligible_layers: Sequence[int],
    widths: Mapping[int, int],
    k: int,
    draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
) -> tuple[NeuronSet, ...]:
    """Sample deterministic layer-count-matched controls from non-target neurons."""

    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError(f"draws must be a positive integer, got {draws!r}")
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError(f"base_seed must be an integer, got {type(base_seed).__name__}")
    base_seed = int(base_seed)
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    _validate_neuron_set(
        target,
        eligible_layers=eligible_layers,
        widths=widths,
        expected_k=k,
    )
    target_counts = per_layer_counts(target, eligible_layers)

    controls: list[NeuronSet] = []
    for draw_index in range(int(draws)):
        rng = np.random.default_rng([base_seed, draw_index, int(k)])
        by_layer: dict[int, tuple[int, ...]] = {}
        for layer in sorted(int(value) for value in eligible_layers):
            count = target_counts[layer]
            if count == 0:
                continue
            width = int(widths[layer])
            targeted = set(target.by_layer.get(layer, ()))
            candidates = np.fromiter(
                (neuron for neuron in range(width) if neuron not in targeted),
                dtype=np.int64,
                count=width - len(targeted),
            )
            if count > len(candidates):
                raise SelectionError(
                    f"Layer {layer} needs {count} random neurons but has only "
                    f"{len(candidates)} non-target candidates"
                )
            chosen = rng.choice(candidates, size=count, replace=False)
            by_layer[layer] = tuple(sorted(int(value) for value in chosen.tolist()))
        control = NeuronSet(
            by_layer,
            source=CONTROL_TYPE_LAYER_RANDOM,
            selection_seed=draw_index,
        )
        _validate_neuron_set(
            control,
            eligible_layers=eligible_layers,
            widths=widths,
            expected_k=k,
        )
        if per_layer_counts(control, eligible_layers) != target_counts:
            raise SelectionError(
                f"Random draw {draw_index} does not preserve the target's layer counts"
            )
        for layer, neurons in control.by_layer.items():
            overlap = set(neurons) & set(target.by_layer.get(layer, ()))
            if overlap:
                raise SelectionError(
                    f"Random draw {draw_index}, layer {layer} overlaps target ids "
                    f"{sorted(overlap)}"
                )
        controls.append(control)
    return tuple(controls)


def build_selection_conditions(
    ranking: FrozenAttributionRanking,
    fractions_percent: Sequence[float | Decimal],
    *,
    control_draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
) -> tuple[SelectionCondition, ...]:
    """Build targeted conditions and their matched controls in a stable order."""

    conditions: list[SelectionCondition] = []
    seen_labels: set[str] = set()
    for fraction_value in fractions_percent:
        label = fraction_label(fraction_value)
        if label in seen_labels:
            raise ValueError(f"Duplicate normalized fraction label {label}")
        seen_labels.add(label)
        fraction = float(_as_decimal(fraction_value))
        k = exact_k(fraction_value, ranking.pool_size)
        target = select_global_top_k(ranking, k)
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


def _condition_dict(condition: SelectionCondition) -> dict[str, Any]:
    return {
        "control_type": condition.control_type,
        "source": condition.neuron_set.source,
        "fraction_percent": condition.fraction_percent,
        "k": condition.k,
        "control_seed": condition.control_seed,
        "per_layer_counts": {
            str(layer): count for layer, count in condition.per_layer_counts.items()
        },
        # NeuronSet rejects empty layer entries.  The diagnostic count mapping above keeps
        # explicit zeros; by_layer deliberately omits them.
        "by_layer": {
            str(layer): list(neurons)
            for layer, neurons in condition.neuron_set.by_layer.items()
        },
    }


def build_neuron_sets_document(
    ranking: FrozenAttributionRanking,
    conditions: Sequence[SelectionCondition],
    *,
    fractions_percent: Sequence[float | Decimal],
    control_draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
    experiment_id: str = "task6_selection",
) -> dict[str, Any]:
    """Create a stable, hash-bearing discovery selection document.

    The default preserves the byte-identical Task-6 smoke artefact.  Stage B supplies a
    distinct experiment id so its full-model selections cannot be mistaken for or written
    over the tracked smoke candidates.
    """

    condition_ids = [condition.condition_id for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise SelectionError("Condition ids must be unique")
    document: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "stage": "discovery",
        "model_id": ranking.model_id,
        "model_revision": ranking.model_revision,
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "selection_method": SELECTION_METHOD,
        "ranking_score": RANKING_SCORE,
        "attribution_sha256": ranking.attribution_sha256,
        "corpus_manifest_sha256": ranking.corpus_manifest_sha256,
        "sink_scope_sha256": ranking.sink_scope_sha256,
        "eligible_mlp_layers": list(ranking.eligible_mlp_layers),
        "mlp_width": {
            str(layer): int(ranking.mlp_width[layer])
            for layer in ranking.eligible_mlp_layers
        },
        "eligible_pool_size": ranking.pool_size,
        "fractions_percent": [float(_as_decimal(value)) for value in fractions_percent],
        "rounding_rule": ROUNDING_RULE,
        "registered_base_seed": int(base_seed),
        "control_draws": int(control_draws),
        "control_type": CONTROL_TYPE_LAYER_RANDOM,
        "control_rng": CONTROL_RNG,
        "control_seed_semantics": "zero-indexed draw index",
        "control_seed_derivation": CONTROL_SEED_DERIVATION,
        "condition_ids": condition_ids,
        "conditions": {
            condition.condition_id: _condition_dict(condition)
            for condition in conditions
        },
        "is_causal_evidence": False,
        "note": (
            "These are attribution-ranked intervention candidates and matched random "
            "controls, not causal neurons. Causal evidence requires Task 7's held-out "
            "suppression comparison."
            if experiment_id == "task6_selection"
            else
            "These are attribution-ranked intervention candidates and matched random "
            "controls, not causal neurons. Causal evidence requires held-out suppression "
            "against the saved layer-count-matched controls."
        ),
    }
    document["neuron_sets_sha256"] = canonical_sha256(document)
    return document


def condition_rows(conditions: Sequence[SelectionCondition]) -> list[dict[str, Any]]:
    """Flat one-row-per-(condition, layer, neuron) representation."""

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for layer, neurons in condition.neuron_set.by_layer.items():
            for neuron in neurons:
                rows.append({
                    "condition_id": condition.condition_id,
                    "control_type": condition.control_type,
                    "fraction_percent": condition.fraction_percent,
                    "k": condition.k,
                    "control_seed": condition.control_seed,
                    "layer": layer,
                    "neuron": neuron,
                })
    return rows


def verify_neuron_sets_document(document: Mapping[str, Any]) -> FrozenNeuronSets:
    """Verify a saved Task-6 document and reconstruct every existing ``NeuronSet``."""

    if document.get("schema") != SCHEMA_VERSION:
        raise SelectionError(
            f"Neuron-set schema {document.get('schema')!r} != {SCHEMA_VERSION!r}"
        )
    stored_sha = str(document.get("neuron_sets_sha256", ""))
    recomputed_sha = canonical_sha256(
        {key: value for key, value in document.items() if key != "neuron_sets_sha256"}
    )
    if stored_sha != recomputed_sha:
        raise SelectionError(
            f"Frozen neuron-set hash mismatch: stored {stored_sha} != recomputed "
            f"{recomputed_sha}"
        )
    if document.get("ranking_score") != RANKING_SCORE:
        raise SelectionError(
            f"Neuron sets use ranking score {document.get('ranking_score')!r}, expected "
            f"{RANKING_SCORE!r}"
        )
    if document.get("rounding_rule") != ROUNDING_RULE:
        raise SelectionError(
            f"Neuron sets use rounding rule {document.get('rounding_rule')!r}, expected "
            f"{ROUNDING_RULE!r}"
        )
    if document.get("control_rng") != CONTROL_RNG:
        raise SelectionError(
            f"Neuron sets use control RNG {document.get('control_rng')!r}, expected "
            f"{CONTROL_RNG!r}"
        )
    if document.get("control_seed_derivation") != CONTROL_SEED_DERIVATION:
        raise SelectionError("Neuron sets use an unregistered control-seed derivation")

    eligible = tuple(int(layer) for layer in document["eligible_mlp_layers"])
    widths = {int(layer): int(width) for layer, width in document["mlp_width"].items()}
    if tuple(sorted(widths)) != tuple(sorted(eligible)):
        raise SelectionError("Saved MLP-width keys do not match eligible layers")
    pool_size = sum(widths.values())
    if int(document.get("eligible_pool_size", -1)) != pool_size:
        raise SelectionError(
            f"Saved eligible_pool_size={document.get('eligible_pool_size')} != {pool_size}"
        )
    fractions = tuple(float(value) for value in document.get("fractions_percent", ()))
    if not fractions or len(fractions) != len(set(fractions)):
        raise SelectionError("fractions_percent must be a non-empty unique grid")
    control_draws = int(document.get("control_draws", -1))
    if control_draws < 1:
        raise SelectionError("control_draws must be positive")
    base_seed = int(document.get("registered_base_seed", -1))
    if base_seed < 0:
        raise SelectionError("registered_base_seed must be non-negative")
    raw_conditions = document.get("conditions")
    condition_ids = list(document.get("condition_ids", ()))
    if (
        not isinstance(raw_conditions, Mapping)
        or len(condition_ids) != len(set(condition_ids))
        or set(condition_ids) != set(raw_conditions)
    ):
        raise SelectionError(
            "condition_ids must uniquely enumerate every saved condition"
        )

    sets: dict[str, NeuronSet] = {}
    records: dict[str, Mapping[str, Any]] = {}
    targeted_by_fraction: dict[float, tuple[str, NeuronSet, dict[int, int]]] = {}
    for condition_id in condition_ids:
        record = raw_conditions[condition_id]
        control_type = str(record["control_type"])
        if control_type not in (CONTROL_TYPE_TARGETED, CONTROL_TYPE_LAYER_RANDOM):
            raise SelectionError(
                f"Condition {condition_id} has unknown control_type {control_type!r}"
            )
        seed = record.get("control_seed")
        if seed is not None:
            seed = int(seed)
        neuron_set = NeuronSet(
            {
                int(layer): tuple(int(neuron) for neuron in neurons)
                for layer, neurons in record["by_layer"].items()
            },
            source=str(record.get("source", control_type)),
            selection_seed=seed,
        )
        k = int(record["k"])
        fraction = float(record["fraction_percent"])
        expected_k = exact_k(fraction, pool_size)
        if k != expected_k:
            raise SelectionError(
                f"Condition {condition_id} has k={k}; {fraction}% of pool {pool_size} "
                f"requires k={expected_k} under the frozen rounding rule"
            )
        _validate_neuron_set(
            neuron_set,
            eligible_layers=eligible,
            widths=widths,
            expected_k=k,
        )
        counts = {int(layer): int(count) for layer, count in record[
            "per_layer_counts"
        ].items()}
        if tuple(sorted(counts)) != tuple(sorted(eligible)):
            raise SelectionError(
                f"Condition {condition_id} per_layer_counts does not cover eligible layers"
            )
        if counts != per_layer_counts(neuron_set, eligible):
            raise SelectionError(
                f"Condition {condition_id} saved counts do not match by_layer"
            )
        if control_type == CONTROL_TYPE_TARGETED:
            if seed is not None or neuron_set.source != CONTROL_TYPE_TARGETED:
                raise SelectionError(
                    f"Targeted condition {condition_id} must have no control seed"
                )
            if fraction in targeted_by_fraction:
                raise SelectionError(f"Duplicate targeted condition for {fraction}%")
            targeted_by_fraction[fraction] = (condition_id, neuron_set, counts)
        else:
            if seed is None or neuron_set.source != CONTROL_TYPE_LAYER_RANDOM:
                raise SelectionError(
                    f"Random condition {condition_id} must carry its draw-index seed"
                )
        sets[condition_id] = neuron_set
        records[condition_id] = record

    if set(targeted_by_fraction) != set(fractions):
        raise SelectionError(
            f"Targeted fractions {sorted(targeted_by_fraction)} do not match the frozen "
            f"grid {list(fractions)}"
        )

    for condition_id, record in records.items():
        if record["control_type"] != CONTROL_TYPE_LAYER_RANDOM:
            continue
        fraction = float(record["fraction_percent"])
        if fraction not in targeted_by_fraction:
            raise SelectionError(
                f"Random condition {condition_id} has no targeted condition at {fraction}%"
            )
        _target_id, target, target_counts = targeted_by_fraction[fraction]
        control = sets[condition_id]
        if per_layer_counts(control, eligible) != target_counts:
            raise SelectionError(
                f"Random condition {condition_id} does not preserve target layer counts"
            )
        for layer, neurons in control.by_layer.items():
            overlap = set(neurons) & set(target.by_layer.get(layer, ()))
            if overlap:
                raise SelectionError(
                    f"Random condition {condition_id}, layer {layer} overlaps target ids "
                    f"{sorted(overlap)}"
                )

    # Recompute each control from its recorded composite seed.  This makes Task 7's loader
    # verify the selection algorithm, not merely the counts and the document checksum.
    for fraction, (_target_id, target, _counts) in targeted_by_fraction.items():
        expected_controls = generate_layer_matched_controls(
            target,
            eligible_layers=eligible,
            widths=widths,
            k=exact_k(fraction, pool_size),
            draws=control_draws,
            base_seed=base_seed,
        )
        saved_by_seed = {
            int(record["control_seed"]): sets[condition_id]
            for condition_id, record in records.items()
            if record["control_type"] == CONTROL_TYPE_LAYER_RANDOM
            and float(record["fraction_percent"]) == fraction
        }
        if set(saved_by_seed) != set(range(control_draws)):
            raise SelectionError(
                f"Random conditions at {fraction}% do not contain exactly draw indices "
                f"0..{control_draws - 1}"
            )
        for draw_index, expected in enumerate(expected_controls):
            if saved_by_seed[draw_index] != expected:
                raise SelectionError(
                    f"Random condition at {fraction}%, draw {draw_index} does not match "
                    "the registered composite RNG seed"
                )
    return FrozenNeuronSets(document=document, neuron_sets=sets)


def load_frozen_neuron_sets(path: Path | str) -> FrozenNeuronSets:
    path = Path(path)
    if not path.is_file():
        raise SelectionError(f"Frozen neuron sets not found: {path}")
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise SelectionError(f"{path} does not contain a JSON object")
    return verify_neuron_sets_document(document)
