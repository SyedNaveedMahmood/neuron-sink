"""Per-sink-layer budgeted neuron selection (Stage C3, amendment A007).

Stage C and Stage C2 select one global top-``k`` list. On Qwen that concentrated the budget in
the last eligible MLP layer, which can only influence the last sink layer, so three of seven
graded sink layers were causally unreachable and moved by exactly zero. Global selection is
not wrong in principle -- it is wrong when the scores being compared are not commensurable
across depth, which they are not here: a deep neuron has a shorter gradient path and a smaller
target set than a shallow one.

Stage C3 therefore spends the budget where the graded metric actually lives. Each registered
sink layer ``j`` receives a share of ``k`` proportional to its weight in the metric --
``w_j = S0_j / sum_j S0_j``, allocated by largest remainder so the shares sum to exactly ``k``
-- and that share is filled only from MLP layers ``l < j``, ranked by the effect measured on
layer ``j`` itself. The union of the resulting set therefore reaches every graded sink layer,
which restores the registered full-scope gate to a metric the intervention can actually move.

Fill order is ascending ``j``. The earliest sink layer has the smallest pool of upstream MLP
layers, so it is the most constrained and picks first; deeper targets have far more candidates
and can absorb the leftovers. A neuron already taken for an earlier target is skipped rather
than double-counted, and any unfilled remainder is redistributed from the global measured
ranking so that ``k`` is met exactly.

Controls, their RNG, their seeds and their re-derivation check are inherited **unchanged** from
:mod:`neuron_sink.selection`; only the targeted set is chosen differently.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ablation_screen import ScreenTable
from .provenance import canonical_sha256, read_json
from .selection import (
    CONTROL_TYPE_LAYER_RANDOM,
    CONTROL_TYPE_TARGETED,
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


SCHEMA_VERSION = "neuron_sets_c3_v1"
SELECTION_METHOD = "per_sink_layer_budget_measured_ablation"
RANKING_SCORE = "measured_delta_sink"
SIGN_REQUIREMENT = "strictly_positive_measured_reduction"
BUDGET_RULE = (
    "largest remainder on w_j = baseline_sink_j / sum_j baseline_sink_j over the registered "
    "sink layers; shares sum to exactly k"
)
FILL_RULE = (
    "ascending target sink layer; candidates ranked by measured_delta_sink at that target; "
    "already-selected neurons skipped; shortfall redistributed from the global measured order"
)


class SelectionC3Error(RuntimeError):
    """Raised when a Stage-C3 selection request or artefact is invalid."""


def sink_layer_budget(
    k: int, sink_layers: Sequence[int], baseline_per_layer: Sequence[float]
) -> dict[int, int]:
    """Split ``k`` across sink layers by metric weight, summing to exactly ``k``.

    Largest remainder rather than rounding, so the shares are integral and total ``k`` with no
    drift. When ``k`` is smaller than the number of sink layers the lowest-weight layers
    necessarily receive zero; that is a property of a budget smaller than the target set, not
    a tie-break, and it is recorded in the returned document.
    """

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 1:
        raise SelectionC3Error(f"k must be at least 1, got {k}")
    layers = [int(layer) for layer in sink_layers]
    weights = [float(value) for value in baseline_per_layer]
    if len(layers) != len(weights) or not layers:
        raise SelectionC3Error(
            f"Need one baseline sink per sink layer, got {len(layers)} layers and "
            f"{len(weights)} baselines"
        )
    if any(value <= 0.0 for value in weights):
        raise SelectionC3Error(
            f"Every registered sink layer must have a positive baseline sink, got {weights}"
        )

    total = sum(weights)
    exact = [k * value / total for value in weights]
    floors = [int(value) for value in exact]
    remainder = k - sum(floors)
    # Largest fractional part first; ties broken by ascending layer id so the allocation is a
    # pure function of (k, layers, baselines).
    order = sorted(
        range(len(layers)), key=lambda i: (-(exact[i] - floors[i]), layers[i])
    )
    for position in order[:remainder]:
        floors[position] += 1
    budget = {layers[i]: floors[i] for i in range(len(layers))}
    if sum(budget.values()) != k:
        raise SelectionC3Error(
            f"Budget {budget} sums to {sum(budget.values())}, expected {k}"
        )
    return budget


def select_per_sink_layer_budget(
    screen: ScreenTable, k: int, *, eligible_layers: Sequence[int]
) -> tuple[NeuronSet, dict[str, Any]]:
    """Fill each sink layer's quota from the MLP layers that can reach it."""

    budget = sink_layer_budget(k, screen.sink_layers, screen.baseline_per_layer)
    allowed = {int(layer) for layer in eligible_layers}

    chosen: dict[int, list[int]] = {}
    taken: set[tuple[int, int]] = set()
    drawn_for: list[dict[str, Any]] = []
    per_target_filled: dict[int, int] = {}

    for target in sorted(budget):
        quota = budget[target]
        per_target_filled[target] = 0
        if quota <= 0:
            continue
        for row in screen.candidates_for_target(target):
            layer = int(row["mlp_layer"])
            neuron = int(row["neuron"])
            if layer not in allowed:
                continue
            key = (layer, neuron)
            if key in taken:
                continue
            taken.add(key)
            chosen.setdefault(layer, []).append(neuron)
            drawn_for.append({
                "mlp_layer": layer,
                "neuron": neuron,
                "target_sink_layer": target,
                "measured_delta_sink": float(row["measured_delta_sink"]),
                "source": "quota",
            })
            per_target_filled[target] += 1
            if per_target_filled[target] == quota:
                break

    shortfall = k - len(taken)
    if shortfall > 0:
        pooled = [
            row for row in screen.rows
            if float(row["measured_delta_sink"]) > 0.0
            and int(row["mlp_layer"]) in allowed
        ]
        pooled.sort(
            key=lambda row: (
                -float(row["measured_delta_sink"]),
                int(row["target_sink_layer"]),
                int(row["mlp_layer"]),
                int(row["neuron"]),
            )
        )
        for row in pooled:
            if shortfall == 0:
                break
            key = (int(row["mlp_layer"]), int(row["neuron"]))
            if key in taken:
                continue
            taken.add(key)
            chosen.setdefault(key[0], []).append(key[1])
            drawn_for.append({
                "mlp_layer": key[0],
                "neuron": key[1],
                "target_sink_layer": int(row["target_sink_layer"]),
                "measured_delta_sink": float(row["measured_delta_sink"]),
                "source": "redistributed",
            })
            shortfall -= 1

    if len(taken) != k:
        raise SelectionC3Error(
            f"Stage C3 could fill only {len(taken)} of {k} neurons with a strictly positive "
            "measured sink reduction. Widen the shortlist rather than relaxing the sign "
            "requirement, or stop and report the shortage."
        )

    neuron_set = NeuronSet(
        {layer: tuple(sorted(neurons)) for layer, neurons in sorted(chosen.items())},
        source=CONTROL_TYPE_TARGETED,
    )
    reached = sorted({
        int(target) for target in screen.sink_layers
        if any(layer < int(target) for layer in chosen)
    })
    diagnostics = {
        "budget": {str(layer): count for layer, count in sorted(budget.items())},
        "filled": {str(layer): per_target_filled.get(layer, 0) for layer in sorted(budget)},
        "redistributed": sum(1 for row in drawn_for if row["source"] == "redistributed"),
        "drawn_for": drawn_for,
        "reachable_sink_layers": reached,
        "unreachable_sink_layers": [
            int(target) for target in screen.sink_layers if int(target) not in reached
        ],
        "reachable_metric_weight": (
            sum(
                value for target, value in zip(screen.sink_layers, screen.baseline_per_layer)
                if int(target) in reached
            ) / sum(screen.baseline_per_layer)
        ),
        # Sum of the MEASURED marginal reductions of the selected neurons. It is the
        # direction check the grid is gated on, and it is a stronger predictor than a
        # first-order score because each term was measured under the real intervention. It
        # is still a sum of marginals, so it is not the joint effect -- measure_joint_effect
        # reports that separately.
        "sum_measured_marginal_delta_sink": sum(
            row["measured_delta_sink"] for row in drawn_for
        ),
    }
    return neuron_set, diagnostics


def build_c3_selection_conditions(
    ranking: Any,
    screen: ScreenTable,
    fractions_percent: Sequence[float | Decimal],
    *,
    control_draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
) -> tuple[tuple[SelectionCondition, ...], dict[str, Any]]:
    """Build the targeted and layer-count-matched random conditions for every fraction."""

    eligible = tuple(int(layer) for layer in ranking.eligible_mlp_layers)
    widths = {int(k): int(v) for k, v in ranking.mlp_width.items()}
    pool_size = ranking.pool_size

    conditions: list[SelectionCondition] = []
    diagnostics: dict[str, Any] = {}
    for fraction in fractions_percent:
        k = exact_k(fraction, pool_size)
        label = fraction_label(fraction)
        target, target_diagnostics = select_per_sink_layer_budget(
            screen, k, eligible_layers=eligible
        )
        diagnostics[label] = target_diagnostics
        conditions.append(SelectionCondition(
            condition_id=f"targeted_{label}",
            control_type=CONTROL_TYPE_TARGETED,
            fraction_percent=float(fraction),
            k=k,
            control_seed=None,
            neuron_set=target,
            per_layer_counts=per_layer_counts(target, eligible),
        ))
        controls = generate_layer_matched_controls(
            target,
            eligible_layers=eligible,
            widths=widths,
            k=k,
            draws=control_draws,
            base_seed=base_seed,
        )
        for draw_index, control in enumerate(controls):
            conditions.append(SelectionCondition(
                condition_id=f"{CONTROL_TYPE_LAYER_RANDOM}_{label}_s{draw_index}",
                control_type=CONTROL_TYPE_LAYER_RANDOM,
                fraction_percent=float(fraction),
                k=k,
                control_seed=draw_index,
                neuron_set=control,
                per_layer_counts=per_layer_counts(control, eligible),
            ))
    return tuple(conditions), diagnostics


def build_c3_neuron_sets_document(
    ranking: Any,
    conditions: Sequence[SelectionCondition],
    screen: ScreenTable,
    selection_diagnostics: Mapping[str, Any],
    *,
    fractions_percent: Sequence[float | Decimal],
    control_draws: int,
    base_seed: int = REGISTERED_BASE_SEED,
    experiment_id: str,
) -> dict[str, Any]:
    """Emit the Stage-C3 neuron-set document with its own schema and re-hash."""

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
    document["amendment"] = "A007"
    document["sign_requirement"] = SIGN_REQUIREMENT
    document["budget_rule"] = BUDGET_RULE
    document["fill_rule"] = FILL_RULE
    document["screen_method"] = "measured_single_neuron_ablation"
    document["screen_alpha"] = screen.alpha
    document["screen_examples"] = screen.n_examples
    document["screen_candidates"] = screen.n_candidates
    document["screen_sha256"] = screen.sha256()
    document["registered_sink_layers"] = list(screen.sink_layers)
    document["baseline_sink_per_layer"] = list(screen.baseline_per_layer)
    document["selection_diagnostics"] = dict(selection_diagnostics)
    document["first_order_direction"] = (
        "positive measured sink reduction at the neuron's own target layer"
    )
    document["neuron_sets_sha256"] = canonical_sha256(
        {k: v for k, v in document.items() if k != "neuron_sets_sha256"}
    )
    return document


def verify_c3_neuron_sets_document(document: Mapping[str, Any]) -> Any:
    """Verify a Stage-C3 document, then delegate to the unchanged registered verifier."""

    for field, expected in (
        ("schema", SCHEMA_VERSION),
        ("selection_method", SELECTION_METHOD),
        ("ranking_score", RANKING_SCORE),
        ("amendment", "A007"),
        ("sign_requirement", SIGN_REQUIREMENT),
        ("budget_rule", BUDGET_RULE),
        ("fill_rule", FILL_RULE),
    ):
        if document.get(field) != expected:
            raise SelectionC3Error(
                f"Stage-C3 neuron sets have {field}={document.get(field)!r}, expected "
                f"{expected!r}"
            )
    stored = document.get("neuron_sets_sha256")
    recomputed = canonical_sha256(
        {k: v for k, v in document.items() if k != "neuron_sets_sha256"}
    )
    if stored != recomputed:
        raise SelectionC3Error(
            f"Stage-C3 neuron-set hash mismatch: stored {stored} != recomputed {recomputed}"
        )
    try:
        return verify_neuron_sets_document(
            document,
            expected_schema=SCHEMA_VERSION,
            expected_selection_method=SELECTION_METHOD,
            expected_ranking_score=RANKING_SCORE,
        )
    except SelectionError as error:
        raise SelectionC3Error(str(error)) from error


def load_c3_neuron_sets(path: Path | str) -> Any:
    """Load and verify a frozen Stage-C3 neuron-set document."""

    path = Path(path)
    if not path.is_file():
        raise SelectionC3Error(f"Stage-C3 neuron sets not found: {path}")
    return verify_c3_neuron_sets_document(read_json(path))


__all__ = [
    "BUDGET_RULE",
    "FILL_RULE",
    "RANKING_SCORE",
    "SCHEMA_VERSION",
    "SELECTION_METHOD",
    "SIGN_REQUIREMENT",
    "SelectionC3Error",
    "build_c3_neuron_sets_document",
    "build_c3_selection_conditions",
    "load_c3_neuron_sets",
    "select_per_sink_layer_budget",
    "sink_layer_budget",
    "verify_c3_neuron_sets_document",
]
