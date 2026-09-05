"""Whole-MLP layer attenuation ceiling (Stage C3 arm B, amendment A007).

``docs/01_PHENOMENON_GATE.md`` registers a layer-level baseline: attenuate the *complete* MLP
intermediate vector of each eligible layer across the same alpha grid and compare with sparse
top-neuron suppression. The registered question is

    Can a small selected neuron set achieve a meaningful fraction of whole-MLP sink reduction
    with substantially less functional drift?

For Stage C3 it answers a second, prior question that the Stage-C null made urgent: how much
of the graded metric is available to this unit type at all?

Be precise about what is and is not a bound here, because it is easy to overclaim.

- **The only rigorous bound is causal reachability.** A set drawn from MLP layers ``L`` cannot
  change attention at any sink layer ``j <= min(L)``; those terms are already computed and stay
  at baseline. So the achievable relative reduction is at most the baseline sink mass of the
  reachable layers divided by the total, recorded per condition as
  ``reachable_metric_weight``. This is arithmetic, not a measurement. It is what showed 41.1%
  of the Stage-C metric to be unreachable.
- **Whole-layer and all-layer attenuation are maximal interventions, not upper bounds.**
  Suppression is *not* monotone in the sink: Stage C is itself a case where suppressing neurons
  *increased* it, so a subset can outperform the full set, and a set spanning several layers can
  outperform any single layer. Treat these conditions as strong empirical reference points -
  "what does this unit type do at this depth when pushed as hard as it goes" - and never as a
  ceiling that a sparse set cannot exceed.

The diagnostic use survives that correction. If attenuating every eligible MLP layer barely
moves a sink layer, that is strong evidence the sink is not MLP-mediated at that depth, and the
honest response is to question the unit type rather than the ranking.

This is deliberately cheap -- one forward per (layer, alpha, example), no gradients, discovery
only -- and it should be run before spending compute on selection. It discovers nothing and
selects nothing, so it is a diagnostic rather than a stage in the gated sequence.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .ablation_screen import METRIC_DTYPE, _sink_vector, baseline_per_layer_sink
from .corpus import NeutralCorpusItem
from .model_adapters import MLPModelAdapter
from .provenance import canonical_sha256
from .sink_metrics import REGISTERED_TARGET_POSITION
from .suppression import NeuronSet, suppress_neurons


SCHEMA_VERSION = "layer_attenuation_ceiling_c3_v1"
METHOD = "whole_mlp_intermediate_attenuation"

#: Pseudo-layer id for the all-eligible-layers condition.
ALL_LAYERS = -1

ROW_FIELDS: tuple[str, ...] = (
    "mlp_layer",
    "alpha",
    "target_sink_layer",
    "baseline_sink",
    "attenuated_sink",
    "delta_sink",
    "relative_sink_reduction",
    "reachable",
    "n_examples",
)


class LayerBaselineError(RuntimeError):
    """Raised when a layer-attenuation request is invalid."""


def layer_attenuation_ceiling(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    items: Sequence[NeutralCorpusItem],
    eligible_layers: Sequence[int],
    sink_layers: Sequence[int],
    *,
    alphas: Sequence[float],
    baseline: np.ndarray | None = None,
    device: torch.device | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
    metric_dtype: torch.dtype | None = METRIC_DTYPE,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Per-sink-layer effect of attenuating each eligible MLP layer in full."""

    if device is None:
        device = next(model.parameters()).device
    layers = [int(layer) for layer in eligible_layers]
    ordered_sink_layers = tuple(int(layer) for layer in sink_layers)
    if not layers or not ordered_sink_layers:
        raise LayerBaselineError("Eligible layers and sink layers must both be non-empty")
    doses = [float(alpha) for alpha in alphas if float(alpha) != 1.0]
    if not doses:
        raise LayerBaselineError(
            "alpha=1.0 is the identity condition and is excluded; supply at least one "
            "non-identity alpha"
        )

    if baseline is None:
        baseline = baseline_per_layer_sink(
            model, items, ordered_sink_layers,
            device=device, target_pos=target_pos, metric_dtype=metric_dtype,
        )
    baseline_mean = baseline.mean(axis=0)
    aggregate_baseline = float(baseline_mean.mean())

    input_batches = [
        torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
        for item in items
    ]

    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    # ``-1`` is the reserved pseudo-layer id for "every eligible MLP layer at once": the
    # maximal intervention this unit type admits, and the reference point for whether the sink
    # is MLP-mediated at all. It is not an upper bound on what a sparse set can achieve.
    sweep: list[int] = [*layers, ALL_LAYERS]
    total = len(sweep) * len(doses)
    step = 0
    for layer in sweep:
        if layer == ALL_LAYERS:
            whole_layer = NeuronSet(
                {
                    eligible: tuple(range(adapter.mlp_width(eligible)))
                    for eligible in layers
                },
                source="all_eligible_layer_attenuation",
            )
            earliest = min(layers)
        else:
            whole_layer = NeuronSet(
                {layer: tuple(range(adapter.mlp_width(layer)))},
                source="layer_attenuation",
            )
            earliest = layer
        for alpha in doses:
            totals = np.zeros(len(ordered_sink_layers), dtype=np.float64)
            with suppress_neurons(adapter, whole_layer, alpha):
                for input_ids in input_batches:
                    totals += _sink_vector(
                        model, input_ids, ordered_sink_layers,
                        target_pos=target_pos, metric_dtype=metric_dtype,
                    )
            attenuated = totals / len(items)
            for position, target in enumerate(ordered_sink_layers):
                delta = float(baseline_mean[position] - attenuated[position])
                rows.append({
                    "mlp_layer": layer,
                    "alpha": alpha,
                    "target_sink_layer": target,
                    "baseline_sink": float(baseline_mean[position]),
                    "attenuated_sink": float(attenuated[position]),
                    "delta_sink": delta,
                    "relative_sink_reduction": (
                        delta / max(float(baseline_mean[position]), 1e-12)
                    ),
                    # Causal ordering: an MLP at layer l cannot move attention at j <= l.
                    # Those rows must measure exactly zero, and are kept as a live check. For
                    # the all-layers condition the relevant l is the earliest eligible layer.
                    "reachable": bool(target > earliest),
                    "n_examples": len(items),
                })
            aggregate_attenuated = float(attenuated.mean())
            aggregates.append({
                "mlp_layer": layer,
                "alpha": alpha,
                "aggregate_baseline_sink": aggregate_baseline,
                "aggregate_attenuated_sink": aggregate_attenuated,
                "aggregate_relative_sink_reduction": (
                    (aggregate_baseline - aggregate_attenuated)
                    / max(aggregate_baseline, 1e-12)
                ),
                "reachable_sink_layers": [
                    int(target) for target in ordered_sink_layers if target > earliest
                ],
                "reachable_metric_weight": (
                    sum(
                        float(value)
                        for target, value in zip(ordered_sink_layers, baseline_mean)
                        if target > earliest
                    ) / max(float(baseline_mean.sum()), 1e-12)
                ),
            })
            step += 1
            if progress is not None:
                progress(step, total)

    unreachable = [
        row for row in rows
        if not row["reachable"] and row["delta_sink"] != 0.0
    ]
    document = {
        "schema": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "discovery",
        "n_examples": len(items),
        "alphas": doses,
        "eligible_mlp_layers": layers,
        "sink_layers": list(ordered_sink_layers),
        "baseline_sink_per_layer": [float(value) for value in baseline_mean],
        "aggregate_baseline_sink": aggregate_baseline,
        "metric_dtype": "float32" if metric_dtype is not None else "model_dtype",
        "rows": rows,
        "aggregates": aggregates,
        "all_layers_condition_id": ALL_LAYERS,
        "maximal_intervention": [
            entry for entry in aggregates if entry["mlp_layer"] == ALL_LAYERS
        ],
        "causal_order_violations": len(unreachable),
        "causal_order_pass": not unreachable,
        "is_causal_evidence": False,
        "note": (
            "These are maximal interventions, not upper bounds: suppression is not monotone "
            "in the sink, so a sparse subset can exceed a whole-layer or all-layer effect. "
            "The one rigorous bound is reachable_metric_weight, which follows from causal "
            "ordering alone. Nothing here claims that any particular neuron matters."
        ),
    }
    document["layer_baseline_sha256"] = canonical_sha256(
        {k: v for k, v in document.items() if k != "layer_baseline_sha256"}
    )
    return document


def ceiling_by_sink_layer(document: Mapping[str, Any], alpha: float = 0.0) -> dict[str, Any]:
    """Best *single-layer* whole-MLP effect on each sink layer, at one alpha.

    Read this precisely. Attenuating all of MLP layer ``l`` bounds what any neuron subset **of
    layer l** can do, so the maximum over ``l`` is the best any *single-layer* set can do. It
    is **not** an upper bound on a set spanning several MLP layers, which can exceed it: in a
    dev check on GPT-2-small a 15-neuron set drawn from layers 6-9 reduced the layer-7 sink by
    21% while the best single-layer whole-MLP attenuation reached 8.5%.

    The all-eligible-layers condition recorded by :func:`layer_attenuation_ceiling` is the
    maximal intervention of this unit type, which is the right reference point for "is the
    sink MLP-mediated at all" -- but it is not an upper bound either, because suppression is
    not monotone in the sink. The only rigorous bound is ``reachable_metric_weight``.
    """

    best: dict[int, dict[str, Any]] = {}
    for row in document["rows"]:
        if float(row["alpha"]) != float(alpha) or not row["reachable"]:
            continue
        target = int(row["target_sink_layer"])
        current = best.get(target)
        if current is None or row["relative_sink_reduction"] > current["best_rsr"]:
            best[target] = {
                "target_sink_layer": target,
                "best_rsr": float(row["relative_sink_reduction"]),
                "best_mlp_layer": int(row["mlp_layer"]),
                "baseline_sink": float(row["baseline_sink"]),
            }
    return {
        "alpha": float(alpha),
        "per_sink_layer": [best[key] for key in sorted(best)],
    }
