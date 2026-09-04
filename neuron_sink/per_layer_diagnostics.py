"""Pure helpers for the amendment-A006 Stage-C per-layer diagnostic."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


PER_LAYER_ROW_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "source_experiment_id",
    "model_id",
    "split",
    "example_id",
    "condition_id",
    "condition_order",
    "fraction_percent",
    "k",
    "alpha",
    "attention_layer",
    "sink_baseline",
    "sink_intervened",
    "delta_sink",
    "relative_sink_reduction",
    "baseline_valid",
    "intervention_valid",
    "valid_forward",
    "forward_runtime_seconds",
)

PER_LAYER_AGGREGATE_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "source_experiment_id",
    "model_id",
    "split",
    "condition_id",
    "condition_order",
    "fraction_percent",
    "k",
    "alpha",
    "attention_layer",
    "n_examples",
    "sink_baseline",
    "sink_intervened",
    "delta_sink",
    "relative_sink_reduction",
    "mean_per_example_relative_sink_reduction",
    "valid_forward_all",
    "runtime_seconds",
)

FIRST_ORDER_FIELDS: tuple[str, ...] = (
    "condition_id",
    "fraction_percent",
    "k",
    "alpha",
    "selected_signed_attr_sum",
    "causal_scope_weighted_signed_attr_sum",
    "predicted_delta_sink",
    "predicted_effect",
)


def per_layer_sink_scores(
    attentions: Sequence[torch.Tensor], sink_layers: Sequence[int]
) -> dict[int, float]:
    """Return the registered sink score independently for each absolute layer."""

    layers = tuple(int(layer) for layer in sink_layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("sink_layers must be non-empty and unique")
    scores: list[torch.Tensor] = []
    for layer in layers:
        if not 0 <= layer < len(attentions):
            raise IndexError(
                f"Layer {layer} outside zero-indexed attention range [0, {len(attentions)})"
            )
        attention = attentions[layer]
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(
                f"Layer {layer} attention must be [1, heads, seq, seq], got "
                f"{tuple(attention.shape)}"
            )
        seq_len = int(attention.shape[-1])
        if attention.shape[-2] != seq_len or seq_len < 2:
            raise ValueError(f"Layer {layer} attention has invalid shape {attention.shape}")
        scores.append(attention[0, :, seq_len // 2 :, 0].mean())
    values = torch.stack(scores).detach().to(device="cpu").tolist()
    return {layer: float(value) for layer, value in zip(layers, values)}


def aggregate_per_layer_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate paired rows using ratio-of-means RSR, matching the primary metric."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    key_fields = (
        "experiment_id",
        "source_experiment_id",
        "model_id",
        "split",
        "condition_id",
        "condition_order",
        "fraction_percent",
        "k",
        "alpha",
        "attention_layer",
    )
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)

    aggregates: list[dict[str, Any]] = []
    for key, group in groups.items():
        baseline = sum(float(row["sink_baseline"]) for row in group) / len(group)
        intervened = sum(float(row["sink_intervened"]) for row in group) / len(group)
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("Per-layer baseline sink must be positive and finite")
        rsrs = [
            (float(row["sink_baseline"]) - float(row["sink_intervened"]))
            / float(row["sink_baseline"])
            for row in group
        ]
        aggregate = dict(zip(key_fields, key))
        aggregate.update({
            "n_examples": len(group),
            "sink_baseline": baseline,
            "sink_intervened": intervened,
            "delta_sink": intervened - baseline,
            "relative_sink_reduction": 1.0 - intervened / baseline,
            "mean_per_example_relative_sink_reduction": sum(rsrs) / len(rsrs),
            "valid_forward_all": all(bool(row["valid_forward"]) for row in group),
            "runtime_seconds": sum(
                float(row["forward_runtime_seconds"]) for row in group
            ),
        })
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda row: (
            int(row["condition_order"]),
            -float(row["alpha"]),
            int(row["attention_layer"]),
        ),
    )


def first_order_predictions(
    attribution_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[Any],
    *,
    sink_layers: Sequence[int],
    seq_len: int,
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    """Compute amendment A006's no-forward aggregate first-order prediction."""

    sink_layer_set = set(int(layer) for layer in sink_layers)
    if not sink_layer_set or seq_len < 1:
        raise ValueError("A non-empty sink scope and positive sequence length are required")
    by_coordinate = {
        (int(row["layer"]), int(row["neuron"])): row for row in attribution_rows
    }
    output: list[dict[str, Any]] = []
    for condition in conditions:
        signed_sum = 0.0
        weighted_sum = 0.0
        selected = 0
        for layer, neurons in condition.neuron_set.by_layer.items():
            for neuron in neurons:
                row = by_coordinate[(int(layer), int(neuron))]
                value = float(row["mean_signed_attr"])
                targets = {
                    int(value)
                    for value in str(row["future_sink_layers"]).split("|")
                    if value != ""
                }
                if not targets or not targets.issubset(sink_layer_set):
                    raise ValueError("Attribution future scope is inconsistent with sink scope")
                signed_sum += value
                weighted_sum += value * len(targets) / len(sink_layer_set)
                selected += 1
        if selected != int(condition.k):
            raise ValueError(
                f"Condition {condition.condition_id} contains {selected}, expected {condition.k}"
            )
        for alpha in alphas:
            predicted = -(1.0 - float(alpha)) * float(seq_len) * weighted_sum
            output.append({
                "condition_id": condition.condition_id,
                "fraction_percent": float(condition.fraction_percent),
                "k": int(condition.k),
                "alpha": float(alpha),
                "selected_signed_attr_sum": signed_sum,
                "causal_scope_weighted_signed_attr_sum": weighted_sum,
                "predicted_delta_sink": predicted,
                "predicted_effect": (
                    "sink_reduction" if predicted < 0.0
                    else "sink_increase" if predicted > 0.0
                    else "identity"
                ),
            })
    return output
