"""Baseline sink map and the registered sink-heavy layer/head selection rule.

The per-layer/per-head statistic is the same received-attention quantity the pinned
upstream metric averages, decomposed rather than redefined. Upstream
``intervention_analysis_legacy.compute_bos_attention_metric`` computes, per layer,
``attn[:, seq_len // 2:, target_pos].mean()`` over heads and second-half queries jointly,
then means those values over the layer band. Because every head sees the same number of
query positions, that per-layer scalar is exactly the mean over heads of this module's
per-head values, so :func:`sink_scalar_from_map` reproduces the upstream scalar on the
same band. ``scripts/map_sink_layers.py`` asserts that equality per example rather than
assuming it.

The selection rule is registered in ``docs/00_MASTER_EXPERIMENT_DESIGN.md``
("Sink-heavy attention scope") and must not be changed after seeing a map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


#: ``configs/experiment_plan.yaml``: sink_metric.preflight_min_sink / absolute_floor.
REGISTERED_SINK_FLOOR = 0.15
REGISTERED_TARGET_POSITION = 0
REGISTERED_QUERY_RULE = "second_half"

#: The design says "top quartile" without fixing the rounding, so it is fixed here and
#: recorded in every frozen scope: ceil(n / 4), at least one layer. For GPT-2-small's 12
#: layers this is 3; for GPT-2-medium's 24 it is 6.
QUARTILE_RULE = "ceil(n/4), minimum 1, ties broken by ascending index"


class SinkPreflightError(RuntimeError):
    """Raised when no layer clears the registered absolute sink floor."""


def top_quartile_size(n: int) -> int:
    if n < 1:
        raise ValueError(f"Need at least one element, got {n}")
    return max(1, math.ceil(n / 4))


def _as_layer_head_seq_seq(attentions: Sequence[Any]) -> list[torch.Tensor]:
    """Normalise a list of attention tensors to ``[num_heads, seq, seq]`` per layer."""

    if len(attentions) == 0:
        raise ValueError("No attention tensors were provided")
    normalised: list[torch.Tensor] = []
    for index, attention in enumerate(attentions):
        if not isinstance(attention, torch.Tensor):
            raise TypeError(
                f"Layer {index} attention must be a torch.Tensor, got "
                f"{type(attention).__name__}"
            )
        if attention.ndim == 4:
            if attention.shape[0] != 1:
                raise ValueError(
                    f"Layer {index} has batch size {attention.shape[0]}; the sink map is "
                    "computed one example at a time"
                )
            attention = attention[0]
        if attention.ndim != 3:
            raise ValueError(
                f"Layer {index} attention must be [heads, seq, seq] or [1, heads, seq, "
                f"seq], got {tuple(attention.shape)}"
            )
        if attention.shape[-1] != attention.shape[-2]:
            raise ValueError(
                f"Layer {index} attention is not square: {tuple(attention.shape)}"
            )
        normalised.append(attention)
    shapes = {tuple(a.shape) for a in normalised}
    if len(shapes) != 1:
        raise ValueError(f"Attention tensors have inconsistent shapes: {sorted(shapes)}")
    return normalised


def per_layer_head_position0_attention(
    attentions: Sequence[Any], *, target_pos: int = REGISTERED_TARGET_POSITION
) -> np.ndarray:
    """Mean attention received by ``target_pos`` from second-half queries.

    Returns a ``[num_layers, num_heads]`` float64 array for one example.
    """

    layers = _as_layer_head_seq_seq(attentions)
    seq_len = int(layers[0].shape[-1])
    if not 0 <= target_pos < seq_len:
        raise ValueError(f"target_pos {target_pos} outside sequence length {seq_len}")
    second_half_start = seq_len // 2
    if second_half_start >= seq_len:
        raise ValueError(f"Sequence length {seq_len} has no second-half query positions")
    rows = [
        attention[:, second_half_start:, target_pos]
        .mean(dim=1)
        .detach()
        .to(dtype=torch.float64, device="cpu")
        .numpy()
        for attention in layers
    ]
    return np.stack(rows, axis=0)


def sink_scalar_from_map(
    layer_head_mean: np.ndarray, layers: Sequence[int] | None = None
) -> float:
    """Mean over the given layers and all heads, reproducing the upstream scalar."""

    array = np.asarray(layer_head_mean, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a [layers, heads] map, got {array.shape}")
    selected = range(array.shape[0]) if layers is None else layers
    indices = list(selected)
    if not indices:
        raise ValueError("At least one layer must be selected")
    for layer in indices:
        if not 0 <= layer < array.shape[0]:
            raise IndexError(
                f"Layer {layer} outside the zero-indexed range [0, {array.shape[0]})"
            )
    # Mean per layer over heads first, then over layers -- upstream's aggregation order.
    return float(np.mean([array[layer].mean() for layer in indices]))


def layer_scores(layer_head_mean: np.ndarray) -> np.ndarray:
    """Per-layer sink score: mean over that layer's heads."""

    array = np.asarray(layer_head_mean, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a [layers, heads] map, got {array.shape}")
    return array.mean(axis=1)


@dataclass(frozen=True)
class SinkScope:
    """The frozen sink-heavy layer/head scope and its derived attribution eligibility."""

    sink_layers: tuple[int, ...]
    sink_heads: Mapping[int, tuple[int, ...]]
    eligible_mlp_layers: tuple[int, ...]
    future_sink_layers: Mapping[int, tuple[int, ...]]
    num_layers: int
    num_heads: int
    floor: float
    quartile_size: int
    rule_applied: str
    n_above_floor: int
    fallback_incomplete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink_layers": list(self.sink_layers),
            "sink_heads": {str(k): list(v) for k, v in sorted(self.sink_heads.items())},
            "eligible_mlp_layers": list(self.eligible_mlp_layers),
            "future_sink_layers": {
                str(k): list(v) for k, v in sorted(self.future_sink_layers.items())
            },
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "absolute_floor": self.floor,
            "quartile_size": self.quartile_size,
            "quartile_rule": QUARTILE_RULE,
            "rule_applied": self.rule_applied,
            "n_layers_above_floor": self.n_above_floor,
            "fallback_incomplete": self.fallback_incomplete,
            "layer_indexing": "zero_indexed",
            "target_position": REGISTERED_TARGET_POSITION,
            "query_rule": REGISTERED_QUERY_RULE,
        }


def select_sink_heavy_layers(
    scores: Sequence[float], *, floor: float = REGISTERED_SINK_FLOOR
) -> tuple[tuple[int, ...], str, int, bool]:
    """Apply the registered rule; returns ``(layers, rule_applied, n_above_floor, partial)``.

    Registered rule (``docs/00_MASTER_EXPERIMENT_DESIGN.md``): a sink-heavy layer is in the
    top quartile of the model's layers **and** scores at least ``floor``. If fewer than two
    layers satisfy both, use the top two layers above ``floor``. If no layer exceeds
    ``floor``, the checkpoint fails sink preflight.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Expected a 1-D non-empty score vector, got shape {values.shape}")
    num_layers = int(values.size)
    order = sorted(range(num_layers), key=lambda layer: (-values[layer], layer))
    quartile = top_quartile_size(num_layers)

    above_floor = [layer for layer in order if values[layer] >= floor]
    if not above_floor:
        raise SinkPreflightError(
            f"No layer reaches the registered sink floor {floor}; the highest layer score "
            f"is {values.max():.6f}. This checkpoint fails sink preflight and must not be "
            "used for neuron claims. Do not lower the floor to obtain a scope."
        )

    both = [layer for layer in order[:quartile] if values[layer] >= floor]
    if len(both) >= 2:
        return tuple(sorted(both)), "top_quartile_and_floor", len(above_floor), False
    selected = above_floor[:2]
    return (
        tuple(sorted(selected)),
        "fallback_top_two_above_floor",
        len(above_floor),
        len(selected) < 2,
    )


def select_sink_heavy_heads(
    layer_head_mean: np.ndarray, sink_layers: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    """Top quartile of heads, by the same statistic, within each sink-heavy layer."""

    array = np.asarray(layer_head_mean, dtype=np.float64)
    num_heads = int(array.shape[1])
    quartile = top_quartile_size(num_heads)
    heads: dict[int, tuple[int, ...]] = {}
    for layer in sink_layers:
        row = array[layer]
        order = sorted(range(num_heads), key=lambda head: (-row[head], head))
        heads[int(layer)] = tuple(sorted(order[:quartile]))
    return heads


def eligible_mlp_layers(
    sink_layers: Sequence[int], num_layers: int
) -> tuple[int, ...]:
    """MLP layers that can be attributed: those with at least one sink layer after them.

    A decoder block computes attention before its MLP, so layer ``l``'s MLP cannot cause
    the attention already produced at layer ``l`` or earlier (``AGENTS.md``, "Causal
    ordering constraint").
    """

    if not sink_layers:
        raise ValueError("sink_layers must not be empty")
    latest = max(int(layer) for layer in sink_layers)
    return tuple(layer for layer in range(num_layers) if layer < latest)


def future_sink_layers_by_mlp_layer(
    sink_layers: Sequence[int], num_layers: int
) -> dict[int, tuple[int, ...]]:
    """For each eligible MLP layer, the strictly later sink-heavy attention layers."""

    ordered = tuple(sorted(int(layer) for layer in sink_layers))
    return {
        layer: tuple(j for j in ordered if j > layer)
        for layer in eligible_mlp_layers(ordered, num_layers)
    }


def build_sink_scope(
    layer_head_mean: np.ndarray, *, floor: float = REGISTERED_SINK_FLOOR
) -> SinkScope:
    """Apply the full registered rule to a baseline map and freeze the resulting scope."""

    array = np.asarray(layer_head_mean, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a [layers, heads] map, got {array.shape}")
    num_layers, num_heads = array.shape
    scores = layer_scores(array)
    layers, rule_applied, n_above_floor, partial = select_sink_heavy_layers(
        scores, floor=floor
    )
    return SinkScope(
        sink_layers=layers,
        sink_heads=select_sink_heavy_heads(array, layers),
        eligible_mlp_layers=eligible_mlp_layers(layers, num_layers),
        future_sink_layers=future_sink_layers_by_mlp_layer(layers, num_layers),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        floor=float(floor),
        quartile_size=top_quartile_size(int(num_layers)),
        rule_applied=rule_applied,
        n_above_floor=int(n_above_floor),
        fallback_incomplete=bool(partial),
    )
