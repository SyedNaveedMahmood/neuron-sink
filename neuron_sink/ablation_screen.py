"""Measured single-neuron ablation screening (Stage C3, amendment A007).

Activation-times-gradient is a first-order estimate of the sink around the neuron's *current*
activation. The registered intervention does not make a small perturbation: it multiplies the
coordinate by ``alpha``, and the primary condition is ``alpha=0``, an ablation all the way to
zero. Nothing guarantees the first-order score ranks neurons correctly that far out.

Stage C3 therefore uses the gradient score only to *shortlist*, and ranks the shortlist by the
effect actually measured under the real intervention, on discovery examples only.

Two properties make this cheap and exact:

1. one ablated forward yields the sink at **every** registered sink layer at once, so a
   candidate is screened against all of its reachable targets for the price of one forward;
2. the comparison is paired and deterministic -- the same examples, the same model, no
   sampling -- so a difference of any size is measured exactly rather than estimated. The only
   noise is generalisation to held-out text, which is what the later splits are for.

The screen measures *marginal* effects: one neuron at a time. A selected set's joint effect
can differ through redundancy or interaction, which is why
:func:`measure_joint_effect` exists and why the Stage-C3 report must record predicted-versus-
actual for the assembled set rather than assuming additivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from .corpus import NeutralCorpusItem
from .model_adapters import MLPModelAdapter
from .provenance import canonical_sha256
from .sink_metrics import REGISTERED_TARGET_POSITION, differentiable_sink_scores_per_layer
from .suppression import NeuronSet, suppress_neurons


SCHEMA_VERSION = "ablation_screen_c3_v1"
SCREEN_METHOD = "measured_single_neuron_ablation"
SCREEN_ALPHA = 0.0
METRIC_DTYPE = torch.float32

ROW_FIELDS: tuple[str, ...] = (
    "mlp_layer",
    "neuron",
    "target_sink_layer",
    "baseline_sink",
    "ablated_sink",
    "measured_delta_sink",
    "measured_rsr",
    "predicted_delta_sink",
    "n_examples",
)


class AblationScreenError(RuntimeError):
    """Raised when a screening request or artefact is invalid."""


@dataclass(frozen=True)
class ScreenTable:
    """Measured marginal ablation effects, one row per candidate and target sink layer."""

    rows: tuple[Mapping[str, Any], ...]
    sink_layers: tuple[int, ...]
    baseline_per_layer: tuple[float, ...]
    n_examples: int
    n_candidates: int
    alpha: float = SCREEN_ALPHA

    def sha256(self) -> str:
        return canonical_sha256([[row[name] for name in ROW_FIELDS] for row in self.rows])

    def candidates_for_target(self, target_sink_layer: int) -> list[Mapping[str, Any]]:
        """Rows attacking one sink layer, best measured reduction first.

        Only strictly positive measured reductions are returned. A candidate that raises the
        sink at its own target is exactly what Stage C selected by accident.
        """

        selected = [
            row for row in self.rows
            if int(row["target_sink_layer"]) == int(target_sink_layer)
            and float(row["measured_delta_sink"]) > 0.0
        ]
        selected.sort(
            key=lambda row: (
                -float(row["measured_delta_sink"]),
                int(row["mlp_layer"]),
                int(row["neuron"]),
            )
        )
        return selected


def _sink_vector(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    sink_layers: Sequence[int],
    *,
    target_pos: int,
    metric_dtype: torch.dtype | None,
) -> np.ndarray:
    """Per-sink-layer position-0 attention for one example, as float64 NumPy."""

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=True,
            use_cache=False,
        )
        scores = differentiable_sink_scores_per_layer(
            output.attentions, sink_layers, None,
            target_pos=target_pos, dtype=metric_dtype,
        )
        return scores.detach().to(torch.float64).cpu().numpy()


def baseline_per_layer_sink(
    model: torch.nn.Module,
    items: Sequence[NeutralCorpusItem],
    sink_layers: Sequence[int],
    *,
    device: torch.device,
    target_pos: int = REGISTERED_TARGET_POSITION,
    metric_dtype: torch.dtype | None = METRIC_DTYPE,
) -> np.ndarray:
    """``[n_examples, n_sink_layers]`` baseline sink, measured once and reused."""

    if not items:
        raise AblationScreenError("At least one example is required")
    rows = []
    for item in items:
        input_ids = torch.tensor(
            [list(item.input_ids)], dtype=torch.long, device=device
        )
        rows.append(_sink_vector(
            model, input_ids, sink_layers,
            target_pos=target_pos, metric_dtype=metric_dtype,
        ))
        del input_ids
    return np.stack(rows, axis=0)


def screen_neurons(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    items: Sequence[NeutralCorpusItem],
    candidates: Iterable[tuple[int, int]],
    sink_layers: Sequence[int],
    *,
    baseline: np.ndarray | None = None,
    predicted: Mapping[tuple[int, int, int], float] | None = None,
    alpha: float = SCREEN_ALPHA,
    device: torch.device | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
    metric_dtype: torch.dtype | None = METRIC_DTYPE,
    progress: Callable[[int, int], None] | None = None,
) -> ScreenTable:
    """Measure each candidate neuron's marginal effect on every reachable sink layer.

    ``candidates`` are ``(mlp_layer, neuron)`` pairs. One ablated forward per candidate per
    example yields the whole per-sink-layer vector, so a candidate is scored against all of
    its causally reachable targets at once. Targets at or before the candidate's own layer are
    skipped: an MLP at layer ``l`` cannot move attention already computed at layer ``j <= l``,
    and the measured delta there must be exactly zero.
    """

    if device is None:
        device = next(model.parameters()).device
    ordered_sink_layers = tuple(int(layer) for layer in sink_layers)
    if not ordered_sink_layers:
        raise AblationScreenError("At least one sink layer is required")
    unique = sorted({(int(layer), int(neuron)) for layer, neuron in candidates})
    if not unique:
        raise AblationScreenError("No candidates to screen")
    for layer, neuron in unique:
        adapter.validate_neuron(layer, neuron)

    if baseline is None:
        baseline = baseline_per_layer_sink(
            model, items, ordered_sink_layers,
            device=device, target_pos=target_pos, metric_dtype=metric_dtype,
        )
    if baseline.shape != (len(items), len(ordered_sink_layers)):
        raise AblationScreenError(
            f"baseline has shape {baseline.shape}, expected "
            f"{(len(items), len(ordered_sink_layers))}"
        )
    baseline_mean = baseline.mean(axis=0)

    input_batches = [
        torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
        for item in items
    ]

    rows: list[dict[str, Any]] = []
    for index, (layer, neuron) in enumerate(unique):
        neuron_set = NeuronSet({layer: (neuron,)}, source="screen")
        totals = np.zeros(len(ordered_sink_layers), dtype=np.float64)
        with suppress_neurons(adapter, neuron_set, alpha):
            for input_ids in input_batches:
                totals += _sink_vector(
                    model, input_ids, ordered_sink_layers,
                    target_pos=target_pos, metric_dtype=metric_dtype,
                )
        ablated_mean = totals / len(items)

        for position, target in enumerate(ordered_sink_layers):
            if target <= layer:
                # Causally unreachable; recorded as skipped rather than as a zero effect.
                continue
            delta = float(baseline_mean[position] - ablated_mean[position])
            rows.append({
                "mlp_layer": layer,
                "neuron": neuron,
                "target_sink_layer": target,
                "baseline_sink": float(baseline_mean[position]),
                "ablated_sink": float(ablated_mean[position]),
                "measured_delta_sink": delta,
                "measured_rsr": delta / max(float(baseline_mean[position]), 1e-12),
                "predicted_delta_sink": float(
                    (predicted or {}).get((layer, neuron, target), float("nan"))
                ),
                "n_examples": len(items),
            })
        if progress is not None:
            progress(index + 1, len(unique))

    return ScreenTable(
        rows=tuple(rows),
        sink_layers=ordered_sink_layers,
        baseline_per_layer=tuple(float(value) for value in baseline_mean),
        n_examples=len(items),
        n_candidates=len(unique),
        alpha=float(alpha),
    )


def measure_joint_effect(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    items: Sequence[NeutralCorpusItem],
    neuron_set: NeuronSet,
    sink_layers: Sequence[int],
    *,
    baseline: np.ndarray | None = None,
    alpha: float = SCREEN_ALPHA,
    device: torch.device | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
    metric_dtype: torch.dtype | None = METRIC_DTYPE,
) -> dict[str, Any]:
    """Measure a whole selected set at once, for the additivity diagnostic.

    Single-neuron screening ranks marginal effects. Comparing the joint measurement with the
    sum of the marginals is the honest way to report how much redundancy the set carries; it
    is a diagnostic, never a selection criterion.
    """

    if device is None:
        device = next(model.parameters()).device
    ordered_sink_layers = tuple(int(layer) for layer in sink_layers)
    if baseline is None:
        baseline = baseline_per_layer_sink(
            model, items, ordered_sink_layers,
            device=device, target_pos=target_pos, metric_dtype=metric_dtype,
        )
    baseline_mean = baseline.mean(axis=0)

    totals = np.zeros(len(ordered_sink_layers), dtype=np.float64)
    with suppress_neurons(adapter, neuron_set, alpha):
        for item in items:
            input_ids = torch.tensor(
                [list(item.input_ids)], dtype=torch.long, device=device
            )
            totals += _sink_vector(
                model, input_ids, ordered_sink_layers,
                target_pos=target_pos, metric_dtype=metric_dtype,
            )
            del input_ids
    ablated_mean = totals / len(items)

    aggregate_baseline = float(baseline_mean.mean())
    aggregate_ablated = float(ablated_mean.mean())
    return {
        "alpha": float(alpha),
        "n_examples": len(items),
        "sink_layers": list(ordered_sink_layers),
        "baseline_per_layer": [float(v) for v in baseline_mean],
        "ablated_per_layer": [float(v) for v in ablated_mean],
        "delta_per_layer": [
            float(b - a) for b, a in zip(baseline_mean, ablated_mean)
        ],
        "rsr_per_layer": [
            float((b - a) / max(float(b), 1e-12))
            for b, a in zip(baseline_mean, ablated_mean)
        ],
        "aggregate_baseline_sink": aggregate_baseline,
        "aggregate_ablated_sink": aggregate_ablated,
        "aggregate_relative_sink_reduction": (
            (aggregate_baseline - aggregate_ablated) / max(aggregate_baseline, 1e-12)
        ),
    }
