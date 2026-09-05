"""Per-target-layer future-sink attribution (Stage C3, amendment A007).

Stage C and Stage C2 differentiate a single scalar, the mean sink over every reachable
sink-heavy attention layer ``j > l``. That aggregate hides which target a neuron actually
serves, and the Stage-C per-layer decomposition showed why it matters: the selected neurons
sat almost entirely in the last eligible MLP layer, so three of the seven graded sink layers
were computed before those neurons fired and moved by *exactly* zero.

This module keeps the terms separate. For eligible MLP layer ``l`` and each reachable sink
layer ``j > l`` it accumulates

``mean over examples and token positions of  a(l,n) * dS_j/da(l,n)``

and the corresponding absolute mean. The signed mean is the primary quantity: for suppression
by ``alpha < 1`` the first-order change is ``delta S_j ~= -(1 - alpha) * (a * dS_j/da)``, so a
*positive* signed score predicts a sink reduction and a negative one predicts the increase
Stage C actually measured.

Two numerical differences from :mod:`neuron_sink.attribution`, both registered by A007:

1. the ``activation * gradient`` product and the per-token reduction are computed in float32
   even when the registered forward is bfloat16 -- :mod:`neuron_sink.evaluation` already does
   this for CE/KL, and the signed score needs it more, because it is a heavily cancelling sum;
2. the sink metric itself is reduced in float32 (``differentiable_sink_scores_per_layer(...,
   dtype=torch.float32)``).

The *backward pass* still runs in the model's registered dtype. Upcasting the metric and the
product improves where precision is cheapest to recover; it does not turn a bfloat16 backward
into a float32 one, and the report must not claim otherwise.

Attribution remains a ranking heuristic. Stage C3 additionally reranks the shortlist by
directly measured ablation (:mod:`neuron_sink.ablation_screen`), because a first-order score
around the current activation is not a reliable predictor of ablating that activation to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .attribution import (
    AttributionError,
    _require_discovery_items,
    capture_mlp_projection_input,
    require_future_targets,
)
from .corpus import NeutralCorpus, assert_no_downstream_source, require_discovery_split
from .model_adapters import MLPModelAdapter
from .provenance import canonical_sha256, read_json
from .sink_metrics import (
    REGISTERED_TARGET_POSITION,
    FrozenSinkScope,
    differentiable_sink_scores_per_layer,
)


SCHEMA_VERSION = "neuron_attribution_c3_v1"

ATTRIBUTION_METHOD = "per_target_signed_activation_x_sink_gradient"
ATTRIBUTION_OBJECTIVE = "causal_order_per_target_sink"
ATTRIBUTION_AGGREGATION = "mean_over_examples_and_tokens"
TOKEN_POSITION_RULE = "all_positions"
RANKING_SCORE = "mean_signed_attr"
SIGN_REQUIREMENT = "strictly_positive"

#: Amendment A007: metric and product arithmetic is float32 regardless of the forward dtype.
METRIC_DTYPE = torch.float32
METRIC_DTYPE_NAME = "float32"

#: Long form: one row per (eligible MLP layer, neuron, reachable sink layer).
ROW_FIELDS: tuple[str, ...] = (
    "mlp_layer",
    "neuron",
    "target_sink_layer",
    "mean_abs_activation",
    "mean_signed_attr",
    "mean_abs_attr",
    "n_examples",
    "n_tokens",
)

INT_ROW_FIELDS = frozenset({
    "mlp_layer", "neuron", "target_sink_layer", "n_examples", "n_tokens",
})
FLOAT_ROW_FIELDS = frozenset({
    "mean_abs_activation", "mean_signed_attr", "mean_abs_attr",
})


class AttributionC3Error(RuntimeError):
    """Raised when a Stage-C3 attribution request or artefact is invalid."""


@dataclass(frozen=True)
class PerTargetExampleAttribution:
    """One example's activation and one gradient per reachable target sink layer."""

    layer: int
    targets: tuple[int, ...]
    sink_per_target: tuple[float, ...]
    activation: torch.Tensor  # detached [sequence, width]
    gradients: torch.Tensor  # [n_targets, sequence, width]


def score_example_per_target(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    input_ids: torch.Tensor,
    layer: int,
    targets: Sequence[int],
    *,
    heads: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
    metric_dtype: torch.dtype | None = METRIC_DTYPE,
) -> PerTargetExampleAttribution:
    """One forward and ``len(targets)`` backwards over a single retained graph.

    The graph starts at layer ``l``'s projection input, because the capture hook hands the
    model a detached leaf, so retaining it across the per-target backwards is cheap: it holds
    only the attention and residual tensors from layer ``l`` onward, a few megabytes at the
    registered sequence length.
    """

    ordered = require_future_targets(layer, targets)
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise AttributionC3Error(
            "Attribution runs at batch size 1 (docs/04_HARDWARE_RUNBOOK.md); got input_ids "
            f"of shape {tuple(input_ids.shape)}"
        )

    with capture_mlp_projection_input(adapter, layer) as captured:
        with torch.enable_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=True,
                use_cache=False,
            )
            activation = captured.get("activation")
            if activation is None:
                raise AttributionC3Error(
                    f"{adapter.get_mlp_intermediate_path(layer)} was never reached; the "
                    "capture hook did not fire"
                )
            scores = differentiable_sink_scores_per_layer(
                output.attentions, ordered, heads,
                target_pos=target_pos, dtype=metric_dtype,
            )
            if not scores.requires_grad:
                raise AttributionC3Error(
                    f"S_j for MLP layer {layer} does not depend on the captured activation. "
                    "The forward most likely ran under torch.inference_mode(), which blocks "
                    "autograd and cannot be re-enabled from inside it."
                )
            gradients = []
            last = len(ordered) - 1
            for index in range(len(ordered)):
                (grad,) = torch.autograd.grad(
                    scores[index], activation, retain_graph=index < last
                )
                gradients.append(grad[0])

    return PerTargetExampleAttribution(
        layer=int(layer),
        targets=ordered,
        sink_per_target=tuple(float(value) for value in scores.detach()),
        activation=activation.detach()[0],
        gradients=torch.stack(gradients, dim=0),
    )


@dataclass(frozen=True)
class TargetAttribution:
    """Accumulated per-neuron scores for one ``(MLP layer, target sink layer)`` pair."""

    layer: int
    target_sink_layer: int
    width: int
    n_examples: int
    n_tokens: int
    mean_abs_activation: np.ndarray
    mean_signed_attr: np.ndarray
    mean_abs_attr: np.ndarray
    mean_sink_target: float
    max_abs_gradient: float
    nonfinite_values: int
    zero_gradient_examples: int

    def diagnostics(self) -> dict[str, Any]:
        positive = int((self.mean_signed_attr > 0.0).sum())
        order = np.argsort(-self.mean_signed_attr, kind="stable")
        return {
            "mlp_layer": self.layer,
            "target_sink_layer": self.target_sink_layer,
            "width": self.width,
            "n_examples": self.n_examples,
            "n_tokens": self.n_tokens,
            "mean_sink_target": self.mean_sink_target,
            "max_abs_gradient": self.max_abs_gradient,
            "nonfinite_values": self.nonfinite_values,
            "zero_gradient_examples": self.zero_gradient_examples,
            "positive_signed_count": positive,
            "positive_signed_fraction": positive / self.width if self.width else 0.0,
            "mean_signed_attr_max": float(self.mean_signed_attr.max()),
            "mean_abs_attr_max": float(self.mean_abs_attr.max()),
            # Net signed influence divided by total absolute influence. A low value means
            # the layer's strongest neurons cancel each other under the direction-blind
            # ranking that Stage C used.
            "signed_coherence": (
                float(self.mean_signed_attr.sum() / self.mean_abs_attr.sum())
                if float(self.mean_abs_attr.sum()) > 0.0 else 0.0
            ),
            "top_neuron": int(order[0]),
            "top_neuron_mean_signed_attr": float(self.mean_signed_attr[order[0]]),
        }


@dataclass(frozen=True)
class PerTargetAttributionResult:
    """The full Stage-C3 discovery ranking input, before any selection."""

    pairs: tuple[TargetAttribution, ...]
    example_ids: tuple[str, ...]
    corpus_id: str
    corpus_manifest_sha256: str
    split: str
    seq_len: int
    target_pos: int
    metric_dtype: str = METRIC_DTYPE_NAME
    checks: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.example_ids)

    @property
    def n_rows(self) -> int:
        return sum(pair.width for pair in self.pairs)

    @property
    def eligible_mlp_layers(self) -> tuple[int, ...]:
        return tuple(sorted({pair.layer for pair in self.pairs}))

    @property
    def target_sink_layers(self) -> tuple[int, ...]:
        return tuple(sorted({pair.target_sink_layer for pair in self.pairs}))


def rank_neurons_per_target(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    corpus: NeutralCorpus,
    future_sink_layers: Mapping[int, Sequence[int]],
    *,
    split: str = "discovery",
    smoke: bool = False,
    max_examples: int | None = None,
    heads: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
    metric_dtype: torch.dtype | None = METRIC_DTYPE,
    device: torch.device | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> PerTargetAttributionResult:
    """Rank every ``(MLP layer, neuron, target sink layer)`` triple on discovery only.

    The anti-leakage guards are the same ones the registered ranking API uses: the split
    argument is rejected unless it is ``discovery``, and every selected item's own frozen
    split role is checked, so a validation or test manifest cannot reach a gradient.
    """

    require_discovery_split(split)
    assert_no_downstream_source(
        str(corpus.source.get("dataset_id", "")), str(corpus.source.get("corpus_id", ""))
    )
    selected = list(corpus.items_for(split, smoke=smoke))
    if max_examples is not None:
        selected = selected[:max_examples]
    if not selected:
        raise AttributionC3Error(f"Split {split!r} is empty; nothing to rank")
    items = _require_discovery_items(selected, split)

    seq_lengths = {item.n_tokens for item in items}
    if len(seq_lengths) != 1:
        raise AttributionC3Error(
            f"Attribution expects one frozen sequence length, got {sorted(seq_lengths)}"
        )
    seq_len = seq_lengths.pop()

    layers = sorted(int(layer) for layer in future_sink_layers)
    if not layers:
        raise AttributionC3Error("No eligible MLP layers were provided")
    targets_by_layer = {
        layer: require_future_targets(layer, future_sink_layers[layer]) for layer in layers
    }

    if device is None:
        device = next(model.parameters()).device
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AttributionC3Error(
            f"{len(trainable)} model parameters still require grad (first: {trainable[0]}). "
            "Call model.requires_grad_(False) before attribution."
        )

    pairs: list[TargetAttribution] = []
    for layer in layers:
        targets = targets_by_layer[layer]
        width = adapter.mlp_width(layer)
        n_targets = len(targets)
        abs_activation = np.zeros(width, dtype=np.float64)
        signed_attr = np.zeros((n_targets, width), dtype=np.float64)
        abs_attr = np.zeros((n_targets, width), dtype=np.float64)
        sink_totals = np.zeros(n_targets, dtype=np.float64)
        max_abs_gradient = np.zeros(n_targets, dtype=np.float64)
        nonfinite = np.zeros(n_targets, dtype=np.int64)
        zero_gradient = np.zeros(n_targets, dtype=np.int64)

        for index, item in enumerate(items):
            input_ids = torch.tensor(
                [list(item.input_ids)], dtype=torch.long, device=device
            )
            scored = score_example_per_target(
                model, adapter, input_ids, layer, targets,
                heads=heads, target_pos=target_pos, metric_dtype=metric_dtype,
            )
            activation = scored.activation.to(torch.float32)
            gradients = scored.gradients.to(torch.float32)
            attribution = activation.unsqueeze(0) * gradients

            abs_activation += (
                activation.abs().to(torch.float64).sum(dim=0).cpu().numpy()
            )
            signed_attr += attribution.to(torch.float64).sum(dim=1).cpu().numpy()
            abs_attr += attribution.abs().to(torch.float64).sum(dim=1).cpu().numpy()
            sink_totals += np.asarray(scored.sink_per_target, dtype=np.float64)

            finite = torch.isfinite(gradients)
            nonfinite += (~finite).sum(dim=(1, 2)).cpu().numpy().astype(np.int64)
            magnitudes = gradients.abs().amax(dim=(1, 2)).cpu().numpy()
            max_abs_gradient = np.maximum(max_abs_gradient, magnitudes)
            zero_gradient += (magnitudes == 0.0).astype(np.int64)

            del scored, activation, gradients, attribution, input_ids
            if progress is not None:
                progress(layer, index + 1, len(items))

        n_tokens = len(items) * seq_len
        for position, target in enumerate(targets):
            pairs.append(TargetAttribution(
                layer=layer,
                target_sink_layer=int(target),
                width=width,
                n_examples=len(items),
                n_tokens=n_tokens,
                mean_abs_activation=abs_activation / n_tokens,
                mean_signed_attr=signed_attr[position] / n_tokens,
                mean_abs_attr=abs_attr[position] / n_tokens,
                mean_sink_target=float(sink_totals[position] / len(items)),
                max_abs_gradient=float(max_abs_gradient[position]),
                nonfinite_values=int(nonfinite[position]),
                zero_gradient_examples=int(zero_gradient[position]),
            ))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return PerTargetAttributionResult(
        pairs=tuple(pairs),
        example_ids=tuple(item.item_id for item in items),
        corpus_id=corpus.corpus_id,
        corpus_manifest_sha256=corpus.manifest_sha256,
        split=split,
        seq_len=seq_len,
        target_pos=int(target_pos),
        metric_dtype=(
            METRIC_DTYPE_NAME if metric_dtype is not None else "model_dtype"
        ),
    )


def attribution_rows(result: PerTargetAttributionResult) -> list[dict[str, Any]]:
    """Long-form rows, emitted in ``(mlp_layer, target_sink_layer, neuron)`` order."""

    rows: list[dict[str, Any]] = []
    for pair in sorted(result.pairs, key=lambda p: (p.layer, p.target_sink_layer)):
        for neuron in range(pair.width):
            rows.append({
                "mlp_layer": pair.layer,
                "neuron": neuron,
                "target_sink_layer": pair.target_sink_layer,
                "mean_abs_activation": float(pair.mean_abs_activation[neuron]),
                "mean_signed_attr": float(pair.mean_signed_attr[neuron]),
                "mean_abs_attr": float(pair.mean_abs_attr[neuron]),
                "n_examples": pair.n_examples,
                "n_tokens": pair.n_tokens,
            })
    return rows


def attribution_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the emitted rows so a rerun can be proved identical."""

    return canonical_sha256([[row[name] for name in ROW_FIELDS] for row in rows])


@dataclass(frozen=True)
class FrozenPerTargetRanking:
    """A verified Stage-C3 long-form ranking, indexed for per-target selection."""

    rows: tuple[Mapping[str, Any], ...]
    eligible_mlp_layers: tuple[int, ...]
    target_sink_layers: tuple[int, ...]
    mlp_width: Mapping[int, int]
    attribution_sha256: str
    corpus_manifest_sha256: str
    sink_scope_sha256: str
    model_id: str
    model_revision: str
    metadata: Mapping[str, Any]

    @property
    def pool_size(self) -> int:
        """Distinct ``(layer, neuron)`` pairs -- the eligible neuron pool, not the row count."""

        return sum(self.mlp_width[layer] for layer in self.eligible_mlp_layers)

    def candidates_for_target(self, target_sink_layer: int) -> list[Mapping[str, Any]]:
        """Rows attacking one sink layer, ordered by descending signed score.

        Only strictly positive scores are returned: a non-positive first-order score predicts
        that suppression leaves the sink unchanged or raises it, which is the Stage-C failure
        this experiment exists to avoid.
        """

        selected = [
            row for row in self.rows
            if int(row["target_sink_layer"]) == int(target_sink_layer)
            and float(row[RANKING_SCORE]) > 0.0
        ]
        selected.sort(
            key=lambda row: (
                -float(row[RANKING_SCORE]), int(row["mlp_layer"]), int(row["neuron"])
            )
        )
        return selected


def _typed_row(row: Mapping[str, str]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    for name in ROW_FIELDS:
        if name not in row:
            raise AttributionC3Error(f"Attribution row is missing field {name!r}")
        value = row[name]
        if name in INT_ROW_FIELDS:
            typed[name] = int(value)
        elif name in FLOAT_ROW_FIELDS:
            typed[name] = float(value)
        else:  # pragma: no cover - defensive; ROW_FIELDS is fully partitioned
            raise AttributionC3Error(f"Field {name!r} has no declared type")
    return typed


def load_frozen_per_target_attribution(
    csv_path: Path | str,
    metadata_path: Path | str,
    *,
    scope: FrozenSinkScope,
    expected_corpus_manifest_sha256: str,
) -> FrozenPerTargetRanking:
    """Load and re-verify a Stage-C3 long-form ranking.

    A separate loader from :func:`neuron_sink.selection.load_frozen_attribution` is required,
    not preferred: that one hard-codes ``ranking_score == "mean_abs_attr"`` and re-derives a
    global ``rank_abs`` column from the absolute order, neither of which exists here.
    """

    import csv as _csv

    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    for path in (csv_path, metadata_path):
        if not path.is_file():
            raise AttributionC3Error(f"Stage-C3 attribution artefact not found: {path}")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, Mapping):
        raise AttributionC3Error(f"{metadata_path} does not contain a JSON object")
    if metadata.get("schema") != SCHEMA_VERSION:
        raise AttributionC3Error(
            f"{metadata_path} has schema {metadata.get('schema')!r}, expected "
            f"{SCHEMA_VERSION!r}"
        )
    if metadata.get("ranking_score") != RANKING_SCORE:
        raise AttributionC3Error(
            f"{metadata_path} ranks by {metadata.get('ranking_score')!r}, expected "
            f"{RANKING_SCORE!r}"
        )
    if metadata.get("corpus_manifest_sha256") != expected_corpus_manifest_sha256:
        raise AttributionC3Error(
            "Stage-C3 attribution was computed on corpus "
            f"{metadata.get('corpus_manifest_sha256')}, but the loaded corpus is "
            f"{expected_corpus_manifest_sha256}"
        )
    if metadata.get("sink_scope_sha256") != scope.sink_scope_sha256:
        raise AttributionC3Error(
            "Stage-C3 attribution was computed under sink scope "
            f"{metadata.get('sink_scope_sha256')}, but the loaded scope is "
            f"{scope.sink_scope_sha256}"
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise AttributionC3Error(
                f"{csv_path} header {tuple(reader.fieldnames or ())} != {ROW_FIELDS}"
            )
        rows = [_typed_row(row) for row in reader]

    recomputed = attribution_sha256(rows)
    if recomputed != metadata.get("attribution_sha256"):
        raise AttributionC3Error(
            f"Stage-C3 attribution hash mismatch: stored "
            f"{metadata.get('attribution_sha256')} != recomputed {recomputed}"
        )

    eligible = tuple(int(layer) for layer in metadata["eligible_mlp_layers"])
    if eligible != scope.eligible_mlp_layers:
        raise AttributionC3Error(
            f"Attribution eligible layers {list(eligible)} != frozen scope "
            f"{list(scope.eligible_mlp_layers)}"
        )
    widths = {int(k): int(v) for k, v in metadata["mlp_width"].items()}

    expected_rows = sum(
        widths[layer] * len(scope.targets_for(layer)) for layer in eligible
    )
    if len(rows) != expected_rows:
        raise AttributionC3Error(
            f"{csv_path} holds {len(rows)} rows, expected {expected_rows}"
        )
    for row in rows:
        layer = int(row["mlp_layer"])
        target = int(row["target_sink_layer"])
        if layer not in widths:
            raise AttributionC3Error(f"Row references ineligible MLP layer {layer}")
        if target not in scope.targets_for(layer):
            raise AttributionC3Error(
                f"MLP layer {layer} row targets sink layer {target}, which is not one of "
                f"its frozen future targets {list(scope.targets_for(layer))}"
            )
        if not 0 <= int(row["neuron"]) < widths[layer]:
            raise AttributionC3Error(
                f"Neuron {row['neuron']} outside layer {layer} range [0, {widths[layer]})"
            )
        for name in FLOAT_ROW_FIELDS:
            if not np.isfinite(row[name]):
                raise AttributionC3Error(f"Non-finite {name} at {row}")

    return FrozenPerTargetRanking(
        rows=tuple(rows),
        eligible_mlp_layers=eligible,
        target_sink_layers=tuple(sorted({int(r["target_sink_layer"]) for r in rows})),
        mlp_width=widths,
        attribution_sha256=str(metadata["attribution_sha256"]),
        corpus_manifest_sha256=str(metadata["corpus_manifest_sha256"]),
        sink_scope_sha256=str(metadata["sink_scope_sha256"]),
        model_id=str(metadata.get("model_id", "")),
        model_revision=str(metadata.get("model_revision", "")),
        metadata=metadata,
    )
