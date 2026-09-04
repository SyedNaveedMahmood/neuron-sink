"""Causal-order-aware future-sink activation-times-gradient neuron attribution.

Registered objective (``docs/00_MASTER_EXPERIMENT_DESIGN.md``, "Causal-order-aware
attribution objective"): ``S_future(l)`` is the mean sink attention over the frozen
sink-heavy attention layers ``j > l``, and for neuron ``n`` in eligible MLP layer ``l``::

    I(l, n) = mean over examples and token positions of | a(l,n) * dS_future(l)/da(l,n) |

``a`` is the tensor entering ``transformer.h[l].mlp.c_proj`` -- the registered neuron
definition (``handover.md`` section 4), the same tensor ``neuron_sink.suppression``
multiplies by ``alpha``. Ranking is by ``mean_abs_attr``; ``mean_signed_attr`` is saved for
analysis and must never be ranked by, because signed scores cancel across tokens.

Token scope is every sequence position, matching the registered aggregation
(``configs/experiment_plan.yaml``: ``aggregation: mean_over_examples_and_tokens``) and the
intervention this ranking predicts, which scales the neuron at all positions
(``suppression_positions: all``). Positions before the second-half query window still carry
gradient, through the key/value route by which an early-position neuron shapes later
attention.

**Attribution is a ranking heuristic, not causal evidence.** Causal evidence comes from
held-out suppression against layer-count-matched random controls, which is a later task. No
function here selects, thresholds, or suppresses anything.

Memory (``docs/04_HARDWARE_RUNBOOK.md``, "Memory discipline" -- attribution is the
highest-risk path): batch size 1, one eligible MLP layer per backward pass, graph and
captured tensors released between examples and layers. :func:`capture_c_proj_input` hands the
model a *detached* leaf, so with model parameters frozen nothing upstream of layer ``l``'s
``c_proj`` requires grad, no graph is built for layers ``0..l-1`` at all, and no embedding
backward ever runs. This does not change the quantity being measured: ``dS_future/da`` is a
partial derivative with respect to ``a``, which does not depend on how ``a`` was produced.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .corpus import (
    RANKING_ALLOWED_SPLITS,
    LeakageError,
    NeutralCorpus,
    NeutralCorpusItem,
    assert_no_downstream_source,
    require_discovery_split,
)
from .model_adapters import GPT2ModelAdapter, ModelStructureError
from .provenance import canonical_sha256
from .sink_metrics import REGISTERED_TARGET_POSITION, differentiable_sink_score


SCHEMA_VERSION = "neuron_attribution_v1"

#: ``docs/05_METRICS_AND_SCHEMAS.md`` "Attribution metrics" and
#: ``configs/experiment_plan.yaml`` ``attribution:``.
ATTRIBUTION_METHOD = "abs_activation_x_sink_gradient"
ATTRIBUTION_OBJECTIVE = "causal_order_future_sink"
ATTRIBUTION_AGGREGATION = "mean_over_examples_and_tokens"
TOKEN_POSITION_RULE = "all_positions"
RANKING_SCORE = "mean_abs_attr"

#: Row schema from ``docs/03_IMPLEMENTATION_SPEC.md`` section 4, plus the two rank columns
#: that make a rerun byte-comparable. Ranking is not selection: no ``k``, no top-k set.
ROW_FIELDS: tuple[str, ...] = (
    "layer",
    "neuron",
    "mean_abs_activation",
    "mean_signed_attr",
    "mean_abs_attr",
    "n_examples",
    "n_tokens",
    "future_sink_layers",
    "rank_abs",
    "rank_abs_in_layer",
)

#: ``future_sink_layers`` is a list inside a CSV cell; a pipe keeps it unquoted and
#: unambiguous. Recorded in the metadata JSON so a reader never has to guess.
FUTURE_LAYER_SEPARATOR = "|"


class AttributionError(RuntimeError):
    """Raised when an attribution request violates the registered objective."""


def require_future_targets(layer: int, targets: Sequence[int]) -> tuple[int, ...]:
    """Enforce causal ordering: MLP layer ``l`` may target only attention layers ``j > l``.

    A decoder block computes attention before its MLP, so layer ``l``'s MLP cannot have
    caused the attention already produced at layer ``l`` or earlier (``AGENTS.md``, "Causal
    ordering constraint").
    """

    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise TypeError(f"Layer id must be an integer, got {type(layer).__name__}")
    ordered = tuple(int(target) for target in targets)
    if not ordered:
        raise AttributionError(
            f"MLP layer {layer} has no future sink layers and is therefore ineligible for "
            "the primary attribution scan"
        )
    if len(set(ordered)) != len(ordered):
        raise AttributionError(f"MLP layer {layer} has duplicate future targets {ordered}")
    same_or_earlier = [target for target in ordered if target <= int(layer)]
    if same_or_earlier:
        raise AttributionError(
            f"MLP layer {layer} cannot be attributed to attention layer(s) "
            f"{same_or_earlier}. A decoder block computes attention before its MLP, so an "
            "MLP at layer l may target only strictly later attention layers j > l."
        )
    return tuple(sorted(ordered))


@contextmanager
def capture_c_proj_input(
    adapter: GPT2ModelAdapter, layer: int
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture one layer's ``mlp.c_proj`` input as a differentiable leaf.

    The hook detaches the incoming tensor and returns the leaf in its place, so the value the
    model computes with is unchanged while the backward pass stops exactly at the registered
    neuron definition. Yields a dict that holds ``"activation"`` once the forward has run.
    """

    layer = adapter.validate_layer(layer)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, args: tuple[object, ...]):
        if not args:
            raise ModelStructureError(
                f"{adapter.get_mlp_intermediate_path(layer)} received no input"
            )
        if "activation" in captured:
            raise AttributionError(
                f"Layer {layer} c_proj ran twice inside one capture; attribution uses one "
                "capture per forward pass so a gradient cannot be attributed to the wrong "
                "activation"
            )
        leaf = adapter.validate_intermediate(layer, args[0]).detach().requires_grad_(True)
        captured["activation"] = leaf
        return (leaf, *args[1:])

    handle = adapter.mlp_projection(layer).register_forward_pre_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@dataclass(frozen=True)
class ExampleAttribution:
    """One example's activation and future-sink gradient at one eligible MLP layer."""

    layer: int
    future_sink_layers: tuple[int, ...]
    sink_future: float
    activation: torch.Tensor  # detached [sequence, width]
    gradient: torch.Tensor  # [sequence, width]


def score_example(
    model: torch.nn.Module,
    adapter: GPT2ModelAdapter,
    input_ids: torch.Tensor,
    layer: int,
    targets: Sequence[int],
    *,
    heads: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
) -> ExampleAttribution:
    """One forward and one backward: ``dS_future(l)/da`` for a single example.

    The forward is the Hugging Face ``output_attentions=True`` path, which is what Task 4's
    sink map was built from and what ``neuron_sink.suppression`` hooks, so attribution and
    every later measurement travel the same route. An enclosing ``torch.no_grad()`` is
    overridden deliberately, but ``torch.inference_mode()`` cannot be re-enabled from inside
    and is reported as an error rather than silently producing no gradient.
    """

    targets = require_future_targets(layer, targets)
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise AttributionError(
            "Attribution runs at batch size 1 (docs/04_HARDWARE_RUNBOOK.md); got input_ids "
            f"of shape {tuple(input_ids.shape)}"
        )

    with capture_c_proj_input(adapter, layer) as captured:
        with torch.enable_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=True,
                use_cache=False,
            )
            activation = captured.get("activation")
            if activation is None:
                raise AttributionError(
                    f"{adapter.get_mlp_intermediate_path(layer)} was never reached; the "
                    "capture hook did not fire"
                )
            score = differentiable_sink_score(
                output.attentions, targets, heads, target_pos=target_pos
            )
            if not score.requires_grad:
                raise AttributionError(
                    f"S_future({layer}) does not depend on the captured activation. The "
                    "forward most likely ran under torch.inference_mode(), which blocks "
                    "autograd and cannot be re-enabled from inside it; use a plain forward. "
                    "The other possibility is that the capture point is not upstream of the "
                    "target attention layers."
                )
            (gradient,) = torch.autograd.grad(score, activation)

    return ExampleAttribution(
        layer=int(layer),
        future_sink_layers=targets,
        sink_future=float(score.detach()),
        activation=activation.detach()[0],
        gradient=gradient[0],
    )


def objective_depends_on_layer(
    model: torch.nn.Module,
    adapter: GPT2ModelAdapter,
    input_ids: torch.Tensor,
    mlp_layer: int,
    attention_layers: Sequence[int],
    *,
    heads: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
) -> bool:
    """Whether a sink objective over ``attention_layers`` is a function of the MLP layer.

    Deliberately skips :func:`require_future_targets`, because probing a same-layer or
    earlier-layer target is the point: it turns the causal-ordering constraint from an
    assumption into a measurement. With model parameters frozen, an objective built from
    attention at layer ``j <= mlp_layer`` is not connected to layer ``mlp_layer``'s ``c_proj``
    input at all, so the returned score carries no ``grad_fn`` and this returns ``False`` --
    a stronger statement than a numerically zero gradient.
    """

    with capture_c_proj_input(adapter, mlp_layer) as captured:
        with torch.enable_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=True,
                use_cache=False,
            )
            if captured.get("activation") is None:
                raise AttributionError(
                    f"{adapter.get_mlp_intermediate_path(mlp_layer)} was never reached; the "
                    "capture hook did not fire"
                )
            score = differentiable_sink_score(
                output.attentions, attention_layers, heads, target_pos=target_pos
            )
            return bool(score.requires_grad)


@dataclass(frozen=True)
class LayerAttribution:
    """Accumulated per-neuron scores for one eligible MLP layer."""

    layer: int
    future_sink_layers: tuple[int, ...]
    width: int
    n_examples: int
    n_tokens: int
    mean_abs_activation: np.ndarray
    mean_signed_attr: np.ndarray
    mean_abs_attr: np.ndarray
    mean_sink_future: float
    max_abs_gradient: float
    nonfinite_values: int
    zero_gradient_examples: int
    peak_memory_allocated_bytes: int = 0

    def diagnostics(self) -> dict[str, Any]:
        """Per-layer summary for the run metadata; a diagnostic, not a scientific result."""

        order = np.argsort(-self.mean_abs_attr, kind="stable")
        return {
            "layer": self.layer,
            "future_sink_layers": list(self.future_sink_layers),
            "width": self.width,
            "n_examples": self.n_examples,
            "n_tokens": self.n_tokens,
            "mean_sink_future": self.mean_sink_future,
            "max_abs_gradient": self.max_abs_gradient,
            "nonfinite_values": self.nonfinite_values,
            "zero_gradient_examples": self.zero_gradient_examples,
            "mean_abs_attr_mean": float(self.mean_abs_attr.mean()),
            "mean_abs_attr_median": float(np.median(self.mean_abs_attr)),
            "mean_abs_attr_max": float(self.mean_abs_attr.max()),
            "top_neuron": int(order[0]),
            "top_neuron_mean_abs_attr": float(self.mean_abs_attr[order[0]]),
            "top_neuron_mean_signed_attr": float(self.mean_signed_attr[order[0]]),
            "peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
        }


@dataclass(frozen=True)
class AttributionResult:
    """The full discovery-split ranking input, before any selection."""

    layers: tuple[LayerAttribution, ...]
    example_ids: tuple[str, ...]
    corpus_id: str
    corpus_manifest_sha256: str
    split: str
    seq_len: int
    target_pos: int
    heads_restricted: bool = False
    checks: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.example_ids)

    @property
    def n_rows(self) -> int:
        return sum(layer.width for layer in self.layers)


def attribution_rows(result: AttributionResult) -> list[dict[str, Any]]:
    """One row per ``(layer, neuron)``, emitted in ``(layer, neuron)`` order.

    ``rank_abs`` is the global rank by ``mean_abs_attr`` over every eligible pair, broken by
    ``(layer, neuron)`` so the order is total and a rerun is byte-comparable. A rank column is
    the ranking this task produces; choosing a ``k`` and cutting a set is a later task.
    """

    rows: list[dict[str, Any]] = []
    for layer_result in result.layers:
        future = FUTURE_LAYER_SEPARATOR.join(
            str(target) for target in layer_result.future_sink_layers
        )
        within = sorted(
            range(layer_result.width),
            key=lambda neuron: (-layer_result.mean_abs_attr[neuron], neuron),
        )
        rank_in_layer = {neuron: index + 1 for index, neuron in enumerate(within)}
        for neuron in range(layer_result.width):
            rows.append({
                "layer": layer_result.layer,
                "neuron": neuron,
                "mean_abs_activation": float(layer_result.mean_abs_activation[neuron]),
                "mean_signed_attr": float(layer_result.mean_signed_attr[neuron]),
                "mean_abs_attr": float(layer_result.mean_abs_attr[neuron]),
                "n_examples": layer_result.n_examples,
                "n_tokens": layer_result.n_tokens,
                "future_sink_layers": future,
                "rank_abs": 0,
                "rank_abs_in_layer": rank_in_layer[neuron],
            })

    ranked = sorted(
        rows, key=lambda row: (-row["mean_abs_attr"], row["layer"], row["neuron"])
    )
    for index, row in enumerate(ranked):
        row["rank_abs"] = index + 1
    rows.sort(key=lambda row: (row["layer"], row["neuron"]))
    return rows


def attribution_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the emitted rows so a rerun can be proved identical."""

    return canonical_sha256([[row[name] for name in ROW_FIELDS] for row in rows])


def _require_discovery_items(
    items: Sequence[NeutralCorpusItem], split: str
) -> tuple[NeutralCorpusItem, ...]:
    """Reject held-out data at the item level, not just by the split argument."""

    for item in items:
        if item.split not in RANKING_ALLOWED_SPLITS:
            raise LeakageError(
                f"Item {item.item_id!r} belongs to the {item.split!r} split but was passed "
                f"to neuron ranking as {split!r}. Ranking may only read "
                f"{sorted(RANKING_ALLOWED_SPLITS)}; validation selects the operating point "
                "and test is used once, after k* is frozen."
            )
    return tuple(items)


def rank_neurons(
    model: torch.nn.Module,
    adapter: GPT2ModelAdapter,
    corpus: NeutralCorpus,
    future_sink_layers: Mapping[int, Sequence[int]],
    *,
    split: str = "discovery",
    smoke: bool = True,
    max_examples: int | None = None,
    heads: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
    target_pos: int = REGISTERED_TARGET_POSITION,
    device: torch.device | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> AttributionResult:
    """Rank every eligible ``(layer, neuron)`` pair on the discovery split.

    This is the ranking API named by ``AGENTS.md`` required test 6: it reads the discovery
    split and nothing else. ``require_discovery_split`` rejects the split argument, and every
    selected item's own frozen split role is checked as well, so a validation or test
    manifest cannot reach a gradient even if the caller mislabels it.

    Layers are processed one at a time with one backward pass per example, and the graph and
    captured tensors are released before the next example, per the runbook memory discipline.
    Accumulation is float64 on the CPU so a long run cannot drift.
    """

    require_discovery_split(split)
    assert_no_downstream_source(
        str(corpus.source.get("dataset_id", "")), str(corpus.source.get("corpus_id", ""))
    )
    selected = list(corpus.items_for(split, smoke=smoke))
    if max_examples is not None:
        selected = selected[:max_examples]
    if not selected:
        raise AttributionError(f"Split {split!r} is empty; nothing to rank")
    items = _require_discovery_items(selected, split)

    seq_lengths = {item.n_tokens for item in items}
    if len(seq_lengths) != 1:
        raise AttributionError(
            f"Attribution expects one frozen sequence length, got {sorted(seq_lengths)}"
        )
    seq_len = seq_lengths.pop()

    layers = sorted(int(layer) for layer in future_sink_layers)
    if not layers:
        raise AttributionError("No eligible MLP layers were provided")
    targets_by_layer = {
        layer: require_future_targets(layer, future_sink_layers[layer]) for layer in layers
    }

    if device is None:
        device = next(model.parameters()).device
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AttributionError(
            f"{len(trainable)} model parameters still require grad (first: {trainable[0]}). "
            "Call model.requires_grad_(False) before attribution so no parameter gradients "
            "are computed."
        )

    results: list[LayerAttribution] = []
    for layer in layers:
        targets = targets_by_layer[layer]
        width = adapter.mlp_width(layer)
        abs_activation = np.zeros(width, dtype=np.float64)
        signed_attr = np.zeros(width, dtype=np.float64)
        abs_attr = np.zeros(width, dtype=np.float64)
        sink_total = 0.0
        max_abs_gradient = 0.0
        nonfinite = 0
        zero_gradient_examples = 0

        for index, item in enumerate(items):
            input_ids = torch.tensor(
                [list(item.input_ids)], dtype=torch.long, device=device
            )
            scored = score_example(
                model, adapter, input_ids, layer, targets,
                heads=heads, target_pos=target_pos,
            )
            activation = scored.activation
            gradient = scored.gradient
            attribution = activation * gradient

            nonfinite += int((~torch.isfinite(activation)).sum().item())
            nonfinite += int((~torch.isfinite(gradient)).sum().item())
            gradient_magnitude = float(gradient.abs().max().item())
            max_abs_gradient = max(max_abs_gradient, gradient_magnitude)
            if gradient_magnitude == 0.0:
                zero_gradient_examples += 1

            abs_activation += activation.abs().to(torch.float64).sum(dim=0).cpu().numpy()
            signed_attr += attribution.to(torch.float64).sum(dim=0).cpu().numpy()
            abs_attr += attribution.abs().to(torch.float64).sum(dim=0).cpu().numpy()
            sink_total += scored.sink_future

            # Release the captured tensors before the next backward pass.
            del scored, activation, gradient, attribution, input_ids
            if progress is not None:
                progress(layer, index + 1, len(items))

        n_tokens = len(items) * seq_len
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        results.append(LayerAttribution(
            layer=layer,
            future_sink_layers=targets,
            width=width,
            n_examples=len(items),
            n_tokens=n_tokens,
            mean_abs_activation=abs_activation / n_tokens,
            mean_signed_attr=signed_attr / n_tokens,
            mean_abs_attr=abs_attr / n_tokens,
            mean_sink_future=sink_total / len(items),
            max_abs_gradient=max_abs_gradient,
            nonfinite_values=nonfinite,
            zero_gradient_examples=zero_gradient_examples,
            peak_memory_allocated_bytes=peak,
        ))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return AttributionResult(
        layers=tuple(results),
        example_ids=tuple(item.item_id for item in items),
        corpus_id=corpus.corpus_id,
        corpus_manifest_sha256=corpus.manifest_sha256,
        split=split,
        seq_len=seq_len,
        target_pos=int(target_pos),
        heads_restricted=heads is not None,
    )
