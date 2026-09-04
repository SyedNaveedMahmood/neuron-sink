"""Scoped suppression of selected registered MLP intermediate coordinates."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import torch
from torch.utils.hooks import RemovableHandle

from .model_adapters import MLPModelAdapter, ModelStructureError


@dataclass(frozen=True)
class NeuronSet:
    """An immutable, zero-indexed collection of MLP intermediate coordinates."""

    by_layer: Mapping[int, tuple[int, ...]]
    source: str
    selection_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("NeuronSet.source must be a non-empty string")
        if self.selection_seed is not None and (
            isinstance(self.selection_seed, bool)
            or not isinstance(self.selection_seed, int)
        ):
            raise TypeError("NeuronSet.selection_seed must be an integer or None")
        if not isinstance(self.by_layer, Mapping):
            raise TypeError("NeuronSet.by_layer must be a mapping")

        normalized: dict[int, tuple[int, ...]] = {}
        for layer, neurons in self.by_layer.items():
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise TypeError(f"Layer id must be an integer, got {type(layer).__name__}")
            if isinstance(neurons, (str, bytes)):
                raise TypeError(f"Neuron ids for layer {layer} must be an iterable of integers")
            try:
                values = tuple(neurons)
            except TypeError as exc:
                raise TypeError(
                    f"Neuron ids for layer {layer} must be an iterable of integers"
                ) from exc
            for neuron in values:
                if isinstance(neuron, bool) or not isinstance(neuron, int):
                    raise TypeError(
                        f"Neuron id must be an integer, got {type(neuron).__name__}"
                    )
            normalized[layer] = values

        object.__setattr__(
            self,
            "by_layer",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    def __hash__(self) -> int:
        return hash((tuple(self.by_layer.items()), self.source, self.selection_seed))


SuppressionObserver = Callable[[int, torch.Tensor, torch.Tensor], None]


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a real number, got {type(alpha).__name__}")
    value = float(alpha)
    if not math.isfinite(value):
        raise ValueError(f"alpha must be finite, got {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"alpha must be in the registered range [0, 1], got {value}")
    return value


class SuppressionContext(AbstractContextManager["SuppressionContext"]):
    """Install temporary MLP output-projection pre-hooks for one forward scope.

    The hook clones the post-activation intermediate tensor, changes only selected
    coordinates at all sequence positions, and returns the replacement input. An
    ``alpha`` of exactly 1.0 validates the request but registers no hooks at all.
    """

    def __init__(
        self,
        adapter: MLPModelAdapter,
        neuron_set: NeuronSet,
        alpha: float,
        *,
        observer: SuppressionObserver | None = None,
    ) -> None:
        if not isinstance(adapter, MLPModelAdapter):
            raise TypeError("adapter must be an MLPModelAdapter")
        adapter.validate_neuron_set(neuron_set)
        self.adapter = adapter
        self.neuron_set = neuron_set
        self.alpha = _validate_alpha(alpha)
        self.observer = observer
        self._handles: list[RemovableHandle] = []
        self._entered = False
        self._closed = False

    @property
    def active_hook_count(self) -> int:
        return len(self._handles)

    def _hook_for_layer(self, layer: int, neurons: tuple[int, ...]):
        alpha = self.alpha

        def hook(_module: torch.nn.Module, args: tuple[object, ...]):
            if not args:
                raise ModelStructureError(
                    f"{self.adapter.get_mlp_intermediate_path(layer)} received no input"
                )
            before = self.adapter.validate_intermediate(layer, args[0])
            after = before.clone()
            if alpha == 0.0:
                after[..., neurons] = 0
            else:
                after[..., neurons] = before[..., neurons] * alpha
            if self.observer is not None:
                # Audit callbacks cannot mutate either the live incoming tensor or the
                # replacement tensor that continues through the model.
                self.observer(layer, before.detach().clone(), after.detach().clone())
            return (after, *args[1:])

        return hook

    def __enter__(self) -> "SuppressionContext":
        if self._entered:
            raise RuntimeError("A SuppressionContext cannot be entered more than once")
        self._entered = True
        if self.alpha == 1.0:
            return self
        try:
            for layer, neurons in self.neuron_set.by_layer.items():
                handle = self.adapter.mlp_projection(layer).register_forward_pre_hook(
                    self._hook_for_layer(layer, neurons)
                )
                self._handles.append(handle)
        except BaseException:
            self._remove_hooks()
            self._closed = True
            raise
        return self

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._remove_hooks()
        self._closed = True
        return None


def suppress_neurons(
    adapter: MLPModelAdapter,
    neuron_set: NeuronSet,
    alpha: float,
    *,
    observer: SuppressionObserver | None = None,
) -> SuppressionContext:
    """Return a scoped suppression context without running a model forward."""

    return SuppressionContext(adapter, neuron_set, alpha, observer=observer)
