"""Model-structure adapters used by neuron-sink interventions.

Only the GPT-2 adapter needed for Task 3 is implemented here. The registered GPT-2
neuron is a coordinate of the post-GELU tensor entering ``mlp.c_proj``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

if TYPE_CHECKING:
    from .suppression import NeuronSet


class ModelStructureError(RuntimeError):
    """Raised when a model does not expose the pinned GPT-2 module structure."""


class GPT2ModelAdapter:
    """Validate and expose GPT-2 MLP post-activation projection inputs."""

    def __init__(self, model: nn.Module, model_id: str | None = None) -> None:
        self.model = model
        self.model_id = model_id or str(
            getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
            or type(model).__name__
        )
        self._blocks = self._resolve_blocks()
        self._widths = tuple(self._validate_block(index) for index in range(len(self._blocks)))

    def _resolve_blocks(self) -> nn.ModuleList:
        transformer = getattr(self.model, "transformer", None)
        blocks = getattr(transformer, "h", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
            raise ModelStructureError(
                "Expected GPT-2 blocks at model.transformer.h as a non-empty ModuleList"
            )
        configured_layers = getattr(getattr(self.model, "config", None), "n_layer", None)
        if configured_layers is not None and int(configured_layers) != len(blocks):
            raise ModelStructureError(
                f"config.n_layer={configured_layers} does not match transformer.h "
                f"length {len(blocks)}"
            )
        return blocks

    def _validate_block(self, layer: int) -> int:
        block = self._blocks[layer]
        mlp = getattr(block, "mlp", None)
        c_fc = getattr(mlp, "c_fc", None)
        c_proj = getattr(mlp, "c_proj", None)
        if not isinstance(c_fc, Conv1D) or not isinstance(c_proj, Conv1D):
            raise ModelStructureError(
                f"Layer {layer} must expose GPT-2 Conv1D modules at "
                "transformer.h[layer].mlp.c_fc and .c_proj"
            )
        if c_fc.weight.ndim != 2 or c_proj.weight.ndim != 2:
            raise ModelStructureError(f"Layer {layer} GPT-2 MLP weights must be rank 2")

        # Transformers Conv1D stores weights as [input_features, output_features].
        fc_output_width = int(c_fc.weight.shape[1])
        projection_input_width = int(c_proj.weight.shape[0])
        if fc_output_width <= 0 or fc_output_width != projection_input_width:
            raise ModelStructureError(
                f"Layer {layer} c_fc output width {fc_output_width} does not match "
                f"c_proj input width {projection_input_width}"
            )
        return projection_input_width

    @property
    def num_layers(self) -> int:
        return len(self._blocks)

    def validate_layer(self, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"Layer id must be an integer, got {type(layer).__name__}")
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(
                f"Layer id {layer} is outside the zero-indexed range "
                f"[0, {self.num_layers})"
            )
        return layer

    def mlp_width(self, layer: int) -> int:
        return self._widths[self.validate_layer(layer)]

    def mlp_projection(self, layer: int) -> Conv1D:
        layer = self.validate_layer(layer)
        return self._blocks[layer].mlp.c_proj

    def get_mlp_intermediate_path(self, layer: int) -> str:
        self.validate_layer(layer)
        return f"transformer.h[{layer}].mlp.c_proj input"

    def validate_neuron(self, layer: int, neuron: int) -> int:
        width = self.mlp_width(layer)
        if isinstance(neuron, bool) or not isinstance(neuron, int):
            raise TypeError(f"Neuron id must be an integer, got {type(neuron).__name__}")
        if neuron < 0 or neuron >= width:
            raise IndexError(
                f"Neuron id {neuron} is outside layer {layer}'s range [0, {width})"
            )
        return neuron

    def validate_neuron_set(self, neuron_set: NeuronSet) -> None:
        from .suppression import NeuronSet

        if not isinstance(neuron_set, NeuronSet):
            raise TypeError("neuron_set must be a NeuronSet")
        if not neuron_set.by_layer:
            raise ValueError("NeuronSet.by_layer must contain at least one layer")
        for layer, neurons in neuron_set.by_layer.items():
            self.validate_layer(layer)
            if not neurons:
                raise ValueError(f"Layer {layer} has no selected neurons")
            if len(neurons) != len(set(neurons)):
                raise ValueError(f"Layer {layer} contains duplicate neuron ids")
            for neuron in neurons:
                self.validate_neuron(layer, neuron)

    def validate_intermediate(self, layer: int, activation: Any) -> torch.Tensor:
        layer = self.validate_layer(layer)
        if not isinstance(activation, torch.Tensor):
            raise ModelStructureError(
                f"{self.get_mlp_intermediate_path(layer)} must be a torch.Tensor"
            )
        if activation.ndim != 3:
            raise ModelStructureError(
                f"{self.get_mlp_intermediate_path(layer)} must have shape "
                f"[batch, sequence, {self.mlp_width(layer)}], got {tuple(activation.shape)}"
            )
        if int(activation.shape[-1]) != self.mlp_width(layer):
            raise ModelStructureError(
                f"{self.get_mlp_intermediate_path(layer)} has width "
                f"{activation.shape[-1]}, expected {self.mlp_width(layer)}"
            )
        return activation
