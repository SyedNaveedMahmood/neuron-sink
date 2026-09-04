"""Architecture-specific access to registered MLP intermediate neurons.

The primary unit is the post-nonlinearity MLP coordinate immediately before the output
projection: ``mlp.c_proj`` input for GPT-2 and the SwiGLU product entering
``mlp.down_proj`` for Qwen2/Qwen2.5. The adapters validate the complete module seam before
any hook is installed so an architecture drift fails loudly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

if TYPE_CHECKING:
    from .suppression import NeuronSet


class ModelStructureError(RuntimeError):
    """Raised when a model does not expose a registered module structure."""


class MLPModelAdapter(ABC):
    """Common validation API for model-specific MLP output-projection inputs."""

    def __init__(self, model: nn.Module, model_id: str | None = None) -> None:
        self.model = model
        self.model_id = model_id or str(
            getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
            or type(model).__name__
        )
        self._blocks = self._resolve_blocks()
        self._widths = tuple(self._validate_block(index) for index in range(len(self._blocks)))

    @abstractmethod
    def _resolve_blocks(self) -> nn.ModuleList:
        """Return the model's zero-indexed decoder blocks after structural validation."""

    @abstractmethod
    def _validate_block(self, layer: int) -> int:
        """Validate one block and return its registered MLP intermediate width."""

    @property
    def num_layers(self) -> int:
        return len(self._blocks)

    @property
    @abstractmethod
    def num_attention_heads(self) -> int:
        """Return the number of query heads emitted in each attention tensor."""

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

    @abstractmethod
    def mlp_projection(self, layer: int) -> nn.Module:
        """Return the output/down projection whose input defines the neuron vector."""

    @abstractmethod
    def get_mlp_intermediate_path(self, layer: int) -> str:
        """Return the human-readable registered hook point."""

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


class GPT2ModelAdapter(MLPModelAdapter):
    """Validate and expose GPT-2 MLP post-GELU projection inputs."""

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
    def num_attention_heads(self) -> int:
        value = getattr(getattr(self.model, "config", None), "n_head", None)
        if value is None or int(value) <= 0:
            raise ModelStructureError("GPT-2 config.n_head must be a positive integer")
        return int(value)

    def mlp_projection(self, layer: int) -> Conv1D:
        layer = self.validate_layer(layer)
        return self._blocks[layer].mlp.c_proj

    def get_mlp_intermediate_path(self, layer: int) -> str:
        self.validate_layer(layer)
        return f"transformer.h[{layer}].mlp.c_proj input"


class Qwen2ModelAdapter(MLPModelAdapter):
    """Validate and expose the Qwen2.5 SwiGLU product entering ``down_proj``.

    The module mapping follows the pinned upstream Qwen harness at
    ``cross_scale_and_architecture/qwen/intervention_analysis_qwen.py``. Qwen uses
    RoPE and grouped-query attention, but neither changes the registered MLP neuron:
    ``SiLU(gate_proj(x)) * up_proj(x)`` immediately before ``down_proj``.
    """

    def _resolve_blocks(self) -> nn.ModuleList:
        inner = getattr(self.model, "model", None)
        blocks = getattr(inner, "layers", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
            raise ModelStructureError(
                "Expected Qwen2 blocks at model.model.layers as a non-empty ModuleList"
            )
        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type != "qwen2":
            raise ModelStructureError(
                f"Expected config.model_type='qwen2', got {model_type!r}"
            )
        configured_layers = getattr(config, "num_hidden_layers", None)
        if configured_layers is not None and int(configured_layers) != len(blocks):
            raise ModelStructureError(
                f"config.num_hidden_layers={configured_layers} does not match "
                f"model.layers length {len(blocks)}"
            )
        return blocks

    def _validate_block(self, layer: int) -> int:
        block = self._blocks[layer]
        mlp = getattr(block, "mlp", None)
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if not all(isinstance(module, nn.Linear) for module in (gate_proj, up_proj, down_proj)):
            raise ModelStructureError(
                f"Layer {layer} must expose Qwen2 nn.Linear modules at "
                "model.layers[layer].mlp.{gate_proj,up_proj,down_proj}"
            )
        width = int(gate_proj.out_features)
        if (
            width <= 0
            or int(up_proj.out_features) != width
            or int(down_proj.in_features) != width
            or int(gate_proj.in_features) != int(up_proj.in_features)
            or int(down_proj.out_features) != int(gate_proj.in_features)
        ):
            raise ModelStructureError(
                f"Layer {layer} Qwen2 SwiGLU/down_proj dimensions are inconsistent"
            )

        attention = getattr(block, "self_attn", None)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not isinstance(getattr(attention, name, None), nn.Linear):
                raise ModelStructureError(
                    f"Layer {layer} Qwen2 attention is missing nn.Linear {name}"
                )
        if getattr(attention.q_proj, "bias", None) is None:
            raise ModelStructureError(
                f"Layer {layer} Qwen2 q_proj has no learned bias; the pinned Qwen2.5 "
                "reference semantics require attention_bias=True"
            )
        for name in ("input_layernorm", "post_attention_layernorm"):
            if not isinstance(getattr(block, name, None), nn.Module):
                raise ModelStructureError(f"Layer {layer} Qwen2 block is missing {name}")
        return width

    @property
    def num_attention_heads(self) -> int:
        value = getattr(getattr(self.model, "config", None), "num_attention_heads", None)
        if value is None or int(value) <= 0:
            raise ModelStructureError(
                "Qwen2 config.num_attention_heads must be a positive integer"
            )
        return int(value)

    def mlp_projection(self, layer: int) -> nn.Linear:
        layer = self.validate_layer(layer)
        return self._blocks[layer].mlp.down_proj

    def get_mlp_intermediate_path(self, layer: int) -> str:
        self.validate_layer(layer)
        return f"model.layers[{layer}].mlp.down_proj input"
