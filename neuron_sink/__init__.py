"""Root package for neuron-sink experiment infrastructure."""

from .attribution import (
    AttributionError,
    AttributionResult,
    attribution_rows,
    attribution_sha256,
    capture_c_proj_input,
    objective_depends_on_layer,
    rank_neurons,
    require_future_targets,
    score_example,
)
from .model_adapters import GPT2ModelAdapter, ModelStructureError
from .suppression import NeuronSet, SuppressionContext, suppress_neurons

__all__ = [
    "AttributionError",
    "AttributionResult",
    "GPT2ModelAdapter",
    "ModelStructureError",
    "NeuronSet",
    "SuppressionContext",
    "attribution_rows",
    "attribution_sha256",
    "capture_c_proj_input",
    "objective_depends_on_layer",
    "rank_neurons",
    "require_future_targets",
    "score_example",
    "suppress_neurons",
]
