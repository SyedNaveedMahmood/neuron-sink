from dataclasses import dataclass

import pytest
import torch

from neuron_sink.per_layer_diagnostics import (
    aggregate_per_layer_rows,
    first_order_predictions,
    per_layer_sink_scores,
)
from neuron_sink.suppression import NeuronSet


def test_per_layer_sink_scores_uses_second_half_position_zero() -> None:
    layers = []
    for value in (0.2, 0.6, 0.9):
        attention = torch.zeros(1, 2, 4, 4)
        attention[:, :, 2:, 0] = value
        layers.append(attention)
    assert per_layer_sink_scores(layers, (0, 2)) == pytest.approx({0: 0.2, 2: 0.9})


def test_aggregate_per_layer_rows_uses_ratio_of_means() -> None:
    common = {
        "experiment_id": "diag",
        "source_experiment_id": "source",
        "model_id": "model",
        "split": "test",
        "condition_id": "targeted",
        "condition_order": 1,
        "fraction_percent": 0.1,
        "k": 2,
        "alpha": 0.0,
        "attention_layer": 4,
        "baseline_valid": True,
        "intervention_valid": True,
        "valid_forward": True,
        "forward_runtime_seconds": 0.5,
    }
    rows = [
        {**common, "sink_baseline": 0.5, "sink_intervened": 0.4},
        {**common, "sink_baseline": 1.0, "sink_intervened": 0.9},
    ]
    result = aggregate_per_layer_rows(rows)[0]
    assert result["sink_baseline"] == pytest.approx(0.75)
    assert result["sink_intervened"] == pytest.approx(0.65)
    assert result["relative_sink_reduction"] == pytest.approx(1.0 - 0.65 / 0.75)
    assert result["runtime_seconds"] == 1.0


@dataclass(frozen=True)
class _Condition:
    condition_id: str
    fraction_percent: float
    k: int
    neuron_set: NeuronSet


def test_first_order_prediction_has_suppression_sign_and_scope_weight() -> None:
    rows = [
        {
            "layer": 1,
            "neuron": 2,
            "mean_signed_attr": -0.01,
            "future_sink_layers": "2|4",
        }
    ]
    condition = _Condition("targeted", 0.1, 1, NeuronSet({1: (2,)}, source="targeted"))
    result = first_order_predictions(
        rows, (condition,), sink_layers=(0, 2, 4, 6), seq_len=10, alphas=(1.0, 0.0)
    )
    assert result[0]["predicted_delta_sink"] == 0.0
    assert result[1]["predicted_delta_sink"] == pytest.approx(0.05)
    assert result[1]["predicted_effect"] == "sink_increase"
