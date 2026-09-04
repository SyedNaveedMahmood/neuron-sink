from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from neuron_sink.attribution import (
    capture_mlp_projection_input,
    objective_depends_on_layer,
    score_example,
)
from neuron_sink.model_adapters import Qwen2ModelAdapter
from neuron_sink.provenance import canonical_sha256
from neuron_sink.stage_c import (
    EXPERIMENT_ID,
    FORMAL_GATE_SCHEMA,
    OPERATING_POINT_SCHEMA,
    StageCError,
    build_operating_point_document,
    evaluate_formal_gate,
    freeze_operating_point,
    stage_c_run_root,
    unlock_test_split,
)
from neuron_sink.suppression import NeuronSet, suppress_neurons


def tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(23)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


def forward(model: Qwen2ForCausalLM, ids: torch.Tensor):
    with torch.inference_mode():
        return model(ids, output_attentions=True, use_cache=False)


class QwenNeuronHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_qwen()
        self.adapter = Qwen2ModelAdapter(self.model, model_id="debug-qwen2")
        self.ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)
        self.neurons = NeuronSet({1: (0, 7, 31)}, source="unit-test")

    def test_adapter_maps_the_swiglu_product_entering_down_proj(self) -> None:
        self.assertEqual(self.adapter.num_layers, 3)
        self.assertEqual(self.adapter.mlp_width(1), 64)
        self.assertEqual(self.adapter.num_attention_heads, 4)
        self.assertEqual(
            self.adapter.get_mlp_intermediate_path(1),
            "model.layers[1].mlp.down_proj input",
        )
        self.assertIs(self.adapter.mlp_projection(1), self.model.model.layers[1].mlp.down_proj)
        with self.assertRaises(IndexError):
            self.adapter.validate_neuron(1, 64)

    def test_grouped_query_attention_returns_query_head_shape(self) -> None:
        output = forward(self.model, self.ids)
        self.assertEqual(
            [tuple(attention.shape) for attention in output.attentions],
            [(1, 4, 8, 8)] * 3,
        )

    def test_alpha_one_is_exact_and_installs_no_hook(self) -> None:
        baseline = forward(self.model, self.ids)
        projection = self.adapter.mlp_projection(1)
        hooks_before = tuple(projection._forward_pre_hooks.items())
        with suppress_neurons(self.adapter, self.neurons, 1.0) as context:
            self.assertEqual(context.active_hook_count, 0)
            identity = forward(self.model, self.ids)
        self.assertEqual(tuple(projection._forward_pre_hooks.items()), hooks_before)
        self.assertTrue(torch.equal(baseline.logits, identity.logits))
        for left, right in zip(baseline.attentions, identity.attentions):
            self.assertTrue(torch.equal(left, right))

    def test_alpha_zero_changes_only_selected_down_proj_input_coordinates(self) -> None:
        records: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def observer(layer: int, before: torch.Tensor, after: torch.Tensor) -> None:
            records[layer] = (before.detach().clone(), after.detach().clone())

        with suppress_neurons(self.adapter, self.neurons, 0.0, observer=observer):
            forward(self.model, self.ids)
        before, after = records[1]
        selected = self.neurons.by_layer[1]
        keep = torch.ones(64, dtype=torch.bool)
        keep[list(selected)] = False
        self.assertEqual(tuple(before.shape), (1, 8, 64))
        self.assertEqual(torch.count_nonzero(after[..., selected]).item(), 0)
        self.assertTrue(torch.equal(before[..., keep], after[..., keep]))

    def test_generic_capture_is_value_preserving_and_differentiable(self) -> None:
        baseline = forward(self.model, self.ids).logits
        with capture_mlp_projection_input(self.adapter, 1) as captured:
            with torch.enable_grad():
                logits = self.model(self.ids, use_cache=False).logits
        activation = captured["activation"]
        self.assertTrue(torch.equal(baseline, logits.detach()))
        self.assertTrue(activation.is_leaf and activation.requires_grad)
        self.assertEqual(tuple(activation.shape), (1, 8, 64))

    def test_future_sink_gradient_and_causal_order_hold_for_qwen(self) -> None:
        scored = score_example(self.model, self.adapter, self.ids, 0, [1, 2])
        self.assertEqual(tuple(scored.activation.shape), (8, 64))
        self.assertEqual(tuple(scored.gradient.shape), (8, 64))
        self.assertTrue(torch.isfinite(scored.gradient).all())
        self.assertGreater(float(scored.gradient.abs().max()), 0.0)
        self.assertTrue(
            objective_depends_on_layer(self.model, self.adapter, self.ids, 1, [2])
        )
        self.assertFalse(
            objective_depends_on_layer(self.model, self.adapter, self.ids, 1, [1])
        )
        self.assertFalse(
            objective_depends_on_layer(self.model, self.adapter, self.ids, 1, [0])
        )

    def test_suppression_leaves_no_state_or_hook_leakage(self) -> None:
        baseline = forward(self.model, self.ids)
        projection = self.adapter.mlp_projection(1)
        hooks_before = tuple(projection._forward_pre_hooks.items())
        with suppress_neurons(self.adapter, self.neurons, 0.0):
            intervened = forward(self.model, self.ids)
        after = forward(self.model, self.ids)
        self.assertFalse(torch.equal(baseline.logits, intervened.logits))
        self.assertTrue(torch.equal(baseline.logits, after.logits))
        self.assertEqual(tuple(projection._forward_pre_hooks.items()), hooks_before)


class StageCBoundaryTests(unittest.TestCase):
    def test_stage_c_paths_are_model_specific_and_separate_from_stage_b(self) -> None:
        path = stage_c_run_root(
            Path("repo"),
            "qwen2.5-1.5b-instruct",
            registered_run=True,
            stamp="run_20260904T000000Z",
        )
        self.assertEqual(path.parts[-3:], ("stage_c_full", "qwen2.5-1.5b-instruct", "run_20260904T000000Z"))
        with self.assertRaises(StageCError):
            stage_c_run_root(
                Path("repo"), "gpt2-small", registered_run=True,
                stamp="run_20260904T000000Z",
            )

    def test_operating_point_builder_relabels_and_rehashes_stage_c(self) -> None:
        stage_b_document = {
            "schema": "stage_b_operating_point_v1",
            "experiment_id": "stage_b_full_phenomenon_v1",
            "value": 7,
            "operating_point_sha256": "old",
        }
        with patch(
            "neuron_sink.stage_c._build_stage_b_operating_point",
            return_value=stage_b_document,
        ):
            document = build_operating_point_document([], [])
        self.assertEqual(document["schema"], OPERATING_POINT_SCHEMA)
        self.assertEqual(document["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(
            document["operating_point_sha256"],
            canonical_sha256({
                key: value
                for key, value in document.items()
                if key != "operating_point_sha256"
            }),
        )

    def test_stage_b_operating_point_cannot_unlock_stage_c_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "operating_point.json"
            from neuron_sink.provenance import write_json

            write_json(path, {"schema": "stage_b_operating_point_v1"})
            with self.assertRaises(StageCError):
                unlock_test_split(
                    path,
                    model_id="qwen",
                    model_revision="r",
                    corpus_manifest_sha256="c",
                    sink_scope_sha256="s",
                    attribution_sha256="a",
                    neuron_sets_sha256="n",
                )

    def test_freeze_refuses_non_stage_c_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(StageCError):
                freeze_operating_point(
                    Path(temp) / "operating_point.json",
                    {"schema": "stage_b_operating_point_v1"},
                )

    def test_dry_run_gate_never_claims_a_scientific_result(self) -> None:
        document = evaluate_formal_gate(
            [],
            [],
            all_identity_pass=True,
            all_validity_pass=True,
            state_leakage_pass=True,
            registered_run=False,
        )
        self.assertEqual(document["schema"], FORMAL_GATE_SCHEMA)
        self.assertEqual(document["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(document["status"], "NOT_EVALUATED_DRY_RUN")
        self.assertFalse(document["test_split_accessed"])
        self.assertNotIn("formal_gate_sha256", document)


if __name__ == "__main__":
    unittest.main()
