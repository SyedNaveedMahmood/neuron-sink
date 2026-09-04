from __future__ import annotations

import os
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neuron_sink import GPT2ModelAdapter, ModelStructureError, NeuronSet, suppress_neurons
from neuron_sink.provenance import require_registered_gpu


def tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(11)
    config = GPT2Config(
        vocab_size=71,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=3,
        n_head=4,
        n_inner=32,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval()


def forward(model: GPT2LMHeadModel, input_ids: torch.Tensor):
    with torch.inference_mode():
        return model(input_ids, output_attentions=True, use_cache=False)


def assert_outputs_equal(test: unittest.TestCase, left, right) -> None:
    test.assertTrue(torch.equal(left.logits, right.logits))
    test.assertEqual(len(left.attentions), len(right.attentions))
    for left_attention, right_attention in zip(left.attentions, right.attentions):
        test.assertTrue(torch.equal(left_attention, right_attention))


class SuppressionUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")
        self.ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)
        self.single = NeuronSet({1: (0, 5, 17)}, source="debug")
        self.multi = NeuronSet({0: (1, 4), 2: (3, 9)}, source="debug")

    def _capture(self):
        records: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def observer(layer: int, before: torch.Tensor, after: torch.Tensor) -> None:
            self.assertNotIn(layer, records)
            records[layer] = (before.detach().clone(), after.detach().clone())

        return records, observer

    def test_neuron_set_is_immutable_and_normalized(self) -> None:
        original = {2: [3, 1], 0: (4,)}
        neuron_set = NeuronSet(original, source="debug", selection_seed=2)
        original[2].append(8)
        self.assertEqual(list(neuron_set.by_layer), [0, 2])
        self.assertEqual(neuron_set.by_layer[2], (3, 1))
        with self.assertRaises(TypeError):
            neuron_set.by_layer[0] = (2,)  # type: ignore[index]
        self.assertIsInstance(hash(neuron_set), int)

    def test_invalid_alpha_values_fail_without_clipping(self) -> None:
        invalid = (float("nan"), float("inf"), -0.01, 1.01)
        for alpha in invalid:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                suppress_neurons(self.adapter, self.single, alpha)
        for alpha in (True, "0.5", None):
            with self.subTest(alpha=alpha), self.assertRaises(TypeError):
                suppress_neurons(self.adapter, self.single, alpha)  # type: ignore[arg-type]

    def test_alpha_one_is_a_strict_hook_free_identity(self) -> None:
        baseline = forward(self.model, self.ids)
        projection = self.adapter.mlp_projection(1)
        hooks_before = tuple(projection._forward_pre_hooks)
        with suppress_neurons(self.adapter, self.single, 1.0) as context:
            self.assertEqual(context.active_hook_count, 0)
            self.assertEqual(tuple(projection._forward_pre_hooks), hooks_before)
            identity = forward(self.model, self.ids)
        assert_outputs_equal(self, baseline, identity)

    def test_alpha_zero_changes_only_selected_coordinates(self) -> None:
        records, observer = self._capture()
        with suppress_neurons(
            self.adapter, self.single, 0.0, observer=observer
        ) as context:
            forward(self.model, self.ids)
            self.assertEqual(context.active_hook_count, 1)
        before, after = records[1]
        selected = self.single.by_layer[1]
        mask = torch.ones(before.shape[-1], dtype=torch.bool)
        mask[list(selected)] = False
        self.assertEqual(before.shape, after.shape)
        self.assertEqual(before.dtype, after.dtype)
        self.assertEqual(before.device, after.device)
        self.assertEqual(torch.count_nonzero(after[..., selected]).item(), 0)
        self.assertTrue(torch.equal(before[..., mask], after[..., mask]))

    def test_alpha_half_scales_only_selected_coordinates(self) -> None:
        records, observer = self._capture()
        with suppress_neurons(self.adapter, self.single, 0.5, observer=observer):
            forward(self.model, self.ids)
        before, after = records[1]
        selected = self.single.by_layer[1]
        mask = torch.ones(before.shape[-1], dtype=torch.bool)
        mask[list(selected)] = False
        self.assertTrue(torch.equal(after[..., selected], before[..., selected] * 0.5))
        self.assertTrue(torch.equal(before[..., mask], after[..., mask]))

    def test_multiple_layers_are_modified_independently(self) -> None:
        records, observer = self._capture()
        with suppress_neurons(self.adapter, self.multi, 0.0, observer=observer):
            forward(self.model, self.ids)
        self.assertEqual(set(records), {0, 2})
        for layer, selected in self.multi.by_layer.items():
            before, after = records[layer]
            mask = torch.ones(before.shape[-1], dtype=torch.bool)
            mask[list(selected)] = False
            self.assertEqual(torch.count_nonzero(after[..., selected]).item(), 0)
            self.assertTrue(torch.equal(before[..., mask], after[..., mask]))

    def test_context_removes_hooks_even_after_forward_error(self) -> None:
        projection = self.adapter.mlp_projection(1)
        hooks_before = tuple(projection._forward_pre_hooks)
        with self.assertRaises(ModelStructureError):
            with suppress_neurons(self.adapter, self.single, 0.0):
                projection(torch.zeros(1, 4, 31))
        self.assertEqual(tuple(projection._forward_pre_hooks), hooks_before)

    def test_no_state_leakage_and_no_parameter_mutation(self) -> None:
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        hooks_before = {
            layer: tuple(self.adapter.mlp_projection(layer)._forward_pre_hooks)
            for layer in self.multi.by_layer
        }
        baseline_before = forward(self.model, self.ids)
        with suppress_neurons(self.adapter, self.multi, 0.0) as context:
            forward(self.model, self.ids)
        self.assertEqual(context.active_hook_count, 0)
        baseline_after = forward(self.model, self.ids)
        assert_outputs_equal(self, baseline_before, baseline_after)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameters_before[name], parameter))
        for layer in self.multi.by_layer:
            self.assertEqual(
                tuple(self.adapter.mlp_projection(layer)._forward_pre_hooks),
                hooks_before[layer],
            )

    def test_suppressed_outputs_are_finite_and_attention_is_normalized(self) -> None:
        with suppress_neurons(self.adapter, self.multi, 0.0):
            output = forward(self.model, self.ids)
        self.assertTrue(torch.isfinite(output.logits).all())
        for attention in output.attentions:
            self.assertTrue(torch.isfinite(attention).all())
            row_error = (attention.sum(dim=-1) - 1.0).abs().max().item()
            self.assertLessEqual(row_error, 1e-6)


@unittest.skipUnless(
    os.environ.get("NEURON_SINK_RUN_GPU_INTEGRATION") == "1",
    "set NEURON_SINK_RUN_GPU_INTEGRATION=1 for the cached GPT-2 CUDA test",
)
class GPT2CudaIntegrationTests(unittest.TestCase):
    def test_real_gpt2_c_proj_input_on_registered_dev_gpu(self) -> None:
        # Amendment A001: both registered dev GPUs are accepted, via one shared list.
        require_registered_gpu("dev")
        cache_dir = os.environ.get("NEURON_SINK_HF_CACHE") or None
        revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
            attn_implementation="eager",
            dtype=torch.float32,
        ).eval().to("cuda:0")
        adapter = GPT2ModelAdapter(model, model_id="openai-community/gpt2")
        self.assertEqual(adapter.num_layers, 12)
        self.assertEqual(adapter.mlp_width(2), 3072)
        records: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def observer(layer: int, before: torch.Tensor, after: torch.Tensor) -> None:
            records[layer] = (before.detach().cpu(), after.detach().cpu())

        neurons = NeuronSet({2: (0, 17), 7: (1, 19)}, source="debug")
        ids = torch.tensor(
            [[5562, 262, 1103, 1885, 86, 505, 17685, 373]], device="cuda:0"
        )
        with suppress_neurons(adapter, neurons, 0.0, observer=observer):
            output = forward(model, ids)
        self.assertEqual(set(records), {2, 7})
        for layer, selected in neurons.by_layer.items():
            before, after = records[layer]
            self.assertEqual(tuple(before.shape), (1, 8, 3072))
            self.assertEqual(torch.count_nonzero(after[..., selected]).item(), 0)
            mask = torch.ones(3072, dtype=torch.bool)
            mask[list(selected)] = False
            self.assertTrue(torch.equal(before[..., mask], after[..., mask]))
        self.assertTrue(torch.isfinite(output.logits).all())


if __name__ == "__main__":
    unittest.main()
