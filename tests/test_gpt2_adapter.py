from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from neuron_sink import GPT2ModelAdapter, ModelStructureError, NeuronSet


def tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=67,
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


class GPT2ModelAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")

    def test_reports_zero_indexed_structure_and_width(self) -> None:
        self.assertEqual(self.adapter.model_id, "debug-gpt2")
        self.assertEqual(self.adapter.num_layers, 3)
        self.assertEqual(self.adapter.mlp_width(0), 32)
        self.assertEqual(self.adapter.mlp_width(2), 32)
        self.assertEqual(
            self.adapter.get_mlp_intermediate_path(1),
            "transformer.h[1].mlp.c_proj input",
        )
        self.assertIs(self.adapter.mlp_projection(1), self.model.transformer.h[1].mlp.c_proj)

    def test_invalid_layers_fail_clearly(self) -> None:
        for layer in (-1, 3):
            with self.subTest(layer=layer), self.assertRaises(IndexError):
                self.adapter.mlp_width(layer)
        for layer in (True, 1.5, "1"):
            with self.subTest(layer=layer), self.assertRaises(TypeError):
                self.adapter.mlp_width(layer)  # type: ignore[arg-type]

    def test_invalid_neurons_fail_clearly(self) -> None:
        for neuron in (-1, 32):
            with self.subTest(neuron=neuron), self.assertRaises(IndexError):
                self.adapter.validate_neuron(0, neuron)
        for neuron in (False, 2.0, "2"):
            with self.subTest(neuron=neuron), self.assertRaises(TypeError):
                self.adapter.validate_neuron(0, neuron)  # type: ignore[arg-type]

    def test_duplicate_or_empty_selection_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.adapter.validate_neuron_set(
                NeuronSet({0: (1, 1)}, source="debug")
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.adapter.validate_neuron_set(NeuronSet({}, source="debug"))
        with self.assertRaisesRegex(ValueError, "no selected"):
            self.adapter.validate_neuron_set(NeuronSet({0: ()}, source="debug"))

    def test_unexpected_intermediate_shape_fails(self) -> None:
        with self.assertRaisesRegex(ModelStructureError, "must have shape"):
            self.adapter.validate_intermediate(0, torch.zeros(4, 32))
        with self.assertRaisesRegex(ModelStructureError, "has width 31"):
            self.adapter.validate_intermediate(0, torch.zeros(1, 4, 31))
        with self.assertRaisesRegex(ModelStructureError, "torch.Tensor"):
            self.adapter.validate_intermediate(0, object())

    def test_unexpected_model_structure_fails(self) -> None:
        with self.assertRaisesRegex(ModelStructureError, "model.transformer.h"):
            GPT2ModelAdapter(nn.Linear(2, 2))


if __name__ == "__main__":
    unittest.main()
