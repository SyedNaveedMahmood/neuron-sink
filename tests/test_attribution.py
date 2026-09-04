from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neuron_sink.attribution import (
    AttributionError,
    _require_discovery_items,
    attribution_rows,
    attribution_sha256,
    capture_c_proj_input,
    objective_depends_on_layer,
    rank_neurons,
    require_future_targets,
    score_example,
)
from neuron_sink.corpus import LeakageError, NeutralCorpus, NeutralCorpusItem
from neuron_sink.model_adapters import GPT2ModelAdapter
from neuron_sink.provenance import canonical_sha256, require_registered_gpu
from neuron_sink.sink_metrics import (
    FrozenScopeError,
    differentiable_sink_score,
    load_frozen_sink_scope,
    per_layer_head_position0_attention,
    sink_scalar_from_map,
)
from neuron_sink.upstream_bridge import is_available, sink_repro_module


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCOPE = ROOT / "configs" / "frozen" / "sink_scope.json"
FROZEN_MANIFEST = ROOT / "configs" / "frozen" / "neutral_corpus_manifest.json"

#: Task 4 froze these; Task 5 consumes them and must never re-derive them.
TASK4_SINK_LAYERS = (7, 9, 10)
TASK4_ELIGIBLE_MLP_LAYERS = tuple(range(10))
TASK4_SCOPE_SHA256 = "b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235"
TASK4_CORPUS_SHA256 = "c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7"


def causal_attention(num_layers: int, num_heads: int, seq_len: int, seed: int = 0):
    """A batch of valid causal, row-normalised attention tensors."""

    generator = torch.Generator().manual_seed(seed)
    tensors = []
    for _ in range(num_layers):
        logits = torch.randn(num_heads, seq_len, seq_len, generator=generator)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        tensors.append(torch.softmax(logits, dim=-1))
    return tensors


def tiny_gpt2() -> GPT2LMHeadModel:
    """The same debug model shape ``tests/test_suppression.py`` uses, with frozen weights."""

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
    model = GPT2LMHeadModel(config).eval()
    model.requires_grad_(False)
    return model


def synthetic_corpus(
    splits: tuple[str, ...] = ("discovery", "discovery", "validation", "test"),
    *,
    seq_len: int = 8,
    dataset_id: str = "unit-test",
    corpus_id: str = "unit_test_corpus",
) -> NeutralCorpus:
    """A tiny in-memory corpus with real split roles, for API-level guards."""

    items = tuple(
        NeutralCorpusItem(
            item_id=f"blk{index}",
            split=split,
            text=f"synthetic item {index}",
            input_ids=tuple((index * 7 + 3 + step) % 60 + 1 for step in range(seq_len)),
            n_tokens=seq_len,
            meta={},
        )
        for index, split in enumerate(splits)
    )
    return NeutralCorpus(
        corpus_id=corpus_id,
        items=items,
        tokenizer_name="gpt2",
        tokenizer_revision=None,
        cut_length=seq_len,
        seed=0,
        source={"dataset_id": dataset_id, "corpus_id": corpus_id},
        upstream_manifest_sha256="0" * 64,
        upstream_provenance={},
    )


class DifferentiableSinkScoreTests(unittest.TestCase):
    """The gradient scorer must be the frozen metric, not a second definition."""

    def setUp(self) -> None:
        self.attentions = causal_attention(12, 12, 40, seed=17)
        self.map = per_layer_head_position0_attention(self.attentions)

    def test_matches_sink_scalar_from_map_on_fixed_tensors(self) -> None:
        for layers in (None, [7, 9, 10], [9, 10], [10], list(range(3, 11)), [0]):
            with self.subTest(layers=layers):
                differentiable = float(differentiable_sink_score(self.attentions, layers))
                frozen = sink_scalar_from_map(self.map, layers)
                # float32 torch against float64 NumPy: far inside upstream METRIC_ATOL 1e-5.
                self.assertLessEqual(abs(differentiable - frozen), 1e-6)

    def test_returns_a_zero_dim_tensor_that_carries_gradient(self) -> None:
        attentions = [tensor.clone().requires_grad_(True) for tensor in self.attentions]
        score = differentiable_sink_score(attentions, [7, 9, 10])
        self.assertIsInstance(score, torch.Tensor)
        self.assertEqual(score.ndim, 0)
        self.assertTrue(score.requires_grad)
        self.assertIsNotNone(score.grad_fn)

    def test_accepts_batched_attention_tensors(self) -> None:
        batched = [tensor.unsqueeze(0) for tensor in self.attentions]
        self.assertAlmostEqual(
            float(differentiable_sink_score(batched, [7, 9, 10])),
            float(differentiable_sink_score(self.attentions, [7, 9, 10])),
            places=9,
        )

    def test_head_restriction_matches_a_hand_computed_mean(self) -> None:
        heads = {7: (2, 10, 11)}
        restricted = float(differentiable_sink_score(self.attentions, [7], heads))
        manual = float(self.attentions[7][[2, 10, 11]][:, 20:, 0].mean())
        self.assertAlmostEqual(restricted, manual, places=9)
        self.assertNotAlmostEqual(
            restricted, float(differentiable_sink_score(self.attentions, [7])), places=9
        )

    def test_invalid_layer_and_head_requests_are_rejected(self) -> None:
        with self.assertRaises(IndexError):
            differentiable_sink_score(self.attentions, [12])
        with self.assertRaises(ValueError):
            differentiable_sink_score(self.attentions, [])
        with self.assertRaises(IndexError):
            differentiable_sink_score(self.attentions, [7], {7: (12,)})
        with self.assertRaises(ValueError):
            differentiable_sink_score(self.attentions, [7], {7: ()})

    @unittest.skipUnless(
        is_available("sink_repro"), "upstream/sink-repro submodule is not checked out"
    )
    def test_matches_the_pinned_upstream_metric_on_the_parity_band(self) -> None:
        legacy = sink_repro_module("intervention_analysis_legacy")
        engine = sink_repro_module("nnsight_engine")
        band_start, band_end = legacy.compute_band(12, "scaled")
        upstream = legacy.compute_bos_attention_metric(
            self.attentions, 12, "mid", target_pos=0,
            layer_start=band_start, layer_end=band_end,
        )
        mine = float(
            differentiable_sink_score(self.attentions, list(range(band_start, band_end)))
        )
        limit = engine.METRIC_ATOL + engine.METRIC_RTOL * abs(upstream)
        self.assertLessEqual(abs(mine - upstream), limit)


class CaptureAndGradientTests(unittest.TestCase):
    """Gradients must reach the registered neuron definition, unchanged in value."""

    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")
        self.ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)

    def _baseline_logits(self) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(self.ids, use_cache=False).logits.clone()

    def test_capture_does_not_change_the_forward(self) -> None:
        baseline = self._baseline_logits()
        with capture_c_proj_input(self.adapter, 1) as captured:
            with torch.enable_grad():
                logits = self.model(self.ids, use_cache=False).logits
            activation = captured["activation"]
        self.assertTrue(torch.equal(baseline, logits.detach()))
        self.assertTrue(activation.is_leaf)
        self.assertTrue(activation.requires_grad)
        self.assertEqual(tuple(activation.shape), (1, 8, 32))

    def test_capture_hook_is_removed_after_the_context(self) -> None:
        projection = self.adapter.mlp_projection(1)
        before = tuple(projection._forward_pre_hooks)
        with capture_c_proj_input(self.adapter, 1):
            self.assertNotEqual(tuple(projection._forward_pre_hooks), before)
        self.assertEqual(tuple(projection._forward_pre_hooks), before)

    def test_gradients_flow_to_the_c_proj_input(self) -> None:
        scored = score_example(self.model, self.adapter, self.ids, 0, [1, 2])
        self.assertEqual(tuple(scored.activation.shape), (8, 32))
        self.assertEqual(tuple(scored.gradient.shape), (8, 32))
        self.assertTrue(torch.isfinite(scored.gradient).all())
        self.assertGreater(float(scored.gradient.abs().max()), 0.0)
        self.assertTrue(0.0 <= scored.sink_future <= 1.0)

    def test_gradient_is_non_zero_at_every_token_position(self) -> None:
        # Positions before the second-half query window still matter, through the key/value
        # route: this is why attribution averages over all positions.
        scored = score_example(self.model, self.adapter, self.ids, 0, [1, 2])
        per_position = scored.gradient.abs().max(dim=1).values
        self.assertTrue(bool((per_position[:4] > 0).all()))

    def test_batch_larger_than_one_is_rejected(self) -> None:
        with self.assertRaises(AttributionError):
            score_example(
                self.model, self.adapter, self.ids.repeat(2, 1), 0, [1, 2]
            )

    def test_an_enclosing_no_grad_is_overridden(self) -> None:
        reference = score_example(self.model, self.adapter, self.ids, 0, [1, 2])
        with torch.no_grad():
            inside = score_example(self.model, self.adapter, self.ids, 0, [1, 2])
        self.assertTrue(torch.equal(reference.gradient, inside.gradient))
        self.assertGreater(float(inside.gradient.abs().max()), 0.0)

    def test_inference_mode_is_reported_rather_than_silently_ungradiented(self) -> None:
        with self.assertRaises(AttributionError):
            with torch.inference_mode():
                ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)
                score_example(self.model, self.adapter, ids, 0, [1, 2])


class CausalOrderTests(unittest.TestCase):
    """An MLP at layer l may target only attention layers j > l."""

    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")
        self.ids = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]], dtype=torch.long)

    def test_same_and_earlier_targets_are_rejected(self) -> None:
        for targets in ([5], [0, 5], [3, 5], [5, 9]):
            with self.subTest(targets=targets):
                with self.assertRaises(AttributionError):
                    require_future_targets(5, targets)

    def test_strictly_later_targets_are_accepted_and_sorted(self) -> None:
        self.assertEqual(require_future_targets(5, [10, 7, 9]), (7, 9, 10))
        self.assertEqual(require_future_targets(9, [10]), (10,))

    def test_empty_and_duplicate_targets_are_rejected(self) -> None:
        with self.assertRaises(AttributionError):
            require_future_targets(0, [])
        with self.assertRaises(AttributionError):
            require_future_targets(0, [7, 7])

    def test_objective_does_not_depend_on_same_or_earlier_attention(self) -> None:
        # Measured, not assumed: with parameters frozen, an objective built from attention at
        # layer j <= l is not a function of layer l's c_proj input at all, so the score
        # carries no grad_fn. This is stronger than a numerically zero gradient.
        for mlp_layer, attention_layers in ((2, [2]), (2, [0]), (1, [1]), (1, [0])):
            with self.subTest(mlp_layer=mlp_layer, attention_layers=attention_layers):
                self.assertFalse(objective_depends_on_layer(
                    self.model, self.adapter, self.ids, mlp_layer, attention_layers
                ))

    def test_objective_depends_on_strictly_later_attention(self) -> None:
        for mlp_layer, attention_layers in ((0, [1, 2]), (0, [2]), (1, [2])):
            with self.subTest(mlp_layer=mlp_layer, attention_layers=attention_layers):
                self.assertTrue(objective_depends_on_layer(
                    self.model, self.adapter, self.ids, mlp_layer, attention_layers
                ))


class RankingApiGuardTests(unittest.TestCase):
    """Ranking may read the discovery split and nothing else (AGENTS.md test 6)."""

    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")
        self.corpus = synthetic_corpus()
        self.future = {0: (1, 2), 1: (2,)}

    def _rank(self, **kwargs):
        return rank_neurons(
            self.model, self.adapter, self.corpus, self.future, **kwargs
        )

    def test_validation_and_test_splits_are_rejected(self) -> None:
        for split in ("validation", "test"):
            with self.subTest(split=split), self.assertRaises(LeakageError):
                self._rank(split=split)

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaises(LeakageError):
            self._rank(split="train")

    def test_held_out_items_are_rejected_at_the_item_level(self) -> None:
        # Second barrier: NeutralCorpus.items_for already filters by frozen split role, so
        # this guard fires only if a caller assembles items itself.
        held_out = [item for item in self.corpus.items if item.split != "discovery"]
        self.assertTrue(held_out)
        with self.assertRaises(LeakageError):
            _require_discovery_items(held_out, "discovery")
        discovery = [item for item in self.corpus.items if item.split == "discovery"]
        self.assertEqual(len(_require_discovery_items(discovery, "discovery")), 2)

    def test_downstream_benchmark_sources_are_rejected(self) -> None:
        for dataset_id in ("openai/gsm8k", "cais/mmlu"):
            corpus = synthetic_corpus(dataset_id=dataset_id)
            with self.subTest(dataset_id=dataset_id), self.assertRaises(LeakageError):
                rank_neurons(self.model, self.adapter, corpus, self.future)
        e1 = synthetic_corpus(corpus_id="e1_parity_mixture")
        with self.assertRaises(LeakageError):
            rank_neurons(self.model, self.adapter, e1, self.future)

    def test_non_causal_target_maps_are_rejected(self) -> None:
        for future in ({1: (1,)}, {2: (0,)}, {0: (0, 2)}, {1: ()}):
            with self.subTest(future=future), self.assertRaises(AttributionError):
                rank_neurons(self.model, self.adapter, self.corpus, future)

    def test_an_empty_layer_map_is_rejected(self) -> None:
        with self.assertRaises(AttributionError):
            rank_neurons(self.model, self.adapter, self.corpus, {})

    def test_unfrozen_parameters_are_rejected(self) -> None:
        model = tiny_gpt2()
        model.requires_grad_(True)
        adapter = GPT2ModelAdapter(model, model_id="debug-gpt2")
        with self.assertRaises(AttributionError):
            rank_neurons(model, adapter, self.corpus, self.future)


class RankingOutputTests(unittest.TestCase):
    """Row schema, ranking order, and determinism."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_gpt2()
        cls.adapter = GPT2ModelAdapter(cls.model, model_id="debug-gpt2")
        cls.corpus = synthetic_corpus()
        cls.future = {0: (1, 2), 1: (2,)}
        cls.result = rank_neurons(cls.model, cls.adapter, cls.corpus, cls.future)
        cls.rows = attribution_rows(cls.result)

    def test_one_row_per_eligible_layer_neuron_pair(self) -> None:
        width = self.adapter.mlp_width(0)
        self.assertEqual(len(self.rows), len(self.future) * width)
        self.assertEqual(self.result.n_rows, len(self.rows))
        seen = {(row["layer"], row["neuron"]) for row in self.rows}
        self.assertEqual(len(seen), len(self.rows))

    def test_neuron_ids_are_in_range_for_every_layer(self) -> None:
        for layer in self.future:
            width = self.adapter.mlp_width(layer)
            ids = sorted(row["neuron"] for row in self.rows if row["layer"] == layer)
            with self.subTest(layer=layer):
                self.assertEqual(ids, list(range(width)))

    def test_only_eligible_layers_appear(self) -> None:
        self.assertEqual({row["layer"] for row in self.rows}, set(self.future))

    def test_future_sink_layers_are_recorded_per_row(self) -> None:
        for row in self.rows:
            targets = [int(part) for part in row["future_sink_layers"].split("|")]
            with self.subTest(layer=row["layer"]):
                self.assertEqual(tuple(targets), self.future[row["layer"]])
                self.assertTrue(all(target > row["layer"] for target in targets))

    def test_token_count_covers_every_position(self) -> None:
        for row in self.rows:
            self.assertEqual(row["n_examples"], 2)
            self.assertEqual(row["n_tokens"], 2 * self.corpus.cut_length)

    def test_scores_are_finite_and_absolute_dominates_signed(self) -> None:
        for row in self.rows:
            self.assertTrue(np.isfinite(row["mean_abs_attr"]))
            self.assertTrue(np.isfinite(row["mean_signed_attr"]))
            self.assertGreaterEqual(row["mean_abs_attr"], 0.0)
            self.assertGreaterEqual(row["mean_abs_activation"], 0.0)
            # mean|x| >= |mean x|; ranking by the absolute score cannot be cancellation.
            self.assertGreaterEqual(
                row["mean_abs_attr"] + 1e-12, abs(row["mean_signed_attr"])
            )

    def test_ranks_are_a_permutation_ordered_by_the_absolute_score(self) -> None:
        self.assertEqual(
            sorted(row["rank_abs"] for row in self.rows), list(range(1, len(self.rows) + 1))
        )
        by_rank = sorted(self.rows, key=lambda row: row["rank_abs"])
        scores = [row["mean_abs_attr"] for row in by_rank]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for layer in self.future:
            width = self.adapter.mlp_width(layer)
            in_layer = sorted(
                row["rank_abs_in_layer"] for row in self.rows if row["layer"] == layer
            )
            with self.subTest(layer=layer):
                self.assertEqual(in_layer, list(range(1, width + 1)))

    def test_rows_are_emitted_in_layer_neuron_order(self) -> None:
        keys = [(row["layer"], row["neuron"]) for row in self.rows]
        self.assertEqual(keys, sorted(keys))

    def test_ranking_is_deterministic_under_a_fixed_seed(self) -> None:
        torch.manual_seed(0)
        repeat = rank_neurons(self.model, self.adapter, self.corpus, self.future)
        repeat_rows = attribution_rows(repeat)
        self.assertEqual(repeat_rows, self.rows)
        self.assertEqual(attribution_sha256(repeat_rows), attribution_sha256(self.rows))
        self.assertEqual(repeat.example_ids, self.result.example_ids)

    def test_gradients_were_actually_observed(self) -> None:
        for layer_result in self.result.layers:
            with self.subTest(layer=layer_result.layer):
                self.assertEqual(layer_result.nonfinite_values, 0)
                self.assertEqual(layer_result.zero_gradient_examples, 0)
                self.assertGreater(layer_result.max_abs_gradient, 0.0)
                self.assertGreater(float(layer_result.mean_abs_attr.max()), 0.0)

    def test_max_examples_limits_the_read(self) -> None:
        limited = rank_neurons(
            self.model, self.adapter, self.corpus, self.future, max_examples=1
        )
        self.assertEqual(limited.n_examples, 1)
        self.assertEqual(limited.layers[0].n_tokens, self.corpus.cut_length)


class FrozenScopeLoadTests(unittest.TestCase):
    """Task 5 consumes Task 4's frozen scope; loading re-verifies it."""

    def setUp(self) -> None:
        if not FROZEN_SCOPE.is_file():
            self.skipTest("configs/frozen/sink_scope.json is not present")
        self.document = json.loads(FROZEN_SCOPE.read_text(encoding="utf-8"))

    def _write(self, document: dict, *, restamp: bool = True) -> Path:
        if restamp:
            document = dict(document)
            document["sink_scope_sha256"] = canonical_sha256(
                {k: v for k, v in document.items() if k != "sink_scope_sha256"}
            )
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(document, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return Path(handle.name)

    def test_the_frozen_scope_matches_what_task4_recorded(self) -> None:
        scope = load_frozen_sink_scope(
            FROZEN_SCOPE, expected_corpus_manifest_sha256=TASK4_CORPUS_SHA256
        )
        self.assertEqual(scope.sink_layers, TASK4_SINK_LAYERS)
        self.assertEqual(scope.eligible_mlp_layers, TASK4_ELIGIBLE_MLP_LAYERS)
        self.assertEqual(scope.sink_scope_sha256, TASK4_SCOPE_SHA256)
        self.assertEqual(scope.num_layers, 12)
        self.assertEqual(scope.num_heads, 12)
        self.assertEqual(scope.seq_len, 40)
        self.assertEqual(scope.targets_for(0), (7, 9, 10))
        self.assertEqual(scope.targets_for(7), (9, 10))
        self.assertEqual(scope.targets_for(9), (10,))

    def test_every_frozen_target_is_strictly_later_than_its_mlp_layer(self) -> None:
        scope = load_frozen_sink_scope(FROZEN_SCOPE)
        for layer, targets in scope.future_sink_layers.items():
            with self.subTest(layer=layer):
                self.assertTrue(targets)
                self.assertTrue(all(target > layer for target in targets))
                self.assertTrue(set(targets) <= set(scope.sink_layers))
        for layer in (10, 11):
            with self.subTest(layer=layer), self.assertRaises(FrozenScopeError):
                scope.targets_for(layer)

    def test_a_modified_document_is_rejected(self) -> None:
        document = dict(self.document)
        document["sink_layers"] = [7, 9, 10, 11]
        with self.assertRaises(FrozenScopeError):
            load_frozen_sink_scope(self._write(document, restamp=False))

    def test_a_mismatched_corpus_hash_is_rejected(self) -> None:
        with self.assertRaises(FrozenScopeError):
            load_frozen_sink_scope(
                FROZEN_SCOPE, expected_corpus_manifest_sha256="0" * 64
            )

    def test_a_non_causal_scope_is_rejected_even_when_correctly_hashed(self) -> None:
        document = dict(self.document)
        future = dict(document["future_sink_layers"])
        future["9"] = [9, 10]  # same-layer target
        document["future_sink_layers"] = future
        with self.assertRaises(FrozenScopeError):
            load_frozen_sink_scope(self._write(document))

    def test_a_target_outside_the_frozen_sink_layers_is_rejected(self) -> None:
        document = dict(self.document)
        future = dict(document["future_sink_layers"])
        future["9"] = [10, 11]
        document["future_sink_layers"] = future
        with self.assertRaises(FrozenScopeError):
            load_frozen_sink_scope(self._write(document))

    def test_a_missing_file_and_a_wrong_schema_are_rejected(self) -> None:
        with self.assertRaises(FrozenScopeError):
            load_frozen_sink_scope(FROZEN_SCOPE.parent / "does_not_exist.json")
        document = dict(self.document)
        document["schema"] = "sink_scope_v99"
        with self.assertRaises(FrozenScopeError):
            load_frozen_sink_scope(self._write(document))


@unittest.skipUnless(
    os.environ.get("NEURON_SINK_RUN_GPU_INTEGRATION") == "1",
    "set NEURON_SINK_RUN_GPU_INTEGRATION=1 for the cached GPT-2 CUDA test",
)
class GPT2CudaAttributionTests(unittest.TestCase):
    """One real GPT-2 example on a registered dev GPU, at the frozen scope."""

    def test_attribution_on_the_real_model_and_frozen_scope(self) -> None:
        if not (FROZEN_SCOPE.is_file() and FROZEN_MANIFEST.is_file()):
            self.skipTest("frozen corpus/scope are not present")
        require_registered_gpu("dev")
        corpus = NeutralCorpus.load(FROZEN_MANIFEST)
        scope = load_frozen_sink_scope(
            FROZEN_SCOPE, expected_corpus_manifest_sha256=corpus.manifest_sha256
        )
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
            cache_dir=os.environ.get("NEURON_SINK_HF_CACHE") or None,
            attn_implementation="eager",
            dtype=torch.float32,
        ).eval().to("cuda:0")
        model.requires_grad_(False)
        adapter = GPT2ModelAdapter(model, model_id="openai-community/gpt2")
        self.assertEqual(adapter.mlp_width(9), 3072)

        item = corpus.items_for("discovery", smoke=True)[0]
        ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device="cuda:0")

        scored = score_example(model, adapter, ids, 9, scope.targets_for(9))
        self.assertEqual(tuple(scored.activation.shape), (40, 3072))
        self.assertEqual(tuple(scored.gradient.shape), (40, 3072))
        self.assertTrue(torch.isfinite(scored.gradient).all())
        self.assertGreater(float(scored.gradient.abs().max()), 0.0)

        # The differentiable objective must equal the frozen metric on the same attentions.
        with torch.inference_mode():
            output = model(ids, output_attentions=True, use_cache=False)
        attentions = [attention[0] for attention in output.attentions]
        frozen = sink_scalar_from_map(
            per_layer_head_position0_attention(attentions), list(scope.targets_for(9))
        )
        self.assertLessEqual(abs(scored.sink_future - frozen), 1e-5)

        # Causal ordering on the real model: layer 9 cannot reach sink layer 7.
        self.assertFalse(objective_depends_on_layer(model, adapter, ids, 9, [7]))
        self.assertTrue(objective_depends_on_layer(model, adapter, ids, 9, [10]))


if __name__ == "__main__":
    unittest.main()
