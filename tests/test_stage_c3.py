from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neuron_sink.ablation_screen import (
    ScreenTable,
    baseline_per_layer_sink,
    measure_joint_effect,
    screen_neurons,
)
from neuron_sink.attribution import score_example
from neuron_sink.attribution_c3 import (
    AttributionC3Error,
    ROW_FIELDS as C3_ROW_FIELDS,
    attribution_rows,
    attribution_sha256,
    load_frozen_per_target_attribution,
    rank_neurons_per_target,
    score_example_per_target,
)
from neuron_sink.corpus import (
    CorpusError,
    LeakageError,
    NeutralCorpus,
    NeutralCorpusItem,
    build_neutral_corpus,
)
from neuron_sink.layer_baseline import ceiling_by_sink_layer, layer_attenuation_ceiling
from neuron_sink.model_adapters import GPT2ModelAdapter
from neuron_sink.provenance import canonical_sha256
from neuron_sink.selection import per_layer_counts
from neuron_sink.selection_c3 import (
    SelectionC3Error,
    build_c3_neuron_sets_document,
    build_c3_selection_conditions,
    select_per_sink_layer_budget,
    sink_layer_budget,
    verify_c3_neuron_sets_document,
)
from neuron_sink.sink_metrics import (
    differentiable_sink_score,
    differentiable_sink_scores_per_layer,
    load_frozen_sink_scope,
    sink_scalar_from_map,
    per_layer_head_position0_attention,
)
from neuron_sink.stage_c2 import OPERATING_POINT_SCHEMA as C2_OPERATING_POINT_SCHEMA
from neuron_sink.stage_c3 import (
    AMENDMENT,
    EXPERIMENT_ID,
    OPERATING_POINT_SCHEMA,
    REGISTERED_BLOCK_INDICES,
    REGISTERED_CORPUS_ID,
    StageC3Error,
    evaluate_formal_gate,
    freeze_operating_point,
    registered_window,
    stage_c3_run_root,
    unlock_test_split,
    verify_fresh_corpus,
)


ROOT = Path(__file__).resolve().parents[1]

#: A deliberately scattered sink scope on a 6-layer model, so reachability actually bites:
#: MLP 4 reaches only sink layer 5, while MLP 0 reaches all three.
SINK_LAYERS = (1, 3, 5)
ELIGIBLE = (0, 1, 2, 3, 4)
FUTURE = {0: (1, 3, 5), 1: (3, 5), 2: (3, 5), 3: (5,), 4: (5,)}


def tiny_gpt2(n_layer: int = 6, n_inner: int = 24) -> GPT2LMHeadModel:
    torch.manual_seed(29)
    config = GPT2Config(
        vocab_size=83,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=n_layer,
        n_head=4,
        n_inner=n_inner,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config).eval()
    model.requires_grad_(False)
    return model


def synthetic_items(count: int = 3, seq_len: int = 8, split: str = "discovery"):
    return tuple(
        NeutralCorpusItem(
            item_id=f"openwebtext:validation:0:blk{600 + index}",
            split=split,
            text=f"synthetic {index}",
            input_ids=tuple((index * 11 + 5 + step) % 70 + 1 for step in range(seq_len)),
            n_tokens=seq_len,
            meta={"purpose": "ppl", "block_index": 600 + index},
        )
        for index in range(count)
    )


def synthetic_corpus(
    *,
    split_size: int = 2,
    seq_len: int = 8,
    block_offset: int = 600,
    purpose: str = "ppl",
    corpus_id: str = REGISTERED_CORPUS_ID,
    skip_blocks: int = 300,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> NeutralCorpus:
    names = ("discovery", "validation", "test")
    items = []
    for position, name in enumerate(names):
        for index in range(split_size):
            block = block_offset + position * split_size + index
            items.append(NeutralCorpusItem(
                item_id=f"openwebtext:validation:0:blk{block}",
                split=name,
                text=f"synthetic {block}",
                input_ids=tuple((block + step) % 70 + 1 for step in range(seq_len)),
                n_tokens=seq_len,
                meta={"purpose": purpose, "block_index": block},
            ))
    return NeutralCorpus(
        corpus_id=corpus_id,
        items=tuple(items),
        tokenizer_name=tokenizer_name,
        tokenizer_revision=None,
        cut_length=seq_len,
        seed=0,
        source={
            "dataset_id": "Skylion007/openwebtext",
            "corpus_id": corpus_id,
            "purpose": purpose,
            "skip_blocks": skip_blocks,
        },
        upstream_manifest_sha256="0" * 64,
        upstream_provenance={},
    )


def write_scope(tmpdir: Path, *, corpus_sha: str = "0" * 64) -> Path:
    document = {
        "schema": "sink_scope_v1",
        "model_id": "debug-gpt2",
        "model_revision": "debug",
        "corpus_manifest_sha256": corpus_sha,
        "sink_layers": list(SINK_LAYERS),
        "sink_heads": {str(layer): [0, 1] for layer in SINK_LAYERS},
        "eligible_mlp_layers": list(ELIGIBLE),
        "future_sink_layers": {str(k): list(v) for k, v in FUTURE.items()},
        "num_layers": 6,
        "num_heads": 4,
        "seq_len": 8,
    }
    document["sink_scope_sha256"] = canonical_sha256(document)
    path = tmpdir / "sink_scope.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class PerLayerSinkScoreTests(unittest.TestCase):
    """Splitting the registered scalar must not change it."""

    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        mask = torch.ones(12, 12, dtype=torch.bool).tril()
        self.attentions = [
            torch.softmax(
                torch.randn(4, 12, 12, generator=generator).masked_fill(
                    ~mask, float("-inf")
                ),
                dim=-1,
            )
            for _ in range(6)
        ]

    def test_mean_of_terms_is_the_registered_aggregate(self) -> None:
        for layers in (None, [1, 3, 5], [5], [0, 1, 2]):
            with self.subTest(layers=layers):
                terms = differentiable_sink_scores_per_layer(self.attentions, layers)
                aggregate = differentiable_sink_score(self.attentions, layers)
                self.assertAlmostEqual(float(terms.mean()), float(aggregate), places=12)

    def test_terms_follow_the_caller_layer_order(self) -> None:
        terms = differentiable_sink_scores_per_layer(self.attentions, [5, 1, 3])
        mapping = per_layer_head_position0_attention(self.attentions)
        for position, layer in enumerate([5, 1, 3]):
            self.assertAlmostEqual(
                float(terms[position]), sink_scalar_from_map(mapping, [layer]), places=6
            )

    def test_float32_upcast_leaves_a_float32_input_unchanged(self) -> None:
        plain = differentiable_sink_score(self.attentions, [1, 3, 5])
        upcast = differentiable_sink_score(
            self.attentions, [1, 3, 5], dtype=torch.float32
        )
        self.assertEqual(float(plain), float(upcast))

    def test_native_bfloat16_quantises_the_metric_but_the_upcast_does_not(self) -> None:
        # The structural claim, independent of any particular fixture: reducing in the model
        # dtype returns a bfloat16 scalar, so the sink is representable only to ~8 mantissa
        # bits. At a real sink of ~0.6 that is a resolution of ~0.0023 -- the same order as a
        # whole matched-random control effect in Stage C.
        low = [attention.to(torch.bfloat16) for attention in self.attentions]
        native = differentiable_sink_score(low, [1, 3, 5])
        upcast = differentiable_sink_score(low, [1, 3, 5], dtype=torch.float32)
        self.assertEqual(native.dtype, torch.bfloat16)
        self.assertEqual(upcast.dtype, torch.float32)
        self.assertEqual(
            float(native), float(native.to(torch.bfloat16)),
            "a bfloat16 reduction cannot carry more than bfloat16 resolution",
        )

    def test_upcast_is_closer_to_the_float32_reference_on_average(self) -> None:
        # A single subset can be lucky, so average the error over every subset.
        low = [attention.to(torch.bfloat16) for attention in self.attentions]
        subsets = [[1], [3], [5], [1, 3], [3, 5], [1, 3, 5], [0, 1, 2, 3, 4, 5]]
        native_error = 0.0
        upcast_error = 0.0
        for layers in subsets:
            reference = float(differentiable_sink_score(self.attentions, layers))
            native_error += abs(
                float(differentiable_sink_score(low, layers)) - reference
            )
            upcast_error += abs(
                float(differentiable_sink_score(low, layers, dtype=torch.float32))
                - reference
            )
        self.assertLess(upcast_error, native_error)

    def test_scores_carry_gradient(self) -> None:
        attentions = [a.clone().requires_grad_(True) for a in self.attentions]
        terms = differentiable_sink_scores_per_layer(attentions, [1, 3, 5])
        self.assertEqual(tuple(terms.shape), (3,))
        self.assertTrue(terms.requires_grad)


class PerTargetAttributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")
        self.ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]], dtype=torch.long)

    def test_per_target_gradients_mean_to_the_aggregate_gradient(self) -> None:
        per_target = score_example_per_target(
            self.model, self.adapter, self.ids, 0, FUTURE[0], metric_dtype=None
        )
        aggregate = score_example(self.model, self.adapter, self.ids, 0, FUTURE[0])
        self.assertEqual(tuple(per_target.gradients.shape), (3, 8, 24))
        difference = (per_target.gradients.mean(dim=0) - aggregate.gradient).abs().max()
        self.assertLess(float(difference), 1e-6)
        self.assertAlmostEqual(
            sum(per_target.sink_per_target) / 3, aggregate.sink_future, places=6
        )

    def test_each_target_gradient_differs(self) -> None:
        scored = score_example_per_target(self.model, self.adapter, self.ids, 0, FUTURE[0])
        first, second = scored.gradients[0], scored.gradients[1]
        self.assertGreater(float((first - second).abs().max()), 0.0)

    def test_non_causal_targets_are_rejected(self) -> None:
        with self.assertRaises(Exception):
            score_example_per_target(self.model, self.adapter, self.ids, 3, [3])

    def test_batch_larger_than_one_is_rejected(self) -> None:
        with self.assertRaises(AttributionC3Error):
            score_example_per_target(
                self.model, self.adapter, self.ids.repeat(2, 1), 0, FUTURE[0]
            )

    def test_inference_mode_is_reported(self) -> None:
        with self.assertRaises(AttributionC3Error):
            with torch.inference_mode():
                ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]], dtype=torch.long)
                score_example_per_target(self.model, self.adapter, ids, 0, FUTURE[0])


class PerTargetRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_gpt2()
        cls.adapter = GPT2ModelAdapter(cls.model, model_id="debug-gpt2")
        cls.corpus = synthetic_corpus(split_size=2, seq_len=8)
        cls.result = rank_neurons_per_target(
            cls.model, cls.adapter, cls.corpus, FUTURE, split="discovery"
        )
        cls.rows = attribution_rows(cls.result)

    def test_one_row_per_layer_neuron_target(self) -> None:
        expected = sum(24 * len(FUTURE[layer]) for layer in ELIGIBLE)
        self.assertEqual(len(self.rows), expected)
        self.assertEqual(self.result.n_rows, expected)

    def test_every_row_is_causally_ordered(self) -> None:
        for row in self.rows:
            self.assertGreater(row["target_sink_layer"], row["mlp_layer"])
            self.assertIn(row["target_sink_layer"], FUTURE[row["mlp_layer"]])

    def test_rows_are_finite_and_absolute_dominates_signed(self) -> None:
        for row in self.rows:
            self.assertTrue(np.isfinite(row["mean_signed_attr"]))
            self.assertGreaterEqual(
                row["mean_abs_attr"] + 1e-12, abs(row["mean_signed_attr"])
            )

    def test_ranking_is_deterministic(self) -> None:
        repeat = rank_neurons_per_target(
            self.model, self.adapter, self.corpus, FUTURE, split="discovery"
        )
        self.assertEqual(attribution_rows(repeat), self.rows)
        self.assertEqual(
            attribution_sha256(attribution_rows(repeat)), attribution_sha256(self.rows)
        )

    def test_both_signs_are_present(self) -> None:
        # The whole point of the amendment: magnitude alone cannot tell these apart.
        signs = {np.sign(row["mean_signed_attr"]) for row in self.rows}
        self.assertIn(1.0, signs)
        self.assertIn(-1.0, signs)

    def test_held_out_splits_are_rejected(self) -> None:
        for split in ("validation", "test"):
            with self.subTest(split=split), self.assertRaises(LeakageError):
                rank_neurons_per_target(
                    self.model, self.adapter, self.corpus, FUTURE, split=split
                )

    def test_round_trip_through_the_loader(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmpdir = Path(raw)
            scope_path = write_scope(
                tmpdir, corpus_sha=self.corpus.manifest_sha256
            )
            scope = load_frozen_sink_scope(scope_path)
            csv_path = tmpdir / "attr.csv"
            import csv as _csv

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = _csv.DictWriter(handle, fieldnames=list(C3_ROW_FIELDS))
                writer.writeheader()
                writer.writerows(self.rows)
            metadata = {
                "schema": "neuron_attribution_c3_v1",
                "ranking_score": "mean_signed_attr",
                "corpus_manifest_sha256": self.corpus.manifest_sha256,
                "sink_scope_sha256": scope.sink_scope_sha256,
                "attribution_sha256": attribution_sha256(self.rows),
                "eligible_mlp_layers": list(ELIGIBLE),
                "mlp_width": {str(layer): 24 for layer in ELIGIBLE},
                "model_id": "debug-gpt2",
                "model_revision": "debug",
            }
            metadata_path = tmpdir / "attr.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            ranking = load_frozen_per_target_attribution(
                csv_path, metadata_path,
                scope=scope,
                expected_corpus_manifest_sha256=self.corpus.manifest_sha256,
            )
            self.assertEqual(ranking.pool_size, 24 * len(ELIGIBLE))
            self.assertEqual(ranking.target_sink_layers, SINK_LAYERS)
            for target in SINK_LAYERS:
                candidates = ranking.candidates_for_target(target)
                self.assertTrue(candidates)
                self.assertTrue(all(
                    row["mlp_layer"] < target for row in candidates
                ))
                self.assertTrue(all(
                    row["mean_signed_attr"] > 0.0 for row in candidates
                ))

            # A tampered score must not survive the hash check.
            bad = list(self.rows)
            bad[0] = dict(bad[0])
            bad[0]["mean_signed_attr"] = bad[0]["mean_signed_attr"] + 1.0
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = _csv.DictWriter(handle, fieldnames=list(C3_ROW_FIELDS))
                writer.writeheader()
                writer.writerows(bad)
            with self.assertRaises(AttributionC3Error):
                load_frozen_per_target_attribution(
                    csv_path, metadata_path,
                    scope=scope,
                    expected_corpus_manifest_sha256=self.corpus.manifest_sha256,
                )


class AblationScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_gpt2()
        cls.adapter = GPT2ModelAdapter(cls.model, model_id="debug-gpt2")
        cls.items = synthetic_items(count=3, seq_len=8)
        cls.baseline = baseline_per_layer_sink(
            cls.model, cls.items, SINK_LAYERS, device=torch.device("cpu")
        )
        cls.candidates = [(layer, neuron) for layer in ELIGIBLE for neuron in range(6)]
        cls.screen = screen_neurons(
            cls.model, cls.adapter, cls.items, cls.candidates, SINK_LAYERS,
            baseline=cls.baseline, device=torch.device("cpu"),
        )

    def test_baseline_shape(self) -> None:
        self.assertEqual(self.baseline.shape, (3, 3))

    def test_only_causally_reachable_targets_are_measured(self) -> None:
        for row in self.screen.rows:
            self.assertGreater(row["target_sink_layer"], row["mlp_layer"])

    def test_row_count_matches_reachable_pairs(self) -> None:
        expected = sum(
            len([t for t in SINK_LAYERS if t > layer]) * 6 for layer in ELIGIBLE
        )
        self.assertEqual(len(self.screen.rows), expected)

    def test_screen_is_deterministic(self) -> None:
        repeat = screen_neurons(
            self.model, self.adapter, self.items, self.candidates, SINK_LAYERS,
            baseline=self.baseline, device=torch.device("cpu"),
        )
        self.assertEqual(repeat.sha256(), self.screen.sha256())

    def test_candidates_for_target_are_positive_and_sorted(self) -> None:
        for target in SINK_LAYERS:
            rows = self.screen.candidates_for_target(target)
            deltas = [row["measured_delta_sink"] for row in rows]
            self.assertEqual(deltas, sorted(deltas, reverse=True))
            self.assertTrue(all(value > 0.0 for value in deltas))

    def test_measured_effects_are_not_all_zero(self) -> None:
        self.assertGreater(
            max(abs(row["measured_delta_sink"]) for row in self.screen.rows), 0.0
        )

    def test_joint_effect_reports_every_sink_layer(self) -> None:
        from neuron_sink.suppression import NeuronSet

        joint = measure_joint_effect(
            self.model, self.adapter, self.items,
            NeuronSet({0: (0, 1), 2: (3,)}, source="targeted"),
            SINK_LAYERS, baseline=self.baseline, device=torch.device("cpu"),
        )
        self.assertEqual(len(joint["rsr_per_layer"]), len(SINK_LAYERS))
        self.assertEqual(joint["sink_layers"], list(SINK_LAYERS))


class BudgetTests(unittest.TestCase):
    LAYERS = [4, 6, 14, 23, 24, 25, 26]
    BASE = [0.570990, 0.580959, 0.627029, 0.668493, 0.609233, 0.672835, 0.594128]

    def test_budget_sums_to_k_for_every_registered_fraction(self) -> None:
        for k in (23, 116, 233, 582, 1165, 2330):
            with self.subTest(k=k):
                budget = sink_layer_budget(k, self.LAYERS, self.BASE)
                self.assertEqual(sum(budget.values()), k)
                self.assertEqual(sorted(budget), sorted(self.LAYERS))

    def test_budget_tracks_metric_weight(self) -> None:
        budget = sink_layer_budget(2330, self.LAYERS, self.BASE)
        total = sum(self.BASE)
        for layer, base in zip(self.LAYERS, self.BASE):
            self.assertAlmostEqual(budget[layer] / 2330, base / total, places=2)

    def test_every_sink_layer_is_funded_when_k_exceeds_the_target_count(self) -> None:
        budget = sink_layer_budget(len(self.LAYERS), self.LAYERS, self.BASE)
        self.assertTrue(all(count >= 1 for count in budget.values()))

    def test_small_k_leaves_low_weight_layers_unfunded(self) -> None:
        budget = sink_layer_budget(3, self.LAYERS, self.BASE)
        self.assertEqual(sum(budget.values()), 3)
        self.assertTrue(any(count == 0 for count in budget.values()))

    def test_invalid_requests_are_rejected(self) -> None:
        with self.assertRaises(SelectionC3Error):
            sink_layer_budget(0, self.LAYERS, self.BASE)
        with self.assertRaises(SelectionC3Error):
            sink_layer_budget(10, self.LAYERS, self.BASE[:-1])
        with self.assertRaises(SelectionC3Error):
            sink_layer_budget(10, self.LAYERS, [0.0] * len(self.LAYERS))
        with self.assertRaises(TypeError):
            sink_layer_budget(True, self.LAYERS, self.BASE)


class SelectionC3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_gpt2()
        cls.adapter = GPT2ModelAdapter(cls.model, model_id="debug-gpt2")
        cls.items = synthetic_items(count=3, seq_len=8)
        cls.candidates = [(layer, neuron) for layer in ELIGIBLE for neuron in range(24)]
        cls.screen = screen_neurons(
            cls.model, cls.adapter, cls.items, cls.candidates, SINK_LAYERS,
            device=torch.device("cpu"),
        )

    def test_selection_reaches_every_sink_layer(self) -> None:
        neuron_set, diagnostics = select_per_sink_layer_budget(
            self.screen, 12, eligible_layers=ELIGIBLE
        )
        self.assertEqual(sum(len(v) for v in neuron_set.by_layer.values()), 12)
        self.assertEqual(diagnostics["unreachable_sink_layers"], [])
        self.assertEqual(diagnostics["reachable_sink_layers"], list(SINK_LAYERS))
        self.assertAlmostEqual(diagnostics["reachable_metric_weight"], 1.0, places=9)

    def test_selection_draws_only_from_reachable_layers(self) -> None:
        _, diagnostics = select_per_sink_layer_budget(
            self.screen, 12, eligible_layers=ELIGIBLE
        )
        for drawn in diagnostics["drawn_for"]:
            if drawn["source"] == "quota":
                self.assertLess(drawn["mlp_layer"], drawn["target_sink_layer"])

    def test_selection_is_deterministic(self) -> None:
        first, _ = select_per_sink_layer_budget(self.screen, 9, eligible_layers=ELIGIBLE)
        second, _ = select_per_sink_layer_budget(self.screen, 9, eligible_layers=ELIGIBLE)
        self.assertEqual(first, second)

    def test_no_duplicate_neurons(self) -> None:
        neuron_set, _ = select_per_sink_layer_budget(
            self.screen, 15, eligible_layers=ELIGIBLE
        )
        pairs = [
            (layer, neuron)
            for layer, neurons in neuron_set.by_layer.items()
            for neuron in neurons
        ]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_conditions_and_document_verify(self) -> None:
        class _Ranking:
            eligible_mlp_layers = ELIGIBLE
            mlp_width = {layer: 24 for layer in ELIGIBLE}
            pool_size = 24 * len(ELIGIBLE)
            attribution_sha256 = "a" * 64
            corpus_manifest_sha256 = "b" * 64
            sink_scope_sha256 = "c" * 64
            model_id = "debug-gpt2"
            model_revision = "debug"

        fractions = (1.0, 2.0)
        conditions, diagnostics = build_c3_selection_conditions(
            _Ranking(), self.screen, fractions, control_draws=3,
        )
        self.assertEqual(len(conditions), len(fractions) * 4)
        document = build_c3_neuron_sets_document(
            _Ranking(), conditions, self.screen, diagnostics,
            fractions_percent=fractions, control_draws=3,
            experiment_id=EXPERIMENT_ID,
        )
        frozen = verify_c3_neuron_sets_document(document)
        self.assertEqual(len(frozen.neuron_sets), len(conditions))

        # Controls preserve the target's per-layer counts exactly and never overlap it.
        for fraction in fractions:
            label = f"f{fraction:.2f}".replace(".", "p")
            target = frozen.neuron_sets[f"targeted_{label}"]
            counts = per_layer_counts(target, ELIGIBLE)
            for draw in range(3):
                control = frozen.neuron_sets[f"layer_random_{label}_s{draw}"]
                self.assertEqual(per_layer_counts(control, ELIGIBLE), counts)
                for layer, neurons in control.by_layer.items():
                    self.assertFalse(set(neurons) & set(target.by_layer.get(layer, ())))

    def test_a_tampered_document_is_rejected(self) -> None:
        class _Ranking:
            eligible_mlp_layers = ELIGIBLE
            mlp_width = {layer: 24 for layer in ELIGIBLE}
            pool_size = 24 * len(ELIGIBLE)
            attribution_sha256 = "a" * 64
            corpus_manifest_sha256 = "b" * 64
            sink_scope_sha256 = "c" * 64
            model_id = "debug-gpt2"
            model_revision = "debug"

        conditions, diagnostics = build_c3_selection_conditions(
            _Ranking(), self.screen, (1.0,), control_draws=2,
        )
        document = build_c3_neuron_sets_document(
            _Ranking(), conditions, self.screen, diagnostics,
            fractions_percent=(1.0,), control_draws=2, experiment_id=EXPERIMENT_ID,
        )
        document["budget_rule"] = "something else"
        with self.assertRaises(SelectionC3Error):
            verify_c3_neuron_sets_document(document)


class LayerCeilingTests(unittest.TestCase):
    def test_ceiling_is_measured_per_sink_layer_and_respects_causal_order(self) -> None:
        model = tiny_gpt2()
        adapter = GPT2ModelAdapter(model, model_id="debug-gpt2")
        items = synthetic_items(count=2, seq_len=8)
        document = layer_attenuation_ceiling(
            model, adapter, items, ELIGIBLE, SINK_LAYERS,
            alphas=(1.0, 0.5, 0.0), device=torch.device("cpu"),
        )
        self.assertTrue(document["causal_order_pass"])
        self.assertEqual(document["alphas"], [0.5, 0.0])
        # Every eligible layer, plus the all-eligible-layers condition, at each alpha.
        self.assertEqual(len(document["aggregates"]), (len(ELIGIBLE) + 1) * 2)

        # The all-layers condition is the only genuine bound on a multi-layer set, and it must
        # dominate every single-layer condition on the aggregate metric at the same alpha.
        from neuron_sink.layer_baseline import ALL_LAYERS

        joint = [
            entry for entry in document["aggregates"]
            if entry["mlp_layer"] == ALL_LAYERS and entry["alpha"] == 0.0
        ]
        self.assertEqual(len(joint), 1)
        self.assertEqual(document["maximal_intervention"], [
            entry for entry in document["aggregates"]
            if entry["mlp_layer"] == ALL_LAYERS
        ])
        # The all-layers condition reaches every sink layer after the earliest eligible MLP
        # layer, so its reachability weight is 1.0. It is deliberately NOT asserted to
        # dominate the single-layer conditions: suppression is not monotone in the sink, and
        # asserting dominance would encode an overclaim the Stage-C result disproves.
        self.assertEqual(joint[0]["reachable_metric_weight"], 1.0)
        self.assertTrue(all(
            entry["reachable_metric_weight"] <= 1.0 + 1e-12
            for entry in document["aggregates"]
        ))
        # Every row is a reachable pair; unreachable pairs are omitted, not zero-filled.
        for row in document["rows"]:
            self.assertEqual(row["reachable"], row["target_sink_layer"] > row["mlp_layer"])
        ceiling = ceiling_by_sink_layer(document, alpha=0.0)
        self.assertEqual(
            [entry["target_sink_layer"] for entry in ceiling["per_sink_layer"]],
            list(SINK_LAYERS),
        )


class CorpusWindowTests(unittest.TestCase):
    def test_skip_blocks_requires_the_ppl_purpose(self) -> None:
        with self.assertRaises(CorpusError):
            build_neutral_corpus(object(), purpose="sink", skip_blocks=300)

    def test_skip_blocks_rejects_bad_values(self) -> None:
        with self.assertRaises(TypeError):
            build_neutral_corpus(object(), purpose="ppl", skip_blocks=True)
        with self.assertRaises(CorpusError):
            build_neutral_corpus(object(), purpose="ppl", skip_blocks=-1)


class StageC3BoundaryTests(unittest.TestCase):
    def test_run_roots_are_separate_from_every_earlier_stage(self) -> None:
        stamp = "run_20260905T000000Z"
        full = stage_c3_run_root(ROOT, "qwen2.5-1.5b-instruct", registered_run=True, stamp=stamp)
        pre = stage_c3_run_root(ROOT, "qwen2.5-1.5b-instruct", registered_run=False, stamp=stamp)
        self.assertIn("stage_c3_full", str(full))
        self.assertIn("stage_c3_preflight", str(pre))
        self.assertNotIn("stage_c2", str(full))
        self.assertNotIn("stage_c_full", str(full))
        gpt2 = stage_c3_run_root(ROOT, "gpt2-small", registered_run=True, stamp=stamp)
        self.assertNotEqual(full, gpt2)
        with self.assertRaises(StageC3Error):
            stage_c3_run_root(ROOT, "gpt2-medium", registered_run=True, stamp=stamp)
        with self.assertRaises(StageC3Error):
            stage_c3_run_root(ROOT, "gpt2-small", registered_run=True, stamp="nope")

    def test_registered_windows_are_model_specific_and_disjoint(self) -> None:
        qwen_id, qwen_blocks, qwen_skip = registered_window("qwen2.5-1.5b-instruct")
        gpt2_id, gpt2_blocks, gpt2_skip = registered_window("gpt2-small")
        self.assertEqual(qwen_blocks, REGISTERED_BLOCK_INDICES)
        self.assertEqual(qwen_skip, 300)
        self.assertEqual(gpt2_skip, 0)
        self.assertNotEqual(qwen_id, gpt2_id)
        # Qwen's C3 window must not touch Stage C (0-299) or Stage C2 (300-599).
        self.assertFalse(set(qwen_blocks) & set(range(0, 600)))

    def test_fresh_corpus_requires_disjointness_from_every_predecessor(self) -> None:
        # Full registered size: the contract checks 3 x 100 blocks at 600-899.
        c3 = synthetic_corpus(split_size=100, block_offset=600)
        stage_c = synthetic_corpus(
            split_size=100, block_offset=0, purpose="sink",
            corpus_id="openwebtext_validation_sink_300", skip_blocks=0,
        )
        stage_c2 = synthetic_corpus(
            split_size=100, block_offset=300,
            corpus_id="openwebtext_validation_ppl_300", skip_blocks=0,
        )
        checks = verify_fresh_corpus(
            c3, [stage_c, stage_c2], model_alias="qwen2.5-1.5b-instruct"
        )
        self.assertEqual(checks["predecessor_count"], 2)
        self.assertTrue(checks["c3_block_window_pass"])
        self.assertTrue(checks["block_indices_disjoint_from_1_pass"])

        with self.assertRaises(StageC3Error):
            verify_fresh_corpus(c3, [], model_alias="qwen2.5-1.5b-instruct")
        # Overlapping the Stage-C2 window must be refused.
        overlapping = synthetic_corpus(split_size=100, block_offset=300)
        with self.assertRaises(StageC3Error):
            verify_fresh_corpus(
                overlapping, [stage_c, stage_c2], model_alias="qwen2.5-1.5b-instruct"
            )

    def test_operating_point_is_relabelled_and_rehashed(self) -> None:
        document = {
            "schema": OPERATING_POINT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "amendment": AMENDMENT,
            "stage": "validation",
            "payload": 1,
        }
        document["operating_point_sha256"] = canonical_sha256(document)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "operating_point.json"
            freeze_operating_point(path, document)
            with self.assertRaises(FileExistsError):
                freeze_operating_point(path, document)

    def test_earlier_stage_operating_points_cannot_unlock_c3(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "operating_point.json"
            foreign = {
                "schema": C2_OPERATING_POINT_SCHEMA,
                "experiment_id": "stage_c2_qwen_signed_replication_v1",
                "amendment": "A005",
                "stage": "validation",
            }
            foreign["operating_point_sha256"] = canonical_sha256(foreign)
            path.write_text(json.dumps(foreign), encoding="utf-8")
            with self.assertRaises(StageC3Error):
                unlock_test_split(
                    path,
                    model_id="x", model_revision="y",
                    corpus_manifest_sha256="a", sink_scope_sha256="b",
                    attribution_sha256="c", neuron_sets_sha256="d",
                )

    def test_missing_operating_point_refuses_the_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(StageC3Error):
                unlock_test_split(Path(raw) / "absent.json")

    def test_freeze_refuses_a_foreign_document(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(StageC3Error):
                freeze_operating_point(
                    Path(raw) / "operating_point.json",
                    {"schema": C2_OPERATING_POINT_SCHEMA},
                )

    def test_dry_run_cannot_claim_a_scientific_gate(self) -> None:
        gate = evaluate_formal_gate(
            [], (), all_identity_pass=True, all_validity_pass=True,
            state_leakage_pass=True, registered_run=False,
        )
        self.assertEqual(gate["status"], "NOT_EVALUATED_DRY_RUN")
        self.assertFalse(gate["test_split_accessed"])
        self.assertEqual(gate["experiment_id"], EXPERIMENT_ID)
        self.assertNotIn("formal_gate_sha256", gate)


if __name__ == "__main__":
    unittest.main()
