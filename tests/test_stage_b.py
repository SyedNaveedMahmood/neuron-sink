from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel

from neuron_sink.evaluation import EVALUATION_PROTOCOL, PHENOMENON_ROW_FIELDS
from neuron_sink.model_adapters import GPT2ModelAdapter
from neuron_sink.selection import (
    CONTROL_TYPE_LAYER_RANDOM,
    CONTROL_TYPE_TARGETED,
    FULL_CONTROL_DRAWS,
    FULL_FRACTIONS_PERCENT,
    FrozenNeuronSets,
    exact_k,
    fraction_label,
    generate_layer_matched_controls,
)
from neuron_sink.stage_b import (
    FULL_ALPHAS,
    OPERATING_POINT_SCHEMA,
    StageBError,
    build_operating_point_document,
    evaluate_formal_gate,
    freeze_operating_point,
    registered_full_conditions,
    stage_b_run_root,
    unlock_test_split,
)
from neuron_sink.suppression import NeuronSet


def full_frozen_grid(width: int = 1000) -> FrozenNeuronSets:
    condition_ids = []
    records = {}
    sets = {}
    for fraction in FULL_FRACTIONS_PERCENT:
        label = fraction_label(fraction)
        k = exact_k(fraction, width)
        target_id = f"targeted_{label}"
        target = NeuronSet({0: tuple(range(k))}, source=CONTROL_TYPE_TARGETED)
        condition_ids.append(target_id)
        sets[target_id] = target
        records[target_id] = {
            "control_type": CONTROL_TYPE_TARGETED,
            "control_seed": None,
            "fraction_percent": fraction,
            "k": k,
        }
        controls = generate_layer_matched_controls(
            target,
            eligible_layers=(0,),
            widths={0: width},
            k=k,
            draws=FULL_CONTROL_DRAWS,
        )
        for seed, control in enumerate(controls):
            condition_id = f"layer_random_{label}_s{seed}"
            condition_ids.append(condition_id)
            sets[condition_id] = control
            records[condition_id] = {
                "control_type": CONTROL_TYPE_LAYER_RANDOM,
                "control_seed": seed,
                "fraction_percent": fraction,
                "k": k,
            }
    return FrozenNeuronSets(
        document={
            "fractions_percent": list(FULL_FRACTIONS_PERCENT),
            "control_draws": FULL_CONTROL_DRAWS,
            "condition_ids": condition_ids,
            "conditions": records,
        },
        neuron_sets=sets,
    )


def valid_row(**updates):
    row = {
        "experiment_id": "stage_b_test",
        "model_id": "debug-gpt2",
        "stage": "validation",
        "example_id": "example-0",
        "condition_id": "targeted_f0p01",
        "condition_order": 1,
        "alpha_order": 5,
        "control_type": "targeted",
        "control_seed": None,
        "fraction": 0.0001,
        "fraction_percent": 0.01,
        "k": 1,
        "alpha": 0.0,
        "sink_baseline": 1.0,
        "sink_intervened": 0.9,
        "delta_sink": -0.1,
        "relative_sink_reduction": 0.1,
        "ce_baseline": 4.0,
        "ce_intervened": 4.05,
        "delta_ce": 0.05,
        "ppl_baseline": 54.598150033144236,
        "ppl_intervened": 57.39745704544619,
        "kl_baseline_to_intervened": 0.01,
        "top1_flip_rate": 0.1,
        "prompt_tokens": 40,
        "prediction_tokens": 39,
        "logits_exact_match": False,
        "attentions_exact_match": False,
        "max_logits_abs_diff": 0.1,
        "max_attention_abs_diff": 0.1,
        "baseline_valid": True,
        "intervention_valid": True,
        "valid_forward": True,
        "nonfinite_logits": 0,
        "nonfinite_attention": 0,
        "all_zero_logits": False,
        "max_attention_row_sum_error": 1e-7,
        "max_causal_future_attention": 0.0,
        "min_attention_value": 0.0,
        "forward_runtime_seconds": 0.01,
    }
    row.update(updates)
    assert set(row) == set(PHENOMENON_ROW_FIELDS)
    return row


def full_rows(stage: str, conditions, n_examples: int, *, winning: bool = True):
    rows = []
    target_max = {
        0.01: 0.05,
        0.05: 0.15 if winning else 0.04,
        0.10: 0.14 if winning else 0.04,
        0.25: 0.13 if winning else 0.04,
        0.50: 0.12 if winning else 0.04,
        1.00: 0.11 if winning else 0.04,
    }
    for example_index in range(n_examples):
        example_id = f"example-{example_index}"
        rows.append(valid_row(
            stage=stage,
            example_id=example_id,
            condition_id="baseline",
            condition_order=0,
            alpha_order=0,
            control_type="baseline",
            control_seed=None,
            fraction=None,
            fraction_percent=None,
            k=None,
            alpha=1.0,
            sink_intervened=1.0,
            delta_sink=0.0,
            relative_sink_reduction=0.0,
            ce_intervened=4.0,
            delta_ce=0.0,
            logits_exact_match=True,
            attentions_exact_match=True,
            max_logits_abs_diff=0.0,
            max_attention_abs_diff=0.0,
        ))
        for condition in conditions:
            maximum = (
                target_max[condition.fraction_percent]
                if condition.control_type == CONTROL_TYPE_TARGETED
                else 0.01
            )
            for alpha_order, alpha in enumerate(FULL_ALPHAS, start=1):
                rsr = maximum * (1.0 - alpha)
                identity = alpha == 1.0
                rows.append(valid_row(
                    stage=stage,
                    example_id=example_id,
                    condition_id=condition.condition_id,
                    condition_order=condition.condition_order,
                    alpha_order=alpha_order,
                    control_type=condition.control_type,
                    control_seed=condition.control_seed,
                    fraction=condition.fraction_percent / 100.0,
                    fraction_percent=condition.fraction_percent,
                    k=condition.k,
                    alpha=alpha,
                    sink_intervened=1.0 - rsr,
                    delta_sink=-rsr,
                    relative_sink_reduction=rsr,
                    ce_intervened=(
                        4.0
                        if identity
                        else 4.05 if condition.control_type == CONTROL_TYPE_TARGETED else 4.01
                    ),
                    delta_ce=(
                        0.0
                        if identity
                        else 0.05 if condition.control_type == CONTROL_TYPE_TARGETED else 0.01
                    ),
                    logits_exact_match=identity,
                    attentions_exact_match=identity,
                    max_logits_abs_diff=0.0 if identity else 0.1,
                    max_attention_abs_diff=0.0 if identity else 0.1,
                ))
    return rows


class FullGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conditions = registered_full_conditions(full_frozen_grid())

    def test_exact_full_fraction_alpha_and_twenty_control_grid(self) -> None:
        self.assertEqual(FULL_ALPHAS, (1.0, 0.75, 0.5, 0.25, 0.0))
        self.assertEqual(len(self.conditions), 126)
        self.assertEqual(
            sum(c.control_type == CONTROL_TYPE_TARGETED for c in self.conditions), 6
        )
        for fraction in FULL_FRACTIONS_PERCENT:
            controls = [
                c for c in self.conditions
                if c.fraction_percent == fraction
                and c.control_type == CONTROL_TYPE_LAYER_RANDOM
            ]
            self.assertEqual(len(controls), 20)
            self.assertEqual({c.control_seed for c in controls}, set(range(20)))

    def test_twenty_controls_are_deterministic(self) -> None:
        first = registered_full_conditions(full_frozen_grid())
        second = registered_full_conditions(full_frozen_grid())
        self.assertEqual(
            [c.neuron_set for c in first], [c.neuron_set for c in second]
        )

    def test_gpt2_small_and_medium_widths_are_validated_independently(self) -> None:
        small = GPT2ModelAdapter(GPT2LMHeadModel(GPT2Config(
            n_layer=2, n_head=2, n_embd=8, n_inner=32, vocab_size=32
        )))
        medium_shape = GPT2ModelAdapter(GPT2LMHeadModel(GPT2Config(
            n_layer=3, n_head=4, n_embd=16, n_inner=64, vocab_size=32
        )))
        self.assertEqual(small.mlp_width(0), 32)
        self.assertEqual(medium_shape.mlp_width(0), 64)
        with self.assertRaises(IndexError):
            small.validate_neuron(0, 32)
        self.assertEqual(medium_shape.validate_neuron(0, 63), 63)

    def test_evaluation_protocol_remains_the_stage_a_protocol(self) -> None:
        self.assertEqual(EVALUATION_PROTOCOL, "neutral_next_token_sink_ce_kl_top1_v1")

    def test_model_artifact_roots_are_separate_and_cannot_overwrite_smoke_freezes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            small = stage_b_run_root(
                directory, "gpt2-small", registered_run=True,
                stamp="run_20260904T000000Z",
            )
            medium = stage_b_run_root(
                directory, "gpt2-medium", registered_run=True,
                stamp="run_20260904T000000Z",
            )
            self.assertNotEqual(small, medium)
            self.assertIn("stage_b_full", small.parts)
            self.assertNotIn("configs", small.parts)
            self.assertNotIn("frozen", small.parts)


class OperatingPointAndGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conditions = registered_full_conditions(full_frozen_grid())

    def test_validation_selects_smallest_qualifying_fraction_only(self) -> None:
        rows = full_rows("validation", self.conditions, 4)
        document = build_operating_point_document(
            rows,
            self.conditions,
            model_id="debug-gpt2",
            model_revision="revision-a",
            corpus_manifest_sha256="corpus",
            sink_scope_sha256="scope",
            attribution_sha256="attribution",
            neuron_sets_sha256="sets",
            bootstrap_resamples=100,
            expected_examples=4,
        )
        self.assertEqual(document["schema"], OPERATING_POINT_SCHEMA)
        self.assertEqual(document["operating_point_type"], "k_star")
        self.assertEqual(document["selected_fraction_percent"], 0.05)
        self.assertFalse(document["exploratory_only"])

    def test_no_qualifier_freezes_k_max_effect_without_tuning(self) -> None:
        rows = full_rows("validation", self.conditions, 4, winning=False)
        document = build_operating_point_document(
            rows,
            self.conditions,
            model_id="debug-gpt2",
            model_revision="revision-a",
            corpus_manifest_sha256="corpus",
            sink_scope_sha256="scope",
            attribution_sha256="attribution",
            neuron_sets_sha256="sets",
            bootstrap_resamples=100,
            expected_examples=4,
        )
        self.assertEqual(document["operating_point_type"], "k_max_effect")
        self.assertTrue(document["exploratory_only"])

    def test_locked_test_refuses_missing_or_non_registered_validation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "operating_point.json"
            with self.assertRaises(StageBError):
                unlock_test_split(
                    missing,
                    model_id="m",
                    model_revision="r",
                    corpus_manifest_sha256="c",
                    sink_scope_sha256="s",
                    attribution_sha256="a",
                    neuron_sets_sha256="n",
                )
            rows = full_rows("validation", self.conditions, 4)
            document = build_operating_point_document(
                rows,
                self.conditions,
                model_id="m",
                model_revision="r",
                corpus_manifest_sha256="c",
                sink_scope_sha256="s",
                attribution_sha256="a",
                neuron_sets_sha256="n",
                bootstrap_resamples=20,
                expected_examples=4,
            )
            freeze_operating_point(missing, document)
            with self.assertRaises(StageBError):
                unlock_test_split(
                    missing,
                    model_id="m",
                    model_revision="r",
                    corpus_manifest_sha256="c",
                    sink_scope_sha256="s",
                    attribution_sha256="a",
                    neuron_sets_sha256="n",
                )
            with self.assertRaises(FileExistsError):
                freeze_operating_point(missing, document)

    def test_formal_gate_pass_and_null_cases(self) -> None:
        passing = evaluate_formal_gate(
            full_rows("test", self.conditions, 4),
            self.conditions,
            all_identity_pass=True,
            all_validity_pass=True,
            state_leakage_pass=True,
            registered_run=True,
            bootstrap_resamples=100,
            expected_examples=4,
        )
        self.assertTrue(passing["formal_gate_pass"])
        self.assertIn(0.05, passing["passing_fractions"])
        null = evaluate_formal_gate(
            full_rows("test", self.conditions, 4, winning=False),
            self.conditions,
            all_identity_pass=True,
            all_validity_pass=True,
            state_leakage_pass=True,
            registered_run=True,
            bootstrap_resamples=100,
            expected_examples=4,
        )
        self.assertFalse(null["formal_gate_pass"])

    def test_dry_run_cannot_emit_scientific_gate(self) -> None:
        result = evaluate_formal_gate(
            [],
            self.conditions,
            all_identity_pass=True,
            all_validity_pass=True,
            state_leakage_pass=True,
            registered_run=False,
        )
        self.assertEqual(result["status"], "NOT_EVALUATED_DRY_RUN")
        self.assertFalse(result["test_split_accessed"])


if __name__ == "__main__":
    unittest.main()
