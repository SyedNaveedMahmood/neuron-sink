from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neuron_sink.evaluation import (
    EVALUATION_PROTOCOL,
    EXPERIMENT_ID,
    PHENOMENON_ROW_FIELDS,
    SMOKE_ALPHAS,
    EvaluationError,
    aggregate_phenomenon_rows,
    evaluate_smoke_gate,
    forward_snapshot,
    paired_metrics,
    registered_smoke_conditions,
    validate_phenomenon_row,
)
from neuron_sink.model_adapters import GPT2ModelAdapter
from neuron_sink.selection import load_frozen_neuron_sets
from neuron_sink.suppression import NeuronSet, suppress_neurons


ROOT = Path(__file__).resolve().parents[1]


def tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(19)
    config = GPT2Config(
        vocab_size=79,
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


def valid_row(**updates):
    row = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": "debug-gpt2",
        "stage": "test",
        "example_id": "example-0",
        "condition_id": "targeted_f0p05",
        "condition_order": 1,
        "alpha_order": 2,
        "control_type": "targeted",
        "control_seed": None,
        "fraction": 0.0005,
        "fraction_percent": 0.05,
        "k": 15,
        "alpha": 0.5,
        "sink_baseline": 0.7,
        "sink_intervened": 0.63,
        "delta_sink": -0.07,
        "relative_sink_reduction": 0.1,
        "ce_baseline": 4.0,
        "ce_intervened": 4.01,
        "delta_ce": 0.01,
        "ppl_baseline": math.exp(4.0),
        "ppl_intervened": math.exp(4.01),
        "kl_baseline_to_intervened": 0.001,
        "top1_flip_rate": 0.1,
        "prompt_tokens": 8,
        "prediction_tokens": 7,
        "logits_exact_match": False,
        "attentions_exact_match": False,
        "max_logits_abs_diff": 0.2,
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
    return row


def synthetic_gate_aggregates(*, target_wins: bool = True):
    rows = []
    k_by_fraction = {0.05: 15, 0.10: 31, 0.25: 77}
    for fraction, k in k_by_fraction.items():
        label = f"f{fraction:.2f}".replace(".", "p")
        for alpha in SMOKE_ALPHAS:
            target_rsr = {1.0: 0.0, 0.5: 0.08, 0.0: 0.16}[alpha]
            if not target_wins and fraction == 0.05 and alpha == 0.0:
                target_rsr = 0.01
            rows.append({
                "stage": "test",
                "condition_id": f"targeted_{label}",
                "control_type": "targeted",
                "control_seed": None,
                "fraction_percent": fraction,
                "k": k,
                "alpha": alpha,
                "relative_sink_reduction": target_rsr,
            })
            for seed in range(5):
                control_rsr = 0.02 + seed * 0.005 if alpha != 1.0 else 0.0
                if not target_wins:
                    control_rsr = 0.2 if alpha != 1.0 else 0.0
                rows.append({
                    "stage": "test",
                    "condition_id": f"layer_random_{label}_s{seed}",
                    "control_type": "layer_random",
                    "control_seed": seed,
                    "fraction_percent": fraction,
                    "k": k,
                    "alpha": alpha,
                    "relative_sink_reduction": control_rsr,
                })
    return rows


class FrozenGridTests(unittest.TestCase):
    def test_real_frozen_grid_has_exact_registered_order(self) -> None:
        frozen = load_frozen_neuron_sets(
            ROOT / "configs" / "frozen" / "neuron_sets.json"
        )
        conditions = registered_smoke_conditions(frozen)
        self.assertEqual(len(conditions), 18)
        self.assertEqual(conditions[0].condition_id, "targeted_f0p05")
        self.assertEqual(conditions[-1].condition_id, "layer_random_f0p25_s4")
        self.assertEqual([condition.condition_order for condition in conditions],
                         list(range(1, 19)))
        self.assertEqual(EVALUATION_PROTOCOL, "neutral_next_token_sink_ce_kl_top1_v1")

    def test_changed_condition_order_is_rejected(self) -> None:
        frozen = load_frozen_neuron_sets(
            ROOT / "configs" / "frozen" / "neuron_sets.json"
        )
        changed = copy.deepcopy(dict(frozen.document))
        changed["condition_ids"] = list(reversed(changed["condition_ids"]))
        # The checksum loader has already done its job; this specifically exercises the
        # Task-7 execution-order guard on an otherwise reconstructed object.
        object.__setattr__(frozen, "document", changed)
        with self.assertRaises(EvaluationError):
            registered_smoke_conditions(frozen)


class ForwardMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.adapter = GPT2ModelAdapter(self.model, model_id="debug-gpt2")
        self.ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)

    def snapshot(self):
        return forward_snapshot(
            self.model,
            self.ids,
            sink_layers=(2,),
            attention_tolerance=1e-5,
            causal_tolerance=1e-7,
        )

    def test_identity_metrics_are_exact(self) -> None:
        baseline = self.snapshot()
        with suppress_neurons(
            self.adapter, NeuronSet({1: (0, 5, 17)}, source="debug"), 1.0
        ):
            identity = self.snapshot()
        metrics = paired_metrics(baseline, identity)
        self.assertTrue(baseline.valid)
        self.assertTrue(identity.valid)
        self.assertTrue(metrics["logits_exact_match"])
        self.assertTrue(metrics["attentions_exact_match"])
        self.assertEqual(metrics["kl_baseline_to_intervened"], 0.0)
        self.assertEqual(metrics["top1_flip_rate"], 0.0)
        self.assertEqual(metrics["prediction_tokens"], 7)

    def test_suppression_produces_finite_paired_metrics(self) -> None:
        baseline = self.snapshot()
        with suppress_neurons(
            self.adapter, NeuronSet({0: (1, 4), 1: (3, 9)}, source="debug"), 0.0
        ):
            intervention = self.snapshot()
        metrics = paired_metrics(baseline, intervention)
        self.assertTrue(metrics["valid_forward"])
        self.assertGreaterEqual(metrics["kl_baseline_to_intervened"], 0.0)
        self.assertTrue(math.isfinite(metrics["relative_sink_reduction"]))


class RowAndAggregateTests(unittest.TestCase):
    def test_schema_accepts_complete_finite_row(self) -> None:
        row = valid_row()
        validate_phenomenon_row(row)
        self.assertEqual(set(row), set(PHENOMENON_ROW_FIELDS))

    def test_schema_rejects_nonfinite_and_bad_prediction_count(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_phenomenon_row(valid_row(ce_intervened=float("nan")))
        with self.assertRaises(EvaluationError):
            validate_phenomenon_row(valid_row(prediction_tokens=8))

    def test_aggregate_uses_ratio_of_aggregate_sink(self) -> None:
        first = valid_row(
            example_id="a", sink_baseline=0.8, sink_intervened=0.4,
            relative_sink_reduction=0.5,
        )
        second = valid_row(
            example_id="b", sink_baseline=0.2, sink_intervened=0.2,
            relative_sink_reduction=0.0,
        )
        aggregate = aggregate_phenomenon_rows([first, second])[0]
        self.assertEqual(aggregate["n_examples"], 2)
        self.assertAlmostEqual(aggregate["sink_baseline"], 0.5)
        self.assertAlmostEqual(aggregate["sink_intervened"], 0.3)
        self.assertAlmostEqual(aggregate["relative_sink_reduction"], 0.4)
        self.assertAlmostEqual(
            aggregate["mean_per_example_relative_sink_reduction"], 0.25
        )


class SmokeGateTests(unittest.TestCase):
    def test_gate_passes_when_target_beats_all_controls_and_dose_is_directional(self) -> None:
        gate = evaluate_smoke_gate(
            synthetic_gate_aggregates(),
            all_split_identity_pass=True,
            all_split_validity_pass=True,
            state_leakage_pass=True,
            registered_run=True,
        )
        self.assertTrue(gate["smoke_gate_pass"])
        self.assertTrue(gate["causal_superiority_pass"])
        self.assertTrue(gate["dose_direction_pass"])
        self.assertEqual(len(gate["passing_superiority_conditions"]), 6)

    def test_gate_is_null_when_no_target_beats_controls(self) -> None:
        gate = evaluate_smoke_gate(
            synthetic_gate_aggregates(target_wins=False),
            all_split_identity_pass=True,
            all_split_validity_pass=True,
            state_leakage_pass=True,
            registered_run=True,
        )
        self.assertFalse(gate["smoke_gate_pass"])
        self.assertFalse(gate["causal_superiority_pass"])

    def test_limited_run_cannot_evaluate_scientific_gate(self) -> None:
        gate = evaluate_smoke_gate(
            [],
            all_split_identity_pass=True,
            all_split_validity_pass=True,
            state_leakage_pass=True,
            registered_run=False,
        )
        self.assertEqual(gate["status"], "NOT_EVALUATED_DRY_RUN")
        self.assertNotIn("smoke_gate_pass", gate)


if __name__ == "__main__":
    unittest.main()
