from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_sink.corpus import NeutralCorpus
from neuron_sink.provenance import canonical_sha256, write_json
from neuron_sink.selection import FrozenAttributionRanking, SelectionError
from neuron_sink.signed_selection import (
    RANKING_SCORE,
    SCHEMA_VERSION,
    SELECTION_METHOD,
    build_signed_neuron_sets_document,
    build_signed_selection_conditions,
    load_signed_neuron_sets,
    select_global_top_k_positive_signed,
    verify_signed_targets,
)
from neuron_sink.stage_c2 import (
    EXPERIMENT_ID,
    FORMAL_GATE_SCHEMA,
    StageC2Error,
    evaluate_formal_gate,
    stage_c2_run_root,
    verify_fresh_corpus,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_C_MANIFEST = (
    ROOT / "configs" / "frozen" / "qwen2_5_1_5b_instruct" /
    "neutral_corpus_manifest.json"
)
C2_MANIFEST = (
    ROOT / "configs" / "frozen" / "qwen2_5_1_5b_instruct_c2" /
    "neutral_corpus_manifest.json"
)


def synthetic_ranking(scores: tuple[float, ...] = (-4, 3, 1, -2, 2, 4)):
    rows = []
    for neuron, signed in enumerate(scores):
        rows.append({
            "layer": neuron // 3,
            "neuron": neuron % 3,
            "mean_abs_activation": 1.0,
            "mean_signed_attr": float(signed),
            "mean_abs_attr": float(abs(signed) * 10 + neuron),
            "n_examples": 2,
            "n_tokens": 8,
            "future_sink_layers": "2",
            "rank_abs": neuron + 1,
            "rank_abs_in_layer": neuron % 3 + 1,
        })
    return FrozenAttributionRanking(
        rows=tuple(rows),
        eligible_mlp_layers=(0, 1),
        mlp_width={0: 3, 1: 3},
        attribution_sha256="attribution",
        corpus_manifest_sha256="corpus",
        sink_scope_sha256="scope",
        model_id="qwen",
        model_revision="revision",
        metadata={},
    )


class SignedSelectionTests(unittest.TestCase):
    def test_ranking_uses_positive_signed_direction_not_absolute_magnitude(self) -> None:
        ranking = synthetic_ranking()
        selected = select_global_top_k_positive_signed(ranking, 3)
        pairs = {
            (layer, neuron)
            for layer, neurons in selected.by_layer.items()
            for neuron in neurons
        }
        self.assertEqual(pairs, {(1, 2), (0, 1), (1, 1)})
        self.assertNotIn((0, 0), pairs)  # largest absolute negative score

    def test_a_nonpositive_cutoff_stops_before_intervention(self) -> None:
        with self.assertRaisesRegex(SelectionError, "strictly positive"):
            select_global_top_k_positive_signed(synthetic_ranking(), 5)

    def test_signed_sets_and_random_controls_round_trip_and_reverify(self) -> None:
        ranking = synthetic_ranking()
        conditions = build_signed_selection_conditions(
            ranking, (10.0, 30.0), control_draws=2, base_seed=0
        )
        document = build_signed_neuron_sets_document(
            ranking,
            conditions,
            fractions_percent=(10.0, 30.0),
            control_draws=2,
            base_seed=0,
            experiment_id=EXPERIMENT_ID,
        )
        self.assertEqual(document["schema"], SCHEMA_VERSION)
        self.assertEqual(document["selection_method"], SELECTION_METHOD)
        self.assertEqual(document["ranking_score"], RANKING_SCORE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sets.json"
            write_json(path, document)
            frozen = load_signed_neuron_sets(path, ranking=ranking)
        self.assertEqual(len(frozen.neuron_sets), 6)
        verify_signed_targets(frozen, ranking)

        changed = json.loads(json.dumps(document))
        changed["conditions"]["targeted_f10p00"]["by_layer"] = {"0": [0]}
        changed["neuron_sets_sha256"] = canonical_sha256({
            key: value for key, value in changed.items()
            if key != "neuron_sets_sha256"
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sets.json"
            write_json(path, changed)
            with self.assertRaises(SelectionError):
                load_signed_neuron_sets(path, ranking=ranking)


class StageC2BoundaryTests(unittest.TestCase):
    def test_output_root_is_separate_from_stage_c(self) -> None:
        path = stage_c2_run_root(
            Path("repo"),
            "qwen2.5-1.5b-instruct",
            registered_run=True,
            stamp="run_20260905T000000Z",
        )
        self.assertEqual(
            path.parts[-3:],
            ("stage_c2_full", "qwen2.5-1.5b-instruct", "run_20260905T000000Z"),
        )
        with self.assertRaises(StageC2Error):
            stage_c2_run_root(
                Path("repo"), "gpt2-small", registered_run=True,
                stamp="run_20260905T000000Z",
            )

    def test_dry_run_gate_cannot_claim_a_result(self) -> None:
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

    @unittest.skipUnless(
        STAGE_C_MANIFEST.is_file() and C2_MANIFEST.is_file(),
        "Qwen Stage-C and Stage-C2 manifests are required",
    )
    def test_frozen_c2_manifest_is_the_fresh_registered_block_window(self) -> None:
        checks = verify_fresh_corpus(
            NeutralCorpus.load(C2_MANIFEST), NeutralCorpus.load(STAGE_C_MANIFEST)
        )
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
