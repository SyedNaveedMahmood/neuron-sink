from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from neuron_sink.attribution import RANKING_SCORE
from neuron_sink.provenance import canonical_sha256, read_json, write_json
from neuron_sink.selection import (
    CONDITION_ROW_FIELDS,
    CONTROL_SEED_DERIVATION,
    FULL_CONTROL_DRAWS,
    FULL_FRACTIONS_PERCENT,
    ROUNDING_RULE,
    SMOKE_CONTROL_DRAWS,
    SMOKE_FRACTIONS_PERCENT,
    FrozenAttributionRanking,
    SelectionError,
    build_neuron_sets_document,
    build_selection_conditions,
    condition_rows,
    exact_k,
    fraction_label,
    generate_layer_matched_controls,
    load_frozen_attribution,
    load_frozen_neuron_sets,
    per_layer_counts,
    select_global_top_k,
    verify_neuron_sets_document,
)
from neuron_sink.sink_metrics import load_frozen_sink_scope
from neuron_sink.suppression import NeuronSet


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "frozen"
FROZEN_MANIFEST = FROZEN / "neutral_corpus_manifest.json"
FROZEN_SCOPE = FROZEN / "sink_scope.json"
FROZEN_ATTRIBUTION = FROZEN / "neuron_attribution.csv"
FROZEN_ATTRIBUTION_META = FROZEN / "neuron_attribution_metadata.json"
FROZEN_NEURON_SETS = FROZEN / "neuron_sets.json"

TASK4_CORPUS_SHA256 = "c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7"
TASK4_SCOPE_SHA256 = "b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235"
TASK5_ATTRIBUTION_SHA256 = "9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692"


def synthetic_ranking(num_layers: int = 3, width: int = 100) -> FrozenAttributionRanking:
    """Small total-order ranking; later layers deliberately dominate the head."""

    rows = []
    for layer in range(num_layers):
        for neuron in range(width):
            score = float(layer * width + neuron + 1)
            rows.append({
                "layer": layer,
                "neuron": neuron,
                "mean_abs_activation": score / 10,
                # Opposite ordering ensures selection cannot accidentally use signed attr.
                "mean_signed_attr": -score * 100,
                "mean_abs_attr": score,
                "n_examples": 2,
                "n_tokens": 16,
                "future_sink_layers": str(num_layers),
                "rank_abs": 0,
                "rank_abs_in_layer": width - neuron,
            })
    ranked = sorted(
        rows, key=lambda row: (-row["mean_abs_attr"], row["layer"], row["neuron"])
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_abs"] = rank
    rows.sort(key=lambda row: (row["layer"], row["neuron"]))
    return FrozenAttributionRanking(
        rows=tuple(rows),
        eligible_mlp_layers=tuple(range(num_layers)),
        mlp_width={layer: width for layer in range(num_layers)},
        attribution_sha256="synthetic",
        corpus_manifest_sha256="corpus",
        sink_scope_sha256="scope",
        model_id="synthetic",
        model_revision="unit-test",
        metadata={},
    )


class ExactKTests(unittest.TestCase):
    def test_all_six_registered_fractions_have_exact_expected_k(self) -> None:
        expected = (3, 15, 31, 77, 154, 307)
        self.assertEqual(
            tuple(exact_k(fraction, 30_720) for fraction in FULL_FRACTIONS_PERCENT),
            expected,
        )

    def test_rounding_rule_is_explicit_half_up_with_minimum_one(self) -> None:
        self.assertIn("ROUND_HALF_UP", ROUNDING_RULE)
        self.assertIn("minimum 1", ROUNDING_RULE)
        self.assertEqual(exact_k(50, 1), 1)  # exactly 0.5, not bankers' rounding
        self.assertEqual(exact_k(0.0001, 10), 1)

    def test_invalid_fraction_pool_and_oversized_k_are_rejected(self) -> None:
        for fraction in (0, -0.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                exact_k(fraction, 10)
        with self.assertRaises(ValueError):
            exact_k(200, 10)
        with self.assertRaises(ValueError):
            exact_k(1, 0)
        with self.assertRaises(TypeError):
            exact_k(1, True)

    def test_fraction_labels_are_stable_and_filename_safe(self) -> None:
        self.assertEqual(fraction_label(0.05), "f0p05")
        self.assertEqual(fraction_label(0.10), "f0p10")
        self.assertEqual(fraction_label(0.25), "f0p25")
        self.assertEqual(fraction_label(1.00), "f1p00")
        with self.assertRaises(ValueError):
            fraction_label(0.001)


class GlobalSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ranking = synthetic_ranking()

    def test_top_k_equals_rank_cut_and_independent_resort(self) -> None:
        selected = select_global_top_k(self.ranking, 7)
        actual = {
            (layer, neuron)
            for layer, neurons in selected.by_layer.items()
            for neuron in neurons
        }
        rank_cut = {
            (row["layer"], row["neuron"])
            for row in self.ranking.rows if row["rank_abs"] <= 7
        }
        independently_sorted = sorted(
            self.ranking.rows,
            key=lambda row: (
                -row["mean_abs_attr"], row["layer"], row["neuron"]
            ),
        )[:7]
        independent = {(row["layer"], row["neuron"]) for row in independently_sorted}
        self.assertEqual(actual, rank_cut)
        self.assertEqual(actual, independent)
        self.assertEqual(selected.source, "targeted")

    def test_signed_diagnostic_does_not_drive_selection(self) -> None:
        selected = select_global_top_k(self.ranking, 1)
        self.assertEqual(dict(selected.by_layer), {2: (99,)})

    def test_targeted_sets_are_nested_as_k_grows(self) -> None:
        sets = []
        for k in (3, 7, 15):
            neuron_set = select_global_top_k(self.ranking, k)
            sets.append({
                (layer, neuron)
                for layer, neurons in neuron_set.by_layer.items()
                for neuron in neurons
            })
        self.assertLess(sets[0], sets[1])
        self.assertLess(sets[1], sets[2])

    def test_zero_count_layers_are_omitted_from_neuron_set(self) -> None:
        selected = select_global_top_k(self.ranking, 3)
        self.assertEqual(list(selected.by_layer), [2])
        self.assertNotIn(0, selected.by_layer)
        self.assertNotIn(1, selected.by_layer)
        self.assertEqual(per_layer_counts(selected, (0, 1, 2)), {0: 0, 1: 0, 2: 3})

    def test_invalid_k_is_rejected(self) -> None:
        for k in (0, -1, self.ranking.pool_size + 1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                select_global_top_k(self.ranking, k)
        with self.assertRaises(TypeError):
            select_global_top_k(self.ranking, True)


class MatchedControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ranking = synthetic_ranking()
        self.target = select_global_top_k(self.ranking, 7)

    def _controls(self, draws: int = 5):
        return generate_layer_matched_controls(
            self.target,
            eligible_layers=self.ranking.eligible_mlp_layers,
            widths=self.ranking.mlp_width,
            k=7,
            draws=draws,
            base_seed=0,
        )

    def test_every_control_matches_layer_counts_and_excludes_target(self) -> None:
        target_counts = per_layer_counts(
            self.target, self.ranking.eligible_mlp_layers
        )
        for control in self._controls():
            self.assertEqual(
                per_layer_counts(control, self.ranking.eligible_mlp_layers),
                target_counts,
            )
            for layer, neurons in control.by_layer.items():
                self.assertFalse(
                    set(neurons) & set(self.target.by_layer.get(layer, ()))
                )

    def test_ids_are_unique_in_range_and_only_eligible_layers_appear(self) -> None:
        for control in self._controls():
            self.assertEqual(control.source, "layer_random")
            for layer, neurons in control.by_layer.items():
                self.assertIn(layer, self.ranking.eligible_mlp_layers)
                self.assertEqual(len(neurons), len(set(neurons)))
                self.assertTrue(
                    all(0 <= neuron < self.ranking.mlp_width[layer] for neuron in neurons)
                )

    def test_fixed_seed_reproduces_all_twenty_draws_exactly(self) -> None:
        left = self._controls(FULL_CONTROL_DRAWS)
        right = self._controls(FULL_CONTROL_DRAWS)
        self.assertEqual(left, right)
        self.assertEqual(
            [control.selection_seed for control in left], list(range(FULL_CONTROL_DRAWS))
        )
        self.assertIn("registered_base_seed", CONTROL_SEED_DERIVATION)
        self.assertIn("control_seed_draw_index", CONTROL_SEED_DERIVATION)
        self.assertIn("k", CONTROL_SEED_DERIVATION)

    def test_different_draw_indices_produce_different_sets(self) -> None:
        signatures = {
            tuple((layer, neurons) for layer, neurons in control.by_layer.items())
            for control in self._controls(FULL_CONTROL_DRAWS)
        }
        self.assertEqual(len(signatures), FULL_CONTROL_DRAWS)

    def test_invalid_draw_and_seed_requests_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._controls(0)
        with self.assertRaises(ValueError):
            generate_layer_matched_controls(
                self.target,
                eligible_layers=self.ranking.eligible_mlp_layers,
                widths=self.ranking.mlp_width,
                k=7,
                draws=1,
                base_seed=-1,
            )


class FrozenAttributionLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            FROZEN_MANIFEST,
            FROZEN_SCOPE,
            FROZEN_ATTRIBUTION,
            FROZEN_ATTRIBUTION_META,
        )
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("Task-4/5 frozen inputs are not present")
        cls.manifest = read_json(FROZEN_MANIFEST)
        cls.scope = load_frozen_sink_scope(
            FROZEN_SCOPE,
            expected_corpus_manifest_sha256=cls.manifest["manifest_sha256"],
        )
        cls.ranking = load_frozen_attribution(
            FROZEN_ATTRIBUTION,
            FROZEN_ATTRIBUTION_META,
            scope=cls.scope,
            expected_corpus_manifest_sha256=cls.manifest["manifest_sha256"],
        )

    def test_frozen_csv_reloads_with_exact_types_and_hash(self) -> None:
        self.assertEqual(self.ranking.attribution_sha256, TASK5_ATTRIBUTION_SHA256)
        self.assertEqual(self.ranking.corpus_manifest_sha256, TASK4_CORPUS_SHA256)
        self.assertEqual(self.ranking.sink_scope_sha256, TASK4_SCOPE_SHA256)
        row = self.ranking.rows[0]
        for field in ("layer", "neuron", "n_examples", "n_tokens", "rank_abs"):
            self.assertIsInstance(row[field], int)
        for field in ("mean_abs_activation", "mean_signed_attr", "mean_abs_attr"):
            self.assertIsInstance(row[field], float)
        self.assertIsInstance(row["future_sink_layers"], str)

    def test_real_pool_layers_ranges_ranks_and_future_targets(self) -> None:
        self.assertEqual(self.ranking.pool_size, 30_720)
        self.assertEqual(self.ranking.eligible_mlp_layers, tuple(range(10)))
        self.assertEqual(set(self.ranking.mlp_width.values()), {3072})
        self.assertEqual(
            sorted(row["rank_abs"] for row in self.ranking.rows),
            list(range(1, 30_721)),
        )
        for layer in self.ranking.eligible_mlp_layers:
            rows = [row for row in self.ranking.rows if row["layer"] == layer]
            self.assertEqual(sorted(row["neuron"] for row in rows), list(range(3072)))
            self.assertTrue(all(
                all(int(target) > layer for target in row["future_sink_layers"].split("|"))
                for row in rows
            ))

    def test_tampered_csv_value_is_rejected_without_regenerating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "attribution.csv"
            lines = FROZEN_ATTRIBUTION.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split(",")
            fields[4] = str(float(fields[4]) + 1.0)
            lines[1] = ",".join(fields)
            tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SelectionError, "hash mismatch"):
                load_frozen_attribution(
                    tampered,
                    FROZEN_ATTRIBUTION_META,
                    scope=self.scope,
                    expected_corpus_manifest_sha256=TASK4_CORPUS_SHA256,
                )

    def test_metadata_corpus_and_scope_mismatches_are_rejected(self) -> None:
        metadata = read_json(FROZEN_ATTRIBUTION_META)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            changed = dict(metadata)
            changed["corpus_manifest_sha256"] = "0" * 64
            write_json(path, changed)
            with self.assertRaisesRegex(SelectionError, "corpus hash"):
                load_frozen_attribution(
                    FROZEN_ATTRIBUTION,
                    path,
                    scope=self.scope,
                    expected_corpus_manifest_sha256=TASK4_CORPUS_SHA256,
                )
            changed = dict(metadata)
            changed["sink_scope_sha256"] = "0" * 64
            write_json(path, changed)
            with self.assertRaisesRegex(SelectionError, "scope hash"):
                load_frozen_attribution(
                    FROZEN_ATTRIBUTION,
                    path,
                    scope=self.scope,
                    expected_corpus_manifest_sha256=TASK4_CORPUS_SHA256,
                )


class RealSelectionAndSerializationTests(FrozenAttributionLoadTests):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.conditions = build_selection_conditions(
            cls.ranking,
            SMOKE_FRACTIONS_PERCENT,
            control_draws=SMOKE_CONTROL_DRAWS,
            base_seed=0,
        )
        cls.document = build_neuron_sets_document(
            cls.ranking,
            cls.conditions,
            fractions_percent=SMOKE_FRACTIONS_PERCENT,
            control_draws=SMOKE_CONTROL_DRAWS,
            base_seed=0,
        )

    def test_smoke_grid_has_stable_names_and_eighteen_conditions(self) -> None:
        self.assertEqual(len(self.conditions), 18)
        expected = []
        for label in ("f0p05", "f0p10", "f0p25"):
            expected.append(f"targeted_{label}")
            expected.extend(f"layer_random_{label}_s{index}" for index in range(5))
        self.assertEqual([condition.condition_id for condition in self.conditions], expected)

    def test_real_global_targets_have_expected_k_are_nested_and_omit_empty_layers(self) -> None:
        targets = [
            condition for condition in self.conditions
            if condition.control_type == "targeted"
        ]
        self.assertEqual([condition.k for condition in targets], [15, 31, 77])
        pair_sets = [
            {
                (layer, neuron)
                for layer, neurons in condition.neuron_set.by_layer.items()
                for neuron in neurons
            }
            for condition in targets
        ]
        self.assertLess(pair_sets[0], pair_sets[1])
        self.assertLess(pair_sets[1], pair_sets[2])
        for condition in targets:
            positive_layers = {
                layer for layer, count in condition.per_layer_counts.items() if count > 0
            }
            self.assertEqual(set(condition.neuron_set.by_layer), positive_layers)

    def test_every_real_set_round_trips_through_neuron_set_and_json(self) -> None:
        verified = verify_neuron_sets_document(self.document)
        self.assertEqual(len(verified.neuron_sets), 18)
        for condition in self.conditions:
            restored = verified.neuron_sets[condition.condition_id]
            self.assertIsInstance(restored, NeuronSet)
            self.assertEqual(restored, condition.neuron_set)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neuron_sets.json"
            write_json(path, self.document)
            loaded = load_frozen_neuron_sets(path)
        self.assertEqual(loaded.neuron_sets, verified.neuron_sets)

    def test_saved_document_is_hash_stable_and_tampering_is_rejected(self) -> None:
        repeat_conditions = build_selection_conditions(
            self.ranking,
            SMOKE_FRACTIONS_PERCENT,
            control_draws=SMOKE_CONTROL_DRAWS,
            base_seed=0,
        )
        repeat = build_neuron_sets_document(
            self.ranking,
            repeat_conditions,
            fractions_percent=SMOKE_FRACTIONS_PERCENT,
            control_draws=SMOKE_CONTROL_DRAWS,
            base_seed=0,
        )
        self.assertEqual(repeat, self.document)
        self.assertEqual(repeat["neuron_sets_sha256"], self.document["neuron_sets_sha256"])

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json(first, self.document)
            write_json(second, repeat)
            self.assertEqual(first.read_bytes(), second.read_bytes())

        tampered = json.loads(json.dumps(self.document))
        tampered["conditions"]["targeted_f0p05"]["k"] = 16
        with self.assertRaisesRegex(SelectionError, "hash mismatch"):
            verify_neuron_sets_document(tampered)

        # A caller cannot make changed controls valid merely by recomputing the document
        # checksum: every random set is regenerated from the recorded composite seed.
        restamped = json.loads(json.dumps(self.document))
        restamped["registered_base_seed"] = 1
        restamped["neuron_sets_sha256"] = canonical_sha256({
            key: value for key, value in restamped.items()
            if key != "neuron_sets_sha256"
        })
        with self.assertRaisesRegex(SelectionError, "composite RNG seed"):
            verify_neuron_sets_document(restamped)

    def test_flat_condition_table_has_registered_schema_and_row_count(self) -> None:
        rows = condition_rows(self.conditions)
        self.assertEqual(tuple(rows[0]), CONDITION_ROW_FIELDS)
        self.assertEqual(len(rows), (15 + 31 + 77) * 6)


class ImportIsolationTests(unittest.TestCase):
    def test_selection_has_no_direct_corpus_downstream_or_benchmark_import(self) -> None:
        source = (ROOT / "neuron_sink" / "selection.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden = ("corpus", "task_eval", "mmlu", "arc", "culturalbench", "gsm8k")
        self.assertFalse(any(
            token in module.lower() for module in imported for token in forbidden
        ))

    def test_task6_runner_is_cpu_only_and_does_not_load_a_model_or_split(self) -> None:
        source = (ROOT / "scripts" / "select_neurons.py").read_text(encoding="utf-8")
        for forbidden in (
            "require_registered_gpu",
            "GPT2LMHeadModel",
            "transformers",
            "NeutralCorpus",
            "items_for(",
            "suppress_neurons",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn('ProvenanceRecorder(device=None, gpu_name="cpu")', source)


@unittest.skipUnless(FROZEN_NEURON_SETS.is_file(), "Task-6 frozen sets not written yet")
class FrozenNeuronSetArtifactTests(unittest.TestCase):
    def test_real_frozen_artifact_loads_and_contains_only_valid_neuron_sets(self) -> None:
        frozen = load_frozen_neuron_sets(FROZEN_NEURON_SETS)
        self.assertEqual(len(frozen.neuron_sets), 18)
        self.assertEqual(frozen.document["ranking_score"], RANKING_SCORE)
        self.assertFalse(frozen.document["is_causal_evidence"])
        for neuron_set in frozen.neuron_sets.values():
            self.assertTrue(neuron_set.by_layer)
            self.assertTrue(set(neuron_set.by_layer) <= set(range(10)))
            for neurons in neuron_set.by_layer.values():
                self.assertTrue(neurons)
                self.assertEqual(len(neurons), len(set(neurons)))
                self.assertTrue(all(0 <= neuron < 3072 for neuron in neurons))


if __name__ == "__main__":
    unittest.main()
