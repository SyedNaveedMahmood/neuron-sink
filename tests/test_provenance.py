from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_sink.provenance import (
    EXPECTED_SINK_KD_COMMIT,
    EXPECTED_SINK_REPRO_COMMIT,
    REGISTERED_GPUS,
    ProvenanceError,
    canonical_json,
    canonical_sha256,
    package_versions,
    prepare_output_dir,
    read_json,
    require_registered_gpu,
    run_stamp,
    write_json,
)


class OutputContractTests(unittest.TestCase):
    def test_a_fresh_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run_a" / "nested"
            created = prepare_output_dir(target)
            self.assertTrue(created.is_dir())

    def test_an_existing_empty_directory_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run_b"
            target.mkdir()
            self.assertTrue(prepare_output_dir(target).is_dir())

    def test_a_non_empty_directory_is_refused(self) -> None:
        # Results are append-only: a completed run is never overwritten.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run_c"
            target.mkdir()
            (target / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_dir(target)

    def test_run_stamps_are_sortable_and_utc(self) -> None:
        stamp = run_stamp()
        self.assertTrue(stamp.startswith("run_"))
        self.assertTrue(stamp.endswith("Z"))


class CanonicalHashTests(unittest.TestCase):
    def test_hash_ignores_key_order(self) -> None:
        left = {"b": 2, "a": [1, 2, {"y": 1, "x": 0}]}
        right = {"a": [1, 2, {"x": 0, "y": 1}], "b": 2}
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))

    def test_hash_is_sensitive_to_values_and_order_of_lists(self) -> None:
        self.assertNotEqual(canonical_sha256([1, 2, 3]), canonical_sha256([1, 3, 2]))
        self.assertNotEqual(canonical_sha256({"a": 1}), canonical_sha256({"a": 2}))

    def test_hash_is_stable_across_calls(self) -> None:
        payload = {"layers": [7, 9, 10], "floor": 0.15}
        self.assertEqual(canonical_sha256(payload), canonical_sha256(payload))

    def test_canonical_json_is_compact_and_sorted(self) -> None:
        self.assertEqual(canonical_json({"b": 1, "a": 2}), '{"a":2,"b":1}')

    def test_numpy_scalars_serialise(self) -> None:
        import numpy as np

        self.assertEqual(canonical_json({"x": np.float64(0.5)}), '{"x":0.5}')
        self.assertEqual(canonical_json({"x": np.int64(3)}), '{"x":3}')

    def test_unsupported_types_raise_rather_than_hashing_a_repr(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json({"x": object()})


class JsonIoTests(unittest.TestCase):
    def test_write_then_read_round_trip(self) -> None:
        import numpy as np

        payload = {"sink_layers": [7, 9, 10], "score": np.float64(0.25)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            write_json(path, payload)
            loaded = read_json(path)
        self.assertEqual(loaded["sink_layers"], [7, 9, 10])
        self.assertAlmostEqual(loaded["score"], 0.25)

    def test_written_json_is_sorted_and_newline_terminated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            write_json(path, {"b": 1, "a": 2})
            text = path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertLess(text.index('"a"'), text.index('"b"'))
        self.assertEqual(json.loads(text), {"a": 2, "b": 1})


class RegisteredHardwareTests(unittest.TestCase):
    def test_amendment_a001_registers_both_development_gpus(self) -> None:
        self.assertIn("NVIDIA GeForce RTX 2060", REGISTERED_GPUS["dev"])
        self.assertIn("NVIDIA GeForce RTX 2060 SUPER", REGISTERED_GPUS["dev"])
        self.assertEqual(REGISTERED_GPUS["full"], ("NVIDIA GeForce RTX 4080 SUPER",))

    def test_an_unknown_profile_is_refused(self) -> None:
        with self.assertRaises(ProvenanceError):
            require_registered_gpu("laptop")

    def test_registered_gpu_names_are_exact_not_substrings(self) -> None:
        # An exact match keeps "RTX 2060" from silently accepting an unregistered card
        # whose name merely contains it.
        for names in REGISTERED_GPUS.values():
            for name in names:
                self.assertTrue(name.startswith("NVIDIA GeForce "))


class PinnedCommitTests(unittest.TestCase):
    def test_expected_commits_match_the_registered_provenance(self) -> None:
        # SOURCE_BRANCHES.md and configs/experiment_plan.yaml record these same hashes.
        self.assertEqual(
            EXPECTED_SINK_REPRO_COMMIT, "9ab67e914464b13863b67527d8ea14068ee9ff10"
        )
        self.assertEqual(
            EXPECTED_SINK_KD_COMMIT, "db114c9c5eb6ffc5de13e444c783408ea7401c62"
        )

    def test_configs_agree_with_the_code_constants(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment_plan.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(EXPECTED_SINK_REPRO_COMMIT, text)
        self.assertIn(EXPECTED_SINK_KD_COMMIT, text)


class PackageVersionTests(unittest.TestCase):
    def test_missing_packages_are_reported_not_raised(self) -> None:
        versions = package_versions(("torch", "definitely-not-installed-xyz"))
        self.assertEqual(versions["definitely-not-installed-xyz"], "not-installed")
        self.assertNotEqual(versions["torch"], "not-installed")


if __name__ == "__main__":
    unittest.main()
