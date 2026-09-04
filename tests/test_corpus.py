from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_sink.corpus import (
    FULL_SPLIT_SIZE,
    REGISTERED_CUT_LENGTH,
    SMOKE_SPLIT_SIZE,
    SPLIT_NAMES,
    CorpusError,
    LeakageError,
    NeutralCorpus,
    NeutralCorpusItem,
    assert_no_downstream_source,
    assign_splits,
    compute_manifest_sha256,
    downstream_overlap_check,
    require_discovery_split,
    verify_disjoint,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "configs" / "frozen" / "neutral_corpus_manifest.json"


def synthetic_corpus(
    split_size: int = SMOKE_SPLIT_SIZE, cut_length: int = 4
) -> NeutralCorpus:
    """A small corpus with the same structure as the real one, built without network."""

    ids = [f"openwebtext:validation:0:blk{index}" for index in range(split_size * 3)]
    splits = assign_splits(ids, split_size=split_size)
    split_of = {item_id: name for name, group in splits.items() for item_id in group}
    items = tuple(
        NeutralCorpusItem(
            item_id=item_id,
            split=split_of[item_id],
            text=f"text-{index}",
            input_ids=tuple(range(index, index + cut_length)),
            n_tokens=cut_length,
            meta={"block_index": index, "purpose": "sink"},
        )
        for index, item_id in enumerate(ids)
    )
    return NeutralCorpus(
        corpus_id=f"openwebtext_validation_sink_{split_size * 3}",
        items=items,
        tokenizer_name="gpt2",
        tokenizer_revision="deadbeef",
        cut_length=cut_length,
        seed=0,
        source={
            "dataset_id": "Skylion007/openwebtext",
            "corpus_id": f"openwebtext_validation_sink_{split_size * 3}",
        },
        upstream_manifest_sha256="upstream-sha",
        upstream_provenance={"provider": "openwebtext_corpus"},
    )


class SplitAssignmentTests(unittest.TestCase):
    def test_splits_are_disjoint_and_correctly_sized(self) -> None:
        corpus = synthetic_corpus(split_size=SMOKE_SPLIT_SIZE)
        splits = corpus.splits
        self.assertEqual(sorted(splits), sorted(SPLIT_NAMES))
        for name in SPLIT_NAMES:
            self.assertEqual(len(splits[name]), SMOKE_SPLIT_SIZE)
        verify_disjoint(splits)
        pooled = [item_id for group in splits.values() for item_id in group]
        self.assertEqual(len(pooled), len(set(pooled)))

    def test_assign_splits_rejects_a_wrong_pool_size(self) -> None:
        with self.assertRaises(CorpusError):
            assign_splits([f"b{i}" for i in range(299)], split_size=100)

    def test_assign_splits_rejects_a_split_smaller_than_the_smoke_size(self) -> None:
        with self.assertRaises(CorpusError):
            assign_splits([f"b{i}" for i in range(3)], split_size=1)

    def test_verify_disjoint_catches_an_overlap(self) -> None:
        with self.assertRaises(CorpusError):
            verify_disjoint({"discovery": ["a", "b"], "test": ["b"]})

    def test_registered_split_sizes(self) -> None:
        ids = [f"b{i}" for i in range(FULL_SPLIT_SIZE * 3)]
        splits = assign_splits(ids)
        self.assertEqual(
            [len(splits[name]) for name in SPLIT_NAMES],
            [FULL_SPLIT_SIZE] * 3,
        )


class SmokeSplitTests(unittest.TestCase):
    def test_smoke_splits_are_prefixes_of_the_full_splits(self) -> None:
        corpus = synthetic_corpus(split_size=FULL_SPLIT_SIZE, cut_length=2)
        for name in SPLIT_NAMES:
            smoke = corpus.smoke_splits[name]
            self.assertEqual(len(smoke), SMOKE_SPLIT_SIZE)
            self.assertEqual(smoke, corpus.splits[name][:SMOKE_SPLIT_SIZE])

    def test_smoke_splits_stay_disjoint(self) -> None:
        corpus = synthetic_corpus(split_size=FULL_SPLIT_SIZE, cut_length=2)
        verify_disjoint(corpus.smoke_splits)

    def test_items_for_honours_the_smoke_flag(self) -> None:
        corpus = synthetic_corpus(split_size=FULL_SPLIT_SIZE, cut_length=2)
        self.assertEqual(len(corpus.items_for("discovery")), FULL_SPLIT_SIZE)
        self.assertEqual(len(corpus.items_for("discovery", smoke=True)), SMOKE_SPLIT_SIZE)
        self.assertTrue(
            all(item.split == "discovery" for item in corpus.items_for("discovery"))
        )
        with self.assertRaises(ValueError):
            corpus.items_for("train")


class AntiLeakageTests(unittest.TestCase):
    def test_ranking_accepts_only_the_discovery_split(self) -> None:
        # AGENTS.md required test 6.
        self.assertEqual(require_discovery_split("discovery"), "discovery")
        for split in ("validation", "test"):
            with self.subTest(split=split), self.assertRaises(LeakageError):
                require_discovery_split(split)

    def test_downstream_datasets_are_refused(self) -> None:
        # AGENTS.md required test 7.
        for dataset_id in (
            "cais/mmlu", "allenai/ai2_arc", "kellycyy/CulturalBench", "openai/gsm8k",
        ):
            with self.subTest(dataset_id=dataset_id), self.assertRaises(LeakageError):
                assert_no_downstream_source(dataset_id)

    def test_the_e1_parity_mixture_is_refused_because_it_contains_gsm8k(self) -> None:
        with self.assertRaises(LeakageError):
            assert_no_downstream_source("mixed", "e1_100x40")

    def test_the_registered_neutral_source_is_accepted(self) -> None:
        check = downstream_overlap_check({
            "dataset_id": "Skylion007/openwebtext",
            "corpus_id": "openwebtext_validation_sink_300",
        })
        self.assertTrue(check["pass"])
        self.assertIn("openai/gsm8k", check["forbidden_dataset_ids"])


class ManifestTests(unittest.TestCase):
    def test_hash_is_deterministic_and_split_sensitive(self) -> None:
        first = synthetic_corpus()
        second = synthetic_corpus()
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)

        moved = tuple(
            NeutralCorpusItem(
                item_id=item.item_id,
                split="test" if item.split == "discovery" else item.split,
                text=item.text,
                input_ids=item.input_ids,
                n_tokens=item.n_tokens,
                meta=dict(item.meta),
            )
            for item in first.items
        )
        self.assertNotEqual(compute_manifest_sha256(moved), first.manifest_sha256)

    def test_hash_ignores_meta_key_order(self) -> None:
        base = synthetic_corpus().items[0]
        reordered = NeutralCorpusItem(
            item_id=base.item_id,
            split=base.split,
            text=base.text,
            input_ids=base.input_ids,
            n_tokens=base.n_tokens,
            meta={"purpose": "sink", "block_index": base.meta["block_index"]},
        )
        self.assertEqual(
            compute_manifest_sha256([base]), compute_manifest_sha256([reordered])
        )

    def test_round_trip_through_disk(self) -> None:
        corpus = synthetic_corpus()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            corpus.save(path)
            loaded = NeutralCorpus.load(path)
        self.assertEqual(loaded.manifest_sha256, corpus.manifest_sha256)
        self.assertEqual(loaded.splits, corpus.splits)
        self.assertEqual(loaded.corpus_id, corpus.corpus_id)
        self.assertEqual(len(loaded), len(corpus))

    def test_a_tampered_manifest_is_rejected_on_load(self) -> None:
        corpus = synthetic_corpus()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            corpus.save(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["items"][0]["input_ids"][0] += 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(CorpusError):
                NeutralCorpus.load(path)

    def test_items_are_immutable(self) -> None:
        item = synthetic_corpus().items[0]
        with self.assertRaises(TypeError):
            item.meta["purpose"] = "ppl"  # type: ignore[index]

    def test_token_count_must_match_the_ids(self) -> None:
        with self.assertRaises(ValueError):
            NeutralCorpusItem(
                item_id="b0", split="discovery", text="x",
                input_ids=(1, 2, 3), n_tokens=4,
            )

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NeutralCorpusItem(
                item_id="b0", split="train", text="x", input_ids=(1,), n_tokens=1,
            )


@unittest.skipUnless(
    FROZEN_MANIFEST.is_file(), "configs/frozen/neutral_corpus_manifest.json not built yet"
)
class FrozenManifestTests(unittest.TestCase):
    """Guard the real frozen artefact against silent drift."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = NeutralCorpus.load(FROZEN_MANIFEST)

    def test_registered_shape(self) -> None:
        self.assertEqual(self.corpus.corpus_id, "openwebtext_validation_sink_300")
        self.assertEqual(len(self.corpus), FULL_SPLIT_SIZE * 3)
        self.assertEqual(self.corpus.cut_length, REGISTERED_CUT_LENGTH)
        self.assertEqual(self.corpus.seed, 0)

    def test_every_example_is_exactly_the_registered_length(self) -> None:
        for item in self.corpus.items:
            self.assertEqual(item.n_tokens, REGISTERED_CUT_LENGTH)
            self.assertEqual(len(item.input_ids), REGISTERED_CUT_LENGTH)

    def test_splits_are_disjoint_and_sized(self) -> None:
        verify_disjoint(self.corpus.splits)
        for name in SPLIT_NAMES:
            self.assertEqual(len(self.corpus.splits[name]), FULL_SPLIT_SIZE)

    def test_smoke_splits_are_prefixes(self) -> None:
        for name in SPLIT_NAMES:
            self.assertEqual(
                self.corpus.smoke_splits[name],
                self.corpus.splits[name][:SMOKE_SPLIT_SIZE],
            )

    def test_source_is_not_a_downstream_benchmark(self) -> None:
        self.assertTrue(downstream_overlap_check(self.corpus.source)["pass"])


if __name__ == "__main__":
    unittest.main()
