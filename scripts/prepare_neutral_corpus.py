#!/usr/bin/env python
"""Task 4a: build and freeze the neutral discovery/validation/test corpus.

This is a thin adapter around the read-only pinned upstream implementation. It reuses
``upstream/sink-kd/common/corpus_providers.py::openwebtext_corpus``, which is the
registered ``openwebtext_validation_sink_300`` construction named by
``configs/experiment_plan.yaml``. Nothing about the corpus is reimplemented here; this
script adds only the project's split assignment, anti-leakage checks, and provenance.

The one parameter this project sets away from the Sink-KD default is ``block_size``: 40
rather than 128, because ``docs/00_MASTER_EXPERIMENT_DESIGN.md`` registers 40 tokens as the
primary sequence length. ``block_size`` does not enter the upstream corpus id, so the
corpus is still exactly ``openwebtext_validation_sink_300``.

No model is loaded and no attention is measured here. Neuron attribution is Task 5 and must
not appear in this stage.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuron_sink.corpus import (  # noqa: E402
    FULL_SPLIT_SIZE,
    REGISTERED_CUT_LENGTH,
    REGISTERED_SEED,
    REGISTERED_SOURCE,
    SMOKE_SPLIT_SIZE,
    SPLIT_NAMES,
    NeutralCorpus,
    build_neutral_corpus,
    verify_disjoint,
)
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    git,
    prepare_output_dir,
    require_pinned_submodules,
    run_stamp,
    write_json,
)


FROZEN_DIR = ROOT / "configs" / "frozen"
FROZEN_MANIFEST = FROZEN_DIR / "neutral_corpus_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the neutral sink corpus and its disjoint split roles."
    )
    parser.add_argument("--model-id", default="gpt2",
                        help="Tokenizer source; must match the model the sink map uses.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cut-length", type=int, default=REGISTERED_CUT_LENGTH)
    parser.add_argument("--seed", type=int, default=REGISTERED_SEED)
    parser.add_argument("--split-size", type=int, default=FULL_SPLIT_SIZE,
                        help="Examples per discovery/validation/test split (registered: 100).")
    parser.add_argument("--cache-dir", type=Path,
                        default=(Path(os.environ["NEURON_SINK_HF_CACHE"])
                                 if os.environ.get("NEURON_SINK_HF_CACHE") else None),
                        help="Hugging Face cache; defaults to $NEURON_SINK_HF_CACHE.")
    parser.add_argument("--block-cache-root", type=Path,
                        default=ROOT / "cache" / "openwebtext_blocks",
                        help="Upstream block-pack cache, so a rerun does not restream "
                             "OpenWebText. Verified by digest on every hit.")
    parser.add_argument("--no-block-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-documents", type=int, default=None,
                        help="DRY RUN ONLY. Resizes the upstream document window and "
                             "therefore produces a different, unregistered corpus.")
    parser.add_argument("--validation-documents", type=int, default=None,
                        help="DRY RUN ONLY. See --train-documents.")
    parser.add_argument("--allow-unregistered-window", action="store_true",
                        help="Required to use the two flags above. Forces --no-freeze.")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Skip writing the tracked configs/frozen/ manifest.")
    parser.add_argument("--skip-repeat", action="store_true",
                        help="Skip the determinism rebuild (which is a block-cache hit).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    unregistered_window = (
        args.train_documents is not None or args.validation_documents is not None
    )
    if unregistered_window and not args.allow_unregistered_window:
        raise SystemExit(
            "--train-documents/--validation-documents change the upstream document window "
            "and produce a corpus that is NOT openwebtext_validation_sink_300. Pass "
            "--allow-unregistered-window to acknowledge this is a dry run."
        )
    freeze = not (args.no_freeze or unregistered_window)

    pool_size = args.split_size * len(SPLIT_NAMES)
    output_dir = prepare_output_dir(
        args.output_dir or ROOT / "results" / "task4_neutral_corpus" / run_stamp()
    )

    submodule_commits = require_pinned_submodules()
    repo_commit = git("rev-parse", "HEAD")
    recorder = ProvenanceRecorder(device=None, gpu_name="cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    info = HfApi().model_info(args.model_id, revision=args.revision)
    resolved_revision = info.sha
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=resolved_revision, cache_dir=cache_dir
    )

    block_cache = None if args.no_block_cache else args.block_cache_root
    if block_cache is not None:
        Path(block_cache).mkdir(parents=True, exist_ok=True)

    print(
        f"Building {REGISTERED_SOURCE['corpus_id']} via upstream openwebtext_corpus: "
        f"{pool_size} blocks x {args.cut_length} tokens, seed={args.seed}"
    )
    if unregistered_window:
        print("  !! DRY RUN: unregistered document window; this corpus will not be frozen")
    print("  streaming OpenWebText to the validation document window (first run is slow)")

    corpus = build_neutral_corpus(
        tokenizer,
        cut_length=args.cut_length,
        seed=args.seed,
        pool_size=pool_size,
        cache_root=block_cache,
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
    )

    # --- checks -------------------------------------------------------------
    splits = corpus.splits
    smoke_splits = corpus.smoke_splits
    verify_disjoint(splits)

    split_sizes = {name: len(ids) for name, ids in splits.items()}
    sizes_pass = all(size == args.split_size for size in split_sizes.values())
    smoke_pass = all(
        len(smoke_splits[name]) == SMOKE_SPLIT_SIZE
        and tuple(smoke_splits[name]) == tuple(splits[name][:SMOKE_SPLIT_SIZE])
        for name in SPLIT_NAMES
    )
    lengths = Counter(item.n_tokens for item in corpus.items)
    lengths_pass = lengths == Counter({args.cut_length: pool_size})
    vocab_size = int(getattr(tokenizer, "vocab_size", 50257))
    ids_in_range = all(
        0 <= token < vocab_size for item in corpus.items for token in item.input_ids
    )
    unique_ids_pass = len({item.item_id for item in corpus.items}) == pool_size

    # Diagnostic only. Blocks are packed spans of documents with EOS separators, so a
    # decoded block is not guaranteed to re-encode to the same ids the way a whole E1
    # example is. This is recorded, not gated.
    roundtrip_matches = 0
    for item in corpus.items:
        encoded = tokenizer(item.text, add_special_tokens=False)["input_ids"]
        if tuple(int(t) for t in encoded) == item.input_ids:
            roundtrip_matches += 1

    # --- determinism --------------------------------------------------------
    repeat_sha = None
    repeat_pass = None
    if not args.skip_repeat:
        print("  rebuilding once for determinism (block-cache hit)")
        repeat = build_neutral_corpus(
            tokenizer,
            cut_length=args.cut_length,
            seed=args.seed,
            pool_size=pool_size,
            cache_root=block_cache,
            train_documents=args.train_documents,
            validation_documents=args.validation_documents,
        )
        repeat_sha = repeat.manifest_sha256
        repeat_pass = (
            repeat_sha == corpus.manifest_sha256
            and repeat.upstream_manifest_sha256 == corpus.upstream_manifest_sha256
            and repeat.splits == splits
        )

    task_pass = all(
        (
            sizes_pass,
            smoke_pass,
            lengths_pass,
            ids_in_range,
            unique_ids_pass,
            repeat_pass is not False,
            corpus.corpus_id == f"openwebtext_validation_sink_{pool_size}",
        )
    )

    # --- outputs ------------------------------------------------------------
    corpus.save(output_dir / "neutral_corpus_manifest.json")

    run_config = {
        "experiment_id": "task4_neutral_corpus",
        "stage": "corpus_freeze",
        "model_id": args.model_id,
        "tokenizer_id": args.model_id,
        "tokenizer_revision": resolved_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "dtype": None,
        "device": "cpu",
        "seed": args.seed,
        "dataset_id": REGISTERED_SOURCE["dataset_id"],
        "dataset_config": None,
        "dataset_split": REGISTERED_SOURCE["document_window"],
        "corpus_id": corpus.corpus_id,
        "manifest_sha256": corpus.manifest_sha256,
        "upstream_manifest_sha256": corpus.upstream_manifest_sha256,
        "seq_len": args.cut_length,
        "pool_size": pool_size,
        "split_sizes": split_sizes,
        "smoke_split_size": SMOKE_SPLIT_SIZE,
        "upstream_provider": "sink-kd common/corpus_providers.py::openwebtext_corpus",
        "upstream_document_window": corpus.upstream_provenance.get("document_window"),
        "block_cache_root": str(block_cache) if block_cache else None,
        "registered_window": not unregistered_window,
        "train_documents_override": args.train_documents,
        "validation_documents_override": args.validation_documents,
    }
    summary = {
        "task4_corpus": "PASS" if task_pass else "FAIL",
        "corpus_id": corpus.corpus_id,
        "manifest_sha256": corpus.manifest_sha256,
        "upstream_manifest_sha256": corpus.upstream_manifest_sha256,
        "pool_size": pool_size,
        "splits": {name: list(ids) for name, ids in splits.items()},
        "smoke_splits": {name: list(ids) for name, ids in smoke_splits.items()},
        "checks": {
            "split_sizes": split_sizes,
            "split_sizes_pass": sizes_pass,
            "splits_disjoint": True,
            "smoke_splits_are_prefixes": smoke_pass,
            "all_sequences_exact_length": lengths_pass,
            "observed_lengths": dict(lengths),
            "token_ids_in_vocab_range": ids_in_range,
            "unique_item_ids": unique_ids_pass,
            "downstream_overlap_check": corpus.to_dict()["downstream_overlap_check"],
            "decode_reencode_roundtrip_matches": roundtrip_matches,
            "decode_reencode_roundtrip_rate": roundtrip_matches / pool_size,
            "deterministic_repeat_sha256": repeat_sha,
            "deterministic_repeat_pass": repeat_pass,
        },
        "upstream_provenance": dict(corpus.upstream_provenance),
    }

    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "provenance.json",
        recorder.finish(repo_commit=repo_commit, submodule_commits=submodule_commits),
    )
    write_json(output_dir / "summary.json", summary)

    if freeze and task_pass:
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        if FROZEN_MANIFEST.exists():
            existing = NeutralCorpus.load(FROZEN_MANIFEST)
            if existing.manifest_sha256 != corpus.manifest_sha256:
                raise SystemExit(
                    f"{FROZEN_MANIFEST} already holds a different frozen corpus "
                    f"({existing.manifest_sha256}). A frozen manifest is immutable; "
                    "register a new experiment id rather than overwriting it."
                )
            print(f"frozen manifest already matches: {FROZEN_MANIFEST}")
        else:
            corpus.save(FROZEN_MANIFEST)
            print(f"frozen manifest written: {FROZEN_MANIFEST}")

    print(f"TASK4_CORPUS={'PASS' if task_pass else 'FAIL'}")
    print(f"corpus_id={corpus.corpus_id}")
    print(f"manifest_sha256={corpus.manifest_sha256}")
    print(f"upstream_manifest_sha256={corpus.upstream_manifest_sha256}")
    print(f"split_sizes={split_sizes}")
    print(f"smoke_splits_are_prefixes={smoke_pass}")
    print(f"deterministic_repeat_pass={repeat_pass}")
    print(f"decode_reencode_roundtrip={roundtrip_matches}/{pool_size}")
    print(f"output_dir={output_dir}")
    return 0 if task_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
