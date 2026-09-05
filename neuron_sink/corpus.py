"""Frozen neutral sink corpus with disjoint discovery/validation/test roles.

The registered source is Sink-KD's ``openwebtext_validation_sink_300``
(``configs/experiment_plan.yaml``), built by
``upstream/sink-kd/common/corpus_providers.py::openwebtext_corpus``. That provider is
called directly rather than reimplemented: its OpenWebText document window
(``datasets_loader.OPENWEBTEXT_SPLIT_WINDOWS``) makes the corpus disjoint from the Sink-KD
training window by construction, and its ``manifest_sha256`` is recorded verbatim here so
a later run can prove it saw byte-identical text.

``block_size`` is set to this project's registered sequence length of 40 rather than the
Sink-KD default of 128 (``docs/00_MASTER_EXPERIMENT_DESIGN.md``, "Primary sequence length:
40 tokens"). ``block_size`` does not enter the upstream corpus id, so the corpus this
module builds is still exactly ``openwebtext_validation_sink_300``.

Terminology: ``document_window`` is OpenWebText's document window (upstream's
``split="validation"`` argument); ``split`` is this project's discovery/validation/test
role. The two are unrelated and must not be conflated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .provenance import canonical_sha256, read_json, write_json
from .upstream_bridge import sink_kd_module


SCHEMA_VERSION = "neutral_sink_corpus_v1"

#: The registered upstream corpus. Any deviation from these values is a different corpus
#: and requires an amendment before results are inspected.
REGISTERED_SOURCE: Mapping[str, Any] = MappingProxyType({
    "provider": "openwebtext_corpus",
    "upstream": "sink_kd",
    "upstream_module": "common/corpus_providers.py",
    "corpus_id": "openwebtext_validation_sink_300",
    "dataset_id": "Skylion007/openwebtext",
    "document_window": "validation",
    "purpose": "sink",
    "n_blocks": 300,
})

SPLIT_NAMES: tuple[str, ...] = ("discovery", "validation", "test")
FULL_SPLIT_SIZE = 100
SMOKE_SPLIT_SIZE = 24
REGISTERED_CUT_LENGTH = 40
REGISTERED_SEED = 0

#: Only the discovery split may reach neuron ranking (AGENTS.md, "Required tests" 6).
RANKING_ALLOWED_SPLITS = frozenset({"discovery"})

#: Downstream evaluation datasets. These must never source neuron discovery
#: (AGENTS.md, "Non-negotiable anti-leakage rules").
FORBIDDEN_DATASET_IDS = frozenset({
    "cais/mmlu",
    "allenai/ai2_arc",
    "kellycyy/culturalbench",
    "openai/gsm8k",
})

#: The Task-2 parity mixture contains GSM8K, so it is parity-only and never a discovery
#: source (handover.md, "Critical leakage rule").
FORBIDDEN_CORPUS_ID_PREFIXES = ("e1_",)


class LeakageError(RuntimeError):
    """Raised when downstream benchmark data or a held-out split reaches ranking code."""


class CorpusError(RuntimeError):
    """Raised when a built corpus does not match the registered contract."""


@dataclass(frozen=True)
class NeutralCorpusItem:
    item_id: str
    split: str
    text: str
    input_ids: tuple[int, ...]
    n_tokens: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_ids", tuple(int(t) for t in self.input_ids))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {self.split!r}; expected {SPLIT_NAMES}")
        if self.n_tokens != len(self.input_ids):
            raise ValueError(
                f"{self.item_id}: n_tokens={self.n_tokens} but "
                f"{len(self.input_ids)} input_ids"
            )


@dataclass(frozen=True)
class NeutralCorpus:
    """A frozen neutral corpus plus its split assignment and provenance."""

    corpus_id: str
    items: tuple[NeutralCorpusItem, ...]
    tokenizer_name: str
    tokenizer_revision: str | None
    cut_length: int
    seed: int
    source: Mapping[str, Any]
    upstream_manifest_sha256: str
    upstream_provenance: Mapping[str, Any]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))
        object.__setattr__(
            self, "upstream_provenance", MappingProxyType(dict(self.upstream_provenance))
        )
        if not self.manifest_sha256:
            object.__setattr__(self, "manifest_sha256", compute_manifest_sha256(self.items))

    def __len__(self) -> int:
        return len(self.items)

    @property
    def splits(self) -> dict[str, tuple[str, ...]]:
        return {
            name: tuple(item.item_id for item in self.items if item.split == name)
            for name in SPLIT_NAMES
        }

    @property
    def smoke_splits(self) -> dict[str, tuple[str, ...]]:
        return {name: ids[:SMOKE_SPLIT_SIZE] for name, ids in self.splits.items()}

    def items_for(self, split: str, *, smoke: bool = False) -> tuple[NeutralCorpusItem, ...]:
        """Return one split's items, in frozen manifest order."""

        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}; expected {SPLIT_NAMES}")
        selected = tuple(item for item in self.items if item.split == split)
        return selected[:SMOKE_SPLIT_SIZE] if smoke else selected

    # --- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "corpus_id": self.corpus_id,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "cut_length": self.cut_length,
            "seed": self.seed,
            "pool_size": len(self.items),
            "source": dict(self.source),
            "splits": {name: list(ids) for name, ids in self.splits.items()},
            "smoke_splits": {name: list(ids) for name, ids in self.smoke_splits.items()},
            "split_sizes": {name: len(ids) for name, ids in self.splits.items()},
            "smoke_split_size": SMOKE_SPLIT_SIZE,
            "downstream_overlap_check": downstream_overlap_check(self.source),
            "upstream_manifest_sha256": self.upstream_manifest_sha256,
            "upstream_provenance": dict(self.upstream_provenance),
            "manifest_sha256": self.manifest_sha256,
            "items": [
                {
                    "item_id": item.item_id,
                    "split": item.split,
                    "text": item.text,
                    "input_ids": list(item.input_ids),
                    "n_tokens": item.n_tokens,
                    "meta": dict(item.meta),
                }
                for item in self.items
            ],
        }

    def save(self, path: Path) -> None:
        write_json(Path(path), self.to_dict())

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "NeutralCorpus":
        if obj.get("schema") != SCHEMA_VERSION:
            raise CorpusError(
                f"Manifest schema {obj.get('schema')!r} != {SCHEMA_VERSION!r}"
            )
        items = tuple(
            NeutralCorpusItem(
                item_id=row["item_id"],
                split=row["split"],
                text=row["text"],
                input_ids=tuple(int(t) for t in row["input_ids"]),
                n_tokens=int(row["n_tokens"]),
                meta=dict(row["meta"]),
            )
            for row in obj["items"]
        )
        recomputed = compute_manifest_sha256(items)
        stored = obj["manifest_sha256"]
        if recomputed != stored:
            raise CorpusError(
                f"Neutral corpus manifest hash mismatch: stored {stored} != "
                f"recomputed {recomputed}. The frozen manifest was modified."
            )
        return NeutralCorpus(
            corpus_id=obj["corpus_id"],
            items=items,
            tokenizer_name=obj["tokenizer_name"],
            tokenizer_revision=obj.get("tokenizer_revision"),
            cut_length=int(obj["cut_length"]),
            seed=int(obj["seed"]),
            source=dict(obj["source"]),
            upstream_manifest_sha256=obj["upstream_manifest_sha256"],
            upstream_provenance=dict(obj.get("upstream_provenance", {})),
            manifest_sha256=stored,
        )

    @staticmethod
    def load(path: Path) -> "NeutralCorpus":
        return NeutralCorpus.from_dict(read_json(Path(path)))


def compute_manifest_sha256(items: Sequence[NeutralCorpusItem]) -> str:
    """Hash item ids, split roles, and token ids -- what a later stage must not vary.

    Mirrors ``corpus_providers.compute_manifest_sha256`` but adds the split role, because
    for this project the split assignment is itself part of what is frozen.
    """

    payload = [
        [item.item_id, item.split, list(item.input_ids), sorted(item.meta.items())]
        for item in items
    ]
    return canonical_sha256(payload)


# --- anti-leakage guards -----------------------------------------------------


def assert_no_downstream_source(dataset_id: str, corpus_id: str = "") -> None:
    """Refuse any corpus sourced from a downstream benchmark or the E1 parity mixture."""

    if str(dataset_id).strip().lower() in FORBIDDEN_DATASET_IDS:
        raise LeakageError(
            f"{dataset_id!r} is a downstream evaluation dataset and must never source "
            "neuron discovery (AGENTS.md anti-leakage rules)."
        )
    lowered = str(corpus_id).strip().lower()
    for prefix in FORBIDDEN_CORPUS_ID_PREFIXES:
        if lowered.startswith(prefix):
            raise LeakageError(
                f"Corpus {corpus_id!r} is the E1 parity mixture, which contains GSM8K. "
                "It is parity-only and must never source neuron discovery."
            )


def downstream_overlap_check(source: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id = str(source.get("dataset_id", ""))
    corpus_id = str(source.get("corpus_id", ""))
    assert_no_downstream_source(dataset_id, corpus_id)
    return {
        "forbidden_dataset_ids": sorted(FORBIDDEN_DATASET_IDS),
        "forbidden_corpus_id_prefixes": list(FORBIDDEN_CORPUS_ID_PREFIXES),
        "checked_dataset_id": dataset_id,
        "checked_corpus_id": corpus_id,
        "pass": True,
    }


def require_discovery_split(split: str) -> str:
    """Gate for ranking-facing APIs: only the discovery split may rank neurons."""

    if split not in RANKING_ALLOWED_SPLITS:
        raise LeakageError(
            f"Split {split!r} must not reach neuron ranking. Ranking may only read "
            f"{sorted(RANKING_ALLOWED_SPLITS)}; validation selects the operating point "
            "and test is used once, after k* is frozen."
        )
    return split


def verify_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    """Raise if any item id appears in more than one split."""

    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for item_id in ids:
            if item_id in seen:
                raise CorpusError(
                    f"Item {item_id!r} appears in both {seen[item_id]!r} and {name!r}; "
                    "discovery/validation/test must be disjoint"
                )
            seen[item_id] = name


# --- construction ------------------------------------------------------------


def assign_splits(
    item_ids: Sequence[str], *, split_size: int = FULL_SPLIT_SIZE
) -> dict[str, tuple[str, ...]]:
    """Partition the frozen pool order into contiguous, disjoint split roles.

    Contiguous windows are how upstream itself guarantees disjointness
    (``corpus_providers.SINK_BLOCKS_RESERVED``), and the pool order is already a seeded
    shuffle of source documents, so a contiguous slice is not an ordered sample.
    """

    required = split_size * len(SPLIT_NAMES)
    if len(item_ids) != required:
        raise CorpusError(
            f"Split assignment needs exactly {required} items "
            f"({len(SPLIT_NAMES)} x {split_size}), got {len(item_ids)}"
        )
    if split_size < SMOKE_SPLIT_SIZE:
        raise CorpusError(
            f"split_size {split_size} is smaller than the smoke size {SMOKE_SPLIT_SIZE}"
        )
    splits = {
        name: tuple(item_ids[index * split_size:(index + 1) * split_size])
        for index, name in enumerate(SPLIT_NAMES)
    }
    verify_disjoint(splits)
    return splits


def build_neutral_corpus(
    tokenizer,
    *,
    cut_length: int = REGISTERED_CUT_LENGTH,
    seed: int = REGISTERED_SEED,
    pool_size: int = FULL_SPLIT_SIZE * len(SPLIT_NAMES),
    purpose: str = str(REGISTERED_SOURCE["purpose"]),
    skip_blocks: int = 0,
    cache_root: Path | None = None,
    train_documents: int | None = None,
    validation_documents: int | None = None,
) -> NeutralCorpus:
    """Build a registered neutral-corpus block window and freeze its split roles.

    ``purpose="sink"`` is the original Stage-B/Stage-C block window. Amendment A005
    registers ``purpose="ppl"`` for Stage C2 so the same pinned upstream provider selects
    its guaranteed-disjoint block window. No other provider purpose is accepted here.

    ``skip_blocks`` (amendment A007) reaches a *third* disjoint window without editing the
    pinned provider. The provider offers only two offsets -- ``0`` for ``sink`` and
    ``SINK_BLOCKS_RESERVED=300`` for ``ppl`` -- but ``ppl`` has no cap on ``n_blocks``, so
    requesting ``pool_size + skip_blocks`` blocks and dropping the first ``skip_blocks``
    yields global block indices ``[300 + skip_blocks, 300 + skip_blocks + pool_size)``.
    Packing is a prefix operation, so the dropped prefix is byte-identical to the corpus a
    smaller request would have produced, and Stage C2 stays exactly reproducible.

    Two superficially similar routes are deliberately **not** used, because both give false
    disjointness: a different ``seed`` reshuffles the *same* documents, and
    ``train_documents``/``validation_documents`` change the document window without being
    encoded in the upstream corpus id.
    """

    if purpose not in ("sink", "ppl"):
        raise CorpusError(
            f"Unsupported OpenWebText purpose {purpose!r}; expected 'sink' or 'ppl'"
        )
    if isinstance(skip_blocks, bool) or not isinstance(skip_blocks, int):
        raise TypeError(f"skip_blocks must be an integer, got {type(skip_blocks).__name__}")
    if skip_blocks < 0:
        raise CorpusError(f"skip_blocks must not be negative, got {skip_blocks}")
    if skip_blocks and purpose != "ppl":
        raise CorpusError(
            "skip_blocks is only defined for purpose='ppl'; the 'sink' window is capped at "
            "SINK_BLOCKS_RESERVED=300 blocks upstream and cannot be offset further"
        )
    requested_blocks = pool_size + skip_blocks

    assert_no_downstream_source(
        REGISTERED_SOURCE["dataset_id"], REGISTERED_SOURCE["corpus_id"]
    )
    providers = sink_kd_module("corpus_providers")
    upstream_corpus = providers.openwebtext_corpus(
        tokenizer,
        REGISTERED_SOURCE["document_window"],
        requested_blocks,
        block_size=cut_length,
        seed=seed,
        purpose=purpose,
        train_documents=train_documents,
        validation_documents=validation_documents,
        cache_root=str(cache_root) if cache_root is not None else None,
    )

    window = REGISTERED_SOURCE["document_window"]
    upstream_id = f"openwebtext_{window}_{purpose}_{requested_blocks}"
    if upstream_corpus.corpus_id != upstream_id:
        raise CorpusError(
            f"Upstream returned corpus_id {upstream_corpus.corpus_id!r}, expected "
            f"{upstream_id!r}"
        )
    if len(upstream_corpus.items) != requested_blocks:
        raise CorpusError(
            f"Upstream returned {len(upstream_corpus.items)} items, expected "
            f"{requested_blocks}"
        )

    # The project corpus id must describe the window this project actually keeps, so a
    # 300-item C3 corpus can never be mistaken for the 600-block upstream request it was
    # sliced out of.
    expected_id = upstream_id if not skip_blocks else f"{upstream_id}_skip{skip_blocks}"
    selected_items = list(upstream_corpus.items)[skip_blocks:]
    if len(selected_items) != pool_size:
        raise CorpusError(
            f"Dropping {skip_blocks} blocks left {len(selected_items)} items, expected "
            f"{pool_size}"
        )
    provider_offset = 0 if purpose == "sink" else 300
    first_expected_block = provider_offset + skip_blocks
    observed_blocks = [int(item.meta["block_index"]) for item in selected_items]
    if observed_blocks != list(
        range(first_expected_block, first_expected_block + pool_size)
    ):
        raise CorpusError(
            f"Kept blocks {observed_blocks[:1]}..{observed_blocks[-1:]} but expected the "
            f"contiguous window [{first_expected_block}, "
            f"{first_expected_block + pool_size})"
        )

    for item in selected_items:
        if item.n_tokens != cut_length or len(item.input_ids) != cut_length:
            raise CorpusError(
                f"{item.item_id} has {item.n_tokens} tokens, expected exactly {cut_length}"
            )

    ordered_ids = [item.item_id for item in selected_items]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise CorpusError("Upstream corpus contains duplicate item ids")
    splits = assign_splits(ordered_ids, split_size=pool_size // len(SPLIT_NAMES))
    split_of = {item_id: name for name, ids in splits.items() for item_id in ids}

    items = tuple(
        NeutralCorpusItem(
            item_id=item.item_id,
            split=split_of[item.item_id],
            text=item.text,
            input_ids=tuple(int(t) for t in item.input_ids),
            n_tokens=int(item.n_tokens),
            meta=dict(item.meta),
        )
        for item in selected_items
    )

    source = dict(REGISTERED_SOURCE)
    source["corpus_id"] = expected_id
    source["purpose"] = purpose
    source["n_blocks"] = pool_size
    source["block_size"] = cut_length
    source["skip_blocks"] = skip_blocks
    source["upstream_n_blocks"] = requested_blocks
    source["block_index_window"] = [
        first_expected_block, first_expected_block + pool_size
    ]
    source["upstream_corpus_id"] = upstream_corpus.corpus_id
    return NeutralCorpus(
        corpus_id=expected_id,
        items=items,
        tokenizer_name=upstream_corpus.tokenizer_name,
        tokenizer_revision=upstream_corpus.tokenizer_revision,
        cut_length=cut_length,
        seed=seed,
        source=source,
        upstream_manifest_sha256=upstream_corpus.manifest_sha256,
        upstream_provenance=dict(upstream_corpus.provenance),
    )
