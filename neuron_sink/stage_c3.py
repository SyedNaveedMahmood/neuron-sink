"""Stage-C3 reachability-aware Qwen replication boundary, registered by amendment A007.

Stage C3 is a sibling of Stage C2, not a child: it delegates its statistics to Stage C, so a
C2 relabel never sits between the C3 document and the registered gate. What changes is
discovery-side selection (per-sink-layer budget, measured-ablation ranking, float32 metric
arithmetic) and the corpus window; the validation rule, the formal gate, the bootstrap, the
alpha grid and the twenty layer-count-matched controls are inherited unchanged.

Two registered models run under this stage id: the Qwen checkpoint whose Stage-C result was
null, and GPT-2-small as a method-validation control. GPT-2-small is a known positive, so if
the new method fails to reproduce or beat its Stage-B effect the method is at fault and no
Qwen number may be interpreted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import FULL_SPLIT_SIZE, NeutralCorpus
from .provenance import canonical_sha256, read_json, write_json
from .stage_b import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FULL_ALPHAS,
    FULL_EXAMPLES_PER_SPLIT,
    FullCondition,
    registered_full_conditions,
)
from .stage_c import (
    EXPERIMENT_ID as STAGE_C_EXPERIMENT_ID,
    OPERATING_POINT_SCHEMA as STAGE_C_OPERATING_POINT_SCHEMA,
    StageCError,
    build_operating_point_document as _build_stage_c_operating_point,
    evaluate_formal_gate as _evaluate_stage_c_gate,
    verify_operating_point_document as _verify_stage_c_operating_point,
)


EXPERIMENT_ID = "stage_c3_reachability_aware_replication_v1"
OPERATING_POINT_SCHEMA = "stage_c3_operating_point_v1"
FORMAL_GATE_SCHEMA = "stage_c3_formal_gate_v1"
AMENDMENT = "A007"

#: Models registered to run under Stage C3, with the block window each one owns.
REGISTERED_MODEL_ALIASES: tuple[str, ...] = ("qwen2.5-1.5b-instruct", "gpt2-small")

#: Qwen: the pinned provider offers only offsets 0 ("sink") and 300 ("ppl"), but "ppl" has no
#: cap on n_blocks, so requesting 600 and dropping the first 300 reaches blocks 600-899.
REGISTERED_CORPUS_ID = "openwebtext_validation_ppl_600_skip300"
REGISTERED_BLOCK_INDICES = tuple(range(600, 900))
REGISTERED_SKIP_BLOCKS = 300

#: GPT-2-small has only ever consumed the GPT-2-tokenized "sink" window (blocks 0-299), so the
#: "ppl" window is fresh for it without any skip.
GPT2_CORPUS_ID = "openwebtext_validation_ppl_300"
GPT2_BLOCK_INDICES = tuple(range(300, 600))
GPT2_SKIP_BLOCKS = 0


class StageC3Error(RuntimeError):
    """Raised when a Stage-C3 artefact or transition violates amendment A007."""


def stage_c3_run_root(
    repository_root: Path | str,
    model_alias: str,
    *,
    registered_run: bool,
    stamp: str,
) -> Path:
    """Return the append-only model-specific Stage-C3 output path."""

    if model_alias not in REGISTERED_MODEL_ALIASES:
        raise StageC3Error(f"Unregistered Stage-C3 model alias {model_alias!r}")
    if not stamp.startswith("run_") or not stamp.endswith("Z"):
        raise StageC3Error(f"Run stamp is not the registered UTC form: {stamp!r}")
    category = "stage_c3_full" if registered_run else "stage_c3_preflight"
    return Path(repository_root).resolve() / "results" / category / model_alias / stamp


def registered_window(model_alias: str) -> tuple[str, tuple[int, ...], int]:
    """The frozen ``(corpus_id, block indices, skip_blocks)`` this model owns in Stage C3."""

    if model_alias == "qwen2.5-1.5b-instruct":
        return REGISTERED_CORPUS_ID, REGISTERED_BLOCK_INDICES, REGISTERED_SKIP_BLOCKS
    if model_alias == "gpt2-small":
        return GPT2_CORPUS_ID, GPT2_BLOCK_INDICES, GPT2_SKIP_BLOCKS
    raise StageC3Error(f"Unregistered Stage-C3 model alias {model_alias!r}")


def verify_fresh_corpus(
    corpus: NeutralCorpus,
    predecessors: Sequence[NeutralCorpus],
    *,
    model_alias: str,
) -> dict[str, Any]:
    """Prove the C3 window is the registered one and disjoint from every predecessor.

    Stage C2 only had one predecessor to exclude. Stage C3 has two for Qwen (the Stage-C
    ``sink`` window and the Stage-C2 ``ppl`` window), so disjointness is checked against every
    corpus this checkpoint has already inspected, not just the most recent one.
    """

    corpus_id, block_indices, skip_blocks = registered_window(model_alias)
    indices = tuple(sorted(int(item.meta.get("block_index", -1)) for item in corpus.items))
    item_ids = {item.item_id for item in corpus.items}

    checks: dict[str, Any] = {
        "c3_corpus_id_pass": corpus.corpus_id == corpus_id,
        "c3_pool_size_pass": len(corpus) == 3 * FULL_SPLIT_SIZE,
        "c3_seed_zero_pass": corpus.seed == 0,
        "c3_purpose_pass": all(
            item.meta.get("purpose") == "ppl" for item in corpus.items
        ),
        "c3_block_window_pass": indices == block_indices,
        "c3_skip_blocks_pass": int(corpus.source.get("skip_blocks", -1)) == skip_blocks,
        "c3_cut_length_pass": all(
            item.n_tokens == corpus.cut_length for item in corpus.items
        ),
    }
    if not predecessors:
        raise StageC3Error(
            "Stage C3 requires at least one predecessor corpus to prove disjointness against"
        )
    for position, previous in enumerate(predecessors):
        previous_indices = {
            int(item.meta.get("block_index", -1)) for item in previous.items
        }
        previous_ids = {item.item_id for item in previous.items}
        checks[f"block_indices_disjoint_from_{position}_pass"] = not (
            set(indices) & previous_indices
        )
        checks[f"item_ids_disjoint_from_{position}_pass"] = not (item_ids & previous_ids)
        checks[f"tokenizer_matches_{position}_pass"] = (
            corpus.tokenizer_name == previous.tokenizer_name
        )
        checks[f"source_dataset_matches_{position}_pass"] = (
            corpus.source.get("dataset_id") == previous.source.get("dataset_id")
        )
    if not all(bool(value) for value in checks.values()):
        raise StageC3Error(f"Stage-C3 fresh-corpus contract failed: {checks}")
    checks["predecessor_count"] = len(predecessors)
    return checks


def _rehash(document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    updated = {key: value for key, value in document.items() if key != hash_field}
    updated[hash_field] = canonical_sha256(updated)
    return updated


def build_operating_point_document(
    validation_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[FullCondition],
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply the unchanged validation rule and emit a Stage-C3-only lock."""

    document = _build_stage_c_operating_point(validation_rows, conditions, **kwargs)
    document["schema"] = OPERATING_POINT_SCHEMA
    document["experiment_id"] = EXPERIMENT_ID
    document["amendment"] = AMENDMENT
    return _rehash(document, "operating_point_sha256")


def verify_operating_point_document(
    document: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    corpus_manifest_sha256: str,
    sink_scope_sha256: str,
    attribution_sha256: str,
    neuron_sets_sha256: str,
) -> Mapping[str, Any]:
    """Re-establish every C3 hash lock before first test-split access."""

    if document.get("schema") != OPERATING_POINT_SCHEMA:
        raise StageC3Error("Missing or unsupported frozen Stage-C3 operating-point schema")
    if (
        document.get("experiment_id") != EXPERIMENT_ID
        or document.get("amendment") != AMENDMENT
    ):
        raise StageC3Error("Operating-point experiment boundary is not Stage C3/A007")
    stored = str(document.get("operating_point_sha256", ""))
    if stored != canonical_sha256({
        key: value for key, value in document.items()
        if key != "operating_point_sha256"
    }):
        raise StageC3Error("Frozen Stage-C3 operating-point hash does not reproduce")

    proxy = dict(document)
    proxy.pop("amendment", None)
    proxy["schema"] = STAGE_C_OPERATING_POINT_SCHEMA
    proxy["experiment_id"] = STAGE_C_EXPERIMENT_ID
    proxy = _rehash(proxy, "operating_point_sha256")
    try:
        _verify_stage_c_operating_point(
            proxy,
            model_id=model_id,
            model_revision=model_revision,
            corpus_manifest_sha256=corpus_manifest_sha256,
            sink_scope_sha256=sink_scope_sha256,
            attribution_sha256=attribution_sha256,
            neuron_sets_sha256=neuron_sets_sha256,
        )
    except StageCError as exc:
        raise StageC3Error(str(exc)) from exc
    return document


def freeze_operating_point(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Write a verified Stage-C3 validation decision exactly once."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Operating-point artefact already exists: {path}")
    if document.get("schema") != OPERATING_POINT_SCHEMA:
        raise StageC3Error("Refusing to freeze a non-Stage-C3 operating point")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(document))
    return path


def unlock_test_split(path: Path | str, **expected: str) -> Mapping[str, Any]:
    """Load and verify the C3 validation artefact required for test access."""

    path = Path(path)
    if not path.is_file():
        raise StageC3Error(
            "Locked Stage-C3 test split refused: no frozen validation operating point"
        )
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise StageC3Error("Operating-point artefact is not a JSON object")
    return verify_operating_point_document(document, **expected)


def evaluate_formal_gate(
    test_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[FullCondition],
    *,
    all_identity_pass: bool,
    all_validity_pass: bool,
    state_leakage_pass: bool,
    registered_run: bool,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_examples: int = FULL_EXAMPLES_PER_SPLIT,
) -> dict[str, Any]:
    """Evaluate the unchanged formal gate under the Stage-C3 schema.

    Stage C2 leaves ``StageCError`` unwrapped here while wrapping it in its operating-point
    verifier; Stage C3 wraps both, so a caller can catch one stage-specific exception type.
    """

    try:
        document = _evaluate_stage_c_gate(
            test_rows,
            conditions,
            all_identity_pass=all_identity_pass,
            all_validity_pass=all_validity_pass,
            state_leakage_pass=state_leakage_pass,
            registered_run=registered_run,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            expected_examples=expected_examples,
        )
    except StageCError as exc:
        raise StageC3Error(str(exc)) from exc
    document["schema"] = FORMAL_GATE_SCHEMA
    document["experiment_id"] = EXPERIMENT_ID
    document["amendment"] = AMENDMENT
    if registered_run:
        document = _rehash(document, "formal_gate_sha256")
    return document


__all__ = [
    "AMENDMENT",
    "EXPERIMENT_ID",
    "FORMAL_GATE_SCHEMA",
    "GPT2_BLOCK_INDICES",
    "GPT2_CORPUS_ID",
    "OPERATING_POINT_SCHEMA",
    "REGISTERED_BLOCK_INDICES",
    "REGISTERED_CORPUS_ID",
    "REGISTERED_MODEL_ALIASES",
    "REGISTERED_SKIP_BLOCKS",
    "StageC3Error",
    "build_operating_point_document",
    "evaluate_formal_gate",
    "freeze_operating_point",
    "registered_full_conditions",
    "registered_window",
    "stage_c3_run_root",
    "unlock_test_split",
    "verify_fresh_corpus",
    "verify_operating_point_document",
]
