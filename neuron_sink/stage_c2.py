"""Stage-C2 signed Qwen replication boundary registered by amendment A005."""

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
    OPERATING_POINT_SCHEMA as STAGE_C_OPERATING_POINT_SCHEMA,
    EXPERIMENT_ID as STAGE_C_EXPERIMENT_ID,
    StageCError,
    build_operating_point_document as _build_stage_c_operating_point,
    evaluate_formal_gate as _evaluate_stage_c_gate,
    verify_operating_point_document as _verify_stage_c_operating_point,
)


EXPERIMENT_ID = "stage_c2_qwen_signed_replication_v1"
OPERATING_POINT_SCHEMA = "stage_c2_operating_point_v1"
FORMAL_GATE_SCHEMA = "stage_c2_formal_gate_v1"
REGISTERED_MODEL_ALIAS = "qwen2.5-1.5b-instruct"
REGISTERED_CORPUS_ID = "openwebtext_validation_ppl_300"
REGISTERED_BLOCK_INDICES = tuple(range(300, 600))


class StageC2Error(RuntimeError):
    """Raised when a Stage-C2 artefact or transition violates amendment A005."""


def stage_c2_run_root(
    repository_root: Path | str,
    model_alias: str,
    *,
    registered_run: bool,
    stamp: str,
) -> Path:
    """Return the append-only model-specific Stage-C2 output path."""

    if model_alias != REGISTERED_MODEL_ALIAS:
        raise StageC2Error(f"Unregistered Stage-C2 model alias {model_alias!r}")
    if not stamp.startswith("run_") or not stamp.endswith("Z"):
        raise StageC2Error(f"Run stamp is not the registered UTC form: {stamp!r}")
    category = "stage_c2_full" if registered_run else "stage_c2_preflight"
    return Path(repository_root).resolve() / "results" / category / model_alias / stamp


def verify_fresh_corpus(
    corpus: NeutralCorpus, stage_c_corpus: NeutralCorpus
) -> dict[str, Any]:
    """Prove that C2 uses the registered blocks 300-599, not Stage-C blocks 0-299."""

    c2_indices = tuple(sorted(int(item.meta.get("block_index", -1)) for item in corpus.items))
    stage_c_indices = tuple(
        sorted(int(item.meta.get("block_index", -1)) for item in stage_c_corpus.items)
    )
    checks = {
        "c2_corpus_id_pass": corpus.corpus_id == REGISTERED_CORPUS_ID,
        "c2_pool_size_pass": len(corpus) == 3 * FULL_SPLIT_SIZE,
        "c2_seed_zero_pass": corpus.seed == 0,
        "c2_purpose_pass": all(
            item.meta.get("purpose") == "ppl" for item in corpus.items
        ),
        "c2_block_window_pass": c2_indices == REGISTERED_BLOCK_INDICES,
        "stage_c_purpose_pass": all(
            item.meta.get("purpose") == "sink" for item in stage_c_corpus.items
        ),
        "block_indices_disjoint_pass": not set(c2_indices) & set(stage_c_indices),
        "item_ids_disjoint_pass": not (
            {item.item_id for item in corpus.items}
            & {item.item_id for item in stage_c_corpus.items}
        ),
        "tokenizer_matches_stage_c_pass": (
            corpus.tokenizer_name == stage_c_corpus.tokenizer_name
        ),
        "source_dataset_matches_pass": (
            corpus.source.get("dataset_id") == stage_c_corpus.source.get("dataset_id")
        ),
    }
    if not all(checks.values()):
        raise StageC2Error(f"Stage-C2 fresh-corpus contract failed: {checks}")
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
    """Apply the unchanged validation rule and emit a Stage-C2-only lock."""

    document = _build_stage_c_operating_point(validation_rows, conditions, **kwargs)
    document["schema"] = OPERATING_POINT_SCHEMA
    document["experiment_id"] = EXPERIMENT_ID
    document["amendment"] = "A005"
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
    """Re-establish every C2 hash lock before first test-split access."""

    if document.get("schema") != OPERATING_POINT_SCHEMA:
        raise StageC2Error("Missing or unsupported frozen Stage-C2 operating-point schema")
    if document.get("experiment_id") != EXPERIMENT_ID or document.get("amendment") != "A005":
        raise StageC2Error("Operating-point experiment boundary is not Stage C2/A005")
    stored = str(document.get("operating_point_sha256", ""))
    if stored != canonical_sha256({
        key: value for key, value in document.items()
        if key != "operating_point_sha256"
    }):
        raise StageC2Error("Frozen Stage-C2 operating-point hash does not reproduce")

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
        raise StageC2Error(str(exc)) from exc
    return document


def freeze_operating_point(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Write a verified Stage-C2 validation decision exactly once."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Operating-point artefact already exists: {path}")
    if document.get("schema") != OPERATING_POINT_SCHEMA:
        raise StageC2Error("Refusing to freeze a non-Stage-C2 operating point")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(document))
    return path


def unlock_test_split(path: Path | str, **expected: str) -> Mapping[str, Any]:
    """Load and verify the C2 validation artefact required for test access."""

    path = Path(path)
    if not path.is_file():
        raise StageC2Error(
            "Locked Stage-C2 test split refused: no frozen validation operating point"
        )
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise StageC2Error("Operating-point artefact is not a JSON object")
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
    """Evaluate the unchanged formal gate under the Stage-C2 schema."""

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
    document["schema"] = FORMAL_GATE_SCHEMA
    document["experiment_id"] = EXPERIMENT_ID
    document["amendment"] = "A005"
    if registered_run:
        document = _rehash(document, "formal_gate_sha256")
    return document


__all__ = [
    "EXPERIMENT_ID",
    "FORMAL_GATE_SCHEMA",
    "OPERATING_POINT_SCHEMA",
    "REGISTERED_BLOCK_INDICES",
    "REGISTERED_CORPUS_ID",
    "REGISTERED_MODEL_ALIAS",
    "StageC2Error",
    "build_operating_point_document",
    "evaluate_formal_gate",
    "freeze_operating_point",
    "registered_full_conditions",
    "stage_c2_run_root",
    "unlock_test_split",
    "verify_fresh_corpus",
    "verify_operating_point_document",
]
