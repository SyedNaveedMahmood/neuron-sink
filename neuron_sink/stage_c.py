"""Stage-C Qwen replication boundary built on the registered phenomenon grid.

Stage C repeats the Stage-B causal experiment on Qwen2.5 with a fresh tokenizer-specific
neutral corpus, sink map, eligible-layer set, attribution ranking, neuron sets, and matched
random controls. Only the statistical rules are shared with Stage B. Stage-C validation
artefacts have their own schema and cannot unlock a Stage-B test split (or vice versa).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import canonical_sha256, read_json, write_json
from .stage_b import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FULL_ALPHAS,
    FULL_EXAMPLES_PER_SPLIT,
    FullCondition,
    OPERATING_POINT_SCHEMA as STAGE_B_OPERATING_POINT_SCHEMA,
    EXPERIMENT_ID as STAGE_B_EXPERIMENT_ID,
    StageBError,
    build_operating_point_document as _build_stage_b_operating_point,
    evaluate_formal_gate as _evaluate_stage_b_gate,
    registered_full_conditions,
    verify_operating_point_document as _verify_stage_b_operating_point,
)


EXPERIMENT_ID = "stage_c_qwen_replication_v1"
OPERATING_POINT_SCHEMA = "stage_c_operating_point_v1"
FORMAL_GATE_SCHEMA = "stage_c_formal_gate_v1"
REGISTERED_MODEL_ALIAS = "qwen2.5-1.5b-instruct"


class StageCError(RuntimeError):
    """Raised when a Stage-C artefact or stage transition is invalid."""


def stage_c_run_root(
    repository_root: Path | str,
    model_alias: str,
    *,
    registered_run: bool,
    stamp: str,
) -> Path:
    """Return the model-separated append-only Stage-C output path."""

    if model_alias != REGISTERED_MODEL_ALIAS:
        raise StageCError(f"Unregistered Stage-C model alias {model_alias!r}")
    if not stamp.startswith("run_") or not stamp.endswith("Z"):
        raise StageCError(f"Run stamp is not the registered UTC form: {stamp!r}")
    category = "stage_c_full" if registered_run else "stage_c_preflight"
    return Path(repository_root).resolve() / "results" / category / model_alias / stamp


def _rehash(document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    updated = {key: value for key, value in document.items() if key != hash_field}
    updated[hash_field] = canonical_sha256(updated)
    return updated


def build_operating_point_document(
    validation_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[FullCondition],
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply the registered validation rule and emit a Stage-C-only lock."""

    document = _build_stage_b_operating_point(validation_rows, conditions, **kwargs)
    document["schema"] = OPERATING_POINT_SCHEMA
    document["experiment_id"] = EXPERIMENT_ID
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
    """Re-establish Stage-C locks before the first test-split access."""

    if document.get("schema") != OPERATING_POINT_SCHEMA:
        raise StageCError("Missing or unsupported frozen Stage-C operating-point schema")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise StageCError("Operating-point experiment id is not Stage C")
    stored = str(document.get("operating_point_sha256", ""))
    if stored != canonical_sha256(
        {key: value for key, value in document.items() if key != "operating_point_sha256"}
    ):
        raise StageCError("Frozen Stage-C operating-point hash does not reproduce")

    expected: dict[str, Any] = {
        "stage": "validation",
        "registered_run": True,
        "model_id": model_id,
        "model_revision": model_revision,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "sink_scope_sha256": sink_scope_sha256,
        "attribution_sha256": attribution_sha256,
        "neuron_sets_sha256": neuron_sets_sha256,
        "n_examples": FULL_EXAMPLES_PER_SPLIT,
        "alpha": 0.0,
    }
    mismatches = {
        key: (document.get(key), value)
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        raise StageCError(f"Operating-point lock mismatch: {mismatches}")
    proxy = dict(document)
    proxy["schema"] = STAGE_B_OPERATING_POINT_SCHEMA
    proxy["experiment_id"] = STAGE_B_EXPERIMENT_ID
    proxy = _rehash(proxy, "operating_point_sha256")
    try:
        _verify_stage_b_operating_point(
            proxy,
            model_id=model_id,
            model_revision=model_revision,
            corpus_manifest_sha256=corpus_manifest_sha256,
            sink_scope_sha256=sink_scope_sha256,
            attribution_sha256=attribution_sha256,
            neuron_sets_sha256=neuron_sets_sha256,
        )
    except StageBError as exc:
        raise StageCError(str(exc)) from exc
    return document


def freeze_operating_point(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Write a Stage-C validation decision exactly once."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Operating-point artefact already exists: {path}")
    verify_schema = document.get("schema") == OPERATING_POINT_SCHEMA
    if not verify_schema:
        raise StageCError("Refusing to freeze a non-Stage-C operating point")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(document))
    return path


def unlock_test_split(path: Path | str, **expected: str) -> Mapping[str, Any]:
    """Load and verify the Stage-C validation artefact required for test access."""

    path = Path(path)
    if not path.is_file():
        raise StageCError(
            "Locked Stage-C test split refused: a frozen validation operating-point "
            "artefact does not exist"
        )
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise StageCError("Operating-point artefact is not a JSON object")
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
    """Evaluate the unchanged phenomenon gate under a Stage-C schema."""

    try:
        document = _evaluate_stage_b_gate(
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
    except StageBError as exc:
        raise StageCError(str(exc)) from exc
    document["schema"] = FORMAL_GATE_SCHEMA
    document["experiment_id"] = EXPERIMENT_ID
    if registered_run:
        document = _rehash(document, "formal_gate_sha256")
    return document


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPERIMENT_ID",
    "FORMAL_GATE_SCHEMA",
    "FULL_ALPHAS",
    "FULL_EXAMPLES_PER_SPLIT",
    "FullCondition",
    "OPERATING_POINT_SCHEMA",
    "REGISTERED_MODEL_ALIAS",
    "StageCError",
    "build_operating_point_document",
    "evaluate_formal_gate",
    "freeze_operating_point",
    "registered_full_conditions",
    "stage_c_run_root",
    "unlock_test_split",
    "verify_operating_point_document",
]
