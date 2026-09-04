#!/usr/bin/env python
"""Run the registered Stage-B, Stage-C, or Stage-C2 phenomenon experiment.

Each invocation handles one checkpoint independently and creates one append-only run
directory.  A dry-run prefix performs discovery and validation only; it never opens the
test split and never emits an operating point or scientific gate.  A registered run uses
all 100 discovery/validation examples, freezes and re-verifies the validation operating
point, and only then reads the 100-example locked test split.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuron_sink.attribution import (  # noqa: E402
    ATTRIBUTION_AGGREGATION,
    ATTRIBUTION_METHOD,
    ATTRIBUTION_OBJECTIVE,
    FUTURE_LAYER_SEPARATOR,
    RANKING_SCORE,
    ROW_FIELDS,
    SCHEMA_VERSION as ATTRIBUTION_SCHEMA,
    TOKEN_POSITION_RULE,
    attribution_rows,
    attribution_sha256,
    objective_depends_on_layer,
    rank_neurons,
)
from neuron_sink.corpus import (  # noqa: E402
    FULL_SPLIT_SIZE,
    NeutralCorpus,
    verify_disjoint,
)
from neuron_sink.evaluation import (  # noqa: E402
    AGGREGATE_FIELDS,
    EVALUATION_PROTOCOL,
    PHENOMENON_ROW_FIELDS,
    aggregate_phenomenon_rows,
    forward_snapshot,
    paired_metrics,
    validate_phenomenon_row,
)
from neuron_sink.model_adapters import (  # noqa: E402
    GPT2ModelAdapter,
    MLPModelAdapter,
    Qwen2ModelAdapter,
)
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    canonical_sha256,
    git,
    prepare_output_dir,
    require_pinned_submodules,
    require_registered_gpu,
    read_json,
    run_stamp,
    write_json,
)
from neuron_sink.selection import (  # noqa: E402
    CONDITION_ROW_FIELDS,
    CONTROL_RNG,
    CONTROL_SEED_DERIVATION,
    FULL_CONTROL_DRAWS,
    FULL_FRACTIONS_PERCENT,
    REGISTERED_BASE_SEED,
    ROUNDING_RULE,
    SELECTION_METHOD,
    build_neuron_sets_document,
    build_selection_conditions,
    condition_rows,
    load_frozen_attribution,
    load_frozen_neuron_sets,
)
from neuron_sink.signed_selection import (  # noqa: E402
    RANKING_SCORE as SIGNED_RANKING_SCORE,
    SELECTION_METHOD as SIGNED_SELECTION_METHOD,
    build_signed_neuron_sets_document,
    build_signed_selection_conditions,
    load_signed_neuron_sets,
    positive_score_count,
)
from neuron_sink.sink_metrics import (  # noqa: E402
    REGISTERED_QUERY_RULE,
    REGISTERED_SINK_FLOOR,
    REGISTERED_TARGET_POSITION,
    build_sink_scope,
    layer_scores,
    load_frozen_sink_scope,
    per_layer_head_position0_attention,
    sink_scalar_from_map,
)
from neuron_sink.stage_b import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID as STAGE_B_EXPERIMENT_ID,
    FULL_ALPHAS,
    FULL_EXAMPLES_PER_SPLIT,
    FullCondition,
    build_operating_point_document as build_stage_b_operating_point,
    evaluate_formal_gate as evaluate_stage_b_gate,
    freeze_operating_point as freeze_stage_b_operating_point,
    phenomenon_rows_sha256,
    registered_full_conditions,
    stage_b_run_root,
    unlock_test_split as unlock_stage_b_test_split,
)
from neuron_sink.stage_c import (  # noqa: E402
    EXPERIMENT_ID as STAGE_C_EXPERIMENT_ID,
    build_operating_point_document as build_stage_c_operating_point,
    evaluate_formal_gate as evaluate_stage_c_gate,
    freeze_operating_point as freeze_stage_c_operating_point,
    stage_c_run_root,
    unlock_test_split as unlock_stage_c_test_split,
)
from neuron_sink.stage_c2 import (  # noqa: E402
    EXPERIMENT_ID as STAGE_C2_EXPERIMENT_ID,
    build_operating_point_document as build_stage_c2_operating_point,
    evaluate_formal_gate as evaluate_stage_c2_gate,
    freeze_operating_point as freeze_stage_c2_operating_point,
    stage_c2_run_root,
    unlock_test_split as unlock_stage_c2_test_split,
    verify_fresh_corpus as verify_stage_c2_fresh_corpus,
)
from neuron_sink.suppression import NeuronSet, suppress_neurons  # noqa: E402
from neuron_sink.upstream_bridge import sink_repro_module  # noqa: E402


GPT2_FROZEN_MANIFEST = ROOT / "configs" / "frozen" / "neutral_corpus_manifest.json"
GPT2_CORPUS_SHA256 = (
    "c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7"
)
QWEN_FROZEN_MANIFEST = (
    ROOT / "configs" / "frozen" / "qwen2_5_1_5b_instruct" /
    "neutral_corpus_manifest.json"
)
QWEN_CORPUS_SHA256 = (
    "e38f7d3e21ef13287228ef5bb661995f0d628f1a61c99475c73ce3649ceb7426"
)
QWEN_C2_FROZEN_MANIFEST = (
    ROOT / "configs" / "frozen" / "qwen2_5_1_5b_instruct_c2" /
    "neutral_corpus_manifest.json"
)
QWEN_C2_CORPUS_SHA256 = (
    "dc9d6e8494923a6462cbd22882bfe0ccf87435525940315e97bdae858dabe8ab"
)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    stage: str
    architecture: str
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dtype: str
    frozen_manifest: Path
    corpus_sha256: str
    experiment_id: str
    expected_layers: int
    expected_width: int
    neuron_hook_point: str
    selection_method: str
    ranking_score: str


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "gpt2-small": ModelSpec(
        alias="gpt2-small",
        stage="stage_b",
        architecture="gpt2",
        model_id="openai-community/gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        tokenizer_id="openai-community/gpt2",
        tokenizer_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        dtype="float32",
        frozen_manifest=GPT2_FROZEN_MANIFEST,
        corpus_sha256=GPT2_CORPUS_SHA256,
        experiment_id=f"{STAGE_B_EXPERIMENT_ID}_gpt2_small",
        expected_layers=12,
        expected_width=3072,
        neuron_hook_point="transformer.h[layer].mlp.c_proj input",
        selection_method=SELECTION_METHOD,
        ranking_score=RANKING_SCORE,
    ),
    "gpt2-medium": ModelSpec(
        alias="gpt2-medium",
        stage="stage_b",
        architecture="gpt2",
        model_id="openai-community/gpt2-medium",
        revision="6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        tokenizer_id="openai-community/gpt2",
        tokenizer_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        dtype="float32",
        frozen_manifest=GPT2_FROZEN_MANIFEST,
        corpus_sha256=GPT2_CORPUS_SHA256,
        experiment_id=f"{STAGE_B_EXPERIMENT_ID}_gpt2_medium",
        expected_layers=24,
        expected_width=4096,
        neuron_hook_point="transformer.h[layer].mlp.c_proj input",
        selection_method=SELECTION_METHOD,
        ranking_score=RANKING_SCORE,
    ),
    "qwen2.5-1.5b-instruct": ModelSpec(
        alias="qwen2.5-1.5b-instruct",
        stage="stage_c",
        architecture="qwen2",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        tokenizer_id="Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        dtype="bfloat16",
        frozen_manifest=QWEN_FROZEN_MANIFEST,
        corpus_sha256=QWEN_CORPUS_SHA256,
        experiment_id=STAGE_C_EXPERIMENT_ID,
        expected_layers=28,
        expected_width=8960,
        neuron_hook_point="model.layers[layer].mlp.down_proj input",
        selection_method=SELECTION_METHOD,
        ranking_score=RANKING_SCORE,
    ),
    "qwen2.5-1.5b-instruct-c2": ModelSpec(
        alias="qwen2.5-1.5b-instruct",
        stage="stage_c2",
        architecture="qwen2",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        tokenizer_id="Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        dtype="bfloat16",
        frozen_manifest=QWEN_C2_FROZEN_MANIFEST,
        corpus_sha256=QWEN_C2_CORPUS_SHA256,
        experiment_id=STAGE_C2_EXPERIMENT_ID,
        expected_layers=28,
        expected_width=8960,
        neuron_hook_point="model.layers[layer].mlp.down_proj input",
        selection_method=SIGNED_SELECTION_METHOD,
        ranking_score=SIGNED_RANKING_SCORE,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registered Stage-B, Stage-C, or Stage-C2 phenomenon experiment."
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the model-specific registered manifest (hash remains enforced).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            Path(os.environ["NEURON_SINK_HF_CACHE"])
            if os.environ.get("NEURON_SINK_HF_CACHE")
            else None
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "20-99 example discovery/validation runtime preflight. It never accesses "
            "test, freezes k*, or emits a formal gate."
        ),
    )
    parser.add_argument(
        "--baseline-preflight",
        action="store_true",
        help=(
            "Stage C only: run the required 100-example baseline sink map, freeze its "
            "model-specific scope, and stop before attribution or intervention."
        ),
    )
    parser.add_argument(
        "--resume-after-validation",
        action="store_true",
        help=(
            "Resume an interrupted registered run whose discovery and validation "
            "artefacts already verify. Requires --output-dir and never overwrites them."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    if args.max_examples is not None and not 20 <= args.max_examples < FULL_SPLIT_SIZE:
        parser.error(f"--max-examples must be in [20, {FULL_SPLIT_SIZE - 1}]")
    if args.baseline_preflight and args.max_examples is not None:
        parser.error("--baseline-preflight and --max-examples are mutually exclusive")
    if args.resume_after_validation and (
        args.baseline_preflight or args.max_examples is not None
    ):
        parser.error(
            "--resume-after-validation cannot be combined with a preflight option"
        )
    if args.resume_after_validation and args.output_dir is None:
        parser.error("--resume-after-validation requires the existing --output-dir")
    if args.baseline_preflight and MODEL_SPECS[args.model].stage not in (
        "stage_c", "stage_c2"
    ):
        parser.error("--baseline-preflight is registered only for Stage C/C2")
    if args.seed != 0:
        parser.error("Stages B, C, and C2 have registered seed 0")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    if path.exists():
        raise FileExistsError(f"Append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _torch_dtype(spec: ModelSpec) -> torch.dtype:
    if spec.dtype == "float32":
        return torch.float32
    if spec.dtype == "bfloat16":
        return torch.bfloat16
    raise AssertionError(f"Unsupported registered dtype {spec.dtype!r}")


def _dtype_audit_tolerance(spec: ModelSpec) -> float:
    """Two representable steps in the registered forward dtype (amendment A003)."""

    return 2.0 * float(torch.finfo(_torch_dtype(spec)).eps)


def _model_adapter(model: torch.nn.Module, spec: ModelSpec) -> MLPModelAdapter:
    if spec.architecture == "gpt2":
        return GPT2ModelAdapter(model, model_id=spec.model_id)
    if spec.architecture == "qwen2":
        return Qwen2ModelAdapter(model, model_id=spec.model_id)
    raise AssertionError(f"Unsupported registered architecture {spec.architecture!r}")


def _stage_api(spec: ModelSpec) -> tuple[Any, Any, Any, Any, Any]:
    if spec.stage == "stage_b":
        return (
            build_stage_b_operating_point,
            freeze_stage_b_operating_point,
            unlock_stage_b_test_split,
            evaluate_stage_b_gate,
            stage_b_run_root,
        )
    if spec.stage == "stage_c":
        return (
            build_stage_c_operating_point,
            freeze_stage_c_operating_point,
            unlock_stage_c_test_split,
            evaluate_stage_c_gate,
            stage_c_run_root,
        )
    if spec.stage == "stage_c2":
        return (
            build_stage_c2_operating_point,
            freeze_stage_c2_operating_point,
            unlock_stage_c2_test_split,
            evaluate_stage_c2_gate,
            stage_c2_run_root,
        )
    raise AssertionError(f"Unsupported registered stage {spec.stage!r}")


def _discover_sink_scope(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    spec: ModelSpec,
    corpus: NeutralCorpus,
    items: Sequence[Any],
    output_dir: Path,
    *,
    legacy: Any,
    engine: Any,
    attention_tolerance: float,
    causal_tolerance: float,
    progress_every: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Map baseline attention and freeze one model-specific sink scope."""

    num_layers = adapter.num_layers
    num_heads = adapter.num_attention_heads
    band_start, band_end = legacy.compute_band(num_layers, "scaled")
    parity_band = list(range(band_start, band_end))
    map_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    example_maps: list[np.ndarray] = []
    per_example: list[dict[str, Any]] = []
    max_map_vs_upstream = 0.0
    nonfinite = 0
    max_row_error = 0.0
    max_future = 0.0

    started = time.perf_counter()
    for index, item in enumerate(items):
        if progress_every > 0 and (index == 0 or (index + 1) % progress_every == 0):
            print(f"  sink map [{index + 1}/{len(items)}] {item.item_id}", flush=True)
        ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_attentions=True,
                use_cache=False,
            )
        attentions = tuple(attention[0].detach() for attention in output.attentions)
        example_map = per_layer_head_position0_attention(attentions)
        if example_map.shape != (num_layers, num_heads):
            raise AssertionError(f"Unexpected sink-map shape {example_map.shape}")
        map_sum += example_map
        example_maps.append(example_map)

        cpu_attentions = [attention.float().cpu() for attention in attentions]
        upstream = float(legacy.compute_bos_attention_metric(
            cpu_attentions,
            num_layers,
            "mid",
            target_pos=REGISTERED_TARGET_POSITION,
            layer_start=band_start,
            layer_end=band_end,
        ))
        mapped = sink_scalar_from_map(example_map, parity_band)
        max_map_vs_upstream = max(max_map_vs_upstream, abs(mapped - upstream))

        stacked = torch.stack(attentions)
        nonfinite += int((~torch.isfinite(stacked)).sum().item())
        max_row_error = max(
            max_row_error, float((stacked.sum(dim=-1) - 1.0).abs().max().item())
        )
        max_future = max(
            max_future, float(torch.triu(stacked, diagonal=1).abs().max().item())
        )
        per_example.append({
            "example_id": item.item_id,
            "split": "discovery",
            "prompt_tokens": item.n_tokens,
            "sink_parity_band": mapped,
            "sink_upstream_hf": upstream,
            "sink_all_layers": sink_scalar_from_map(example_map),
            **{
                f"layer_{layer}": float(example_map[layer].mean())
                for layer in range(num_layers)
            },
        })
        del output, attentions, stacked, ids

    layer_head_mean = map_sum / len(items)
    scores = layer_scores(layer_head_mean)
    selected = build_sink_scope(layer_head_mean, floor=REGISTERED_SINK_FLOOR)
    mean_parity = float(np.mean([row["sink_parity_band"] for row in per_example]))
    metric_limit = max(
        float(engine.METRIC_ATOL + engine.METRIC_RTOL * max(abs(mean_parity), 1e-12)),
        _dtype_audit_tolerance(spec),
    )
    checks = {
        "map_decomposition_vs_upstream_max_abs_diff": max_map_vs_upstream,
        "metric_tolerance": metric_limit,
        "map_decomposition_pass": max_map_vs_upstream <= metric_limit,
        "nonfinite_attention_values": nonfinite,
        "max_attention_row_sum_error": max_row_error,
        "max_causal_future_attention": max_future,
        "attention_validity_pass": bool(
            nonfinite == 0
            and max_row_error <= attention_tolerance
            and max_future <= causal_tolerance
        ),
        "scope_complete_pass": bool(
            selected.sink_layers
            and selected.eligible_mlp_layers
            and not selected.fallback_incomplete
        ),
    }
    if not all(value for key, value in checks.items() if key.endswith("_pass")):
        raise RuntimeError(f"{spec.stage} sink-map checks failed: {checks}")

    document: dict[str, Any] = {
        "schema": "sink_scope_v1",
        "experiment_id": spec.experiment_id,
        "stage": "discovery",
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "dtype": spec.dtype,
        "corpus_id": corpus.corpus_id,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "split": "discovery",
        "split_mode": "full_100" if len(items) == FULL_SPLIT_SIZE else "dry_run_prefix",
        "n_examples": len(items),
        "seq_len": corpus.cut_length,
        "example_ids": [item.item_id for item in items],
        "selection_rule": (
            "layer sink score in the top quartile AND >= absolute_floor; if fewer than "
            "two qualify, the top two above the floor; if none, sink preflight fails"
        ),
        "per_layer_sink": [float(value) for value in scores],
        "per_layer_per_head_sink": [
            [float(value) for value in row] for row in layer_head_mean
        ],
        **selected.to_dict(),
    }
    document["sink_scope_sha256"] = canonical_sha256(document)
    _write_csv(output_dir / "per_example_sink.csv", per_example, tuple(per_example[0]))
    _write_new_json(output_dir / "sink_map.json", {
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "per_layer_per_head_sink": document["per_layer_per_head_sink"],
        "per_layer_sink": document["per_layer_sink"],
        "n_examples": len(items),
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
    })
    scope_path = output_dir / "sink_scope.json"
    _write_new_json(scope_path, document)
    frozen = load_frozen_sink_scope(
        scope_path, expected_corpus_manifest_sha256=corpus.manifest_sha256
    )
    return frozen, np.stack(example_maps), checks


def _run_attribution(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    spec: ModelSpec,
    corpus: NeutralCorpus,
    items: Sequence[Any],
    scope: Any,
    example_maps: np.ndarray,
    output_dir: Path,
    *,
    progress_every: int,
) -> tuple[Any, dict[str, Any]]:
    """Run the registered layer-wise full-discovery attribution and freeze it."""

    eligible = list(scope.eligible_mlp_layers)
    probe_ids = torch.tensor(
        [list(items[0].input_ids)], dtype=torch.long, device=model.device
    )
    causal_probe: list[dict[str, Any]] = []
    for layer in eligible:
        targets = list(scope.targets_for(layer))
        earlier = [target for target in scope.sink_layers if target <= layer]
        causal_probe.append({
            "mlp_layer": layer,
            "future_targets": targets,
            "future_targets_depend": objective_depends_on_layer(
                model, adapter, probe_ids, layer, targets
            ),
            "same_layer_depends": objective_depends_on_layer(
                model, adapter, probe_ids, layer, [layer]
            ),
            "earlier_sink_layers": earlier,
            "earlier_sink_depends": (
                objective_depends_on_layer(model, adapter, probe_ids, layer, earlier)
                if earlier
                else None
            ),
        })
    del probe_ids
    causal_pass = all(
        row["future_targets_depend"] is True
        and row["same_layer_depends"] is False
        and row["earlier_sink_depends"] in (False, None)
        for row in causal_probe
    )
    if not causal_pass:
        raise RuntimeError(f"{spec.stage} causal-order probe failed before attribution")

    def progress(layer: int, done: int, total: int) -> None:
        if progress_every > 0 and (done == 1 or done % progress_every == 0):
            print(f"  attribution layer {layer} [{done}/{total}]", flush=True)

    started = time.perf_counter()
    max_examples = None if len(items) == FULL_SPLIT_SIZE else len(items)
    result = rank_neurons(
        model,
        adapter,
        corpus,
        scope.future_sink_layers,
        split="discovery",
        smoke=False,
        max_examples=max_examples,
        target_pos=REGISTERED_TARGET_POSITION,
        device=model.device,
        progress=progress,
    )
    rows = attribution_rows(result)
    rows_hash = attribution_sha256(rows)
    widths = {layer: adapter.mlp_width(layer) for layer in eligible}
    expected_rows = sum(widths.values())
    target_means = {
        layer: float(example_maps[:, list(scope.targets_for(layer)), :].mean())
        for layer in eligible
    }
    objective_diff = max(
        abs(layer.mean_sink_future - target_means[layer.layer])
        for layer in result.layers
    )
    gradient_pass = all(
        layer.nonfinite_values == 0
        and layer.zero_gradient_examples == 0
        and layer.max_abs_gradient > 0.0
        for layer in result.layers
    )
    rows_pass = bool(
        len(rows) == expected_rows
        and {int(row["layer"]) for row in rows} == set(eligible)
        and all(0 <= int(row["neuron"]) < widths[int(row["layer"])] for row in rows)
        and sorted(int(row["rank_abs"]) for row in rows)
        == list(range(1, expected_rows + 1))
    )
    checks = {
        "causal_order_pass": causal_pass,
        "gradient_pass": gradient_pass,
        "rows_pass": rows_pass,
        "objective_vs_frozen_sink_map_max_abs_diff": objective_diff,
        "objective_tolerance": max(1e-6, _dtype_audit_tolerance(spec)),
        "objective_pass": objective_diff <= max(1e-6, _dtype_audit_tolerance(spec)),
        "discovery_only_pass": result.split == "discovery",
        "expected_rows": expected_rows,
    }
    if not all(value for key, value in checks.items() if key.endswith("_pass")):
        raise RuntimeError(f"{spec.stage} attribution checks failed: {checks}")

    per_layer = [layer.diagnostics() for layer in result.layers]
    metadata: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "experiment_id": spec.experiment_id,
        "stage": "discovery",
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "dtype": spec.dtype,
        "corpus_id": corpus.corpus_id,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "sink_scope_sha256": scope.sink_scope_sha256,
        "split": "discovery",
        "split_mode": "full_100" if len(items) == FULL_SPLIT_SIZE else "dry_run_prefix",
        "n_examples": len(items),
        "n_tokens": len(items) * corpus.cut_length,
        "seq_len": corpus.cut_length,
        "example_ids": list(result.example_ids),
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "neuron_hook_point": spec.neuron_hook_point,
        "attribution_method": ATTRIBUTION_METHOD,
        "attribution_objective": ATTRIBUTION_OBJECTIVE,
        "attribution_aggregation": ATTRIBUTION_AGGREGATION,
        "token_position_rule": TOKEN_POSITION_RULE,
        "ranking_score": RANKING_SCORE,
        "sink_target_position": REGISTERED_TARGET_POSITION,
        "sink_query_rule": REGISTERED_QUERY_RULE,
        "sink_layers": list(scope.sink_layers),
        "sink_heads_used": "all_heads",
        "eligible_mlp_layers": eligible,
        "future_sink_layers": {
            str(layer): list(scope.targets_for(layer)) for layer in eligible
        },
        "mlp_width": {str(layer): widths[layer] for layer in eligible},
        "n_rows": len(rows),
        "row_fields": list(ROW_FIELDS),
        "future_sink_layers_separator": FUTURE_LAYER_SEPARATOR,
        "attribution_sha256": rows_hash,
        "per_layer": per_layer,
        "causal_order_probe": causal_probe,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "is_causal_evidence": False,
    }
    csv_path = output_dir / "neuron_attribution.csv"
    metadata_path = output_dir / "neuron_attribution_metadata.json"
    _write_csv(csv_path, rows, ROW_FIELDS)
    _write_new_json(metadata_path, metadata)
    ranking = load_frozen_attribution(
        csv_path,
        metadata_path,
        scope=scope,
        expected_corpus_manifest_sha256=corpus.manifest_sha256,
    )
    return ranking, checks


def _select_full_grid(
    ranking: Any,
    output_dir: Path,
    *,
    experiment_id: str,
    signed: bool,
) -> tuple[Any, tuple[FullCondition, ...]]:
    if signed:
        conditions = build_signed_selection_conditions(
            ranking,
            FULL_FRACTIONS_PERCENT,
            control_draws=FULL_CONTROL_DRAWS,
            base_seed=REGISTERED_BASE_SEED,
        )
        document = build_signed_neuron_sets_document(
            ranking,
            conditions,
            fractions_percent=FULL_FRACTIONS_PERCENT,
            control_draws=FULL_CONTROL_DRAWS,
            base_seed=REGISTERED_BASE_SEED,
            experiment_id=experiment_id,
        )
    else:
        conditions = build_selection_conditions(
            ranking,
            FULL_FRACTIONS_PERCENT,
            control_draws=FULL_CONTROL_DRAWS,
            base_seed=REGISTERED_BASE_SEED,
        )
        document = build_neuron_sets_document(
            ranking,
            conditions,
            fractions_percent=FULL_FRACTIONS_PERCENT,
            control_draws=FULL_CONTROL_DRAWS,
            base_seed=REGISTERED_BASE_SEED,
            experiment_id=experiment_id,
        )
    json_path = output_dir / "neuron_sets.json"
    _write_new_json(json_path, document)
    _write_csv(output_dir / "neuron_sets.csv", condition_rows(conditions), CONDITION_ROW_FIELDS)
    frozen = (
        load_signed_neuron_sets(json_path, ranking=ranking)
        if signed
        else load_frozen_neuron_sets(json_path)
    )
    return frozen, registered_full_conditions(frozen)


def _load_stage_neuron_sets(
    spec: ModelSpec, path: Path, *, ranking: Any
) -> Any:
    """Load and cross-check the stage-specific frozen target-selection method."""

    if spec.stage == "stage_c2":
        return load_signed_neuron_sets(path, ranking=ranking)
    return load_frozen_neuron_sets(path)


def _row(
    *,
    experiment_id: str,
    model_id: str,
    stage: str,
    example_id: str,
    condition_id: str,
    condition_order: int,
    alpha_order: int,
    control_type: str,
    control_seed: int | None,
    fraction_percent: float | None,
    k: int | None,
    alpha: float,
    prompt_tokens: int,
    forward_runtime_seconds: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "stage": stage,
        "example_id": example_id,
        "condition_id": condition_id,
        "condition_order": condition_order,
        "alpha_order": alpha_order,
        "control_type": control_type,
        "control_seed": control_seed,
        "fraction": None if fraction_percent is None else fraction_percent / 100.0,
        "fraction_percent": fraction_percent,
        "k": k,
        "alpha": alpha,
        "prompt_tokens": prompt_tokens,
        "forward_runtime_seconds": forward_runtime_seconds,
        **dict(metrics),
    }
    validate_phenomenon_row(
        row, allowed_stages=(stage,), allowed_alphas=FULL_ALPHAS
    )
    return row


def _cpu_reference(snapshot: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    return (
        snapshot.logits.detach().cpu().clone(),
        tuple(attention.detach().cpu().clone() for attention in snapshot.attentions),
    )


def _reference_difference(reference: Any, snapshot: Any) -> tuple[float, float]:
    logits, attentions = reference
    return (
        float((logits - snapshot.logits.detach().cpu()).abs().max().item()),
        max(
            float((left - right.detach().cpu()).abs().max().item())
            for left, right in zip(attentions, snapshot.attentions)
        ),
    )


def _evaluate_split(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    spec: ModelSpec,
    items: Sequence[Any],
    scope: Any,
    conditions: Sequence[FullCondition],
    output_dir: Path,
    *,
    stage: str,
    experiment_id: str,
    attention_tolerance: float,
    causal_tolerance: float,
    progress_every: int,
    reference: Any | None,
    reference_ids: torch.Tensor | None,
) -> dict[str, Any]:
    """Evaluate one complete split while retaining no vocabulary logits across forwards."""

    rows: list[dict[str, Any]] = []
    forward_count = 0
    identity_pass = True
    validity_pass = True
    split_started = time.perf_counter()
    for example_index, item in enumerate(items):
        if progress_every > 0 and (
            example_index == 0 or (example_index + 1) % progress_every == 0
        ):
            print(
                f"  {stage} [{example_index + 1}/{len(items)}] {item.item_id}",
                flush=True,
            )
        input_ids = torch.tensor(
            [list(item.input_ids)], dtype=torch.long, device=model.device
        )
        started = time.perf_counter()
        baseline = forward_snapshot(
            model,
            input_ids,
            sink_layers=scope.sink_layers,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
        )
        baseline_elapsed = time.perf_counter() - started
        forward_count += 1
        if reference is None:
            reference = _cpu_reference(baseline)
            reference_ids = input_ids.detach().cpu().clone()
        baseline_metrics = paired_metrics(baseline, baseline)
        baseline_row = _row(
            experiment_id=experiment_id,
            model_id=spec.model_id,
            stage=stage,
            example_id=item.item_id,
            condition_id="baseline",
            condition_order=0,
            alpha_order=0,
            control_type="baseline",
            control_seed=None,
            fraction_percent=None,
            k=None,
            alpha=1.0,
            prompt_tokens=item.n_tokens,
            forward_runtime_seconds=baseline_elapsed,
            metrics=baseline_metrics,
        )
        rows.append(baseline_row)
        identity_pass &= bool(
            baseline_row["logits_exact_match"]
            and baseline_row["attentions_exact_match"]
        )
        validity_pass &= bool(baseline_row["valid_forward"])

        for condition in conditions:
            hooks_before = {
                layer: tuple(adapter.mlp_projection(layer)._forward_pre_hooks.items())
                for layer in condition.neuron_set.by_layer
            }
            for alpha_order, alpha in enumerate(FULL_ALPHAS, start=1):
                started = time.perf_counter()
                with suppress_neurons(adapter, condition.neuron_set, alpha) as context:
                    expected_hooks = 0 if alpha == 1.0 else len(
                        condition.neuron_set.by_layer
                    )
                    if context.active_hook_count != expected_hooks:
                        raise AssertionError("Suppression hook count does not match selection")
                    intervention = forward_snapshot(
                        model,
                        input_ids,
                        sink_layers=scope.sink_layers,
                        attention_tolerance=attention_tolerance,
                        causal_tolerance=causal_tolerance,
                    )
                for layer, expected in hooks_before.items():
                    actual = tuple(adapter.mlp_projection(layer)._forward_pre_hooks.items())
                    if actual != expected:
                        raise AssertionError(
                            f"Hook leakage after {condition.condition_id}, layer {layer}"
                        )
                metrics = paired_metrics(baseline, intervention)
                row = _row(
                    experiment_id=experiment_id,
                    model_id=spec.model_id,
                    stage=stage,
                    example_id=item.item_id,
                    condition_id=condition.condition_id,
                    condition_order=condition.condition_order,
                    alpha_order=alpha_order,
                    control_type=condition.control_type,
                    control_seed=condition.control_seed,
                    fraction_percent=condition.fraction_percent,
                    k=condition.k,
                    alpha=alpha,
                    prompt_tokens=item.n_tokens,
                    forward_runtime_seconds=time.perf_counter() - started,
                    metrics=metrics,
                )
                rows.append(row)
                if alpha == 1.0:
                    identity_pass &= bool(
                        row["logits_exact_match"] and row["attentions_exact_match"]
                    )
                validity_pass &= bool(row["valid_forward"])
                forward_count += 1
                del intervention
        del baseline, input_ids

    torch.cuda.synchronize(model.device)
    aggregates = aggregate_phenomenon_rows(
        rows, allowed_stages=(stage,), allowed_alphas=FULL_ALPHAS
    )
    _write_csv(output_dir / "per_example.csv", rows, PHENOMENON_ROW_FIELDS)
    _write_csv(output_dir / "aggregate.csv", aggregates, AGGREGATE_FIELDS)
    _write_new_json(output_dir / "aggregate.json", {
        "stage": stage,
        "rows_sha256": phenomenon_rows_sha256(rows),
        "rows": aggregates,
    })
    runtime = time.perf_counter() - split_started
    _write_new_json(output_dir / "summary.json", {
        "stage": stage,
        "n_examples": len(items),
        "condition_count": len(conditions),
        "alpha_grid": list(FULL_ALPHAS),
        "per_example_rows": len(rows),
        "forward_count": forward_count,
        "runtime_seconds": runtime,
        "forwards_per_second": forward_count / max(runtime, 1e-12),
        "identity_exact_pass": identity_pass,
        "validity_pass": validity_pass,
        "rows_sha256": phenomenon_rows_sha256(rows),
    })
    return {
        "rows": rows,
        "aggregates": aggregates,
        "forward_count": forward_count,
        "runtime_seconds": runtime,
        "identity_pass": identity_pass,
        "validity_pass": validity_pass,
        "reference": reference,
        "reference_ids": reference_ids,
    }


def _qwen_adapter_preflight(
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    item: Any,
    scope: Any,
    *,
    attention_tolerance: float,
    causal_tolerance: float,
) -> dict[str, Any]:
    """Audit the real Qwen hook and one future-sink gradient without ranking neurons."""

    if not isinstance(adapter, Qwen2ModelAdapter):
        raise TypeError("The Stage-C adapter preflight requires Qwen2ModelAdapter")
    layer = max(scope.eligible_mlp_layers)
    targets = list(scope.targets_for(layer))
    width = adapter.mlp_width(layer)
    probe = NeuronSet({layer: (0, width - 1)}, source="adapter_preflight_not_ranking")
    ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=model.device)
    baseline = forward_snapshot(
        model,
        ids,
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    projection = adapter.mlp_projection(layer)
    hooks_before = tuple(projection._forward_pre_hooks.items())
    with suppress_neurons(adapter, probe, 1.0) as context:
        identity_hook_count = context.active_hook_count
        identity = forward_snapshot(
            model,
            ids,
            sink_layers=scope.sink_layers,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
        )
    identity_metrics = paired_metrics(baseline, identity)

    hook_audit: dict[str, Any] = {}

    def observer(_layer: int, before: torch.Tensor, after: torch.Tensor) -> None:
        keep = torch.ones(width, dtype=torch.bool, device=before.device)
        keep[[0, width - 1]] = False
        hook_audit.update({
            "captured_shape": list(before.shape),
            "captured_dtype": str(before.dtype).removeprefix("torch."),
            "selected_zero": bool(torch.count_nonzero(after[..., [0, width - 1]]) == 0),
            "unselected_exact": bool(torch.equal(before[..., keep], after[..., keep])),
        })

    with suppress_neurons(adapter, probe, 0.0, observer=observer) as context:
        active_suppression_hook_count = context.active_hook_count
        intervened = forward_snapshot(
            model,
            ids,
            sink_layers=scope.sink_layers,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
        )
    intervention_metrics = paired_metrics(baseline, intervened)
    state = forward_snapshot(
        model,
        ids,
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    state_logits_diff = float((baseline.logits - state.logits).abs().max().item())
    state_attention_diff = max(
        float((left - right).abs().max().item())
        for left, right in zip(baseline.attentions, state.attentions)
    )

    future_depends = objective_depends_on_layer(model, adapter, ids, layer, targets)
    same_layer_depends = objective_depends_on_layer(model, adapter, ids, layer, [layer])
    hook_cleanup = tuple(projection._forward_pre_hooks.items()) == hooks_before
    checks = {
        "probe_layer": layer,
        "future_sink_layers": targets,
        "probe_neurons": [0, width - 1],
        "identity_hook_free": identity_hook_count == 0,
        "identity_logits_exact": bool(identity_metrics["logits_exact_match"]),
        "identity_attentions_exact": bool(identity_metrics["attentions_exact_match"]),
        "active_suppression_hook_count": active_suppression_hook_count,
        **hook_audit,
        "alpha_zero_changes_logits": not bool(intervention_metrics["logits_exact_match"]),
        "hook_cleanup": hook_cleanup,
        "state_logits_max_abs_diff": state_logits_diff,
        "state_attention_max_abs_diff": state_attention_diff,
        "state_leakage_absent": state_logits_diff == 0.0 and state_attention_diff == 0.0,
        "future_sink_objective_depends": future_depends,
        "same_layer_sink_objective_depends": same_layer_depends,
    }
    checks["adapter_preflight_pass"] = bool(
        checks["identity_hook_free"]
        and checks["identity_logits_exact"]
        and checks["identity_attentions_exact"]
        and active_suppression_hook_count == 1
        and checks.get("selected_zero") is True
        and checks.get("unselected_exact") is True
        and checks["alpha_zero_changes_logits"]
        and hook_cleanup
        and checks["state_leakage_absent"]
        and future_depends
        and same_layer_depends is False
    )
    if not checks["adapter_preflight_pass"]:
        raise RuntimeError(f"Stage-C Qwen adapter preflight failed: {checks}")
    return checks


_ROW_STRING_FIELDS = {
    "experiment_id",
    "model_id",
    "stage",
    "example_id",
    "condition_id",
    "control_type",
}
_ROW_INTEGER_FIELDS = {
    "condition_order",
    "alpha_order",
    "control_seed",
    "k",
    "prompt_tokens",
    "prediction_tokens",
    "nonfinite_logits",
    "nonfinite_attention",
}
_ROW_BOOLEAN_FIELDS = {
    "logits_exact_match",
    "attentions_exact_match",
    "baseline_valid",
    "intervention_valid",
    "valid_forward",
    "all_zero_logits",
}


def _read_phenomenon_csv(path: Path, *, stage: str) -> list[dict[str, Any]]:
    """Restore typed per-example rows from an append-only split CSV."""

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PHENOMENON_ROW_FIELDS:
            raise RuntimeError(f"Unexpected phenomenon columns in {path}")
        for raw in reader:
            row: dict[str, Any] = {}
            for field in PHENOMENON_ROW_FIELDS:
                value = raw[field]
                if value == "":
                    row[field] = None
                elif field in _ROW_STRING_FIELDS:
                    row[field] = value
                elif field in _ROW_INTEGER_FIELDS:
                    row[field] = int(value)
                elif field in _ROW_BOOLEAN_FIELDS:
                    if value not in ("True", "False"):
                        raise RuntimeError(f"Invalid boolean {value!r} in {path}")
                    row[field] = value == "True"
                else:
                    row[field] = float(value)
            validate_phenomenon_row(
                row, allowed_stages=(stage,), allowed_alphas=FULL_ALPHAS
            )
            rows.append(row)
    return rows


def _verified_completed_split(
    output_dir: Path, stage: str
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    split_dir = output_dir / stage / "suppression"
    summary = read_json(split_dir / "summary.json")
    rows = _read_phenomenon_csv(split_dir / "per_example.csv", stage=stage)
    rows_hash = phenomenon_rows_sha256(rows)
    if summary.get("stage") != stage or summary.get("rows_sha256") != rows_hash:
        raise RuntimeError(f"Completed {stage} artefact hash does not reproduce")
    if int(summary.get("n_examples", -1)) != FULL_EXAMPLES_PER_SPLIT:
        raise RuntimeError(f"Completed {stage} artefact is not the registered 100 examples")
    if int(summary.get("condition_count", -1)) != 126:
        raise RuntimeError(f"Completed {stage} artefact has the wrong condition grid")
    if int(summary.get("per_example_rows", -1)) != len(rows):
        raise RuntimeError(f"Completed {stage} row count does not match its summary")
    return summary, rows


def _resume_after_validation(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    model: torch.nn.Module,
    adapter: MLPModelAdapter,
    spec: ModelSpec,
    corpus: NeutralCorpus,
    discovery_items: Sequence[Any],
    device: torch.device,
    recorder: ProvenanceRecorder,
    repo_commit: str,
    repo_status: str,
    submodule_commits: Mapping[str, str],
    attention_tolerance: float,
    causal_tolerance: float,
) -> int:
    """Verify a stopped run and continue only its still-locked Qwen test phase."""

    if spec.stage not in ("stage_c", "stage_c2"):
        raise RuntimeError("Exact after-validation resume is registered only for Stage C/C2")
    for relative in ("operating_point.json", "formal_gate.json", "provenance.json", "summary.json"):
        if (output_dir / relative).exists():
            raise FileExistsError(
                f"Resume refused because final artefact already exists: {relative}"
            )
    if (output_dir / "test").exists():
        raise FileExistsError("Resume refused because the test output path already exists")

    run_config = read_json(output_dir / "run_config.json")
    required_config = {
        "experiment_id": spec.experiment_id,
        "registered_run": True,
        "run_mode": "registered_full_100",
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "tokenizer_id": spec.tokenizer_id,
        "tokenizer_revision": spec.tokenizer_revision,
        "dtype": spec.dtype,
        "manifest_sha256": corpus.manifest_sha256,
        "examples_per_split": FULL_EXAMPLES_PER_SPLIT,
        "seq_len": corpus.cut_length,
        "batch_size": 1,
        "selection_method": spec.selection_method,
        "ranking_score": spec.ranking_score,
    }
    config_mismatches = {
        key: (run_config.get(key), value)
        for key, value in required_config.items()
        if run_config.get(key) != value
    }
    if config_mismatches:
        raise RuntimeError(f"Resume run-config lock mismatch: {config_mismatches}")

    discovery_dir = output_dir / "discovery"
    scope = load_frozen_sink_scope(
        discovery_dir / "sink_scope.json",
        expected_corpus_manifest_sha256=corpus.manifest_sha256,
    )
    ranking = load_frozen_attribution(
        discovery_dir / "neuron_attribution.csv",
        discovery_dir / "neuron_attribution_metadata.json",
        scope=scope,
        expected_corpus_manifest_sha256=corpus.manifest_sha256,
    )
    frozen_sets = _load_stage_neuron_sets(
        spec, discovery_dir / "neuron_sets.json", ranking=ranking
    )
    conditions = registered_full_conditions(frozen_sets)
    if tuple(run_config.get("condition_ids", ())) != tuple(
        condition.condition_id for condition in conditions
    ):
        raise RuntimeError("Resume condition order differs from run_config.json")
    for condition in conditions:
        adapter.validate_neuron_set(condition.neuron_set)

    hash_locks = {
        "sink_scope_sha256": scope.sink_scope_sha256,
        "attribution_sha256": ranking.attribution_sha256,
        "neuron_sets_sha256": frozen_sets.document["neuron_sets_sha256"],
    }
    lock_mismatches = {
        key: (run_config.get(key), value)
        for key, value in hash_locks.items()
        if run_config.get(key) != value
    }
    if lock_mismatches:
        raise RuntimeError(f"Resume discovery hash mismatch: {lock_mismatches}")

    discovery_summary, discovery_rows = _verified_completed_split(
        output_dir, "discovery"
    )
    del discovery_rows
    validation_summary, validation_rows = _verified_completed_split(
        output_dir, "validation"
    )
    sink_map = read_json(discovery_dir / "sink_map.json")
    attribution_metadata = read_json(
        discovery_dir / "neuron_attribution_metadata.json"
    )

    (
        build_operating_point,
        freeze_operating_point,
        unlock_test_split,
        evaluate_formal_gate,
        _,
    ) = _stage_api(spec)
    operating_point_path = output_dir / "operating_point.json"
    operating_point = build_operating_point(
        validation_rows,
        conditions,
        model_id=spec.model_id,
        model_revision=spec.revision,
        corpus_manifest_sha256=corpus.manifest_sha256,
        sink_scope_sha256=scope.sink_scope_sha256,
        attribution_sha256=ranking.attribution_sha256,
        neuron_sets_sha256=str(frozen_sets.document["neuron_sets_sha256"]),
    )
    freeze_operating_point(operating_point_path, operating_point)
    unlock_test_split(
        operating_point_path,
        model_id=spec.model_id,
        model_revision=spec.revision,
        corpus_manifest_sha256=corpus.manifest_sha256,
        sink_scope_sha256=scope.sink_scope_sha256,
        attribution_sha256=ranking.attribution_sha256,
        neuron_sets_sha256=str(frozen_sets.document["neuron_sets_sha256"]),
    )
    del validation_rows

    reference_ids = torch.tensor(
        [list(discovery_items[0].input_ids)], dtype=torch.long, device=device
    )
    reference_snapshot = forward_snapshot(
        model,
        reference_ids,
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    reference = _cpu_reference(reference_snapshot)
    del reference_snapshot

    # First access to the frozen test role occurs only after the Stage-C lock verifies.
    test_items = list(corpus.items_for("test", smoke=False))
    if len(test_items) != FULL_EXAMPLES_PER_SPLIT:
        raise RuntimeError("Locked test split does not contain exactly 100 examples")
    print(
        f"Resume locks verified; opening the {spec.stage} test split once", flush=True
    )
    test_result = _evaluate_split(
        model,
        adapter,
        spec,
        test_items,
        scope,
        conditions,
        output_dir / "test" / "suppression",
        stage="test",
        experiment_id=spec.experiment_id,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
        progress_every=args.progress_every,
        reference=reference,
        reference_ids=reference_ids.detach().cpu(),
    )

    final_probe = forward_snapshot(
        model,
        reference_ids,
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    state_logits_diff, state_attention_diff = _reference_difference(reference, final_probe)
    state_leakage_pass = state_logits_diff == 0.0 and state_attention_diff == 0.0
    all_identity_pass = bool(
        discovery_summary["identity_exact_pass"]
        and validation_summary["identity_exact_pass"]
        and test_result["identity_pass"]
    )
    all_validity_pass = bool(
        discovery_summary["validity_pass"]
        and validation_summary["validity_pass"]
        and test_result["validity_pass"]
    )
    gate = evaluate_formal_gate(
        test_result["rows"],
        conditions,
        all_identity_pass=all_identity_pass,
        all_validity_pass=all_validity_pass,
        state_leakage_pass=state_leakage_pass,
        registered_run=True,
    )
    _write_new_json(output_dir / "formal_gate.json", gate)

    provenance = recorder.finish(
        repo_commit=repo_commit, submodule_commits=submodule_commits
    )
    provenance.update({
        "repo_dirty_at_run": bool(repo_status),
        "repo_status_at_run": repo_status.splitlines(),
        "resume_after_validation": True,
        "resumed_output_dir": str(output_dir),
    })
    _write_new_json(output_dir / "provenance.json", provenance)
    pre_resume_runtime = float(
        sink_map.get("runtime_seconds", 0.0)
        + attribution_metadata.get("runtime_seconds", 0.0)
        + discovery_summary["runtime_seconds"]
        + validation_summary["runtime_seconds"]
    )
    summary = {
        f"{spec.stage}_execution": "COMPLETE_RESUMED_AFTER_VALIDATION",
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "run_mode": "registered_full_100",
        "registered_run": True,
        "formal_gate_status": gate["status"],
        "formal_gate_pass": gate.get("formal_gate_pass"),
        "test_split_accessed": True,
        "n_examples_per_executed_split": FULL_EXAMPLES_PER_SPLIT,
        "sink_layers": list(scope.sink_layers),
        "eligible_mlp_layers": list(scope.eligible_mlp_layers),
        "eligible_pool_size": ranking.pool_size,
        "condition_count": len(conditions),
        "alpha_grid": list(FULL_ALPHAS),
        "forward_count_scientific_grid": int(
            discovery_summary["forward_count"]
            + validation_summary["forward_count"]
            + test_result["forward_count"]
        ),
        "resume_reference_forward_count": 2,
        "runtime_seconds_by_split": {
            "discovery": discovery_summary["runtime_seconds"],
            "validation": validation_summary["runtime_seconds"],
            "test": test_result["runtime_seconds"],
        },
        "measured_component_runtime_seconds_before_resume": pre_resume_runtime,
        "resume_runtime_seconds": provenance["runtime_seconds"],
        "runtime_seconds_total_measured_components": (
            pre_resume_runtime + provenance["runtime_seconds"]
        ),
        "peak_memory_allocated_bytes_resume_process": provenance[
            "peak_memory_allocated_bytes"
        ],
        "peak_memory_reserved_bytes_resume_process": provenance[
            "peak_memory_reserved_bytes"
        ],
        "resume": {
            "reason": "explicit verified resume after completed validation",
            "test_was_unopened_before_resume": True,
            "existing_artifacts_reused_without_overwrite": True,
            "verified_hash_locks": hash_locks,
        },
        "checks": {
            **dict(sink_map["checks"]),
            **dict(attribution_metadata["checks"]),
            "full_condition_grid_pass": len(conditions) == 126,
            "all_identity_exact_pass": all_identity_pass,
            "all_forward_validity_pass": all_validity_pass,
            "state_leakage_pass": state_leakage_pass,
            "state_leakage_max_logits_abs_diff": state_logits_diff,
            "state_leakage_max_attention_abs_diff": state_attention_diff,
            "discovery_rows_hash_reverified": True,
            "validation_rows_hash_reverified": True,
            "operating_point_verified_before_test": True,
        },
    }
    _write_new_json(output_dir / "summary.json", summary)
    _write_new_json(output_dir / "resume.json", {
        "status": "COMPLETE",
        "reason": summary["resume"]["reason"],
        "test_split_accessed": True,
        "operating_point_sha256": operating_point["operating_point_sha256"],
        "formal_gate_sha256": gate["formal_gate_sha256"],
        "existing_artifacts_reused_without_overwrite": True,
    })

    print(f"{spec.stage.upper()}_EXECUTION=COMPLETE_RESUMED_AFTER_VALIDATION", flush=True)
    print(f"FORMAL_GATE={gate['status']}", flush=True)
    print("test_split_accessed=True", flush=True)
    print(f"identity_exact={all_identity_pass}", flush=True)
    print(f"validity_pass={all_validity_pass}", flush=True)
    print(f"state_leakage_pass={state_leakage_pass}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    return 0


def _run(args: argparse.Namespace, output_dir: Path) -> int:
    spec = MODEL_SPECS[args.model]
    manifest_path = args.manifest or spec.frozen_manifest
    (
        build_operating_point,
        freeze_operating_point,
        unlock_test_split,
        evaluate_formal_gate,
        _,
    ) = _stage_api(spec)
    registered_run = args.max_examples is None and not args.baseline_preflight
    n_examples = (
        FULL_EXAMPLES_PER_SPLIT
        if registered_run or args.baseline_preflight
        else args.max_examples
    )
    assert n_examples is not None
    submodule_commits = require_pinned_submodules()
    repo_commit = git("rev-parse", "HEAD")
    repo_status = git("status", "--porcelain", "--untracked-files=all")
    device, gpu_name, total_vram = require_registered_gpu("full")
    recorder = ProvenanceRecorder(device=device, gpu_name=gpu_name)
    _set_determinism(args.seed)

    corpus = NeutralCorpus.load(manifest_path)
    if corpus.manifest_sha256 != spec.corpus_sha256:
        raise RuntimeError(
            f"Frozen neutral-corpus manifest hash is not registered for {spec.alias}"
        )
    if (
        corpus.tokenizer_name != spec.tokenizer_id
        or corpus.tokenizer_revision not in (None, spec.tokenizer_revision)
    ):
        raise RuntimeError(
            "Frozen neutral corpus tokenizer does not match the registered model input"
        )
    verify_disjoint(corpus.splits)
    fresh_corpus_checks: dict[str, Any] = {}
    if spec.stage == "stage_c2":
        stage_c_corpus = NeutralCorpus.load(QWEN_FROZEN_MANIFEST)
        fresh_corpus_checks = verify_stage_c2_fresh_corpus(corpus, stage_c_corpus)
    discovery_items = list(corpus.items_for("discovery", smoke=False))[:n_examples]
    validation_items = list(corpus.items_for("validation", smoke=False))[:n_examples]
    if len(discovery_items) != n_examples or len(validation_items) != n_examples:
        raise RuntimeError("Frozen corpus does not contain the required split sizes")

    legacy = sink_repro_module("intervention_analysis_legacy")
    engine = sink_repro_module("nnsight_engine")
    dtype = _torch_dtype(spec)
    attention_tolerance = max(
        float(engine.ATTN_ATOL + engine.ATTN_RTOL),
        _dtype_audit_tolerance(spec),
    )
    causal_tolerance = float(engine.ATTN_ATOL)

    from transformers import AutoModelForCausalLM

    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    print(
        f"Loading {spec.model_id}@{spec.revision} {spec.dtype}/eager on {gpu_name}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=cache_dir,
        attn_implementation="eager",
        dtype=dtype,
    ).eval().to(device)
    if model.training or getattr(model.config, "_attn_implementation", None) != "eager":
        raise AssertionError(f"{spec.stage} requires model.eval() with eager attention")
    actual_revision = getattr(model.config, "_commit_hash", None)
    if actual_revision != spec.revision:
        raise AssertionError(
            f"Loaded revision {actual_revision!r} != registered {spec.revision!r}"
        )
    adapter = _model_adapter(model, spec)
    if adapter.num_layers != spec.expected_layers:
        raise AssertionError(
            f"{spec.alias} has {adapter.num_layers} layers, expected {spec.expected_layers}"
        )
    if any(
        adapter.mlp_width(layer) != spec.expected_width
        for layer in range(adapter.num_layers)
    ):
        raise AssertionError(f"{spec.alias} MLP width differs from {spec.expected_width}")

    if args.resume_after_validation:
        model.requires_grad_(False)
        return _resume_after_validation(
            args=args,
            output_dir=output_dir,
            model=model,
            adapter=adapter,
            spec=spec,
            corpus=corpus,
            discovery_items=discovery_items,
            device=device,
            recorder=recorder,
            repo_commit=repo_commit,
            repo_status=repo_status,
            submodule_commits=submodule_commits,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
        )

    experiment_id = spec.experiment_id
    discovery_dir = output_dir / "discovery"
    discovery_dir.mkdir()
    print(f"Discovery: sink map on {n_examples} examples", flush=True)
    scope, example_maps, sink_checks = _discover_sink_scope(
        model,
        adapter,
        spec,
        corpus,
        discovery_items,
        discovery_dir,
        legacy=legacy,
        engine=engine,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
        progress_every=args.progress_every,
    )
    model.requires_grad_(False)
    if args.baseline_preflight:
        adapter_checks = _qwen_adapter_preflight(
            model,
            adapter,
            discovery_items[0],
            scope,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
        )
        provenance = recorder.finish(
            repo_commit=repo_commit, submodule_commits=submodule_commits
        )
        provenance["repo_dirty_at_run"] = bool(repo_status)
        provenance["repo_status_at_run"] = repo_status.splitlines()
        per_layer_sink = example_maps.mean(axis=(0, 2))
        summary = {
            "stage_c_baseline_preflight": "PASS",
            "model_alias": spec.alias,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "dtype": spec.dtype,
            "n_examples": n_examples,
            "test_split_accessed": False,
            "attribution_run": False,
            "intervention_run": False,
            "maximum_layer_sink": float(per_layer_sink.max()),
            "maximum_layer": int(per_layer_sink.argmax()),
            "registered_sink_floor": REGISTERED_SINK_FLOOR,
            "sink_preflight_pass": bool(per_layer_sink.max() >= REGISTERED_SINK_FLOOR),
            "sink_layers": list(scope.sink_layers),
            "eligible_mlp_layers": list(scope.eligible_mlp_layers),
            "sink_scope_sha256": scope.sink_scope_sha256,
            "manifest_sha256": corpus.manifest_sha256,
            "checks": {
                **fresh_corpus_checks,
                **sink_checks,
                **adapter_checks,
            },
            "runtime_seconds_total": provenance["runtime_seconds"],
            "peak_memory_allocated_bytes": provenance["peak_memory_allocated_bytes"],
            "peak_memory_reserved_bytes": provenance["peak_memory_reserved_bytes"],
        }
        _write_new_json(output_dir / "run_config.json", {
            "experiment_id": spec.experiment_id,
            "stage": "discovery_baseline_preflight",
            "run_mode": "registered_baseline_100",
            "registered_run": False,
            "model_alias": spec.alias,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "tokenizer_id": spec.tokenizer_id,
            "tokenizer_revision": spec.tokenizer_revision,
            "dtype": spec.dtype,
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": total_vram,
            "seed": args.seed,
            "dataset_id": corpus.source.get("dataset_id"),
            "dataset_split": ["discovery"],
            "manifest_sha256": corpus.manifest_sha256,
            "examples_per_split": n_examples,
            "seq_len": corpus.cut_length,
            "batch_size": 1,
            "sink_metric_definition": (
                "mean attention probability to key position 0 from second-half query "
                "positions, using all query heads"
            ),
            "sink_target_position": REGISTERED_TARGET_POSITION,
            "sink_query_rule": REGISTERED_QUERY_RULE,
            "sink_layers": list(scope.sink_layers),
            "sink_heads": None,
            "eligible_mlp_layers": list(scope.eligible_mlp_layers),
            "sink_scope_sha256": scope.sink_scope_sha256,
            "neuron_definition": "mlp_intermediate_pre_output_projection",
            "neuron_hook_point": spec.neuron_hook_point,
            "attention_implementation": "eager",
            "attention_row_sum_tolerance": attention_tolerance,
            "causal_future_attention_tolerance": causal_tolerance,
            "test_split_access_allowed": False,
            "source_semantics": {
                "sink_metric": (
                    "upstream/sink-repro/common/intervention_analysis_legacy.py::"
                    "compute_bos_attention_metric"
                ),
                "qwen_module_map": (
                    "upstream/sink-repro/cross_scale_and_architecture/qwen/"
                    "intervention_analysis_qwen.py"
                ),
            },
        })
        _write_new_json(output_dir / "provenance.json", provenance)
        _write_new_json(output_dir / "summary.json", summary)
        print("STAGE_C_BASELINE_PREFLIGHT=PASS", flush=True)
        print(f"maximum_layer_sink={summary['maximum_layer_sink']:.9f}", flush=True)
        print(f"sink_layers={summary['sink_layers']}", flush=True)
        print(f"output_dir={output_dir}", flush=True)
        return 0
    print(
        f"Discovery: attribution over eligible layers {list(scope.eligible_mlp_layers)}",
        flush=True,
    )
    ranking, attribution_checks = _run_attribution(
        model,
        adapter,
        spec,
        corpus,
        discovery_items,
        scope,
        example_maps,
        discovery_dir,
        progress_every=args.progress_every,
    )
    frozen_sets, conditions = _select_full_grid(
        ranking,
        discovery_dir,
        experiment_id=experiment_id,
        signed=spec.stage == "stage_c2",
    )
    for condition in conditions:
        adapter.validate_neuron_set(condition.neuron_set)

    run_config = {
        "experiment_id": experiment_id,
        "stage": "discovery|validation|test" if registered_run else "discovery|validation",
        "run_mode": "registered_full_100" if registered_run else "dry_run_prefix",
        "registered_run": registered_run,
        "test_split_access_allowed_at_start": False,
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_requested_id": spec.model_id,
        "model_revision": spec.revision,
        "model_revision_resolution": "Hugging Face model_info SHA resolved before outputs",
        "tokenizer_id": spec.tokenizer_id,
        "tokenizer_revision": spec.tokenizer_revision,
        "tokenizer_revision_in_frozen_manifest": corpus.tokenizer_revision,
        "dtype": spec.dtype,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total_vram,
        "seed": args.seed,
        "dataset_id": corpus.source.get("dataset_id"),
        "dataset_config": None,
        "dataset_split": ["discovery", "validation", "test"] if registered_run else ["discovery", "validation"],
        "manifest_sha256": corpus.manifest_sha256,
        "examples_per_split": n_examples,
        "seq_len": corpus.cut_length,
        "batch_size": 1,
        "sink_metric_definition": (
            "mean attention probability to key position 0 from second-half query "
            "positions over model-specific frozen sink-heavy layers and all heads"
        ),
        "sink_target_position": REGISTERED_TARGET_POSITION,
        "sink_query_rule": REGISTERED_QUERY_RULE,
        "sink_layers": list(scope.sink_layers),
        "sink_heads": None,
        "diagnostic_sink_heads": {
            str(layer): list(heads) for layer, heads in scope.sink_heads.items()
        },
        "sink_scope_sha256": scope.sink_scope_sha256,
        "eligible_mlp_layers": list(scope.eligible_mlp_layers),
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "neuron_hook_point": spec.neuron_hook_point,
        "suppression_positions": "all",
        "selection_method": spec.selection_method,
        "attribution_method": ATTRIBUTION_METHOD,
        "attribution_objective": ATTRIBUTION_OBJECTIVE,
        "ranking_score": spec.ranking_score,
        "ranking_sign_requirement": (
            "strictly_positive" if spec.stage == "stage_c2" else None
        ),
        "positive_signed_score_count": (
            positive_score_count(ranking) if spec.stage == "stage_c2" else None
        ),
        "attribution_sha256": ranking.attribution_sha256,
        "neuron_sets_sha256": frozen_sets.document["neuron_sets_sha256"],
        "neuron_sets_file": "discovery/neuron_sets.json",
        "condition_ids": [condition.condition_id for condition in conditions],
        "fractions_percent": list(FULL_FRACTIONS_PERCENT),
        "alphas": list(FULL_ALPHAS),
        "control_type": "baseline|targeted|layer_random_grid",
        "control_draws": FULL_CONTROL_DRAWS,
        "control_seed_range": list(range(FULL_CONTROL_DRAWS)),
        "control_rng": CONTROL_RNG,
        "control_seed_derivation": CONTROL_SEED_DERIVATION,
        "rounding_rule": ROUNDING_RULE,
        "evaluation_protocol_version": EVALUATION_PROTOCOL,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "attention_implementation": "eager",
        "attention_row_sum_tolerance": attention_tolerance,
        "causal_future_attention_tolerance": causal_tolerance,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "repo_dirty_at_run": bool(repo_status),
        "repo_status_at_run": repo_status.splitlines(),
        "source_semantics": {
            "sink_metric": "upstream/sink-repro/common/intervention_analysis_legacy.py::compute_bos_attention_metric",
            "neutral_corpus": "upstream/sink-kd/common/corpus_providers.py::openwebtext_corpus",
            "qwen_module_map": (
                "upstream/sink-repro/cross_scale_and_architecture/qwen/"
                "intervention_analysis_qwen.py"
                if spec.architecture == "qwen2"
                else None
            ),
        },
    }
    _write_new_json(output_dir / "run_config.json", run_config)

    reference = None
    reference_ids = None
    split_results: dict[str, dict[str, Any]] = {}
    for stage, items in (
        ("discovery", discovery_items),
        ("validation", validation_items),
    ):
        print(f"Evaluation: {stage} full grid", flush=True)
        result = _evaluate_split(
            model,
            adapter,
            spec,
            items,
            scope,
            conditions,
            output_dir / stage / "suppression",
            stage=stage,
            experiment_id=experiment_id,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
            progress_every=args.progress_every,
            reference=reference,
            reference_ids=reference_ids,
        )
        reference = result["reference"]
        reference_ids = result["reference_ids"]
        split_results[stage] = result

    operating_point_path = output_dir / "operating_point.json"
    if registered_run:
        operating_point = build_operating_point(
            split_results["validation"]["rows"],
            conditions,
            model_id=spec.model_id,
            model_revision=spec.revision,
            corpus_manifest_sha256=corpus.manifest_sha256,
            sink_scope_sha256=scope.sink_scope_sha256,
            attribution_sha256=ranking.attribution_sha256,
            neuron_sets_sha256=str(frozen_sets.document["neuron_sets_sha256"]),
        )
        freeze_operating_point(operating_point_path, operating_point)
        unlock_kwargs = {
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "sink_scope_sha256": scope.sink_scope_sha256,
            "attribution_sha256": ranking.attribution_sha256,
            "neuron_sets_sha256": str(frozen_sets.document["neuron_sets_sha256"]),
        }
        unlock_test_split(operating_point_path, **unlock_kwargs)
        # This is the first corpus API access to the locked test role in this runner.
        test_items = list(corpus.items_for("test", smoke=False))
        if len(test_items) != FULL_EXAMPLES_PER_SPLIT:
            raise RuntimeError("Locked test split does not contain exactly 100 examples")
        print("Validation artefact frozen and verified; opening locked test once", flush=True)
        test_result = _evaluate_split(
            model,
            adapter,
            spec,
            test_items,
            scope,
            conditions,
            output_dir / "test" / "suppression",
            stage="test",
            experiment_id=experiment_id,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
            progress_every=args.progress_every,
            reference=reference,
            reference_ids=reference_ids,
        )
        split_results["test"] = test_result
    else:
        _write_new_json(output_dir / "preflight.json", {
            "status": "NOT_EVALUATED_DRY_RUN",
            "test_split_accessed": False,
            "operating_point_written": False,
            "formal_gate_written": False,
            "n_examples_per_executed_split": n_examples,
        })

    if reference is None or reference_ids is None:
        raise AssertionError("No baseline reference was captured")
    final_probe = forward_snapshot(
        model,
        reference_ids.to(device),
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    state_logits_diff, state_attention_diff = _reference_difference(reference, final_probe)
    state_leakage_pass = state_logits_diff == 0.0 and state_attention_diff == 0.0
    del final_probe

    all_identity_pass = all(
        bool(result["identity_pass"]) for result in split_results.values()
    )
    all_validity_pass = all(
        bool(result["validity_pass"]) for result in split_results.values()
    )
    if registered_run:
        gate = evaluate_formal_gate(
            split_results["test"]["rows"],
            conditions,
            all_identity_pass=all_identity_pass,
            all_validity_pass=all_validity_pass,
            state_leakage_pass=state_leakage_pass,
            registered_run=True,
        )
        _write_new_json(output_dir / "formal_gate.json", gate)
    else:
        gate = evaluate_formal_gate(
            [],
            conditions,
            all_identity_pass=all_identity_pass,
            all_validity_pass=all_validity_pass,
            state_leakage_pass=state_leakage_pass,
            registered_run=False,
        )

    provenance = recorder.finish(
        repo_commit=repo_commit, submodule_commits=submodule_commits
    )
    provenance["repo_dirty_at_run"] = bool(repo_status)
    provenance["repo_status_at_run"] = repo_status.splitlines()
    _write_new_json(output_dir / "provenance.json", provenance)

    forward_count = sum(result["forward_count"] for result in split_results.values()) + 1
    split_runtime = {
        stage: result["runtime_seconds"] for stage, result in split_results.items()
    }
    summary: dict[str, Any] = {
        f"{spec.stage}_execution": "COMPLETE",
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "run_mode": run_config["run_mode"],
        "registered_run": registered_run,
        "formal_gate_status": gate["status"],
        "formal_gate_pass": gate.get("formal_gate_pass"),
        "test_split_accessed": registered_run,
        "n_examples_per_executed_split": n_examples,
        "sink_layers": list(scope.sink_layers),
        "eligible_mlp_layers": list(scope.eligible_mlp_layers),
        "eligible_pool_size": ranking.pool_size,
        "condition_count": len(conditions),
        "alpha_grid": list(FULL_ALPHAS),
        "forward_count": forward_count,
        "runtime_seconds_by_split": split_runtime,
        "runtime_seconds_total": provenance["runtime_seconds"],
        "peak_memory_allocated_bytes": provenance["peak_memory_allocated_bytes"],
        "peak_memory_reserved_bytes": provenance["peak_memory_reserved_bytes"],
        "checks": {
            **fresh_corpus_checks,
            **sink_checks,
            **attribution_checks,
            "full_condition_grid_pass": len(conditions) == 126,
            "all_identity_exact_pass": all_identity_pass,
            "all_forward_validity_pass": all_validity_pass,
            "state_leakage_pass": state_leakage_pass,
            "state_leakage_max_logits_abs_diff": state_logits_diff,
            "state_leakage_max_attention_abs_diff": state_attention_diff,
            "separate_model_artifact_path_pass": spec.alias in str(output_dir),
            "smoke_artifact_overwrite_prevented": not str(output_dir).startswith(
                str((ROOT / "configs" / "frozen").resolve())
            ),
            "selection_method_is_registered": (
                frozen_sets.document.get("selection_method")
                == spec.selection_method
                and frozen_sets.document.get("ranking_score")
                == spec.ranking_score
            ),
        },
    }
    if not registered_run:
        executed_seconds = sum(split_runtime.values())
        executed_examples = len(split_results) * n_examples
        summary["preflight_estimate"] = {
            "measured_evaluation_seconds": executed_seconds,
            "measured_examples_across_splits": executed_examples,
            "estimated_three_split_full_grid_seconds": (
                executed_seconds / max(executed_examples, 1)
                * 3
                * FULL_EXAMPLES_PER_SPLIT
            ),
            "actual_csv_bytes": sum(
                path.stat().st_size for path in output_dir.rglob("*.csv")
            ),
            "estimated_full_output_bytes": int(
                sum(path.stat().st_size for path in output_dir.rglob("*.csv"))
                * (3 * FULL_EXAMPLES_PER_SPLIT)
                / max(executed_examples, 1)
            ),
            "test_split_accessed": False,
            "scientific_gate_evaluated": False,
        }
    _write_new_json(output_dir / "summary.json", summary)

    print(f"{spec.stage.upper()}_EXECUTION=COMPLETE model={spec.alias}", flush=True)
    print(f"FORMAL_GATE={gate['status']}", flush=True)
    print(f"test_split_accessed={registered_run}", flush=True)
    print(f"identity_exact={all_identity_pass}", flush=True)
    print(f"validity_pass={all_validity_pass}", flush=True)
    print(f"state_leakage_pass={state_leakage_pass}", flush=True)
    print(f"wall_seconds={provenance['runtime_seconds']:.3f}", flush=True)
    print(
        f"peak_memory_allocated_bytes={provenance['peak_memory_allocated_bytes']}",
        flush=True,
    )
    print(f"output_dir={output_dir}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    spec = MODEL_SPECS[args.model]
    stage_run_root = _stage_api(spec)[4]
    if args.resume_after_validation:
        output_dir = args.output_dir.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {output_dir}")
    else:
        output_dir = prepare_output_dir(
            args.output_dir
            or stage_run_root(
                ROOT,
                spec.alias,
                registered_run=(
                    args.max_examples is None and not args.baseline_preflight
                ),
                stamp=run_stamp(),
            )
        )
    try:
        return _run(args, output_dir)
    except torch.cuda.OutOfMemoryError as exc:
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        _write_new_json(output_dir / "oom.json", {
            "status": "OOM_STOPPED_NO_SCIENTIFIC_CHANGE",
            "model_alias": spec.alias,
            "error": str(exc),
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "allowed_next_actions": [
                "reduce batch or microbatch size",
                "process layers serially",
                "release graphs",
                "stream results",
            ],
        })
        print(
            f"OOM: allocated_peak={peak_allocated} reserved_peak={peak_reserved}; "
            f"recorded in {output_dir / 'oom.json'}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
