#!/usr/bin/env python
"""Run amendment A006's descriptive per-layer decomposition of completed Stage C."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuron_sink.corpus import FULL_SPLIT_SIZE, NeutralCorpus, verify_disjoint  # noqa: E402
from neuron_sink.evaluation import forward_snapshot  # noqa: E402
from neuron_sink.model_adapters import Qwen2ModelAdapter  # noqa: E402
from neuron_sink.per_layer_diagnostics import (  # noqa: E402
    FIRST_ORDER_FIELDS,
    PER_LAYER_AGGREGATE_FIELDS,
    PER_LAYER_ROW_FIELDS,
    aggregate_per_layer_rows,
    first_order_predictions,
    per_layer_sink_scores,
)
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    canonical_sha256,
    git,
    prepare_output_dir,
    read_json,
    require_pinned_submodules,
    require_registered_gpu,
    run_stamp,
    write_json,
)
from neuron_sink.selection import (  # noqa: E402
    CONTROL_TYPE_TARGETED,
    load_frozen_attribution,
    load_frozen_neuron_sets,
)
from neuron_sink.sink_metrics import load_frozen_sink_scope  # noqa: E402
from neuron_sink.stage_b import FULL_ALPHAS, registered_full_conditions  # noqa: E402
from neuron_sink.stage_c import EXPERIMENT_ID as SOURCE_EXPERIMENT_ID  # noqa: E402
from neuron_sink.suppression import suppress_neurons  # noqa: E402
from neuron_sink.upstream_bridge import sink_repro_module  # noqa: E402


EXPERIMENT_ID = "stage_c_posthoc_per_layer_v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_ALIAS = "qwen2.5-1.5b-instruct"
MANIFEST = (
    ROOT / "configs" / "frozen" / "qwen2_5_1_5b_instruct" /
    "neutral_corpus_manifest.json"
)
MANIFEST_SHA256 = "e38f7d3e21ef13287228ef5bb661995f0d628f1a61c99475c73ce3649ceb7426"
DEFAULT_SOURCE_RUN = (
    ROOT / "results" / "stage_c_full" / MODEL_ALIAS / "run_20260904T160405Z"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            Path(os.environ["NEURON_SINK_HF_CACHE"])
            if os.environ.get("NEURON_SINK_HF_CACHE") else None
        ),
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 0:
        parser.error("Amendment A006 has fixed seed 0")
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _load_source_sink_index(path: Path) -> dict[tuple[str, str, float], float]:
    index: dict[tuple[str, str, float], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = str(row["condition_id"])
            if condition != "baseline" and not condition.startswith("targeted_"):
                continue
            key = (str(row["example_id"]), condition, float(row["alpha"]))
            if key in index:
                raise RuntimeError(f"Duplicate completed Stage-C row {key}")
            field = "sink_baseline" if condition == "baseline" else "sink_intervened"
            index[key] = float(row[field])
    return index


def _cpu_reference(snapshot: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    return (
        snapshot.logits.detach().cpu().clone(),
        tuple(attention.detach().cpu().clone() for attention in snapshot.attentions),
    )


def _reference_difference(
    reference: tuple[torch.Tensor, tuple[torch.Tensor, ...]], snapshot: Any
) -> tuple[float, float]:
    logits, attentions = reference
    return (
        float((logits - snapshot.logits.detach().cpu()).abs().max().item()),
        max(
            float((left - right.detach().cpu()).abs().max().item())
            for left, right in zip(attentions, snapshot.attentions)
        ),
    )


def main() -> int:
    args = _parse_args()
    source_run = args.source_run.resolve()
    output_dir = prepare_output_dir(
        args.output_dir
        or ROOT / "results" / "stage_c_posthoc_per_layer" / MODEL_ALIAS / run_stamp()
    )
    submodule_commits = require_pinned_submodules()
    repo_commit = git("rev-parse", "HEAD")
    repo_status = git("status", "--porcelain", "--untracked-files=all")
    device, gpu_name, total_vram = require_registered_gpu("full")
    recorder = ProvenanceRecorder(device=device, gpu_name=gpu_name)
    _set_determinism(args.seed)

    required_source = (
        "run_config.json",
        "formal_gate.json",
        "summary.json",
        "discovery/sink_scope.json",
        "discovery/neuron_attribution.csv",
        "discovery/neuron_attribution_metadata.json",
        "discovery/neuron_sets.json",
        "test/suppression/per_example.csv",
    )
    for relative in required_source:
        if not (source_run / relative).is_file():
            raise FileNotFoundError(f"Completed Stage-C artifact missing: {relative}")
    source_config = read_json(source_run / "run_config.json")
    required_config = {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "registered_run": True,
        "run_mode": "registered_full_100",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "manifest_sha256": MANIFEST_SHA256,
        "examples_per_split": FULL_SPLIT_SIZE,
        "seq_len": 40,
        "ranking_score": "mean_abs_attr",
    }
    mismatches = {
        key: (source_config.get(key), expected)
        for key, expected in required_config.items()
        if source_config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Source Stage-C configuration mismatch: {mismatches}")

    corpus = NeutralCorpus.load(MANIFEST)
    if corpus.manifest_sha256 != MANIFEST_SHA256:
        raise RuntimeError("Stage-C frozen corpus hash differs from amendment A006")
    verify_disjoint(corpus.splits)
    test_items = list(corpus.items_for("test", smoke=False))
    if len(test_items) != FULL_SPLIT_SIZE:
        raise RuntimeError("Stage-C test role does not contain exactly 100 examples")

    discovery_dir = source_run / "discovery"
    scope = load_frozen_sink_scope(
        discovery_dir / "sink_scope.json",
        expected_corpus_manifest_sha256=MANIFEST_SHA256,
    )
    ranking = load_frozen_attribution(
        discovery_dir / "neuron_attribution.csv",
        discovery_dir / "neuron_attribution_metadata.json",
        scope=scope,
        expected_corpus_manifest_sha256=MANIFEST_SHA256,
    )
    frozen_sets = load_frozen_neuron_sets(discovery_dir / "neuron_sets.json")
    conditions = tuple(
        condition
        for condition in registered_full_conditions(frozen_sets)
        if condition.control_type == CONTROL_TYPE_TARGETED
    )
    if len(conditions) != 6:
        raise RuntimeError("Amendment A006 requires the six frozen targeted sets")
    if source_config["sink_scope_sha256"] != scope.sink_scope_sha256:
        raise RuntimeError("Source run's sink-scope lock does not reproduce")
    if source_config["attribution_sha256"] != ranking.attribution_sha256:
        raise RuntimeError("Source run's attribution lock does not reproduce")
    if source_config["neuron_sets_sha256"] != frozen_sets.document["neuron_sets_sha256"]:
        raise RuntimeError("Source run's neuron-set lock does not reproduce")

    predictions = first_order_predictions(
        ranking.rows,
        conditions,
        sink_layers=scope.sink_layers,
        seq_len=corpus.cut_length,
        alphas=FULL_ALPHAS,
    )
    _write_csv(output_dir / "first_order_predictions.csv", predictions, FIRST_ORDER_FIELDS)
    _write_new_json(output_dir / "first_order_predictions.json", {
        "schema": "stage_c_first_order_prediction_v1",
        "formula": (
            "-(1-alpha)*seq_len*sum(mean_signed_attr*"
            "future_sink_layer_count/sink_layer_count)"
        ),
        "prediction_scope": "registered aggregate sink only; not individual layers",
        "rows_sha256": canonical_sha256(predictions),
        "rows": predictions,
    })

    source_hashes = {
        relative: _file_sha256(source_run / relative) for relative in required_source
    }
    run_config = {
        "schema": "stage_c_posthoc_per_layer_config_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "amendment": "A006",
        "inferential_status": "post_hoc_descriptive_only",
        "can_change_stage_c_gate": False,
        "can_change_stage_c2": False,
        "source_run": str(source_run),
        "source_artifact_sha256": source_hashes,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total_vram,
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "seed": args.seed,
        "dataset_id": corpus.source.get("dataset_id"),
        "dataset_config": None,
        "dataset_split": "test",
        "manifest_sha256": corpus.manifest_sha256,
        "prompt_evaluation_protocol_version": "stage_c_posthoc_per_layer_v1",
        "examples": len(test_items),
        "seq_len": corpus.cut_length,
        "batch_size": 1,
        "sink_metric_definition": (
            "per attention layer: mean probability to key position 0 from second-half "
            "queries over all heads"
        ),
        "sink_layers": list(scope.sink_layers),
        "sink_scope_sha256": scope.sink_scope_sha256,
        "neuron_scoring_method": source_config["attribution_method"],
        "ranking_score": source_config["ranking_score"],
        "selected_target_condition_ids": [c.condition_id for c in conditions],
        "selected_neuron_ids_by_condition": {
            condition.condition_id: {
                str(layer): list(neurons)
                for layer, neurons in condition.neuron_set.by_layer.items()
            }
            for condition in conditions
        },
        "control_selection_seed": frozen_sets.document["registered_base_seed"],
        "control_draw_seeds": list(range(int(frozen_sets.document["control_draws"]))),
        "control_neuron_ids": "frozen in source discovery/neuron_sets.json; not rerun",
        "suppression_alphas": list(FULL_ALPHAS),
        "suppression_positions": "all",
        "neuron_hook_point": "model.layers[layer].mlp.down_proj input",
        "first_order_prediction_file": "first_order_predictions.json",
        "repo_dirty_at_run": bool(repo_status),
        "repo_status_at_run": repo_status.splitlines(),
    }
    _write_new_json(output_dir / "run_config.json", run_config)

    engine = sink_repro_module("nnsight_engine")
    attention_tolerance = max(
        float(engine.ATTN_ATOL + engine.ATTN_RTOL),
        2.0 * float(torch.finfo(torch.bfloat16).eps),
    )
    causal_tolerance = float(engine.ATTN_ATOL)
    from transformers import AutoModelForCausalLM

    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    print(f"Loading {MODEL_ID}@{MODEL_REVISION} bfloat16/eager on {gpu_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).eval().to(device)
    if model.training or getattr(model.config, "_attn_implementation", None) != "eager":
        raise AssertionError("A006 requires model.eval() with eager attention")
    if getattr(model.config, "_commit_hash", None) != MODEL_REVISION:
        raise AssertionError("Loaded model revision differs from amendment A006")
    adapter = Qwen2ModelAdapter(model, model_id=MODEL_ID)
    if adapter.num_layers != 28 or any(adapter.mlp_width(layer) != 8960 for layer in range(28)):
        raise AssertionError("Loaded Qwen architecture differs from completed Stage C")
    for condition in conditions:
        adapter.validate_neuron_set(condition.neuron_set)
    model.requires_grad_(False)

    source_sink = _load_source_sink_index(
        source_run / "test" / "suppression" / "per_example.csv"
    )
    rows: list[dict[str, Any]] = []
    reference = None
    reference_ids = None
    identity_exact = True
    valid_forward_all = True
    hook_leakage_pass = True
    max_source_baseline_diff = 0.0
    max_source_intervention_diff = 0.0
    started = time.perf_counter()
    forward_count = 0
    for index, item in enumerate(test_items):
        if args.progress_every > 0 and (
            index == 0 or (index + 1) % args.progress_every == 0
        ):
            print(f"  per-layer test [{index + 1}/{len(test_items)}] {item.item_id}", flush=True)
        input_ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
        baseline = forward_snapshot(
            model,
            input_ids,
            sink_layers=scope.sink_layers,
            attention_tolerance=attention_tolerance,
            causal_tolerance=causal_tolerance,
        )
        forward_count += 1
        baseline_layers = per_layer_sink_scores(baseline.attentions, scope.sink_layers)
        max_source_baseline_diff = max(
            max_source_baseline_diff,
            abs(baseline.sink - source_sink[(item.item_id, "baseline", 1.0)]),
        )
        if reference is None:
            reference = _cpu_reference(baseline)
            reference_ids = input_ids.detach().cpu().clone()

        for condition in conditions:
            hooks_before = {
                layer: tuple(adapter.mlp_projection(layer)._forward_pre_hooks.items())
                for layer in condition.neuron_set.by_layer
            }
            for alpha in FULL_ALPHAS:
                condition_started = time.perf_counter()
                with suppress_neurons(adapter, condition.neuron_set, alpha) as context:
                    expected_hooks = 0 if alpha == 1.0 else len(condition.neuron_set.by_layer)
                    if context.active_hook_count != expected_hooks:
                        raise AssertionError("Suppression hook count differs from selection")
                    intervention = forward_snapshot(
                        model,
                        input_ids,
                        sink_layers=scope.sink_layers,
                        attention_tolerance=attention_tolerance,
                        causal_tolerance=causal_tolerance,
                    )
                elapsed = time.perf_counter() - condition_started
                forward_count += 1
                for layer, expected in hooks_before.items():
                    actual = tuple(adapter.mlp_projection(layer)._forward_pre_hooks.items())
                    if actual != expected:
                        hook_leakage_pass = False
                        raise AssertionError(
                            f"Hook leakage after {condition.condition_id}, layer {layer}"
                        )
                if alpha == 1.0:
                    identity_exact &= bool(
                        torch.equal(baseline.logits, intervention.logits)
                        and all(
                            torch.equal(left, right)
                            for left, right in zip(
                                baseline.attentions, intervention.attentions
                            )
                        )
                    )
                valid_forward_all &= bool(baseline.valid and intervention.valid)
                layer_scores = per_layer_sink_scores(
                    intervention.attentions, scope.sink_layers
                )
                max_source_intervention_diff = max(
                    max_source_intervention_diff,
                    abs(
                        intervention.sink
                        - source_sink[(item.item_id, condition.condition_id, float(alpha))]
                    ),
                )
                for layer in scope.sink_layers:
                    sink_baseline = baseline_layers[layer]
                    sink_intervened = layer_scores[layer]
                    rows.append({
                        "experiment_id": EXPERIMENT_ID,
                        "source_experiment_id": SOURCE_EXPERIMENT_ID,
                        "model_id": MODEL_ID,
                        "split": "test",
                        "example_id": item.item_id,
                        "condition_id": condition.condition_id,
                        "condition_order": condition.condition_order,
                        "fraction_percent": condition.fraction_percent,
                        "k": condition.k,
                        "alpha": alpha,
                        "attention_layer": layer,
                        "sink_baseline": sink_baseline,
                        "sink_intervened": sink_intervened,
                        "delta_sink": sink_intervened - sink_baseline,
                        "relative_sink_reduction": (
                            sink_baseline - sink_intervened
                        ) / max(sink_baseline, 1e-12),
                        "baseline_valid": baseline.valid,
                        "intervention_valid": intervention.valid,
                        "valid_forward": baseline.valid and intervention.valid,
                        "forward_runtime_seconds": elapsed,
                    })
                del intervention
        del baseline, input_ids

    if reference is None or reference_ids is None:
        raise AssertionError("No diagnostic reference was captured")
    final_probe = forward_snapshot(
        model,
        reference_ids.to(device),
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    forward_count += 1
    state_logits_diff, state_attention_diff = _reference_difference(reference, final_probe)
    state_leakage_pass = state_logits_diff == 0.0 and state_attention_diff == 0.0
    del final_probe, model
    torch.cuda.synchronize(device)

    aggregates = aggregate_per_layer_rows(rows)
    runtime_seconds = time.perf_counter() - started
    _write_csv(output_dir / "per_example.csv", rows, PER_LAYER_ROW_FIELDS)
    _write_csv(
        output_dir / "aggregate.csv", aggregates, PER_LAYER_AGGREGATE_FIELDS
    )
    _write_new_json(output_dir / "aggregate.json", {
        "schema": "stage_c_posthoc_per_layer_aggregate_v1",
        "rows_sha256": canonical_sha256(aggregates),
        "rows": aggregates,
    })

    checks = {
        "source_run_complete": read_json(source_run / "summary.json").get(
            "stage_c_execution"
        ) in ("COMPLETE", "COMPLETE_RESUMED_AFTER_VALIDATION"),
        "source_gate_unchanged": True,
        "source_artifact_hashes_recorded": True,
        "source_baseline_max_abs_diff": max_source_baseline_diff,
        "source_intervention_max_abs_diff": max_source_intervention_diff,
        "source_aggregate_reproduction_pass": (
            max_source_baseline_diff == 0.0 and max_source_intervention_diff == 0.0
        ),
        "alpha_one_identity_exact_pass": identity_exact,
        "all_forward_validity_pass": valid_forward_all,
        "hook_leakage_pass": hook_leakage_pass,
        "state_leakage_pass": state_leakage_pass,
        "state_leakage_max_logits_abs_diff": state_logits_diff,
        "state_leakage_max_attention_abs_diff": state_attention_diff,
        "row_count_pass": len(rows) == 100 * 6 * 5 * len(scope.sink_layers),
        "aggregate_count_pass": len(aggregates) == 6 * 5 * len(scope.sink_layers),
    }
    if not all(bool(value) for key, value in checks.items() if key.endswith("_pass")):
        raise RuntimeError(f"A006 diagnostic checks failed: {checks}")
    provenance = recorder.finish(repo_commit=repo_commit, submodule_commits=submodule_commits)
    provenance.update({
        "repo_dirty_at_run": bool(repo_status),
        "repo_status_at_run": repo_status.splitlines(),
        "source_run": str(source_run),
        "amendment": "A006",
    })
    _write_new_json(output_dir / "provenance.json", provenance)
    summary = {
        "status": "COMPLETE_POST_HOC_DESCRIPTIVE",
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_run": str(source_run),
        "can_change_stage_c_gate": False,
        "can_change_stage_c2": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "manifest_sha256": MANIFEST_SHA256,
        "split": "test",
        "n_examples": len(test_items),
        "sink_layers": list(scope.sink_layers),
        "targeted_condition_count": len(conditions),
        "alpha_grid": list(FULL_ALPHAS),
        "forward_count": forward_count,
        "runtime_seconds_measurement_loop": runtime_seconds,
        "runtime_seconds_total": provenance["runtime_seconds"],
        "peak_memory_allocated_bytes": provenance["peak_memory_allocated_bytes"],
        "peak_memory_reserved_bytes": provenance["peak_memory_reserved_bytes"],
        "per_example_rows": len(rows),
        "aggregate_rows": len(aggregates),
        "per_example_rows_sha256": canonical_sha256(rows),
        "aggregate_rows_sha256": canonical_sha256(aggregates),
        "first_order_rows_sha256": canonical_sha256(predictions),
        "checks": checks,
    }
    _write_new_json(output_dir / "summary.json", summary)
    print("STAGE_C_POSTHOC_PER_LAYER=COMPLETE", flush=True)
    print(f"identity_exact={identity_exact}", flush=True)
    print(f"validity_pass={valid_forward_all}", flush=True)
    print(f"source_reproduction={checks['source_aggregate_reproduction_pass']}", flush=True)
    print(f"wall_seconds={provenance['runtime_seconds']:.3f}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
