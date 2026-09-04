#!/usr/bin/env python
"""Task 7: registered GPT-2-small 24/24/24 suppression smoke experiment.

Consumes the immutable Task-4 corpus/sink scope and Task-6 target/control sets.  It
does not rank, select, redraw, or import downstream benchmark code.  Every split is
saved separately, but the automatic plausibility gate reads the held-out test split
only.
"""

from __future__ import annotations

import argparse
import csv
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

from neuron_sink.corpus import (  # noqa: E402
    SMOKE_SPLIT_SIZE,
    NeutralCorpus,
    verify_disjoint,
)
from neuron_sink.evaluation import (  # noqa: E402
    AGGREGATE_FIELDS,
    EVALUATION_PROTOCOL,
    EXPERIMENT_ID,
    PHENOMENON_ROW_FIELDS,
    SMOKE_ALPHAS,
    SMOKE_SPLITS,
    aggregate_phenomenon_rows,
    evaluate_smoke_gate,
    forward_snapshot,
    paired_metrics,
    registered_smoke_conditions,
    validate_phenomenon_row,
)
from neuron_sink.model_adapters import GPT2ModelAdapter  # noqa: E402
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    git,
    prepare_output_dir,
    require_pinned_submodules,
    require_registered_gpu,
    run_stamp,
    write_json,
)
from neuron_sink.selection import (  # noqa: E402
    RANKING_SCORE,
    SELECTION_METHOD,
    load_frozen_neuron_sets,
)
from neuron_sink.sink_metrics import (  # noqa: E402
    REGISTERED_QUERY_RULE,
    REGISTERED_TARGET_POSITION,
    load_frozen_sink_scope,
)
from neuron_sink.suppression import suppress_neurons  # noqa: E402
from neuron_sink.upstream_bridge import sink_repro_module  # noqa: E402


FROZEN_DIR = ROOT / "configs" / "frozen"
FROZEN_MANIFEST = FROZEN_DIR / "neutral_corpus_manifest.json"
FROZEN_SCOPE = FROZEN_DIR / "sink_scope.json"
FROZEN_NEURON_SETS = FROZEN_DIR / "neuron_sets.json"

DEFAULT_MODEL_ID = "gpt2"
CANONICAL_MODEL_ID = "openai-community/gpt2"
DEFAULT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_CORPUS_SHA256 = "c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7"
EXPECTED_SCOPE_SHA256 = "b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235"
EXPECTED_ATTRIBUTION_SHA256 = "9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692"
EXPECTED_NEURON_SETS_SHA256 = "4fa22a2c68c8c3e56ed13b4f1c481b7b43d963b0190a619cacdc7c03c2672165"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered Task-7 GPT-2-small suppression smoke grid."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--manifest", type=Path, default=FROZEN_MANIFEST)
    parser.add_argument("--scope", type=Path, default=FROZEN_SCOPE)
    parser.add_argument("--neuron-sets", type=Path, default=FROZEN_NEURON_SETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(Path(os.environ["NEURON_SINK_HF_CACHE"])
                 if os.environ.get("NEURON_SINK_HF_CACHE") else None),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Dry-run prefix per split for runtime/VRAM checks. A limited run never "
            "evaluates the scientific gate."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    if args.max_examples is not None and not 1 <= args.max_examples <= SMOKE_SPLIT_SIZE:
        parser.error(f"--max-examples must be in [1, {SMOKE_SPLIT_SIZE}]")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _artifact_checks(corpus, scope, frozen_sets) -> dict[str, bool]:
    document = frozen_sets.document
    checks = {
        "corpus_hash_pass": corpus.manifest_sha256 == EXPECTED_CORPUS_SHA256,
        "scope_hash_pass": scope.sink_scope_sha256 == EXPECTED_SCOPE_SHA256,
        "attribution_hash_pass": (
            str(document.get("attribution_sha256")) == EXPECTED_ATTRIBUTION_SHA256
        ),
        "neuron_sets_hash_pass": (
            str(document.get("neuron_sets_sha256")) == EXPECTED_NEURON_SETS_SHA256
        ),
        "scope_corpus_link_pass": (
            scope.corpus_manifest_sha256 == corpus.manifest_sha256
        ),
        "sets_corpus_link_pass": (
            str(document.get("corpus_manifest_sha256")) == corpus.manifest_sha256
        ),
        "sets_scope_link_pass": (
            str(document.get("sink_scope_sha256")) == scope.sink_scope_sha256
        ),
        "scope_model_pass": (
            scope.model_id == CANONICAL_MODEL_ID
            and scope.model_revision == DEFAULT_MODEL_REVISION
        ),
        "sets_model_pass": (
            str(document.get("model_id")) == CANONICAL_MODEL_ID
            and str(document.get("model_revision")) == DEFAULT_MODEL_REVISION
        ),
        "sequence_length_pass": (
            corpus.cut_length == 40 and scope.seq_len == corpus.cut_length
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(
            "Frozen Task-7 input verification failed: " + ", ".join(failed)
        )
    return checks


def _row(
    *,
    stage: str,
    example_id: str,
    model_id: str,
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
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "stage": stage,
        "example_id": example_id,
        "condition_id": condition_id,
        "condition_order": condition_order,
        "alpha_order": alpha_order,
        "control_type": control_type,
        "control_seed": control_seed,
        "fraction": (
            None if fraction_percent is None else fraction_percent / 100.0
        ),
        "fraction_percent": fraction_percent,
        "k": k,
        "alpha": alpha,
        "prompt_tokens": prompt_tokens,
        "forward_runtime_seconds": forward_runtime_seconds,
        **dict(metrics),
    }
    validate_phenomenon_row(row)
    return row


def _cpu_reference(snapshot) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    return (
        snapshot.logits.detach().cpu().clone(),
        tuple(attention.detach().cpu().clone() for attention in snapshot.attentions),
    )


def _reference_difference(reference, snapshot) -> tuple[float, float]:
    reference_logits, reference_attentions = reference
    logits_diff = float(
        (reference_logits - snapshot.logits.detach().cpu()).abs().max().item()
    )
    attention_diff = max(
        float((left - right.detach().cpu()).abs().max().item())
        for left, right in zip(reference_attentions, snapshot.attentions)
    )
    return logits_diff, attention_diff


def main() -> int:
    args = _parse_args()
    registered_run = args.max_examples is None
    output_dir = prepare_output_dir(
        args.output_dir or ROOT / "results" / "task7_gpt2_smoke" / run_stamp()
    )
    submodule_commits = require_pinned_submodules()
    repo_commit = git("rev-parse", "HEAD")
    repo_status = git("status", "--porcelain", "--untracked-files=all")

    device, gpu_name, total_vram = require_registered_gpu("dev")
    recorder = ProvenanceRecorder(device=device, gpu_name=gpu_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    corpus = NeutralCorpus.load(args.manifest)
    verify_disjoint(corpus.smoke_splits)
    scope = load_frozen_sink_scope(
        args.scope, expected_corpus_manifest_sha256=corpus.manifest_sha256
    )
    frozen_sets = load_frozen_neuron_sets(args.neuron_sets)
    artifact_checks = _artifact_checks(corpus, scope, frozen_sets)
    conditions = registered_smoke_conditions(frozen_sets)

    items_by_split = {
        split: list(corpus.items_for(split, smoke=True)) for split in SMOKE_SPLITS
    }
    for split, items in items_by_split.items():
        expected_ids = tuple(corpus.smoke_splits[split])
        actual_ids = tuple(item.item_id for item in items)
        if len(items) != SMOKE_SPLIT_SIZE or actual_ids != expected_ids:
            raise SystemExit(
                f"{split} smoke split is not the frozen {SMOKE_SPLIT_SIZE}-item prefix"
            )
        if args.max_examples is not None:
            items_by_split[split] = items[:args.max_examples]

    engine = sink_repro_module("nnsight_engine")
    attention_tolerance = float(engine.ATTN_ATOL + engine.ATTN_RTOL)
    causal_tolerance = float(engine.ATTN_ATOL)

    from transformers import GPT2LMHeadModel

    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    print(
        f"Loading {args.model_id}@{args.revision} in float32/eager on {gpu_name}",
        flush=True,
    )
    model = GPT2LMHeadModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=cache_dir,
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval().to(device)
    if model.training:
        raise AssertionError("model.eval() did not take effect")
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "eager":
        raise AssertionError(
            f"Expected eager attention, got {attention_implementation!r}"
        )
    actual_revision = getattr(model.config, "_commit_hash", None)
    if actual_revision is not None and actual_revision != args.revision:
        raise AssertionError(
            f"Loaded model revision {actual_revision} != requested {args.revision}"
        )

    adapter = GPT2ModelAdapter(model, model_id=CANONICAL_MODEL_ID)
    if adapter.num_layers != scope.num_layers:
        raise AssertionError(
            f"Model has {adapter.num_layers} layers, scope has {scope.num_layers}"
        )
    for condition in conditions:
        adapter.validate_neuron_set(condition.neuron_set)

    n_per_split = {split: len(items) for split, items in items_by_split.items()}
    expected_forwards = sum(n_per_split.values()) * (
        1 + len(conditions) * len(SMOKE_ALPHAS)
    )
    mode = "registered_24_per_split" if registered_run else "dry_run_prefix"
    print(
        f"Task 7 mode={mode}; splits={n_per_split}; conditions={len(conditions)}; "
        f"alphas={SMOKE_ALPHAS}; expected_forwards={expected_forwards}",
        flush=True,
    )

    run_config = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "discovery|validation|test",
        "run_mode": mode,
        "registered_run": registered_run,
        "model_id": CANONICAL_MODEL_ID,
        "model_requested_id": args.model_id,
        "model_revision": args.revision,
        "tokenizer_id": corpus.tokenizer_name,
        "tokenizer_revision": corpus.tokenizer_revision,
        "dtype": "float32",
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total_vram,
        "seed": args.seed,
        "dataset_id": corpus.source.get("dataset_id"),
        "dataset_config": None,
        "dataset_split": list(SMOKE_SPLITS),
        "manifest_sha256": corpus.manifest_sha256,
        "split_example_ids": {
            split: [item.item_id for item in items]
            for split, items in items_by_split.items()
        },
        "examples_per_split": n_per_split,
        "seq_len": corpus.cut_length,
        "batch_size": 1,
        "sink_metric_definition": (
            "mean attention probability to key position 0 from second-half query "
            "positions over frozen sink-heavy layers and all heads"
        ),
        "sink_target_position": REGISTERED_TARGET_POSITION,
        "sink_query_rule": REGISTERED_QUERY_RULE,
        "sink_layers": list(scope.sink_layers),
        "sink_heads": None,
        "sink_scope_sha256": scope.sink_scope_sha256,
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "suppression_positions": "all",
        "selection_method": SELECTION_METHOD,
        "ranking_score": RANKING_SCORE,
        "attribution_sha256": frozen_sets.document["attribution_sha256"],
        "neuron_sets_sha256": frozen_sets.document["neuron_sets_sha256"],
        "neuron_sets_file": "neuron_sets.json",
        "condition_ids": [condition.condition_id for condition in conditions],
        "fractions_percent": list(frozen_sets.document["fractions_percent"]),
        "k_by_fraction_percent": {
            f"{condition.fraction_percent:.2f}": condition.k
            for condition in conditions
            if condition.control_type == "targeted"
        },
        "alphas": list(SMOKE_ALPHAS),
        "neuron_fraction": None,
        "k": None,
        "alpha": None,
        "control_type": "baseline|targeted|layer_random_grid",
        "control_draws": frozen_sets.document["control_draws"],
        "control_seed": None,
        "control_seed_range": list(range(frozen_sets.document["control_draws"])),
        "evaluation_protocol_version": EVALUATION_PROTOCOL,
        "ce_positions": "next_token_positions_0_through_seq_len_minus_2",
        "kl_direction": "p_baseline||p_intervened",
        "top1_positions": "next_token_positions_0_through_seq_len_minus_2",
        "attention_implementation": attention_implementation,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "expected_forward_count": expected_forwards,
        "artifact_checks": artifact_checks,
        "repo_dirty_at_run": bool(repo_status),
        "repo_status_at_run": repo_status.splitlines(),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "neuron_sets.json", dict(frozen_sets.document))

    all_rows: list[dict[str, Any]] = []
    split_runtime: dict[str, float] = {}
    reference = None
    reference_ids: torch.Tensor | None = None
    forward_count = 0

    for split in SMOKE_SPLITS:
        split_rows: list[dict[str, Any]] = []
        items = items_by_split[split]
        split_start = time.perf_counter()
        for example_index, item in enumerate(items):
            if args.progress_every > 0 and (
                example_index == 0
                or (example_index + 1) % args.progress_every == 0
            ):
                print(
                    f"  {split} [{example_index + 1}/{len(items)}] {item.item_id}",
                    flush=True,
                )
            input_ids = torch.tensor(
                [list(item.input_ids)], dtype=torch.long, device=device
            )

            baseline_start = time.perf_counter()
            baseline = forward_snapshot(
                model,
                input_ids,
                sink_layers=scope.sink_layers,
                attention_tolerance=attention_tolerance,
                causal_tolerance=causal_tolerance,
            )
            baseline_elapsed = time.perf_counter() - baseline_start
            forward_count += 1
            if reference is None:
                reference = _cpu_reference(baseline)
                reference_ids = input_ids.detach().cpu().clone()

            baseline_metrics = paired_metrics(baseline, baseline)
            split_rows.append(_row(
                stage=split,
                example_id=item.item_id,
                model_id=CANONICAL_MODEL_ID,
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
            ))

            for condition in conditions:
                hooks_before = {
                    layer: tuple(adapter.mlp_projection(layer)._forward_pre_hooks.items())
                    for layer in condition.neuron_set.by_layer
                }
                for alpha_order, alpha in enumerate(SMOKE_ALPHAS, start=1):
                    intervention_start = time.perf_counter()
                    with suppress_neurons(
                        adapter, condition.neuron_set, alpha
                    ) as context:
                        expected_hooks = (
                            0 if alpha == 1.0 else len(condition.neuron_set.by_layer)
                        )
                        if context.active_hook_count != expected_hooks:
                            raise AssertionError(
                                f"{condition.condition_id} alpha={alpha}: active hooks "
                                f"{context.active_hook_count} != {expected_hooks}"
                            )
                        intervention = forward_snapshot(
                            model,
                            input_ids,
                            sink_layers=scope.sink_layers,
                            attention_tolerance=attention_tolerance,
                            causal_tolerance=causal_tolerance,
                        )
                    for layer, expected in hooks_before.items():
                        actual = tuple(
                            adapter.mlp_projection(layer)._forward_pre_hooks.items()
                        )
                        if actual != expected:
                            raise AssertionError(
                                f"Hook leakage after {condition.condition_id}, layer {layer}"
                            )
                    metrics = paired_metrics(baseline, intervention)
                    intervention_elapsed = time.perf_counter() - intervention_start
                    forward_count += 1
                    split_rows.append(_row(
                        stage=split,
                        example_id=item.item_id,
                        model_id=CANONICAL_MODEL_ID,
                        condition_id=condition.condition_id,
                        condition_order=condition.condition_order,
                        alpha_order=alpha_order,
                        control_type=condition.control_type,
                        control_seed=condition.control_seed,
                        fraction_percent=condition.fraction_percent,
                        k=condition.k,
                        alpha=alpha,
                        prompt_tokens=item.n_tokens,
                        forward_runtime_seconds=intervention_elapsed,
                        metrics=metrics,
                    ))
                    del intervention
            del baseline

        torch.cuda.synchronize(device)
        split_runtime[split] = time.perf_counter() - split_start
        expected_rows = len(items) * (1 + len(conditions) * len(SMOKE_ALPHAS))
        if len(split_rows) != expected_rows:
            raise AssertionError(
                f"{split} produced {len(split_rows)} rows, expected {expected_rows}"
            )
        _write_csv(
            output_dir / f"per_example_{split}.csv",
            split_rows,
            PHENOMENON_ROW_FIELDS,
        )
        all_rows.extend(split_rows)
        print(
            f"  {split} complete: rows={len(split_rows)} "
            f"seconds={split_runtime[split]:.3f}",
            flush=True,
        )

    if reference is None or reference_ids is None:
        raise AssertionError("No baseline reference was captured")
    final_probe = forward_snapshot(
        model,
        reference_ids.to(device),
        sink_layers=scope.sink_layers,
        attention_tolerance=attention_tolerance,
        causal_tolerance=causal_tolerance,
    )
    forward_count += 1
    state_logits_diff, state_attention_diff = _reference_difference(
        reference, final_probe
    )
    state_leakage_pass = bool(
        state_logits_diff == 0.0 and state_attention_diff == 0.0
    )
    del final_probe

    _write_csv(output_dir / "per_example.csv", all_rows, PHENOMENON_ROW_FIELDS)
    aggregates = aggregate_phenomenon_rows(all_rows)
    _write_csv(output_dir / "aggregate.csv", aggregates, AGGREGATE_FIELDS)
    write_json(output_dir / "aggregate.json", {"rows": aggregates})

    identity_rows = [
        row for row in all_rows
        if row["control_type"] == "baseline" or float(row["alpha"]) == 1.0
    ]
    all_split_identity_pass = bool(all(
        row["logits_exact_match"] and row["attentions_exact_match"]
        for row in identity_rows
    ))
    all_split_validity_pass = bool(all(
        row["valid_forward"] for row in all_rows
    ))
    gate = evaluate_smoke_gate(
        aggregates,
        all_split_identity_pass=all_split_identity_pass,
        all_split_validity_pass=all_split_validity_pass,
        state_leakage_pass=state_leakage_pass,
        registered_run=registered_run,
    )
    write_json(output_dir / "smoke_gate.json", gate)

    provenance = recorder.finish(
        repo_commit=repo_commit, submodule_commits=submodule_commits
    )
    provenance["repo_dirty_at_run"] = bool(repo_status)
    provenance["repo_status_at_run"] = repo_status.splitlines()
    write_json(output_dir / "provenance.json", provenance)

    summary = {
        "task7_execution": "COMPLETE",
        "run_mode": mode,
        "scientific_smoke_gate": gate["status"],
        "smoke_gate_pass": gate.get("smoke_gate_pass"),
        "n_splits": len(SMOKE_SPLITS),
        "examples_per_split": n_per_split,
        "condition_count": len(conditions),
        "alpha_grid": list(SMOKE_ALPHAS),
        "per_example_rows": len(all_rows),
        "forward_count": forward_count,
        "expected_grid_forward_count": expected_forwards,
        "extra_state_leakage_probe_forwards": 1,
        "runtime_seconds_by_split": split_runtime,
        "runtime_seconds_total": provenance["runtime_seconds"],
        "peak_memory_allocated_bytes": provenance[
            "peak_memory_allocated_bytes"
        ],
        "peak_memory_reserved_bytes": provenance[
            "peak_memory_reserved_bytes"
        ],
        "checks": {
            **artifact_checks,
            "frozen_split_order_pass": True,
            "registered_condition_order_pass": True,
            "registered_alpha_order_pass": True,
            "all_split_identity_exact_pass": all_split_identity_pass,
            "all_forward_validity_pass": all_split_validity_pass,
            "state_leakage_pass": state_leakage_pass,
            "state_leakage_max_logits_abs_diff": state_logits_diff,
            "state_leakage_max_attention_abs_diff": state_attention_diff,
            "output_row_schema_pass": True,
        },
        "is_causal_evidence": bool(
            registered_run and gate.get("smoke_gate_pass", False)
        ),
        "interpretation": (
            "A PASS is a permissive RTX-2060 plausibility result, not the formal "
            "100-example/20-control causal confirmation. A null is valid and must not "
            "trigger tuning of the registered fractions, alphas, scope, or controls."
        ),
    }
    write_json(output_dir / "summary.json", summary)

    print("TASK7_EXECUTION=COMPLETE", flush=True)
    print(f"TASK7_SMOKE_GATE={gate['status']}", flush=True)
    if gate.get("passing_superiority_conditions"):
        for passing in gate["passing_superiority_conditions"]:
            print(
                "  passing condition: "
                f"fraction={passing['fraction_percent']}% k={passing['k']} "
                f"alpha={passing['alpha']} "
                f"target_rsr={passing['target_relative_sink_reduction']:.9f} "
                f"max_random_rsr={passing['max_random_relative_sink_reduction']:.9f}",
                flush=True,
            )
    print(f"identity_exact={all_split_identity_pass}", flush=True)
    print(f"validity_pass={all_split_validity_pass}", flush=True)
    print(f"state_leakage_pass={state_leakage_pass}", flush=True)
    print(f"rows={len(all_rows)} forwards={forward_count}", flush=True)
    print(f"wall_seconds={provenance['runtime_seconds']:.3f}", flush=True)
    print(
        f"peak_memory_allocated_bytes={provenance['peak_memory_allocated_bytes']}",
        flush=True,
    )
    print(f"output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
