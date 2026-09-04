#!/usr/bin/env python
"""Task 5: causal-order-aware future-sink activation-times-gradient neuron ranking.

For each eligible MLP layer ``l`` frozen by Task 4, this scores every neuron by

``I(l, n) = mean over examples and token positions of | a(l,n) * dS_future(l)/da(l,n) |``

where ``a`` is the tensor entering ``transformer.h[l].mlp.c_proj`` and ``S_future(l)`` is the
sink metric restricted to the frozen sink-heavy attention layers ``j > l``. Both the corpus
and the sink scope are **consumed** from ``configs/frozen/``; neither is recomputed, and the
scope's own corpus hash is checked against the manifest actually loaded.

The run has three phases, in this order:

1. **Scorer verification.** ``differentiable_sink_score`` is checked against the frozen
   ``sink_scalar_from_map`` and against the pinned upstream ``compute_bos_attention_metric``
   on this run's own attention tensors, before any gradient is trusted.
2. **Causal-order probe.** For every eligible layer, a sink objective built from attention at
   layer ``j <= l`` is shown not to be a function of that layer's ``c_proj`` input at all,
   while the frozen ``j > l`` targets are. The constraint is measured, not assumed.
3. **Attribution**, one eligible MLP layer per backward pass at batch size 1.

Discovery split only: ``require_discovery_split`` rejects validation and test inside the
ranking API, so a held-out example cannot reach a gradient.

No selection, no controls, no suppression. Attribution is a ranking heuristic, not causal
evidence; that is Task 6 and Task 7 work.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

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
    SCHEMA_VERSION,
    TOKEN_POSITION_RULE,
    attribution_rows,
    attribution_sha256,
    objective_depends_on_layer,
    rank_neurons,
)
from neuron_sink.corpus import NeutralCorpus, require_discovery_split  # noqa: E402
from neuron_sink.model_adapters import GPT2ModelAdapter  # noqa: E402
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    git,
    prepare_output_dir,
    read_json,
    require_pinned_submodules,
    require_registered_gpu,
    run_stamp,
    write_json,
)
from neuron_sink.sink_metrics import (  # noqa: E402
    REGISTERED_TARGET_POSITION,
    differentiable_sink_score,
    load_frozen_sink_scope,
    per_layer_head_position0_attention,
    sink_scalar_from_map,
)
from neuron_sink.upstream_bridge import sink_repro_module  # noqa: E402


FROZEN_DIR = ROOT / "configs" / "frozen"
FROZEN_MANIFEST = FROZEN_DIR / "neutral_corpus_manifest.json"
FROZEN_SCOPE = FROZEN_DIR / "sink_scope.json"
FROZEN_ATTRIBUTION = FROZEN_DIR / "neuron_attribution.csv"
FROZEN_ATTRIBUTION_META = FROZEN_DIR / "neuron_attribution_metadata.json"
DEFAULT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Future-sink activation-times-gradient neuron ranking (discovery only)."
    )
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--manifest", type=Path, default=FROZEN_MANIFEST)
    parser.add_argument("--scope", type=Path, default=FROZEN_SCOPE)
    parser.add_argument("--split", default="discovery",
                        help="Ranking-facing stage; only 'discovery' is permitted.")
    parser.add_argument("--smoke", action="store_true", default=True,
                        help="Use the 24-example smoke prefix (RTX 2060 phase default).")
    parser.add_argument("--full", dest="smoke", action="store_false",
                        help="Use the full 100-example discovery split.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Memory/runtime preflight; not a registered configuration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path,
                        default=(Path(os.environ["NEURON_SINK_HF_CACHE"])
                                 if os.environ.get("NEURON_SINK_HF_CACHE") else None))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--progress-every", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # Rejected here as well as inside the ranking API, so a held-out split cannot even
    # allocate an output directory.
    require_discovery_split(args.split)

    output_dir = prepare_output_dir(
        args.output_dir or ROOT / "results" / "task5_attribution" / run_stamp()
    )
    submodule_commits = require_pinned_submodules()
    repo_commit = git("rev-parse", "HEAD")

    device, gpu_name, total_vram = require_registered_gpu("dev")
    recorder = ProvenanceRecorder(device=device, gpu_name=gpu_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    legacy = sink_repro_module("intervention_analysis_legacy")
    engine = sink_repro_module("nnsight_engine")

    if not args.manifest.is_file():
        raise SystemExit(
            f"Frozen neutral corpus manifest not found: {args.manifest}. "
            "Run scripts/prepare_neutral_corpus.py first."
        )
    corpus = NeutralCorpus.load(args.manifest)
    # Fails loudly if the scope was frozen against a different corpus, if its own hash no
    # longer reproduces, or if any future target is not strictly later than its MLP layer.
    scope = load_frozen_sink_scope(
        args.scope, expected_corpus_manifest_sha256=corpus.manifest_sha256
    )

    items = list(corpus.items_for(args.split, smoke=args.smoke))
    if args.max_examples is not None:
        items = items[:args.max_examples]
    if not items:
        raise SystemExit(f"Split {args.split!r} is empty in {args.manifest}")

    from transformers import GPT2LMHeadModel

    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    print(f"Loading {args.model_id}@{args.revision} in float32 on {gpu_name}")
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
        raise AssertionError(f"Expected eager attention, got {attention_implementation!r}")

    # No parameter gradients are ever wanted: the only gradient taken is d S_future / d a.
    model.requires_grad_(False)
    still_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if still_trainable:
        raise AssertionError(f"{len(still_trainable)} parameters still require grad")

    adapter = GPT2ModelAdapter(model, model_id="openai-community/gpt2")
    num_layers = adapter.num_layers
    if num_layers != scope.num_layers:
        raise SystemExit(
            f"Model has {num_layers} layers but the frozen scope has {scope.num_layers}"
        )
    eligible = list(scope.eligible_mlp_layers)
    widths = {layer: adapter.mlp_width(layer) for layer in eligible}
    band_start, band_end = legacy.compute_band(num_layers, "scaled")
    parity_band = list(range(band_start, band_end))

    print(
        f"Ranking {len(eligible) * sum(widths.values()) // max(len(eligible), 1)} "
        f"neurons x {len(eligible)} eligible MLP layers on {len(items)} {args.split} "
        f"examples ({'smoke' if args.smoke else 'full'} split), seq_len={corpus.cut_length}"
    )
    print(f"  sink-heavy layers {list(scope.sink_layers)}; eligible MLP layers {eligible}")

    # --- phase 1: verify the differentiable scorer before trusting any gradient ---------
    print("Phase 1: differentiable scorer vs the frozen metric and upstream")
    max_all_layer_diff = 0.0
    max_parity_band_diff = 0.0
    max_target_diff = 0.0
    frozen_target_totals = {layer: 0.0 for layer in eligible}
    parity_band_values: list[float] = []

    for item in items:
        ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_attentions=True,
                use_cache=False,
            )
        attentions = [attention[0].detach().float().cpu() for attention in output.attentions]
        example_map = per_layer_head_position0_attention(
            attentions, target_pos=REGISTERED_TARGET_POSITION
        )

        max_all_layer_diff = max(max_all_layer_diff, abs(
            float(differentiable_sink_score(attentions, None))
            - sink_scalar_from_map(example_map, None)
        ))
        upstream_band = legacy.compute_bos_attention_metric(
            attentions, num_layers, "mid", target_pos=REGISTERED_TARGET_POSITION,
            layer_start=band_start, layer_end=band_end,
        )
        differentiable_band = float(differentiable_sink_score(attentions, parity_band))
        max_parity_band_diff = max(
            max_parity_band_diff, abs(differentiable_band - float(upstream_band))
        )
        parity_band_values.append(differentiable_band)

        for layer in eligible:
            targets = list(scope.targets_for(layer))
            frozen_value = sink_scalar_from_map(example_map, targets)
            frozen_target_totals[layer] += frozen_value
            max_target_diff = max(max_target_diff, abs(
                float(differentiable_sink_score(attentions, targets)) - frozen_value
            ))
        del output, attentions, ids

    frozen_target_mean = {
        layer: total / len(items) for layer, total in frozen_target_totals.items()
    }
    mean_parity_band = float(np.mean(parity_band_values))
    metric_limit = engine.METRIC_ATOL + engine.METRIC_RTOL * max(
        abs(mean_parity_band), 1e-12
    )
    scorer_parity_pass = max(
        max_all_layer_diff, max_parity_band_diff, max_target_diff
    ) <= metric_limit
    print(
        f"  max abs diff: all-layer {max_all_layer_diff:.6g}, parity band "
        f"{max_parity_band_diff:.6g}, future targets {max_target_diff:.6g} "
        f"(tolerance {metric_limit:.6g})"
    )
    if not scorer_parity_pass:
        raise SystemExit(
            "The differentiable sink scorer does not reproduce the frozen metric within the "
            "upstream tolerance. Stopping before any gradient is taken; do not loosen the "
            "tolerance."
        )

    # --- phase 2: measure the causal-ordering constraint --------------------------------
    print("Phase 2: causal-order probe")
    probe_ids = torch.tensor([list(items[0].input_ids)], dtype=torch.long, device=device)
    causal_probe: list[dict] = []
    for layer in eligible:
        targets = list(scope.targets_for(layer))
        earlier = [j for j in scope.sink_layers if j <= layer]
        record = {
            "mlp_layer": layer,
            "future_targets": targets,
            "future_targets_depend": objective_depends_on_layer(
                model, adapter, probe_ids, layer, targets
            ),
            # The registered prohibition: never attribute layer l to its own pre-MLP
            # attention (AGENTS.md, "Causal ordering constraint").
            "same_layer_attention": layer,
            "same_layer_depends": objective_depends_on_layer(
                model, adapter, probe_ids, layer, [layer]
            ),
            "earlier_sink_layers": earlier,
            "earlier_sink_depends": (
                objective_depends_on_layer(model, adapter, probe_ids, layer, earlier)
                if earlier else None
            ),
        }
        causal_probe.append(record)
    del probe_ids

    causal_order_pass = all(
        record["future_targets_depend"] is True
        and record["same_layer_depends"] is False
        and record["earlier_sink_depends"] in (False, None)
        and all(target > record["mlp_layer"] for target in record["future_targets"])
        for record in causal_probe
    )
    print(f"  future targets reachable / same-and-earlier unreachable: {causal_order_pass}")

    # --- phase 3: attribution ------------------------------------------------------------
    print("Phase 3: attribution, one eligible MLP layer per backward pass")

    def progress(layer: int, done: int, total: int) -> None:
        if args.progress_every > 0 and (done == 1 or done % args.progress_every == 0):
            print(f"  layer {layer} [{done}/{total}]")

    result = rank_neurons(
        model,
        adapter,
        corpus,
        scope.future_sink_layers,
        split=args.split,
        smoke=args.smoke,
        max_examples=args.max_examples,
        target_pos=REGISTERED_TARGET_POSITION,
        device=device,
        progress=progress,
    )
    torch.cuda.synchronize(device)

    rows = attribution_rows(result)
    rows_sha256 = attribution_sha256(rows)

    # --- checks --------------------------------------------------------------------------
    expected_rows = sum(widths[layer] for layer in eligible)
    rows_pass = (
        len(rows) == expected_rows
        and {row["layer"] for row in rows} == set(eligible)
        and all(
            sorted(row["neuron"] for row in rows if row["layer"] == layer)
            == list(range(widths[layer]))
            for layer in eligible
        )
        and all(
            np.isfinite(row["mean_abs_attr"])
            and np.isfinite(row["mean_signed_attr"])
            and np.isfinite(row["mean_abs_activation"])
            and row["mean_abs_attr"] >= 0.0
            and row["mean_abs_attr"] + 1e-12 >= abs(row["mean_signed_attr"])
            for row in rows
        )
        and sorted(row["rank_abs"] for row in rows) == list(range(1, len(rows) + 1))
    )

    gradient_pass = all(
        layer_result.nonfinite_values == 0
        and layer_result.zero_gradient_examples == 0
        and layer_result.max_abs_gradient > 0.0
        and float(layer_result.mean_abs_attr.max()) > 0.0
        for layer_result in result.layers
    )

    # The scalar that was actually differentiated must be the frozen metric.
    objective_diff = max(
        abs(layer_result.mean_sink_future - frozen_target_mean[layer_result.layer])
        for layer_result in result.layers
    )
    objective_pass = objective_diff <= metric_limit

    expected_ids = tuple(item.item_id for item in items)
    discovery_pass = (
        args.split == "discovery"
        and result.split == "discovery"
        and result.example_ids == expected_ids
        and all(item.split == "discovery" for item in items)
        and result.corpus_manifest_sha256 == corpus.manifest_sha256
    )
    provenance_pass = scope.corpus_manifest_sha256 == corpus.manifest_sha256

    task_pass = all((
        scorer_parity_pass, causal_order_pass, rows_pass, gradient_pass,
        objective_pass, discovery_pass, provenance_pass,
    ))

    # --- outputs -------------------------------------------------------------------------
    per_layer = [layer_result.diagnostics() for layer_result in result.layers]
    peak_layer_allocated = max(
        (entry["peak_memory_allocated_bytes"] for entry in per_layer), default=0
    )

    metadata = {
        "schema": SCHEMA_VERSION,
        "experiment_id": "task5_attribution",
        "stage": "discovery",
        "model_id": "openai-community/gpt2",
        "model_revision": args.revision,
        "dtype": "float32",
        "corpus_id": corpus.corpus_id,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "sink_scope_sha256": scope.sink_scope_sha256,
        "split": args.split,
        "split_mode": "smoke_24" if args.smoke else "full_100",
        "n_examples": len(items),
        "n_tokens": len(items) * corpus.cut_length,
        "seq_len": corpus.cut_length,
        "example_ids": list(result.example_ids),
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "neuron_hook_point": "transformer.h[layer].mlp.c_proj input",
        "attribution_method": ATTRIBUTION_METHOD,
        "attribution_objective": ATTRIBUTION_OBJECTIVE,
        "attribution_aggregation": ATTRIBUTION_AGGREGATION,
        "token_position_rule": TOKEN_POSITION_RULE,
        "ranking_score": RANKING_SCORE,
        "sink_target_position": REGISTERED_TARGET_POSITION,
        "sink_query_rule": "second_half",
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
        "attribution_sha256": rows_sha256,
        "per_layer": per_layer,
        "causal_order_probe": causal_probe,
        "is_causal_evidence": False,
        "note": (
            "Attribution is a ranking heuristic, not causal evidence. No neuron here has "
            "been shown to cause the sink; that requires held-out suppression against "
            "layer-count-matched random controls."
        ),
    }

    run_config = {
        "experiment_id": "task5_attribution",
        "stage": "discovery",
        "model_id": "openai-community/gpt2",
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
        "dataset_split": args.split,
        "manifest_sha256": corpus.manifest_sha256,
        "sink_scope_sha256": scope.sink_scope_sha256,
        "seq_len": corpus.cut_length,
        "n_examples": len(items),
        "batch_size": 1,
        "sink_target_position": REGISTERED_TARGET_POSITION,
        "sink_query_rule": "second_half",
        "sink_layers": list(scope.sink_layers),
        "sink_heads": None,
        "parity_band": parity_band,
        "eligible_mlp_layers": eligible,
        "future_sink_layers": {
            str(layer): list(scope.targets_for(layer)) for layer in eligible
        },
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "selection_method": ATTRIBUTION_METHOD,
        "attribution_objective": ATTRIBUTION_OBJECTIVE,
        "token_position_rule": TOKEN_POSITION_RULE,
        "neuron_fraction": None,
        "k": None,
        "alpha": None,
        "control_type": "baseline",
        "control_seed": None,
        "attention_implementation": attention_implementation,
        "execution_engine": "huggingface eager forward with output_attentions, "
                            "torch.autograd.grad to the mlp.c_proj input",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }

    summary = {
        "task5_attribution": "PASS" if task_pass else "FAIL",
        "attribution_sha256": rows_sha256,
        "n_rows": len(rows),
        "n_examples": len(items),
        "eligible_mlp_layers": eligible,
        "checks": {
            "scorer_all_layer_max_abs_diff": max_all_layer_diff,
            "scorer_parity_band_vs_upstream_max_abs_diff": max_parity_band_diff,
            "scorer_future_targets_max_abs_diff": max_target_diff,
            "metric_tolerance": metric_limit,
            "scorer_parity_pass": scorer_parity_pass,
            "objective_vs_frozen_metric_max_abs_diff": objective_diff,
            "objective_pass": objective_pass,
            "causal_order_pass": causal_order_pass,
            "gradient_pass": gradient_pass,
            "rows_pass": rows_pass,
            "expected_rows": expected_rows,
            "discovery_only_pass": discovery_pass,
            "scope_corpus_provenance_pass": provenance_pass,
            "nonfinite_values": sum(r.nonfinite_values for r in result.layers),
            "zero_gradient_examples": sum(
                r.zero_gradient_examples for r in result.layers
            ),
        },
        "per_layer": per_layer,
        "aggregate": {
            "mean_sink_parity_band": mean_parity_band,
            "mean_sink_future_by_layer": {
                str(r.layer): r.mean_sink_future for r in result.layers
            },
            "top_rows": sorted(rows, key=lambda row: row["rank_abs"])[:20],
        },
        "peak_layer_memory_allocated_bytes": peak_layer_allocated,
        "is_causal_evidence": False,
    }

    with (output_dir / "neuron_attribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "attribution_metadata.json", metadata)
    write_json(output_dir / "summary.json", summary)

    provenance = recorder.finish(
        repo_commit=repo_commit, submodule_commits=submodule_commits
    )
    write_json(output_dir / "provenance.json", provenance)

    if not args.no_freeze and task_pass and args.max_examples is None:
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        if FROZEN_ATTRIBUTION_META.exists():
            existing = read_json(FROZEN_ATTRIBUTION_META)
            if existing.get("attribution_sha256") != rows_sha256:
                raise SystemExit(
                    f"{FROZEN_ATTRIBUTION_META} already holds a different frozen ranking "
                    f"({existing.get('attribution_sha256')}). A frozen ranking is "
                    "immutable; register a new experiment id rather than overwriting it."
                )
            print(f"frozen ranking already matches: {FROZEN_ATTRIBUTION}")
        else:
            with FROZEN_ATTRIBUTION.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
                writer.writeheader()
                writer.writerows(rows)
            write_json(FROZEN_ATTRIBUTION_META, metadata)
            print(f"frozen ranking written: {FROZEN_ATTRIBUTION}")

    print(f"TASK5_ATTRIBUTION={'PASS' if task_pass else 'FAIL'}")
    print(f"n_examples={len(items)} n_rows={len(rows)} (expected {expected_rows})")
    print(f"eligible_mlp_layers={eligible}")
    print(f"scorer_parity_pass={scorer_parity_pass} max_abs_diff="
          f"{max(max_all_layer_diff, max_parity_band_diff, max_target_diff):.6g} "
          f"tolerance={metric_limit:.6g}")
    print(f"objective_vs_frozen_metric_max_abs_diff={objective_diff:.6g}")
    print(f"causal_order_pass={causal_order_pass}")
    print(f"gradient_pass={gradient_pass} rows_pass={rows_pass} "
          f"discovery_only_pass={discovery_pass}")
    for entry in per_layer:
        print(
            f"  layer {entry['layer']:>2} targets {entry['future_sink_layers']} "
            f"S_future={entry['mean_sink_future']:.6f} "
            f"mean|a*g|={entry['mean_abs_attr_mean']:.6e} "
            f"max={entry['mean_abs_attr_max']:.6e} top_neuron={entry['top_neuron']}"
        )
    print(f"attribution_sha256={rows_sha256}")
    print(f"wall_seconds={provenance['runtime_seconds']:.3f}")
    print(f"peak_memory_allocated_bytes={provenance['peak_memory_allocated_bytes']}")
    print(f"peak_memory_reserved_bytes={provenance['peak_memory_reserved_bytes']}")
    print(f"output_dir={output_dir}")
    return 0 if task_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
