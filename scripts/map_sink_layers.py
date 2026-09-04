#!/usr/bin/env python
"""Task 4b: baseline GPT-2-small per-layer/per-head sink map and frozen sink scope.

Runs the **baseline only** on the discovery split of the frozen neutral corpus, records the
attention received by position 0 per layer and head, applies the registered sink-heavy layer
rule from ``docs/00_MASTER_EXPERIMENT_DESIGN.md``, and freezes the resulting scope.

No suppression, no gradients, no neuron attribution. Attribution is Task 5, and it may only
read the discovery split this script also uses.

Two forward paths are run per example and cross-checked against each other:

1. the Hugging Face forward with ``output_attentions=True`` -- the path every later
   intervention uses, since ``neuron_sink.suppression`` hooks a real ``nn.Module``; and
2. the pinned upstream manual baseline
   (``intervention_analysis_legacy.intervention_a_baseline``) that produced the Task-2
   parity number.

The map is built from path 1 so the frozen scope matches the execution path that will
measure interventions. Path 2 exists to prove that choice did not move the metric: for every
example the map's scalar over the upstream ``[3, 11)`` band is compared against upstream
``compute_bos_attention_metric`` on both paths. That band is parity-only; the sink-heavy
rule is applied over all layers, as the master design requires.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuron_sink.corpus import NeutralCorpus, require_discovery_split  # noqa: E402
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    canonical_sha256,
    git,
    prepare_output_dir,
    require_pinned_submodules,
    require_registered_gpu,
    run_stamp,
    write_json,
)
from neuron_sink.sink_metrics import (  # noqa: E402
    REGISTERED_SINK_FLOOR,
    REGISTERED_TARGET_POSITION,
    build_sink_scope,
    layer_scores,
    per_layer_head_position0_attention,
    sink_scalar_from_map,
)
from neuron_sink.upstream_bridge import sink_repro_module  # noqa: E402


FROZEN_DIR = ROOT / "configs" / "frozen"
FROZEN_MANIFEST = FROZEN_DIR / "neutral_corpus_manifest.json"
FROZEN_SCOPE = FROZEN_DIR / "sink_scope.json"
DEFAULT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline per-layer/per-head sink map on the discovery split."
    )
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--manifest", type=Path, default=FROZEN_MANIFEST)
    parser.add_argument("--split", default="discovery",
                        help="Ranking-facing stage; only 'discovery' is permitted.")
    parser.add_argument("--smoke", action="store_true", default=True,
                        help="Use the 24-example smoke prefix (RTX 2060 phase default).")
    parser.add_argument("--full", dest="smoke", action="store_false",
                        help="Use the full 100-example discovery split.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Runtime-estimation dry run; not a registered configuration.")
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
    require_discovery_split(args.split)

    output_dir = prepare_output_dir(
        args.output_dir or ROOT / "results" / "task4_sink_map" / run_stamp()
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

    num_layers = len(model.transformer.h)
    num_heads = int(model.config.n_head)
    band_start, band_end = legacy.compute_band(num_layers, "scaled")
    parity_band = list(range(band_start, band_end))

    print(
        f"Mapping {len(items)} {args.split} examples "
        f"({'smoke' if args.smoke else 'full'} split), seq_len={corpus.cut_length}, "
        f"layers={num_layers}, heads={num_heads}"
    )

    map_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    per_example_rows: list[dict] = []
    max_hf_vs_upstream = 0.0
    max_map_vs_upstream_hf = 0.0
    max_hf_vs_manual_attention = 0.0
    nonfinite = 0
    max_row_sum_error = 0.0
    max_future_attention = 0.0

    for index, item in enumerate(items):
        if args.progress_every > 0 and (
            index == 0 or (index + 1) % args.progress_every == 0
        ):
            print(f"  baseline [{index + 1}/{len(items)}] {item.item_id}")

        ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(ids)

        # Path 1: the Hugging Face forward that later interventions hook.
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )
        hf_attentions = [a[0].detach().float().cpu() for a in output.attentions]

        # Path 2: the pinned upstream manual baseline that produced the Task-2 number.
        inputs = {"input_ids": ids, "attention_mask": attention_mask}
        pos_enc, token_embeddings = legacy.get_initial_embeddings(model, inputs)
        with torch.no_grad():
            manual_attentions = legacy.intervention_a_baseline(
                model, token_embeddings, pos_enc
            )

        example_map = per_layer_head_position0_attention(
            hf_attentions, target_pos=REGISTERED_TARGET_POSITION
        )
        if example_map.shape != (num_layers, num_heads):
            raise AssertionError(
                f"{item.item_id}: map shape {example_map.shape} != "
                f"{(num_layers, num_heads)}"
            )
        map_sum += example_map

        upstream_hf = legacy.compute_bos_attention_metric(
            hf_attentions, num_layers, "mid", target_pos=REGISTERED_TARGET_POSITION,
            layer_start=band_start, layer_end=band_end,
        )
        upstream_manual = legacy.compute_bos_attention_metric(
            manual_attentions, num_layers, "mid", target_pos=REGISTERED_TARGET_POSITION,
            layer_start=band_start, layer_end=band_end,
        )
        map_band_scalar = sink_scalar_from_map(example_map, parity_band)

        max_map_vs_upstream_hf = max(
            max_map_vs_upstream_hf, abs(map_band_scalar - upstream_hf)
        )
        max_hf_vs_upstream = max(max_hf_vs_upstream, abs(upstream_hf - upstream_manual))
        max_hf_vs_manual_attention = max(
            max_hf_vs_manual_attention,
            max(
                float((a - b.detach().float().cpu()).abs().max().item())
                for a, b in zip(hf_attentions, manual_attentions)
            ),
        )

        for attention in hf_attentions:
            nonfinite += int((~torch.isfinite(attention)).sum().item())
            max_row_sum_error = max(
                max_row_sum_error,
                float((attention.sum(dim=-1) - 1.0).abs().max().item()),
            )
            max_future_attention = max(
                max_future_attention,
                float(torch.triu(attention, diagonal=1).abs().max().item()),
            )

        per_example_rows.append({
            "example_id": item.item_id,
            "split": item.split,
            "prompt_tokens": item.n_tokens,
            "sink_parity_band": map_band_scalar,
            "sink_upstream_hf": float(upstream_hf),
            "sink_upstream_manual": float(upstream_manual),
            "sink_all_layers": sink_scalar_from_map(example_map, None),
            **{
                f"layer_{layer}": float(example_map[layer].mean())
                for layer in range(num_layers)
            },
        })

    torch.cuda.synchronize(device)

    layer_head_mean = map_sum / len(items)
    scores = layer_scores(layer_head_mean)
    scope = build_sink_scope(layer_head_mean, floor=REGISTERED_SINK_FLOOR)

    mean_parity_band = float(np.mean([r["sink_parity_band"] for r in per_example_rows]))
    mean_upstream_manual = float(
        np.mean([r["sink_upstream_manual"] for r in per_example_rows])
    )

    metric_limit = engine.METRIC_ATOL + engine.METRIC_RTOL * max(
        abs(mean_parity_band), 1e-12
    )
    attention_limit = engine.ATTN_ATOL + engine.ATTN_RTOL

    decomposition_pass = max_map_vs_upstream_hf <= metric_limit
    path_agreement_pass = max_hf_vs_upstream <= metric_limit
    attention_pass = (
        nonfinite == 0
        and max_row_sum_error <= attention_limit
        and max_future_attention <= engine.ATTN_ATOL
    )
    scope_pass = (
        len(scope.sink_layers) >= 1
        and all(scores[layer] >= REGISTERED_SINK_FLOOR for layer in scope.sink_layers)
        and len(scope.eligible_mlp_layers) >= 1
        and not scope.fallback_incomplete
    )
    finite_pass = bool(np.isfinite(layer_head_mean).all())

    task_pass = all(
        (decomposition_pass, path_agreement_pass, attention_pass, scope_pass, finite_pass)
    )

    scope_document = {
        "schema": "sink_scope_v1",
        "model_id": "openai-community/gpt2",
        "model_revision": args.revision,
        "dtype": "float32",
        "corpus_id": corpus.corpus_id,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "split": args.split,
        "split_mode": "smoke_24" if args.smoke else "full_100",
        "n_examples": len(items),
        "seq_len": corpus.cut_length,
        "example_ids": [item.item_id for item in items],
        "selection_rule": (
            "layer sink score in the top quartile AND >= absolute_floor; if fewer than "
            "two qualify, the top two above the floor; if none, sink preflight fails"
        ),
        "per_layer_sink": [float(v) for v in scores],
        "per_layer_per_head_sink": [[float(v) for v in row] for row in layer_head_mean],
        **scope.to_dict(),
    }
    scope_document["sink_scope_sha256"] = canonical_sha256(
        {k: v for k, v in scope_document.items() if k != "sink_scope_sha256"}
    )

    run_config = {
        "experiment_id": "task4_sink_map",
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
        "seq_len": corpus.cut_length,
        "n_examples": len(items),
        "sink_target_position": REGISTERED_TARGET_POSITION,
        "sink_query_rule": "second_half",
        "sink_layers": list(scope.sink_layers),
        "sink_heads": {str(k): list(v) for k, v in sorted(scope.sink_heads.items())},
        "sink_absolute_floor": REGISTERED_SINK_FLOOR,
        "parity_band": parity_band,
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "selection_method": None,
        "neuron_fraction": None,
        "k": None,
        "alpha": None,
        "control_type": "baseline",
        "control_seed": None,
        "attention_implementation": attention_implementation,
        "execution_engine": "huggingface eager forward, cross-checked against the pinned "
                            "Sink-Repro manual baseline",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sink_scope_sha256": scope_document["sink_scope_sha256"],
    }

    summary = {
        "task4_sink_map": "PASS" if task_pass else "FAIL",
        "sink_scope": scope_document,
        "aggregate": {
            "mean_sink_parity_band": mean_parity_band,
            "mean_sink_upstream_manual": mean_upstream_manual,
            "mean_sink_all_layers": float(np.mean(scores)),
            "per_layer_sink": [float(v) for v in scores],
        },
        "checks": {
            "map_decomposition_vs_upstream_max_abs_diff": max_map_vs_upstream_hf,
            "map_decomposition_pass": decomposition_pass,
            "hf_vs_manual_metric_max_abs_diff": max_hf_vs_upstream,
            "hf_vs_manual_attention_max_abs_diff": max_hf_vs_manual_attention,
            "forward_path_agreement_pass": path_agreement_pass,
            "metric_tolerance": metric_limit,
            "attention_tolerance": attention_limit,
            "nonfinite_attention_values": nonfinite,
            "max_attention_row_sum_error": max_row_sum_error,
            "max_causal_future_attention": max_future_attention,
            "attention_validity_pass": attention_pass,
            "map_finite": finite_pass,
            "scope_pass": scope_pass,
        },
        "runtime": {
            "n_examples": len(items),
        },
    }

    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "sink_map.json", {
        "per_layer_per_head_sink": [[float(v) for v in row] for row in layer_head_mean],
        "per_layer_sink": [float(v) for v in scores],
        "num_layers": num_layers,
        "num_heads": num_heads,
        "n_examples": len(items),
    })
    write_json(output_dir / "sink_scope.json", scope_document)

    with (output_dir / "per_example_sink.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_example_rows[0]))
        writer.writeheader()
        writer.writerows(per_example_rows)

    provenance = recorder.finish(
        repo_commit=repo_commit, submodule_commits=submodule_commits
    )
    write_json(output_dir / "provenance.json", provenance)

    if not args.no_freeze and task_pass and args.max_examples is None:
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        if FROZEN_SCOPE.exists():
            import json as _json

            existing = _json.loads(FROZEN_SCOPE.read_text(encoding="utf-8"))
            if existing.get("sink_scope_sha256") != scope_document["sink_scope_sha256"]:
                raise SystemExit(
                    f"{FROZEN_SCOPE} already holds a different frozen scope "
                    f"({existing.get('sink_scope_sha256')}). A frozen scope is immutable; "
                    "register a new experiment id rather than overwriting it."
                )
            print(f"frozen scope already matches: {FROZEN_SCOPE}")
        else:
            write_json(FROZEN_SCOPE, scope_document)
            print(f"frozen scope written: {FROZEN_SCOPE}")

    print(f"TASK4_SINK_MAP={'PASS' if task_pass else 'FAIL'}")
    print(f"n_examples={len(items)}")
    print(f"per_layer_sink={[round(float(v), 6) for v in scores]}")
    print(f"sink_layers={list(scope.sink_layers)} (rule={scope.rule_applied})")
    print(f"sink_heads={ {k: list(v) for k, v in sorted(scope.sink_heads.items())} }")
    print(f"eligible_mlp_layers={list(scope.eligible_mlp_layers)}")
    print(f"mean_sink_parity_band={mean_parity_band:.9f}")
    print(f"map_vs_upstream_max_abs_diff={max_map_vs_upstream_hf:.6g}")
    print(f"hf_vs_manual_metric_max_abs_diff={max_hf_vs_upstream:.6g}")
    print(f"hf_vs_manual_attention_max_abs_diff={max_hf_vs_manual_attention:.6g}")
    print(f"sink_scope_sha256={scope_document['sink_scope_sha256']}")
    print(f"wall_seconds={provenance['runtime_seconds']:.3f}")
    print(f"peak_memory_allocated_bytes={provenance['peak_memory_allocated_bytes']}")
    print(f"output_dir={output_dir}")
    return 0 if task_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
