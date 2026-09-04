#!/usr/bin/env python
"""Run the pinned Sink-Repro GPT-2-small baseline, without interventions.

This is a thin Task-2 adapter around the read-only upstream implementation. It reuses:

* ``corpus_providers.frozen_e1_corpus`` for the E1 dataset/manifest construction;
* ``intervention_analysis_legacy.intervention_a_baseline`` for the frozen manual
  baseline forward;
* ``intervention_analysis_legacy.compute_band`` for the GPT-2 layer band; and
* ``intervention_analysis_legacy.compute_bos_attention_metric`` for the canonical sink
  scalar (also re-exported by ``intervention_analysis``).

The upstream dataset CLI runs all registered interventions. Task 2 permits only the
baseline, so this adapter calls the baseline function directly and writes its own small,
ignored result bundle outside the submodule.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import GPT2LMHeadModel, GPT2Tokenizer


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "sink-repro"
UPSTREAM_COMMON = UPSTREAM / "common"
EXPECTED_SINK_REPRO_COMMIT = "9ab67e914464b13863b67527d8ea14068ee9ff10"
EXPECTED_SINK_KD_COMMIT = "db114c9c5eb6ffc5de13e444c783408ea7401c62"

if str(UPSTREAM_COMMON) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_COMMON))

# Imports below are intentionally from the pinned submodule, not local copies.
from corpus_providers import frozen_e1_corpus  # noqa: E402
from datasets_loader import DATASET_SPECS  # noqa: E402
# OpenAI GPT-2 has both configuration flags handled by intervention_analysis.py disabled,
# so that wrapper explicitly intends to dispatch here unchanged. At the pinned commit its
# rebinding of _legacy.manual_self_attention_new makes that dispatch recurse. Importing the
# frozen legacy module directly avoids only that wrapper defect; the GPT-2 computation is
# the exact function body that produced the registered reference.
from intervention_analysis_legacy import (  # noqa: E402
    compute_band,
    compute_bos_attention_metric,
    get_initial_embeddings,
    intervention_a_baseline,
)
from nnsight_engine import (  # noqa: E402
    ATTN_ATOL,
    ATTN_RTOL,
    METRIC_ATOL,
    METRIC_RTOL,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    def _numpy_scalar(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_numpy_scalar,
        ) + "\n",
        encoding="utf-8",
    )


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    return ROOT / "results" / "task2_gpt2_sink_parity" / stamp


def _prepare_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Task results are append-only."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "nnsight", "datasets", "numpy", "pandas")
    return {name: importlib.metadata.version(name) for name in names}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the canonical GPT-2-small Sink-Repro baseline."
    )
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Examples per E1 dataset (reference: 100).")
    parser.add_argument("--cut-length", type=int, default=40,
                        help="Exact token length (reference: 40).")
    parser.add_argument("--seed", type=int, default=0,
                        help="E1 manifest seed (reference artifact: 0).")
    parser.add_argument("--repeat-size", type=int, default=3,
                        help="Fixed leading examples rerun for determinism.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Hugging Face cache directory; place it off a low-space drive.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _baseline_maps(model: GPT2LMHeadModel, input_ids: list[int]) -> list[torch.Tensor]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    pos_enc, token_embeddings = get_initial_embeddings(model, inputs)
    with torch.no_grad():
        return intervention_a_baseline(model, token_embeddings, pos_enc)


def main() -> int:
    args = _parse_args()
    if args.sample_size <= 0 or args.cut_length < 2 or args.repeat_size <= 0:
        raise ValueError("sample-size and repeat-size must be positive; cut-length >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 requires the RTX 2060 SUPER to be usable by PyTorch")

    output_dir = _prepare_output_dir(args.output_dir or _default_output_dir())
    started_at = _utc_now()
    wall_start = time.perf_counter()

    repo_commit = _git("rev-parse", "HEAD")
    sink_repro_commit = _git("rev-parse", "HEAD", cwd=UPSTREAM)
    sink_kd_commit = _git("rev-parse", "HEAD", cwd=ROOT / "upstream" / "sink-kd")
    if sink_repro_commit != EXPECTED_SINK_REPRO_COMMIT:
        raise RuntimeError(
            f"sink-repro is {sink_repro_commit}; expected {EXPECTED_SINK_REPRO_COMMIT}"
        )
    if sink_kd_commit != EXPECTED_SINK_KD_COMMIT:
        raise RuntimeError(
            f"sink-kd is {sink_kd_commit}; expected {EXPECTED_SINK_KD_COMMIT}"
        )
    if _git("status", "--porcelain", cwd=UPSTREAM):
        raise RuntimeError("upstream/sink-repro is modified; refusing parity run")
    if _git("status", "--porcelain", cwd=ROOT / "upstream" / "sink-kd"):
        raise RuntimeError("upstream/sink-kd is modified; refusing parity run")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "RTX 2060 SUPER" not in gpu_name:
        raise RuntimeError(f"Expected RTX 2060 SUPER, found {gpu_name!r}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    info = HfApi().model_info(args.model_id, revision=args.revision)
    resolved_revision = info.sha
    resolved_model_id = info.id
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None

    print(
        f"Loading {args.model_id} ({resolved_model_id}@{resolved_revision}) "
        f"in float32 on {gpu_name}"
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_id, revision=resolved_revision, cache_dir=cache_dir
    )
    model = GPT2LMHeadModel.from_pretrained(
        args.model_id,
        revision=resolved_revision,
        cache_dir=cache_dir,
        attn_implementation="eager",
    )
    model.to(device)
    model.to(torch.float32)
    model.eval()
    if model.training:
        raise AssertionError("model.eval() did not take effect")
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "eager":
        raise AssertionError(
            f"Expected eager attention, got {attention_implementation!r}"
        )

    num_layers = len(model.transformer.h)
    num_heads = int(model.config.n_head)
    band = compute_band(num_layers, "scaled")
    layer_start, layer_end = band

    print(
        f"Building frozen E1 corpus: {args.sample_size} examples/dataset, "
        f"length={args.cut_length}, seed={args.seed}"
    )
    corpus = frozen_e1_corpus(
        tokenizer,
        sample_size=args.sample_size,
        cut_length=args.cut_length,
        seed=args.seed,
    )
    corpus.save(output_dir / "sample_manifest.json")

    n_examples = len(corpus)
    if n_examples != args.sample_size * len(DATASET_SPECS):
        raise AssertionError(
            f"Expected {args.sample_size * len(DATASET_SPECS)} examples, got {n_examples}"
        )

    per_example_sink: list[float] = []
    layer_head_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    first_half_key_sum = np.zeros(args.cut_length // 2, dtype=np.float64)
    first_half_key_count = 0
    dataset_counts: Counter[str] = Counter()
    observed_lengths: Counter[int] = Counter()
    nonfinite_values = 0
    max_row_sum_error = 0.0
    max_future_attention = 0.0
    minimum_attention = math.inf
    maximum_attention = -math.inf

    for example_index, item in enumerate(corpus.items):
        if args.progress_every > 0 and (
            example_index == 0 or (example_index + 1) % args.progress_every == 0
        ):
            print(f"  baseline [{example_index + 1}/{n_examples}] {item.item_id}")

        input_ids = [int(token) for token in item.input_ids]
        observed_lengths[len(input_ids)] += 1
        dataset_counts[str(item.meta["dataset"])] += 1
        if len(input_ids) != args.cut_length:
            raise AssertionError(
                f"{item.item_id} has {len(input_ids)} tokens, expected {args.cut_length}"
            )
        retokenized = tokenizer(
            item.text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if list(map(int, retokenized)) != input_ids:
            raise AssertionError(f"Tokenization drift for {item.item_id}")

        maps = _baseline_maps(model, input_ids)
        if len(maps) != num_layers:
            raise AssertionError(f"Expected {num_layers} attention layers, got {len(maps)}")

        sink = compute_bos_attention_metric(
            maps,
            num_layers,
            "mid",
            target_pos=0,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        if not math.isfinite(sink):
            raise AssertionError(f"Non-finite sink score for {item.item_id}: {sink}")
        per_example_sink.append(float(sink))

        for layer_index, attention in enumerate(maps):
            expected_shape = (num_heads, args.cut_length, args.cut_length)
            if tuple(attention.shape) != expected_shape:
                raise AssertionError(
                    f"Layer {layer_index} shape {tuple(attention.shape)} != {expected_shape}"
                )
            finite = torch.isfinite(attention)
            nonfinite_values += int((~finite).sum().item())
            minimum_attention = min(minimum_attention, float(attention.min().item()))
            maximum_attention = max(maximum_attention, float(attention.max().item()))
            row_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
            max_row_sum_error = max(max_row_sum_error, row_error)
            future = float(torch.triu(attention, diagonal=1).abs().max().item())
            max_future_attention = max(max_future_attention, future)

            half = args.cut_length // 2
            per_head = attention[:, half:, 0].mean(dim=1).numpy()
            layer_head_sum[layer_index] += per_head.astype(np.float64)
            if layer_start <= layer_index < layer_end:
                by_key = attention[:, half:, :half].mean(dim=(0, 1)).numpy()
                first_half_key_sum += by_key.astype(np.float64)
                first_half_key_count += 1

    torch.cuda.synchronize(device)

    layer_head_mean = layer_head_sum / n_examples
    layer_mean = layer_head_mean.mean(axis=1)
    reproduced_sink = float(np.mean(per_example_sink))
    first_half_key_mean = first_half_key_sum / first_half_key_count
    other_prefix_mean = float(first_half_key_mean[1:].mean())
    position0_concentration_ratio = (
        float(first_half_key_mean[0] / other_prefix_mean)
        if other_prefix_mean > 0
        else math.inf
    )
    position0_rank = int(
        np.where(np.argsort(-first_half_key_mean) == 0)[0][0] + 1
    )

    repeat_count = min(args.repeat_size, n_examples)
    repeat_scores: list[float] = []
    repeat_max_attention_difference = 0.0
    for index in range(repeat_count):
        maps = _baseline_maps(model, list(corpus.items[index].input_ids))
        repeat_scores.append(
            compute_bos_attention_metric(
                maps,
                num_layers,
                "mid",
                target_pos=0,
                layer_start=layer_start,
                layer_end=layer_end,
            )
        )
        # The first execution's full maps are not retained; rerun once more to compare
        # attention tensors directly while keeping peak host memory bounded.
        comparison_maps = _baseline_maps(model, list(corpus.items[index].input_ids))
        repeat_max_attention_difference = max(
            repeat_max_attention_difference,
            max(float((a - b).abs().max().item())
                for a, b in zip(maps, comparison_maps)),
        )

    repeat_max_sink_difference = max(
        abs(repeat_scores[index] - per_example_sink[index])
        for index in range(repeat_count)
    )

    reference_manifest = json.loads(
        (UPSTREAM / "BASELINE_HASHES.json").read_text(encoding="utf-8")
    )
    reference_block = reference_manifest["reference_results"]["gpt2_small_table1"]
    reference = reference_block["interventions"]["int_a"]
    tolerances = reference_manifest["reference_results"]["tolerances"]
    reference_sink = float(reference["mean_bos_attention"])
    metric_atol = float(tolerances["metric_atol"])
    metric_rtol = float(tolerances["metric_rtol"])
    absolute_difference = abs(reproduced_sink - reference_sink)
    allowed_difference = metric_atol + metric_rtol * abs(reference_sink)

    exact_reference_protocol = (
        args.sample_size == int(reference_block["_source"]["sample_size"])
        and args.cut_length == int(reference_block["_source"]["cut_length"])
        and args.seed == int(reference_block["_source"]["seed"])
        and list(band) == list(reference_block["_source"]["band"])
    )
    reference_pass = exact_reference_protocol and absolute_difference <= allowed_difference
    probability_rows_pass = (
        nonfinite_values == 0
        and minimum_attention >= 0.0
        and maximum_attention <= 1.0
        and max_row_sum_error <= ATTN_ATOL + ATTN_RTOL
        and max_future_attention <= ATTN_ATOL
    )
    lengths_pass = observed_lengths == Counter({args.cut_length: n_examples})
    position0_pass = position0_rank == 1 and first_half_key_mean[0] > other_prefix_mean
    repeat_limit = METRIC_ATOL + METRIC_RTOL * max(
        max(abs(score) for score in per_example_sink[:repeat_count]), 1e-12
    )
    repeat_pass = (
        repeat_max_sink_difference <= repeat_limit
        and repeat_max_attention_difference <= ATTN_ATOL + ATTN_RTOL
    )

    sink_repro_clean_after = not bool(_git("status", "--porcelain", cwd=UPSTREAM))
    sink_kd_clean_after = not bool(
        _git("status", "--porcelain", cwd=ROOT / "upstream" / "sink-kd")
    )
    task2_pass = all(
        (
            reference_pass,
            probability_rows_pass,
            lengths_pass,
            position0_pass,
            repeat_pass,
            sink_repro_clean_after,
            sink_kd_clean_after,
        )
    )

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finished_at = _utc_now()

    run_config = {
        "experiment_id": "task2_gpt2_sink_parity",
        "stage": "parity",
        "model_id": args.model_id,
        "model_resolved_id": resolved_model_id,
        "model_revision": resolved_revision,
        "tokenizer_id": args.model_id,
        "tokenizer_revision": resolved_revision,
        "dtype": "float32",
        "device": str(device),
        "gpu_name": gpu_name,
        "seed": args.seed,
        "dataset_id": "Sink-Repro E1 benchmark mixture",
        "dataset_config": [
            {key: spec.get(key) for key in ("name", "hf_path", "config", "split")}
            for spec in DATASET_SPECS
        ],
        "dataset_split": [spec["split"] for spec in DATASET_SPECS],
        "manifest_sha256": corpus.manifest_sha256,
        "sample_size_per_dataset": args.sample_size,
        "n_examples": n_examples,
        "seq_len": args.cut_length,
        "sink_target_position": 0,
        "sink_query_rule": "second_half",
        "sink_query_positions": list(range(args.cut_length // 2, args.cut_length)),
        "sink_layers": list(range(layer_start, layer_end)),
        "sink_heads": None,
        "sink_aggregation": "per-example mean over heads, second-half queries, and layers; then mean over examples",
        "execution_engine": "Sink-Repro manual baseline",
        "upstream_function": "common/intervention_analysis_legacy.py: intervention_a_baseline + compute_bos_attention_metric (publicly re-exported by common/intervention_analysis.py)",
        "attention_implementation": attention_implementation,
        "model_eval": not model.training,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "quantized": False,
    }
    provenance = {
        "repo_commit": repo_commit,
        "sink_repro_commit": sink_repro_commit,
        "sink_kd_commit": sink_kd_commit,
        "python": platform.python_version(),
        **_package_versions(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "command": " ".join(sys.argv),
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": wall_seconds,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
    }
    summary = {
        "task2_parity": "PASS" if task2_pass else "FAIL",
        "reference": {
            "mean_bos_attention": reference_sink,
            "stderr": float(reference["stderr"]),
            "n_examples": int(reference["n_examples"]),
            "metric_atol": metric_atol,
            "metric_rtol": metric_rtol,
            "allowed_difference": allowed_difference,
            "source": reference_block["_source"],
        },
        "results": {
            "reproduced_baseline_sink": reproduced_sink,
            "absolute_difference": absolute_difference,
            "n_examples": n_examples,
            "per_layer_sink": [float(value) for value in layer_mean],
            "per_layer_per_head_sink": [
                [float(value) for value in row] for row in layer_head_mean
            ],
        },
        "repeatability": {
            "n_examples": repeat_count,
            "original_sink": per_example_sink[:repeat_count],
            "repeated_sink": [float(value) for value in repeat_scores],
            "max_abs_sink_difference": repeat_max_sink_difference,
            "max_abs_attention_difference": repeat_max_attention_difference,
            "pass": repeat_pass,
        },
        "sanity_checks": {
            "exact_reference_protocol": exact_reference_protocol,
            "reference_tolerance": reference_pass,
            "probability_rows": probability_rows_pass,
            "no_nan_or_inf": nonfinite_values == 0,
            "sequence_lengths": lengths_pass,
            "position0_concentration": position0_pass,
            "deterministic_repeat": repeat_pass,
            "sink_repro_clean": sink_repro_clean_after,
            "sink_kd_clean": sink_kd_clean_after,
            "max_attention_row_sum_error": max_row_sum_error,
            "max_causal_future_attention": max_future_attention,
            "minimum_attention": minimum_attention,
            "maximum_attention": maximum_attention,
            "nonfinite_values": nonfinite_values,
            "observed_sequence_lengths": dict(observed_lengths),
            "dataset_counts": dict(dataset_counts),
            "position0_sink": float(first_half_key_mean[0]),
            "other_first_half_positions_mean": other_prefix_mean,
            "position0_to_other_ratio": position0_concentration_ratio,
            "position0_rank_among_first_half_positions": position0_rank,
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        },
    }

    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "summary.json", summary)

    print(f"TASK2_PARITY={'PASS' if task2_pass else 'FAIL'}")
    print(f"reference_sink={reference_sink:.9f}")
    print(f"reproduced_sink={reproduced_sink:.9f}")
    print(f"absolute_difference={absolute_difference:.9g}")
    print(f"allowed_difference={allowed_difference:.9g}")
    print(f"manifest_sha256={corpus.manifest_sha256}")
    print(f"wall_seconds={wall_seconds:.3f}")
    print(f"peak_memory_allocated_bytes={peak_allocated}")
    print(f"peak_memory_reserved_bytes={peak_reserved}")
    print(f"output_dir={output_dir}")
    return 0 if task2_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
