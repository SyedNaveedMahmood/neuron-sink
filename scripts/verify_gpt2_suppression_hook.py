#!/usr/bin/env python
"""Run the Task-3 GPT-2 MLP suppression correctness audit only.

This script uses arbitrary, deterministic ``source="debug"`` coordinates. It performs
no attribution, ranking, neuron selection, scientific suppression sweep, or downstream
evaluation. The sink scalar is imported directly from the pinned Sink-Repro function so
the alpha=1 comparison retains Task-2 metric semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import GPT2LMHeadModel


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "sink-repro"
UPSTREAM_COMMON = UPSTREAM / "common"
EXPECTED_SINK_REPRO_COMMIT = "9ab67e914464b13863b67527d8ea14068ee9ff10"
EXPECTED_SINK_KD_COMMIT = "db114c9c5eb6ffc5de13e444c783408ea7401c62"
DEFAULT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

for source_path in (ROOT, UPSTREAM_COMMON):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from neuron_sink import GPT2ModelAdapter, NeuronSet, suppress_neurons  # noqa: E402
from neuron_sink.provenance import require_registered_gpu  # noqa: E402
from intervention_analysis_legacy import (  # noqa: E402
    compute_band,
    compute_bos_attention_metric,
)
from nnsight_engine import ATTN_ATOL, ATTN_RTOL, METRIC_ATOL, METRIC_RTOL  # noqa: E402


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    return ROOT / "results" / "task3_gpt2_suppression_hook" / stamp


def _prepare_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _find_task2_manifest(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Task-2 manifest does not exist: {candidate}")
        return candidate
    root = ROOT / "results" / "task2_gpt2_sink_parity"
    for run_dir in sorted(root.glob("run_*"), reverse=True):
        manifest = run_dir / "sample_manifest.json"
        summary = run_dir / "summary.json"
        if manifest.is_file() and summary.is_file():
            status = json.loads(summary.read_text(encoding="utf-8")).get("task2_parity")
            if status == "PASS":
                return manifest
    raise FileNotFoundError("No completed PASS Task-2 manifest was found")


def _manifest_input(manifest_path: Path) -> tuple[list[int], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        raise ValueError(f"Task-2 manifest contains no items: {manifest_path}")
    input_ids = [int(token) for token in items[0]["input_ids"]]
    if len(input_ids) != int(manifest["cut_length"]):
        raise ValueError("First Task-2 fixture length does not match manifest cut_length")
    return input_ids, manifest


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(tensor.numpy())))
    return digest.hexdigest()


def _snapshot(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> dict[str, Any]:
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    if output.attentions is None:
        raise AssertionError("GPT-2 forward did not return attention tensors")
    return {
        "logits": output.logits.detach().cpu(),
        "attentions": tuple(attention[0].detach().cpu() for attention in output.attentions),
    }


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _attention_difference(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> float:
    return max(_max_difference(a, b) for a, b in zip(left, right))


def _sink(snapshot: dict[str, Any], num_layers: int) -> float:
    layer_start, layer_end = compute_band(num_layers, "scaled")
    return compute_bos_attention_metric(
        list(snapshot["attentions"]),
        num_layers,
        "mid",
        target_pos=0,
        layer_start=layer_start,
        layer_end=layer_end,
    )


def _capture_records():
    records: dict[int, dict[str, Any]] = {}

    def observer(layer: int, before: torch.Tensor, after: torch.Tensor) -> None:
        if layer in records:
            raise AssertionError(f"Layer {layer} hook ran more than once")
        records[layer] = {
            "before": before.detach().cpu().clone(),
            "after": after.detach().cpu().clone(),
            "before_shape": list(before.shape),
            "after_shape": list(after.shape),
            "before_dtype": str(before.dtype),
            "after_dtype": str(after.dtype),
            "before_device": str(before.device),
            "after_device": str(after.device),
        }

    return records, observer


def _audit_coordinates(
    records: dict[int, dict[str, Any]],
    neuron_set: NeuronSet,
    alpha: float,
) -> dict[str, Any]:
    selected_exact_zero = True
    max_scaling_error = 0.0
    max_unselected_difference = 0.0
    shape_preserved = True
    dtype_preserved = True
    device_preserved = True
    per_layer: dict[str, Any] = {}
    for layer, neurons in neuron_set.by_layer.items():
        record = records[layer]
        before = record["before"]
        after = record["after"]
        mask = torch.ones(before.shape[-1], dtype=torch.bool)
        mask[list(neurons)] = False
        scaling_error = _max_difference(after[..., neurons], before[..., neurons] * alpha)
        unselected_difference = _max_difference(after[..., mask], before[..., mask])
        exact_zero = bool(torch.count_nonzero(after[..., neurons]).item() == 0)
        selected_exact_zero = selected_exact_zero and exact_zero
        max_scaling_error = max(max_scaling_error, scaling_error)
        max_unselected_difference = max(max_unselected_difference, unselected_difference)
        shape_preserved = shape_preserved and record["before_shape"] == record["after_shape"]
        dtype_preserved = dtype_preserved and record["before_dtype"] == record["after_dtype"]
        device_preserved = device_preserved and record["before_device"] == record["after_device"]
        per_layer[str(layer)] = {
            "neurons": list(neurons),
            "shape": record["after_shape"],
            "dtype": record["after_dtype"],
            "device": record["after_device"],
            "selected_exact_zero": exact_zero,
            "max_scaling_error": scaling_error,
            "max_unselected_difference": unselected_difference,
        }
    return {
        "observed_layers": sorted(records),
        "selected_exact_zero": selected_exact_zero,
        "max_scaling_error": max_scaling_error,
        "max_unselected_difference": max_unselected_difference,
        "shape_preserved": shape_preserved,
        "dtype_preserved": dtype_preserved,
        "device_preserved": device_preserved,
        "per_layer": per_layer,
    }


def _validate_bad_inputs(adapter: GPT2ModelAdapter) -> dict[str, Any]:
    cases = {
        "invalid_layer": lambda: suppress_neurons(
            adapter, NeuronSet({adapter.num_layers: (0,)}, source="debug"), 0.0
        ),
        "negative_neuron": lambda: suppress_neurons(
            adapter, NeuronSet({0: (-1,)}, source="debug"), 0.0
        ),
        "outside_neuron": lambda: suppress_neurons(
            adapter, NeuronSet({0: (adapter.mlp_width(0),)}, source="debug"), 0.0
        ),
        "nan_alpha": lambda: suppress_neurons(
            adapter, NeuronSet({0: (0,)}, source="debug"), math.nan
        ),
        "negative_alpha": lambda: suppress_neurons(
            adapter, NeuronSet({0: (0,)}, source="debug"), -0.1
        ),
        "above_one_alpha": lambda: suppress_neurons(
            adapter, NeuronSet({0: (0,)}, source="debug"), 1.1
        ),
    }
    results: dict[str, Any] = {}
    for name, operation in cases.items():
        try:
            operation()
        except (IndexError, TypeError, ValueError) as exc:
            results[name] = {
                "pass": True,
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        else:
            results[name] = {"pass": False, "exception": None, "message": None}
    results["pass"] = all(item["pass"] for item in results.values())
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit only the GPT-2 suppression hook")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(Path(os.environ["NEURON_SINK_HF_CACHE"])
                 if os.environ.get("NEURON_SINK_HF_CACHE") else None),
        help="Hugging Face cache directory; defaults to $NEURON_SINK_HF_CACHE, else "
             "the Hugging Face default. Keep it off a low-space system drive.",
    )
    parser.add_argument("--task2-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started_at = _utc_now()
    wall_start = time.perf_counter()
    output_dir = _prepare_output_dir(args.output_dir or _default_output_dir())

    repo_commit = _git("rev-parse", "HEAD")
    sink_repro_commit = _git("rev-parse", "HEAD", cwd=UPSTREAM)
    sink_kd_path = ROOT / "upstream" / "sink-kd"
    sink_kd_commit = _git("rev-parse", "HEAD", cwd=sink_kd_path)
    if sink_repro_commit != EXPECTED_SINK_REPRO_COMMIT:
        raise RuntimeError(f"sink-repro is {sink_repro_commit}; expected {EXPECTED_SINK_REPRO_COMMIT}")
    if sink_kd_commit != EXPECTED_SINK_KD_COMMIT:
        raise RuntimeError(f"sink-kd is {sink_kd_commit}; expected {EXPECTED_SINK_KD_COMMIT}")
    if _git("status", "--porcelain", cwd=UPSTREAM):
        raise RuntimeError("upstream/sink-repro is modified")
    if _git("status", "--porcelain", cwd=sink_kd_path):
        raise RuntimeError("upstream/sink-kd is modified")
    # Amendment A001: registered dev GPUs are enumerated in neuron_sink.provenance.
    device, gpu_name, _total_vram = require_registered_gpu("dev")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    manifest_path = _find_task2_manifest(args.task2_manifest)
    fixture_ids, manifest = _manifest_input(manifest_path)
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    print(f"Loading {args.model_id}@{args.revision} from {cache_dir or 'the default HF cache'}")
    model = GPT2LMHeadModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=cache_dir,
        local_files_only=False,
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval().to(device)
    if model.training:
        raise AssertionError("model.eval() did not take effect")
    adapter = GPT2ModelAdapter(model, model_id="openai-community/gpt2")
    if adapter.num_layers != 12:
        raise AssertionError(f"Expected 12 GPT-2-small layers, found {adapter.num_layers}")
    widths = [adapter.mlp_width(layer) for layer in range(adapter.num_layers)]
    if len(set(widths)) != 1:
        raise AssertionError(f"Expected one GPT-2 MLP width, found {widths}")

    # These are arbitrary deterministic DEBUG ids, not ranked or scientifically selected.
    debug_neurons = NeuronSet(
        {2: (0, 17, 31), 7: (1, 19, 47)},
        source="debug",
        selection_seed=None,
    )
    adapter.validate_neuron_set(debug_neurons)
    input_ids = torch.tensor([fixture_ids], dtype=torch.long, device=device)
    hook_keys_before = {
        layer: tuple(adapter.mlp_projection(layer)._forward_pre_hooks)
        for layer in debug_neurons.by_layer
    }
    parameter_hash_before = _parameter_sha256(model)

    baseline_before = _snapshot(model, input_ids)
    with suppress_neurons(adapter, debug_neurons, 1.0) as identity_context:
        identity_hook_count = identity_context.active_hook_count
        identity = _snapshot(model, input_ids)

    half_records, half_observer = _capture_records()
    with suppress_neurons(
        adapter, debug_neurons, 0.5, observer=half_observer
    ) as half_context:
        _snapshot(model, input_ids)
        half_active_hook_count = half_context.active_hook_count
    half_audit = _audit_coordinates(half_records, debug_neurons, 0.5)

    zero_records, zero_observer = _capture_records()
    with suppress_neurons(
        adapter, debug_neurons, 0.0, observer=zero_observer
    ) as zero_context:
        suppressed = _snapshot(model, input_ids)
        zero_active_hook_count = zero_context.active_hook_count
    zero_audit = _audit_coordinates(zero_records, debug_neurons, 0.0)

    baseline_after = _snapshot(model, input_ids)
    parameter_hash_after = _parameter_sha256(model)
    hook_keys_after = {
        layer: tuple(adapter.mlp_projection(layer)._forward_pre_hooks)
        for layer in debug_neurons.by_layer
    }

    baseline_sink = _sink(baseline_before, adapter.num_layers)
    identity_sink = _sink(identity, adapter.num_layers)
    identity_logits_equal = torch.equal(baseline_before["logits"], identity["logits"])
    identity_attentions_equal = all(
        torch.equal(left, right)
        for left, right in zip(baseline_before["attentions"], identity["attentions"])
    )
    identity_logits_max_diff = _max_difference(
        baseline_before["logits"], identity["logits"]
    )
    identity_attention_max_diff = _attention_difference(
        baseline_before["attentions"], identity["attentions"]
    )
    identity_sink_diff = abs(baseline_sink - identity_sink)
    identity_limit = METRIC_ATOL + METRIC_RTOL * abs(baseline_sink)
    identity_pass = (
        identity_hook_count == 0
        and identity_logits_equal
        and identity_attentions_equal
        and identity_logits_max_diff == 0.0
        and identity_attention_max_diff <= ATTN_ATOL + ATTN_RTOL
        and identity_sink_diff <= identity_limit
    )

    zero_pass = (
        zero_active_hook_count == len(debug_neurons.by_layer)
        and zero_audit["selected_exact_zero"]
        and zero_audit["max_scaling_error"] == 0.0
        and zero_audit["max_unselected_difference"] == 0.0
        and zero_audit["shape_preserved"]
        and zero_audit["dtype_preserved"]
        and zero_audit["device_preserved"]
    )
    half_pass = (
        half_active_hook_count == len(debug_neurons.by_layer)
        and half_audit["max_scaling_error"] <= torch.finfo(torch.float32).eps
        and half_audit["max_unselected_difference"] == 0.0
        and half_audit["shape_preserved"]
        and half_audit["dtype_preserved"]
        and half_audit["device_preserved"]
    )
    multi_layer_pass = (
        zero_audit["observed_layers"] == list(debug_neurons.by_layer)
        and half_audit["observed_layers"] == list(debug_neurons.by_layer)
    )
    validation = _validate_bad_inputs(adapter)

    before_after_logits_diff = _max_difference(
        baseline_before["logits"], baseline_after["logits"]
    )
    before_after_attention_diff = _attention_difference(
        baseline_before["attentions"], baseline_after["attentions"]
    )
    before_after_attentions_equal = all(
        torch.equal(left, right)
        for left, right in zip(
            baseline_before["attentions"], baseline_after["attentions"]
        )
    )
    hooks_removed = hook_keys_before == hook_keys_after
    parameters_unchanged = parameter_hash_before == parameter_hash_after
    leakage_pass = (
        before_after_logits_diff == 0.0
        and before_after_attentions_equal
        and before_after_attention_diff <= ATTN_ATOL + ATTN_RTOL
        and hooks_removed
        and parameters_unchanged
        and half_context.active_hook_count == 0
        and zero_context.active_hook_count == 0
    )

    nonfinite_logits = int((~torch.isfinite(suppressed["logits"])).sum().item())
    nonfinite_attentions = sum(
        int((~torch.isfinite(attention)).sum().item())
        for attention in suppressed["attentions"]
    )
    max_row_sum_error = max(
        float((attention.sum(dim=-1) - 1.0).abs().max().item())
        for attention in suppressed["attentions"]
    )
    output_validity_pass = (
        nonfinite_logits == 0
        and nonfinite_attentions == 0
        and max_row_sum_error <= ATTN_ATOL + ATTN_RTOL
    )

    sink_repro_clean = not bool(_git("status", "--porcelain", cwd=UPSTREAM))
    sink_kd_clean = not bool(_git("status", "--porcelain", cwd=sink_kd_path))
    task3_pass = all(
        (
            identity_pass,
            zero_pass,
            half_pass,
            multi_layer_pass,
            validation["pass"],
            leakage_pass,
            output_validity_pass,
            sink_repro_clean,
            sink_kd_clean,
        )
    )

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "nnsight", "datasets")
    }
    summary = {
        "task3_hook": "PASS" if task3_pass else "FAIL",
        "source_provenance": {
            "repo_starting_commit": repo_commit,
            "sink_repro_commit": sink_repro_commit,
            "sink_kd_commit": sink_kd_commit,
        },
        "model": {
            "model_id": "openai-community/gpt2",
            "requested_id": args.model_id,
            "revision": args.revision,
            "dtype": "float32",
            "attention_implementation": model.config._attn_implementation,
            "device": str(device),
            "gpu_name": gpu_name,
            "eval_mode": not model.training,
            "seed": args.seed,
        },
        "fixture": {
            "source": str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "item_id": manifest["items"][0]["item_id"],
            "sequence_length": len(fixture_ids),
        },
        "hook": {
            "module_path": "transformer.h[layer].mlp.c_proj",
            "tensor_location": "forward-pre-hook input after c_fc and GELU",
            "tensor_shape": zero_audit["per_layer"]["2"]["shape"],
            "num_layers": adapter.num_layers,
            "mlp_widths": widths,
            "debug_neuron_set": {
                str(layer): list(neurons)
                for layer, neurons in debug_neurons.by_layer.items()
            },
        },
        "identity": {
            "torch_equal_logits": identity_logits_equal,
            "torch_equal_attentions": identity_attentions_equal,
            "max_abs_logits_difference": identity_logits_max_diff,
            "max_abs_attention_difference": identity_attention_max_diff,
            "baseline_sink": baseline_sink,
            "identity_sink": identity_sink,
            "absolute_sink_difference": identity_sink_diff,
            "active_hook_count": identity_hook_count,
            "pass": identity_pass,
        },
        "alpha_zero": {**zero_audit, "pass": zero_pass},
        "alpha_half": {**half_audit, "pass": half_pass},
        "multi_layer": {
            "selected_layers": list(debug_neurons.by_layer),
            "pass": multi_layer_pass,
        },
        "input_validation": validation,
        "state_integrity": {
            "baseline_before_after_max_logits_difference": before_after_logits_diff,
            "baseline_before_after_max_attention_difference": before_after_attention_diff,
            "baseline_before_after_attentions_equal": before_after_attentions_equal,
            "hooks_removed": hooks_removed,
            "active_hook_count_after": half_context.active_hook_count + zero_context.active_hook_count,
            "parameter_sha256_before": parameter_hash_before,
            "parameter_sha256_after": parameter_hash_after,
            "parameters_unchanged": parameters_unchanged,
            "pass": leakage_pass,
        },
        "output_validity": {
            "nonfinite_logits": nonfinite_logits,
            "nonfinite_attention_values": nonfinite_attentions,
            "max_attention_row_sum_error": max_row_sum_error,
            "pass": output_validity_pass,
        },
        "source_cleanliness": {
            "sink_repro_clean": sink_repro_clean,
            "sink_kd_clean": sink_kd_clean,
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        },
    }
    provenance = {
        "repo_commit": repo_commit,
        "sink_repro_commit": sink_repro_commit,
        "sink_kd_commit": sink_kd_commit,
        "python": platform.python_version(),
        **package_versions,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "command": " ".join(sys.argv),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "runtime_seconds": wall_seconds,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "provenance.json", provenance)

    print(f"TASK3_HOOK={'PASS' if task3_pass else 'FAIL'}")
    print(f"hook_shape={tuple(zero_audit['per_layer']['2']['shape'])}")
    print(f"mlp_width={widths[0]}")
    print(f"identity_logits_equal={identity_logits_equal}")
    print(f"identity_logits_max_diff={identity_logits_max_diff}")
    print(f"identity_attention_max_diff={identity_attention_max_diff}")
    print(f"identity_sink_diff={identity_sink_diff}")
    print(f"alpha_zero_max_unselected_diff={zero_audit['max_unselected_difference']}")
    print(f"alpha_half_max_scaling_error={half_audit['max_scaling_error']}")
    print(f"baseline_before_after_logits_diff={before_after_logits_diff}")
    print(f"hooks_removed={hooks_removed}")
    print(f"parameters_unchanged={parameters_unchanged}")
    print(f"max_attention_row_sum_error={max_row_sum_error}")
    print(f"wall_seconds={wall_seconds}")
    print(f"peak_memory_allocated_bytes={peak_allocated}")
    print(f"peak_memory_reserved_bytes={peak_reserved}")
    print(f"output_dir={output_dir}")
    return 0 if task3_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
