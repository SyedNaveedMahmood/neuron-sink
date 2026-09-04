"""Neutral-text suppression evaluation and the Task-7 smoke gate.

This module implements the metrics registered in ``docs/05_METRICS_AND_SCHEMAS.md``
without retaining a run's vocabulary-sized logits.  One baseline snapshot and one
intervention snapshot are live at a time; the caller writes scalar paired rows after
each forward.  The intervention itself remains in :mod:`neuron_sink.suppression`.

Task 7 is the first causal comparison in the project.  Discovery rows are diagnostic,
validation rows are kept separate, and :func:`evaluate_smoke_gate` only admits aggregate
rows whose stage is the locked ``test`` split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .selection import (
    CONTROL_TYPE_LAYER_RANDOM,
    CONTROL_TYPE_TARGETED,
    SMOKE_CONTROL_DRAWS,
    SMOKE_FRACTIONS_PERCENT,
    FrozenNeuronSets,
)
from .sink_metrics import differentiable_sink_score
from .suppression import NeuronSet


SMOKE_ALPHAS: tuple[float, ...] = (1.0, 0.5, 0.0)
SMOKE_SPLITS: tuple[str, ...] = ("discovery", "validation", "test")
EXPERIMENT_ID = "task7_gpt2_smoke"
EVALUATION_PROTOCOL = "neutral_next_token_sink_ce_kl_top1_v1"

PHENOMENON_ROW_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "model_id",
    "stage",
    "example_id",
    "condition_id",
    "condition_order",
    "alpha_order",
    "control_type",
    "control_seed",
    "fraction",
    "fraction_percent",
    "k",
    "alpha",
    "sink_baseline",
    "sink_intervened",
    "delta_sink",
    "relative_sink_reduction",
    "ce_baseline",
    "ce_intervened",
    "delta_ce",
    "ppl_baseline",
    "ppl_intervened",
    "kl_baseline_to_intervened",
    "top1_flip_rate",
    "prompt_tokens",
    "prediction_tokens",
    "logits_exact_match",
    "attentions_exact_match",
    "max_logits_abs_diff",
    "max_attention_abs_diff",
    "baseline_valid",
    "intervention_valid",
    "valid_forward",
    "nonfinite_logits",
    "nonfinite_attention",
    "all_zero_logits",
    "max_attention_row_sum_error",
    "max_causal_future_attention",
    "min_attention_value",
    "forward_runtime_seconds",
)

AGGREGATE_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "model_id",
    "stage",
    "condition_id",
    "condition_order",
    "alpha_order",
    "control_type",
    "control_seed",
    "fraction",
    "fraction_percent",
    "k",
    "alpha",
    "n_examples",
    "sink_baseline",
    "sink_intervened",
    "delta_sink",
    "relative_sink_reduction",
    "mean_per_example_relative_sink_reduction",
    "ce_baseline",
    "ce_intervened",
    "delta_ce",
    "ppl_baseline",
    "ppl_intervened",
    "kl_baseline_to_intervened",
    "top1_flip_rate",
    "identity_exact_all",
    "valid_forward_all",
    "max_logits_abs_diff",
    "max_attention_abs_diff",
    "max_attention_row_sum_error",
    "max_causal_future_attention",
    "nonfinite_logits",
    "nonfinite_attention",
    "all_zero_logits_count",
    "runtime_seconds",
)


class EvaluationError(RuntimeError):
    """Raised when the registered evaluation grid or outputs are malformed."""


@dataclass(frozen=True)
class SmokeCondition:
    """One verified Task-6 condition in its frozen Task-7 execution order."""

    condition_id: str
    condition_order: int
    control_type: str
    control_seed: int | None
    fraction_percent: float
    k: int
    neuron_set: NeuronSet


@dataclass(frozen=True)
class ForwardSnapshot:
    """Live tensors plus scalar validity diagnostics for one model forward."""

    logits: torch.Tensor
    attentions: tuple[torch.Tensor, ...]
    sink: float
    ce: float
    valid: bool
    nonfinite_logits: int
    nonfinite_attention: int
    all_zero_logits: bool
    max_attention_row_sum_error: float
    max_causal_future_attention: float
    min_attention_value: float


def registered_smoke_conditions(frozen: FrozenNeuronSets) -> tuple[SmokeCondition, ...]:
    """Validate and return the exact 3-target/15-control frozen smoke grid."""

    document = frozen.document
    fractions = tuple(float(value) for value in document.get("fractions_percent", ()))
    if fractions != SMOKE_FRACTIONS_PERCENT:
        raise EvaluationError(
            f"Frozen fractions {fractions} do not match registered smoke grid "
            f"{SMOKE_FRACTIONS_PERCENT}"
        )
    if int(document.get("control_draws", -1)) != SMOKE_CONTROL_DRAWS:
        raise EvaluationError(
            f"Frozen control_draws={document.get('control_draws')} does not match "
            f"registered smoke count {SMOKE_CONTROL_DRAWS}"
        )

    raw_records = document.get("conditions")
    condition_ids = tuple(document.get("condition_ids", ()))
    if not isinstance(raw_records, Mapping):
        raise EvaluationError("Frozen neuron-set document has no conditions mapping")

    expected_order: list[str] = []
    for fraction in SMOKE_FRACTIONS_PERCENT:
        label = f"f{fraction:.2f}".replace(".", "p")
        expected_order.append(f"targeted_{label}")
        expected_order.extend(
            f"layer_random_{label}_s{seed}" for seed in range(SMOKE_CONTROL_DRAWS)
        )
    if condition_ids != tuple(expected_order):
        raise EvaluationError(
            "Frozen condition order differs from the registered Task-6 order: "
            f"{condition_ids} != {tuple(expected_order)}"
        )

    conditions: list[SmokeCondition] = []
    for order, condition_id in enumerate(condition_ids, start=1):
        record = raw_records[condition_id]
        control_type = str(record["control_type"])
        seed = record.get("control_seed")
        if seed is not None:
            seed = int(seed)
        conditions.append(SmokeCondition(
            condition_id=condition_id,
            condition_order=order,
            control_type=control_type,
            control_seed=seed,
            fraction_percent=float(record["fraction_percent"]),
            k=int(record["k"]),
            neuron_set=frozen.neuron_sets[condition_id],
        ))

    for fraction in SMOKE_FRACTIONS_PERCENT:
        same = [condition for condition in conditions
                if condition.fraction_percent == fraction]
        targets = [condition for condition in same
                   if condition.control_type == CONTROL_TYPE_TARGETED]
        controls = [condition for condition in same
                    if condition.control_type == CONTROL_TYPE_LAYER_RANDOM]
        if len(targets) != 1 or len(controls) != SMOKE_CONTROL_DRAWS:
            raise EvaluationError(
                f"Fraction {fraction}% needs one target and {SMOKE_CONTROL_DRAWS} "
                f"controls, got {len(targets)} and {len(controls)}"
            )
        if {condition.control_seed for condition in controls} != set(
            range(SMOKE_CONTROL_DRAWS)
        ):
            raise EvaluationError(
                f"Fraction {fraction}% does not contain control seeds "
                f"0..{SMOKE_CONTROL_DRAWS - 1}"
            )
    return tuple(conditions)


def _attention_diagnostics(
    attentions: Sequence[torch.Tensor],
) -> tuple[int, float, float, float]:
    for layer, attention in enumerate(attentions):
        if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
            raise EvaluationError(
                f"Layer {layer} attention must be [batch, heads, seq, seq], got "
                f"{type(attention).__name__} {getattr(attention, 'shape', None)}"
            )
    # Stack once so validity uses four device synchronisations per forward rather than
    # four per layer. GPT-2-small's full attention stack is under 1 MiB at seq_len=40.
    stacked = torch.stack(tuple(attentions), dim=0)
    return (
        int((~torch.isfinite(stacked)).sum().item()),
        float((stacked.sum(dim=-1) - 1.0).abs().max().item()),
        float(torch.triu(stacked, diagonal=1).abs().max().item()),
        float(stacked.min().item()),
    )


def forward_snapshot(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    sink_layers: Sequence[int],
    attention_tolerance: float,
    causal_tolerance: float,
) -> ForwardSnapshot:
    """Run one inference-only forward and compute registered scalar diagnostics."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError(
            f"input_ids must have shape [1, sequence>=2], got {tuple(input_ids.shape)}"
        )
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
        logits = output.logits.detach()
        attentions = tuple(attention.detach() for attention in output.attentions)
        sink = float(
            differentiable_sink_score(attentions, layers=sink_layers).detach().item()
        )
        # Metric arithmetic is float32 even when the registered forward is bfloat16.
        # This does not change model outputs; it prevents large-vocabulary CE/KL
        # reductions from inheriting avoidable low-precision accumulation error.
        shifted_logits = logits[:, :-1, :].float().contiguous()
        shifted_labels = input_ids[:, 1:].contiguous()
        ce = float(F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            reduction="mean",
        ).item())

    nonfinite_logits = int((~torch.isfinite(logits)).sum().item())
    all_zero_logits = bool(torch.count_nonzero(logits).item() == 0)
    (
        nonfinite_attention,
        max_row_error,
        max_future,
        min_attention,
    ) = _attention_diagnostics(attentions)
    valid = bool(
        math.isfinite(sink)
        and math.isfinite(ce)
        and nonfinite_logits == 0
        and nonfinite_attention == 0
        and not all_zero_logits
        and max_row_error <= attention_tolerance
        and max_future <= causal_tolerance
        and min_attention >= -attention_tolerance
    )
    return ForwardSnapshot(
        logits=logits,
        attentions=attentions,
        sink=sink,
        ce=ce,
        valid=valid,
        nonfinite_logits=nonfinite_logits,
        nonfinite_attention=nonfinite_attention,
        all_zero_logits=all_zero_logits,
        max_attention_row_sum_error=max_row_error,
        max_causal_future_attention=max_future,
        min_attention_value=min_attention,
    )


def _max_attention_difference(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> float:
    if len(left) != len(right):
        raise EvaluationError(
            f"Attention layer counts differ: {len(left)} != {len(right)}"
        )
    return float(
        (torch.stack(tuple(left), dim=0) - torch.stack(tuple(right), dim=0))
        .abs()
        .max()
        .item()
    )


def paired_metrics(
    baseline: ForwardSnapshot, intervention: ForwardSnapshot
) -> dict[str, Any]:
    """Compare two snapshots over the registered next-token positions."""

    if baseline.logits.shape != intervention.logits.shape:
        raise EvaluationError(
            f"Logit shapes differ: {baseline.logits.shape} != "
            f"{intervention.logits.shape}"
        )
    baseline_next = baseline.logits[:, :-1, :].float()
    intervention_next = intervention.logits[:, :-1, :].float()
    with torch.inference_mode():
        baseline_log_probs = F.log_softmax(baseline_next, dim=-1)
        intervention_log_probs = F.log_softmax(intervention_next, dim=-1)
        kl_per_token = torch.sum(
            baseline_log_probs.exp()
            * (baseline_log_probs - intervention_log_probs),
            dim=-1,
        )
        kl = max(0.0, float(kl_per_token.mean().item()))
        flip = float(
            (baseline_next.argmax(dim=-1) != intervention_next.argmax(dim=-1))
            .to(dtype=torch.float32)
            .mean()
            .item()
        )

    max_logits_diff = float(
        (baseline.logits - intervention.logits).abs().max().item()
    )
    max_attention_diff = _max_attention_difference(
        baseline.attentions, intervention.attentions
    )
    return {
        "sink_baseline": baseline.sink,
        "sink_intervened": intervention.sink,
        "delta_sink": intervention.sink - baseline.sink,
        "relative_sink_reduction": (
            (baseline.sink - intervention.sink) / max(baseline.sink, 1e-12)
        ),
        "ce_baseline": baseline.ce,
        "ce_intervened": intervention.ce,
        "delta_ce": intervention.ce - baseline.ce,
        "ppl_baseline": math.exp(baseline.ce),
        "ppl_intervened": math.exp(intervention.ce),
        "kl_baseline_to_intervened": kl,
        "top1_flip_rate": flip,
        "logits_exact_match": max_logits_diff == 0.0,
        "attentions_exact_match": max_attention_diff == 0.0,
        "max_logits_abs_diff": max_logits_diff,
        "max_attention_abs_diff": max_attention_diff,
        "baseline_valid": baseline.valid,
        "intervention_valid": intervention.valid,
        "valid_forward": baseline.valid and intervention.valid,
        "nonfinite_logits": (
            baseline.nonfinite_logits + intervention.nonfinite_logits
        ),
        "nonfinite_attention": (
            baseline.nonfinite_attention + intervention.nonfinite_attention
        ),
        "all_zero_logits": intervention.all_zero_logits,
        "max_attention_row_sum_error": max(
            baseline.max_attention_row_sum_error,
            intervention.max_attention_row_sum_error,
        ),
        "max_causal_future_attention": max(
            baseline.max_causal_future_attention,
            intervention.max_causal_future_attention,
        ),
        "min_attention_value": min(
            baseline.min_attention_value, intervention.min_attention_value
        ),
        "prediction_tokens": int(baseline_next.shape[1]),
    }


def validate_phenomenon_row(
    row: Mapping[str, Any],
    *,
    allowed_stages: Sequence[str] = SMOKE_SPLITS,
    allowed_alphas: Sequence[float] = SMOKE_ALPHAS,
) -> None:
    """Validate one machine-readable row before it is saved or aggregated.

    The defaults preserve Task 7's exact smoke contract.  Stage B supplies its larger
    registered stage/alpha grids explicitly; accepting those values is never implicit.
    """

    missing = [field for field in PHENOMENON_ROW_FIELDS if field not in row]
    if missing:
        raise EvaluationError(f"Per-example row is missing fields: {missing}")
    stages = tuple(str(value) for value in allowed_stages)
    alphas = tuple(float(value) for value in allowed_alphas)
    if row["stage"] not in stages:
        raise EvaluationError(
            f"Stage {row['stage']!r} is not in the registered grid {stages}"
        )
    if float(row["alpha"]) not in alphas:
        raise EvaluationError(
            f"Alpha {row['alpha']!r} is not in the registered grid {alphas}"
        )
    if int(row["prompt_tokens"]) < 2:
        raise EvaluationError("prompt_tokens must be at least 2")
    if int(row["prediction_tokens"]) != int(row["prompt_tokens"]) - 1:
        raise EvaluationError("prediction_tokens must equal prompt_tokens - 1")
    numeric = (
        "sink_baseline", "sink_intervened", "delta_sink",
        "relative_sink_reduction", "ce_baseline", "ce_intervened", "delta_ce",
        "ppl_baseline", "ppl_intervened", "kl_baseline_to_intervened",
        "top1_flip_rate", "max_logits_abs_diff", "max_attention_abs_diff",
        "max_attention_row_sum_error", "max_causal_future_attention",
        "min_attention_value", "forward_runtime_seconds",
    )
    nonfinite = [field for field in numeric if not math.isfinite(float(row[field]))]
    if nonfinite:
        raise EvaluationError(f"Per-example row has non-finite fields: {nonfinite}")
    if float(row["kl_baseline_to_intervened"]) < 0.0:
        raise EvaluationError("KL must be non-negative")
    if not 0.0 <= float(row["top1_flip_rate"]) <= 1.0:
        raise EvaluationError("top1_flip_rate must be in [0, 1]")


def aggregate_phenomenon_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_stages: Sequence[str] = SMOKE_SPLITS,
    allowed_alphas: Sequence[float] = SMOKE_ALPHAS,
) -> list[dict[str, Any]]:
    """Aggregate paired rows by split, condition, and alpha."""

    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    order: list[tuple[str, str, float]] = []
    for row in rows:
        validate_phenomenon_row(
            row, allowed_stages=allowed_stages, allowed_alphas=allowed_alphas
        )
        key = (str(row["stage"]), str(row["condition_id"]), float(row["alpha"]))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    aggregates: list[dict[str, Any]] = []
    for key in order:
        members = grouped[key]
        first = members[0]
        sink_baseline = float(np.mean([float(row["sink_baseline"]) for row in members]))
        sink_intervened = float(np.mean([
            float(row["sink_intervened"]) for row in members
        ]))
        ce_baseline = float(np.mean([float(row["ce_baseline"]) for row in members]))
        ce_intervened = float(np.mean([
            float(row["ce_intervened"]) for row in members
        ]))
        aggregates.append({
            "experiment_id": first["experiment_id"],
            "model_id": first["model_id"],
            "stage": first["stage"],
            "condition_id": first["condition_id"],
            "condition_order": first["condition_order"],
            "alpha_order": first["alpha_order"],
            "control_type": first["control_type"],
            "control_seed": first["control_seed"],
            "fraction": first["fraction"],
            "fraction_percent": first["fraction_percent"],
            "k": first["k"],
            "alpha": first["alpha"],
            "n_examples": len(members),
            "sink_baseline": sink_baseline,
            "sink_intervened": sink_intervened,
            "delta_sink": sink_intervened - sink_baseline,
            "relative_sink_reduction": (
                (sink_baseline - sink_intervened) / max(sink_baseline, 1e-12)
            ),
            "mean_per_example_relative_sink_reduction": float(np.mean([
                float(row["relative_sink_reduction"]) for row in members
            ])),
            "ce_baseline": ce_baseline,
            "ce_intervened": ce_intervened,
            "delta_ce": ce_intervened - ce_baseline,
            "ppl_baseline": math.exp(ce_baseline),
            "ppl_intervened": math.exp(ce_intervened),
            "kl_baseline_to_intervened": float(np.mean([
                float(row["kl_baseline_to_intervened"]) for row in members
            ])),
            "top1_flip_rate": float(np.mean([
                float(row["top1_flip_rate"]) for row in members
            ])),
            "identity_exact_all": bool(all(
                bool(row["logits_exact_match"])
                and bool(row["attentions_exact_match"])
                for row in members
            )),
            "valid_forward_all": bool(all(
                bool(row["valid_forward"]) for row in members
            )),
            "max_logits_abs_diff": max(
                float(row["max_logits_abs_diff"]) for row in members
            ),
            "max_attention_abs_diff": max(
                float(row["max_attention_abs_diff"]) for row in members
            ),
            "max_attention_row_sum_error": max(
                float(row["max_attention_row_sum_error"]) for row in members
            ),
            "max_causal_future_attention": max(
                float(row["max_causal_future_attention"]) for row in members
            ),
            "nonfinite_logits": sum(
                int(row["nonfinite_logits"]) for row in members
            ),
            "nonfinite_attention": sum(
                int(row["nonfinite_attention"]) for row in members
            ),
            "all_zero_logits_count": sum(
                int(bool(row["all_zero_logits"])) for row in members
            ),
            "runtime_seconds": float(sum(
                float(row["forward_runtime_seconds"]) for row in members
            )),
        })
    return aggregates


def evaluate_smoke_gate(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    all_split_identity_pass: bool,
    all_split_validity_pass: bool,
    state_leakage_pass: bool,
    registered_run: bool,
    dose_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate the predefined Phase-1 gate on the held-out test aggregates only."""

    if not registered_run:
        return {
            "status": "NOT_EVALUATED_DRY_RUN",
            "registered_run": False,
            "test_split_only": True,
        }

    test_rows = [row for row in aggregates if row["stage"] == "test"]
    indexed = {
        (str(row["condition_id"]), float(row["alpha"])): row
        for row in test_rows
        if row["control_type"] != "baseline"
    }
    expected_cells = 18 * len(SMOKE_ALPHAS)
    if len(indexed) != expected_cells:
        raise EvaluationError(
            f"Held-out gate needs {expected_cells} intervention cells, got "
            f"{len(indexed)}"
        )

    superiority: list[dict[str, Any]] = []
    dose_direction: list[dict[str, Any]] = []
    for fraction in SMOKE_FRACTIONS_PERCENT:
        label = f"f{fraction:.2f}".replace(".", "p")
        target_id = f"targeted_{label}"
        for alpha in (0.5, 0.0):
            target = indexed[(target_id, alpha)]
            controls = [
                indexed[(f"layer_random_{label}_s{seed}", alpha)]
                for seed in range(SMOKE_CONTROL_DRAWS)
            ]
            target_rsr = float(target["relative_sink_reduction"])
            control_rsr = [
                float(control["relative_sink_reduction"]) for control in controls
            ]
            superiority.append({
                "fraction_percent": fraction,
                "k": int(target["k"]),
                "alpha": alpha,
                "target_condition_id": target_id,
                "target_relative_sink_reduction": target_rsr,
                "random_relative_sink_reductions": control_rsr,
                "max_random_relative_sink_reduction": max(control_rsr),
                "target_minus_max_random": target_rsr - max(control_rsr),
                "passes_all_five_controls": bool(target_rsr > max(control_rsr)),
            })

        rsr_identity = float(indexed[(target_id, 1.0)]["relative_sink_reduction"])
        rsr_half = float(indexed[(target_id, 0.5)]["relative_sink_reduction"])
        rsr_full = float(indexed[(target_id, 0.0)]["relative_sink_reduction"])
        between_endpoints = bool(
            min(rsr_identity, rsr_full) - dose_tolerance
            <= rsr_half
            <= max(rsr_identity, rsr_full) + dose_tolerance
        )
        registered_direction = bool(
            rsr_identity <= rsr_half + dose_tolerance
            and rsr_half <= rsr_full + dose_tolerance
        )
        dose_direction.append({
            "fraction_percent": fraction,
            "k": int(indexed[(target_id, 0.0)]["k"]),
            "relative_sink_reduction_alpha_1": rsr_identity,
            "relative_sink_reduction_alpha_0p5": rsr_half,
            "relative_sink_reduction_alpha_0": rsr_full,
            "half_between_identity_and_full": between_endpoints,
            "registered_reduction_direction": registered_direction,
        })

    superiority_pass = any(row["passes_all_five_controls"] for row in superiority)
    dose_pass = any(row["registered_reduction_direction"] for row in dose_direction)
    gate_pass = bool(
        superiority_pass
        and dose_pass
        and all_split_identity_pass
        and all_split_validity_pass
        and state_leakage_pass
    )
    return {
        "status": "PASS" if gate_pass else "NULL_OR_INVALID",
        "registered_run": True,
        "test_split_only": True,
        "criterion": (
            "at least one targeted non-identity condition has held-out relative sink "
            "reduction greater than every one of its five layer-count-matched random "
            "controls; at least one target has alpha=0.5 directionally between identity "
            "and full suppression; identity, validity, and state-leakage checks pass"
        ),
        "causal_superiority_pass": superiority_pass,
        "dose_direction_pass": dose_pass,
        "all_split_identity_pass": all_split_identity_pass,
        "all_split_validity_pass": all_split_validity_pass,
        "state_leakage_pass": state_leakage_pass,
        "smoke_gate_pass": gate_pass,
        "superiority_conditions": superiority,
        "dose_direction": dose_direction,
        "passing_superiority_conditions": [
            row for row in superiority if row["passes_all_five_controls"]
        ],
    }
