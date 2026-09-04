#!/usr/bin/env python
"""Task 6: global top-k selection and layer-count-matched random controls.

This is CPU-only bookkeeping over Task 5's frozen discovery ranking.  It does not load a
model, run a forward pass, open a corpus split, suppress a neuron, or evaluate a held-out
example.  The three smoke fractions and five random draws are frozen for Task 7; the same
selection API is also exercised with all six fractions and twenty draws for Stage-B reuse.

Attribution is a ranking heuristic, not causal evidence.  The neuron sets written here are
intervention candidates; Task 7's held-out targeted-vs-random suppression is the first
causal test.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuron_sink.attribution import RANKING_SCORE  # noqa: E402
from neuron_sink.provenance import (  # noqa: E402
    ProvenanceRecorder,
    git,
    prepare_output_dir,
    read_json,
    require_pinned_submodules,
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
    SMOKE_CONTROL_DRAWS,
    SMOKE_FRACTIONS_PERCENT,
    build_neuron_sets_document,
    build_selection_conditions,
    condition_rows,
    exact_k,
    load_frozen_attribution,
    load_frozen_neuron_sets,
    verify_neuron_sets_document,
)
from neuron_sink.sink_metrics import load_frozen_sink_scope  # noqa: E402


FROZEN_DIR = ROOT / "configs" / "frozen"
FROZEN_MANIFEST = FROZEN_DIR / "neutral_corpus_manifest.json"
FROZEN_SCOPE = FROZEN_DIR / "sink_scope.json"
FROZEN_ATTRIBUTION = FROZEN_DIR / "neuron_attribution.csv"
FROZEN_ATTRIBUTION_META = FROZEN_DIR / "neuron_attribution_metadata.json"
FROZEN_NEURON_SETS = FROZEN_DIR / "neuron_sets.json"

EXPECTED_CORPUS_SHA256 = "c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7"
EXPECTED_SCOPE_SHA256 = "b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235"
EXPECTED_ATTRIBUTION_SHA256 = "9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692"
EXPECTED_ELIGIBLE_LAYERS = tuple(range(10))
EXPECTED_MLP_WIDTH = 3072
EXPECTED_POOL_SIZE = 30_720


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select global top-k neurons and matched random controls (CPU only)."
    )
    parser.add_argument("--manifest", type=Path, default=FROZEN_MANIFEST)
    parser.add_argument("--scope", type=Path, default=FROZEN_SCOPE)
    parser.add_argument("--attribution", type=Path, default=FROZEN_ATTRIBUTION)
    parser.add_argument("--attribution-metadata", type=Path,
                        default=FROZEN_ATTRIBUTION_META)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-freeze", action="store_true")
    return parser.parse_args()


def _condition_signature(condition) -> tuple:
    return tuple(
        (layer, tuple(neurons))
        for layer, neurons in condition.neuron_set.by_layer.items()
    )


def _selection_checks(ranking, conditions) -> dict[str, Any]:
    targets = [condition for condition in conditions if condition.control_type == "targeted"]
    controls = [
        condition for condition in conditions if condition.control_type == "layer_random"
    ]
    target_by_fraction = {condition.fraction_percent: condition for condition in targets}

    exact_k_all = {
        f"{fraction:.2f}": exact_k(fraction, ranking.pool_size)
        for fraction in FULL_FRACTIONS_PERCENT
    }
    k_check = exact_k_all == {
        "0.01": 3,
        "0.05": 15,
        "0.10": 31,
        "0.25": 77,
        "0.50": 154,
        "1.00": 307,
    }

    target_pairs = {
        condition.fraction_percent: {
            (layer, neuron)
            for layer, neurons in condition.neuron_set.by_layer.items()
            for neuron in neurons
        }
        for condition in targets
    }
    ordered_pairs = [target_pairs[fraction] for fraction in SMOKE_FRACTIONS_PERCENT]
    nested = all(left < right for left, right in zip(ordered_pairs, ordered_pairs[1:]))

    controls_match = True
    controls_exclude = True
    ids_valid = True
    draws_distinct = True
    for fraction in SMOKE_FRACTIONS_PERCENT:
        target = target_by_fraction[fraction]
        same_fraction = [
            condition for condition in controls
            if condition.fraction_percent == fraction
        ]
        signatures = {_condition_signature(condition) for condition in same_fraction}
        draws_distinct &= len(signatures) == SMOKE_CONTROL_DRAWS
        for condition in same_fraction:
            controls_match &= condition.per_layer_counts == target.per_layer_counts
            for layer, neurons in condition.neuron_set.by_layer.items():
                controls_exclude &= not bool(
                    set(neurons) & set(target.neuron_set.by_layer.get(layer, ()))
                )
                ids_valid &= layer in ranking.eligible_mlp_layers
                ids_valid &= len(neurons) == len(set(neurons))
                ids_valid &= all(
                    0 <= neuron < ranking.mlp_width[layer] for neuron in neurons
                )

    # Acceptance criterion: the reusable API can generate 20 deterministic draws.  These
    # are checked but not frozen in the smoke artefact.
    twenty_a = build_selection_conditions(
        ranking,
        FULL_FRACTIONS_PERCENT,
        control_draws=FULL_CONTROL_DRAWS,
        base_seed=REGISTERED_BASE_SEED,
    )
    twenty_b = build_selection_conditions(
        ranking,
        FULL_FRACTIONS_PERCENT,
        control_draws=FULL_CONTROL_DRAWS,
        base_seed=REGISTERED_BASE_SEED,
    )
    twenty_deterministic = condition_rows(twenty_a) == condition_rows(twenty_b)

    return {
        "frozen_input_pool_size_pass": ranking.pool_size == EXPECTED_POOL_SIZE,
        "frozen_input_eligible_layers_pass": (
            ranking.eligible_mlp_layers == EXPECTED_ELIGIBLE_LAYERS
        ),
        "frozen_input_widths_pass": all(
            width == EXPECTED_MLP_WIDTH for width in ranking.mlp_width.values()
        ),
        "exact_k_all_registered": exact_k_all,
        "exact_k_pass": k_check,
        "rounding_rule": ROUNDING_RULE,
        "targeted_sets_nested": nested,
        "control_layer_counts_match": controls_match,
        "controls_exclude_own_target": controls_exclude,
        "control_ids_unique_in_range_and_eligible": ids_valid,
        "five_draws_distinct_per_fraction": draws_distinct,
        "twenty_draw_api_deterministic": twenty_deterministic,
        "twenty_draw_api_condition_count": len(twenty_a),
        "no_model_forward_or_corpus_split_read": True,
    }


def main() -> int:
    args = _parse_args()
    output_dir = prepare_output_dir(
        args.output_dir or ROOT / "results" / "task6_selection" / run_stamp()
    )
    submodule_commits = require_pinned_submodules()
    repo_commit = git("rev-parse", "HEAD")
    recorder = ProvenanceRecorder(device=None, gpu_name="cpu")

    manifest = read_json(args.manifest)
    if not isinstance(manifest, Mapping):
        raise SystemExit(f"{args.manifest} does not contain a JSON object")
    corpus_sha = str(manifest.get("manifest_sha256", ""))
    if corpus_sha != EXPECTED_CORPUS_SHA256:
        raise SystemExit(
            f"Frozen corpus hash {corpus_sha} != registered Task-4 hash "
            f"{EXPECTED_CORPUS_SHA256}"
        )

    scope = load_frozen_sink_scope(
        args.scope, expected_corpus_manifest_sha256=corpus_sha
    )
    if scope.sink_scope_sha256 != EXPECTED_SCOPE_SHA256:
        raise SystemExit(
            f"Frozen scope hash {scope.sink_scope_sha256} != registered Task-4 hash "
            f"{EXPECTED_SCOPE_SHA256}"
        )

    ranking = load_frozen_attribution(
        args.attribution,
        args.attribution_metadata,
        scope=scope,
        expected_corpus_manifest_sha256=corpus_sha,
    )
    if ranking.attribution_sha256 != EXPECTED_ATTRIBUTION_SHA256:
        raise SystemExit(
            f"Frozen attribution hash {ranking.attribution_sha256} != registered Task-5 "
            f"hash {EXPECTED_ATTRIBUTION_SHA256}"
        )

    print(
        f"Loaded {ranking.pool_size} typed attribution rows; hash reproduced "
        f"{ranking.attribution_sha256}"
    )
    conditions = build_selection_conditions(
        ranking,
        SMOKE_FRACTIONS_PERCENT,
        control_draws=SMOKE_CONTROL_DRAWS,
        base_seed=REGISTERED_BASE_SEED,
    )
    document = build_neuron_sets_document(
        ranking,
        conditions,
        fractions_percent=SMOKE_FRACTIONS_PERCENT,
        control_draws=SMOKE_CONTROL_DRAWS,
        base_seed=REGISTERED_BASE_SEED,
    )
    # Round-trip through the exact representation Task 7 will load.
    verified = verify_neuron_sets_document(document)
    flat_rows = condition_rows(conditions)
    checks = _selection_checks(ranking, conditions)
    task_pass = (
        len(conditions) == 18
        and len(verified.neuron_sets) == 18
        and all(value is True for key, value in checks.items() if key.endswith("pass"))
        and checks["targeted_sets_nested"] is True
        and checks["control_layer_counts_match"] is True
        and checks["controls_exclude_own_target"] is True
        and checks["control_ids_unique_in_range_and_eligible"] is True
        and checks["five_draws_distinct_per_fraction"] is True
        and checks["twenty_draw_api_deterministic"] is True
    )

    target_summaries: dict[str, Any] = {}
    for condition in conditions:
        if condition.control_type != "targeted":
            continue
        target_summaries[condition.condition_id] = {
            "fraction_percent": condition.fraction_percent,
            "k": condition.k,
            "per_layer_counts": {
                str(layer): count
                for layer, count in condition.per_layer_counts.items()
            },
            "ranked_pairs": [
                [layer, neuron]
                for layer, neurons in condition.neuron_set.by_layer.items()
                for neuron in neurons
            ],
        }

    run_config = {
        "experiment_id": "task6_selection",
        "stage": "discovery",
        "operation": "global_top_k_and_layer_matched_random_selection",
        "model_id": ranking.model_id,
        "model_revision": ranking.model_revision,
        "tokenizer_id": manifest.get("tokenizer_name"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "dtype": None,
        "device": "cpu",
        "gpu_name": "cpu",
        "seed": REGISTERED_BASE_SEED,
        "dataset_id": manifest.get("source", {}).get("dataset_id"),
        "dataset_config": None,
        "dataset_split": "frozen_discovery_ranking_only_no_examples_read",
        "manifest_sha256": corpus_sha,
        "attribution_sha256": ranking.attribution_sha256,
        "sink_scope_sha256": scope.sink_scope_sha256,
        "seq_len": manifest.get("cut_length"),
        "sink_target_position": scope.document.get("target_position"),
        "sink_query_rule": scope.document.get("query_rule"),
        "sink_layers": list(scope.sink_layers),
        "sink_heads": None,
        "eligible_mlp_layers": list(ranking.eligible_mlp_layers),
        "eligible_pool_size": ranking.pool_size,
        "neuron_definition": "mlp_intermediate_pre_output_projection",
        "selection_method": SELECTION_METHOD,
        "ranking_score": RANKING_SCORE,
        "fractions_percent": list(SMOKE_FRACTIONS_PERCENT),
        "k_by_fraction_percent": {
            f"{fraction:.2f}": exact_k(fraction, ranking.pool_size)
            for fraction in SMOKE_FRACTIONS_PERCENT
        },
        "rounding_rule": ROUNDING_RULE,
        "neuron_fraction": None,
        "k": None,
        "alpha": None,
        "control_type": "targeted|layer_random_grid",
        "condition_control_types": ["targeted", "layer_random"],
        "control_draws": SMOKE_CONTROL_DRAWS,
        "control_seed": None,
        "control_seed_range": list(range(SMOKE_CONTROL_DRAWS)),
        "control_rng": CONTROL_RNG,
        "control_seed_derivation": CONTROL_SEED_DERIVATION,
        "model_loaded": False,
        "corpus_examples_read": False,
    }
    summary = {
        "task6_selection": "PASS" if task_pass else "FAIL",
        "neuron_sets_sha256": document["neuron_sets_sha256"],
        "attribution_sha256_reproduced": ranking.attribution_sha256,
        "eligible_pool_size": ranking.pool_size,
        "n_conditions": len(conditions),
        "n_targeted_conditions": 3,
        "n_layer_random_conditions": 15,
        "targeted": target_summaries,
        "checks": checks,
        "is_causal_evidence": False,
    }

    write_json(output_dir / "neuron_sets.json", document)
    with (output_dir / "neuron_sets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONDITION_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(flat_rows)
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "summary.json", summary)
    provenance = recorder.finish(
        repo_commit=repo_commit, submodule_commits=submodule_commits
    )
    write_json(output_dir / "provenance.json", provenance)

    if not args.no_freeze and task_pass:
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        if FROZEN_NEURON_SETS.exists():
            existing = load_frozen_neuron_sets(FROZEN_NEURON_SETS)
            existing_sha = existing.document["neuron_sets_sha256"]
            if existing_sha != document["neuron_sets_sha256"]:
                raise SystemExit(
                    f"{FROZEN_NEURON_SETS} already holds different frozen neuron sets "
                    f"({existing_sha}). A frozen artefact is immutable; register a new "
                    "experiment id rather than overwriting it."
                )
            print(f"frozen neuron sets already match: {FROZEN_NEURON_SETS}")
        else:
            write_json(FROZEN_NEURON_SETS, document)
            print(f"frozen neuron sets written: {FROZEN_NEURON_SETS}")

    print(f"TASK6_SELECTION={'PASS' if task_pass else 'FAIL'}")
    print(f"attribution_sha256={ranking.attribution_sha256}")
    print(f"eligible_pool_size={ranking.pool_size}")
    for condition_id, target in target_summaries.items():
        print(
            f"  {condition_id}: k={target['k']} "
            f"per_layer_counts={target['per_layer_counts']}"
        )
    print(f"conditions={len(conditions)} (3 targeted + 15 layer-random)")
    print(f"neuron_sets_sha256={document['neuron_sets_sha256']}")
    print(f"twenty_draw_api_deterministic={checks['twenty_draw_api_deterministic']}")
    print(f"wall_seconds={provenance['runtime_seconds']:.3f}")
    print(f"output_dir={output_dir}")
    return 0 if task_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
