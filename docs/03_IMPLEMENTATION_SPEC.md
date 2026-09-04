# Implementation Specification

This file describes the intended code architecture. It is a design contract, not implementation.

## New package layout

Create a project package, for example:

```text
neuron_sink/
  __init__.py
  provenance.py
  model_adapters.py
  sink_metrics.py
  corpus.py
  attribution.py
  selection.py
  suppression.py
  evaluation.py
  stats.py
  schemas.py
  task_eval/
    __init__.py
    base.py
    mmlu.py
    arc.py
    culturalbench.py
    gsm8k.py
scripts/
  verify_upstream_parity.py
  prepare_manifests.py
  map_sink_layers.py
  rank_neurons.py
  run_suppression_sweep.py
  choose_operating_point.py
  run_task_drift.py
  run_sink_kd_comparison.py
  aggregate_results.py
configs/
  experiment_plan.yaml
  downstream_tasks.yaml
```

Do not place new implementation inside the pinned submodules.

## 1. Model adapters

Expose a common interface for GPT-2 and Qwen2.5.

Required adapter methods/properties:

```python
model_id: str
num_layers: int
mlp_width(layer: int) -> int
attention_geometry(layer: int) -> dict
get_mlp_intermediate_path(layer: int)
get_attention_probabilities_path(layer: int)
eligible_mlp_layers(sink_layers: list[int]) -> list[int]
```

### GPT-2 neuron hook

Primary hook point: tensor entering `transformer.h[layer].mlp.c_proj`.

This is the post-activation MLP intermediate vector. Verify exact NNsight path against the pinned Transformers 5.3.0 model structure before full runs.

### Qwen2.5 neuron hook

Primary hook point: tensor entering `model.layers[layer].mlp.down_proj`.

This is the post-SwiGLU gated intermediate vector. Verify exact NNsight path in the installed model implementation.

The adapter must fail loudly if the expected module path/shape is absent.

## 2. Sink metric adapter

The default sink scalar must call or exactly reproduce upstream semantics:

- attention to key position 0;
- query positions from `seq_len // 2` onward;
- average over registered attention layers and heads.

For parity, directly call the upstream `compute_bos_attention_metric` where possible.

For gradient attribution, implement a differentiable torch version that is numerically checked against the upstream scalar on fixed attention tensors. Do not use `.item()` before gradient construction.

Suggested API:

```python
def differentiable_sink_score(attentions, layers, heads=None, target_pos=0):
    ...
```

## 3. Sink-map discovery

`map_sink_layers.py` should:

1. run baseline only;
2. save per-example/per-layer/per-head position-0 received attention;
3. aggregate discovery split;
4. apply the registered sink-heavy layer rule;
5. freeze selected sink layers/heads to JSON;
6. hash the JSON and include it in later manifests.

Do not recompute sink-heavy layers separately for targeted and random conditions.

## 4. Attribution engine

Primary attribution target is the causal-order-aware `S_future(layer)`.

Implement layer-wise scoring to bound VRAM:

1. choose one eligible MLP layer `l`;
2. capture its MLP intermediate activation with gradients;
3. compute attention outputs in later registered sink-heavy layers;
4. build differentiable `S_future(l)`;
5. backprop once;
6. accumulate `abs(a * grad)` over examples and token positions;
7. release graph/tensors before moving to the next layer.

The scorer must never touch validation/test manifests.

Output one row per `(layer, neuron)`:

```text
layer
neuron
mean_abs_activation
mean_signed_attr
mean_abs_attr
n_examples
n_tokens
future_sink_layers
```

Save as Parquet or CSV plus a metadata JSON.

## 5. Global selection

`selection.py` should:

- merge eligible layer rankings;
- compute total number of eligible neurons;
- derive exact `k` for each registered fraction;
- select global top-k by `mean_abs_attr`;
- record per-layer counts;
- generate 20 layer-count-matched random sets with fixed seeds;
- optionally generate activation-matched controls only in the confirmatory stage.

A control set may not include a targeted neuron for that same top-k condition.

## 6. Suppression engine

Suggested API:

```python
@dataclass(frozen=True)
class NeuronSet:
    by_layer: dict[int, tuple[int, ...]]
    source: str  # targeted | layer_random | activation_random
    selection_seed: int | None


def run_with_suppression(model, inputs, neuron_set: NeuronSet, alpha: float):
    ...
```

At each selected layer, clone/edit only if needed and multiply selected MLP intermediate coordinates by alpha.

Requirements:

- `alpha=1` exact identity path test;
- no permanent weight mutation;
- no state leakage between examples/conditions;
- same model checkpoint object may be reused only if the hook context is guaranteed to unwind cleanly;
- attention must remain causal and normalized.

## 7. Neutral functional evaluation

For every suppression run compute:

- sink score;
- token-level CE on the same neutral example when labels are available;
- KL from baseline next-token distribution;
- top-1 next-token prediction flip rate.

Store paired per-example baseline and intervention metrics. Aggregate later.

## 8. Downstream adapters

The downstream evaluator should use one shared model/suppression wrapper so B0/T1/T2/R1 differ only by intervention state.

Prefer deterministic likelihood scoring for MMLU/ARC/CulturalBench-Easy. GSM8K requires deterministic greedy generation.

If `lm-evaluation-harness` is used, pin its commit/version and write a thin model adapter that applies NNsight suppression during `loglikelihood` and generation. Do not accept a benchmark implementation whose baseline cannot be reproduced twice deterministically.

CulturalBench is not currently guaranteed to exist in the chosen lm-eval version. A small direct adapter is acceptable, but its prompt/scoring must be frozen in `downstream_tasks.yaml` and unit-tested.

## 9. Statistics

`stats.py` should implement:

- paired bootstrap over examples;
- bootstrap CI for target-minus-random sink reduction;
- Spearman dose-response;
- task accuracy/EM paired bootstrap;
- random-control percentile calculation;
- MMLU/CulturalBench macro grouping.

Use NumPy RNG objects with explicit seeds; do not rely on module-global random state.

## 10. Sink-KD adapter

Do not reimplement Sink-KD training in the first milestone.

`run_sink_kd_comparison.py` should initially accept paths to completed upstream checkpoints and:

1. resolve the checkpoint/model config;
2. run the same neutral sink-map procedure;
3. rank each checkpoint's neurons independently;
4. run the same targeted/random suppression sweep;
5. compare effect profiles, not neuron ids.

If required checkpoints are absent, emit a clear `blocked_missing_checkpoint` record rather than silently training.

## 11. Output directory contract

Recommended structure:

```text
results/
  manifests/
  parity/
  phenomenon/
    gpt2/
    gpt2-medium/
    qwen2.5-1.5b-instruct/
  downstream/
    qwen2.5-1.5b-instruct/
      mmlu/
      arc_challenge/
      culturalbench_easy/
      culturalbench_hard/
      gsm8k/
  sink_kd/
  aggregate/
```

Every run directory must contain `run_config.json`, `provenance.json`, and machine-readable per-example results.

## 12. Implementation milestone order

Do not parallelize scientific stages that depend on gates.

M0. provenance + schema + manifest utilities.

M1. GPT-2 adapter + differentiable sink metric + parity tests.

M2. GPT-2 MLP hook + identity/zero suppression tests.

M3. discovery-only attribution scorer.

M4. top-k selection + layer-matched random controls.

M5. RTX 2060 SUPER smoke end-to-end.

M6. RTX 4080 SUPER full GPT-2-small/medium phenomenon runs.

M7. operating-point selection and locked test evaluation.

M8. Qwen2.5 adapter/preflight + independent neuron localization.

M9. downstream benchmark adapters and smoke tests.

M10. full downstream B0/T1/T2/R1 runs.

M11. Sink-KD checkpoint comparison.

M12. aggregation/figures.
