# Metrics and Result Schemas

## Core sink metrics

### Baseline sink

`S0 = mean attention probability to key position 0 from second-half query positions over the registered attention scope.`

### Intervened sink

`S1` is the same quantity under a neuron suppression condition.

### Absolute sink change

`delta_sink = S1 - S0`

Negative means the sink weakened.

### Relative sink reduction

`relative_sink_reduction = (S0 - S1) / max(S0, eps)`

Positive means the sink weakened. Use `eps=1e-12` only to avoid numerical division error; a model with negligible S0 should already have failed the sink preflight.

## Functional metrics on neutral text

- token CE baseline/intervened;
- `delta_ce = CE_intervened - CE_baseline`;
- PPL baseline/intervened, reported but not used for additive drift statistics;
- KL(`p_baseline || p_intervened`) over next-token distributions;
- top-1 token flip rate.

### Sink-selectivity ratio

Exploratory only:

`SSR = relative_sink_reduction / (abs(delta_ce) + 1e-4)`

Do not make this the sole headline metric because ratios become unstable near zero CE drift.

## Attribution metrics

Per neuron:

- `mean_abs_activation`;
- `mean_signed_attr = mean(a * grad)`;
- `mean_abs_attr = mean(abs(a * grad))` — primary ranking score;
- layer id;
- neuron id;
- list/hash of future sink layers;
- example/token count.

## Random-control metrics

For every targeted set, store all random control ids and results, not just their mean.

Aggregate:

- median random RSR;
- 5th/95th percentile random RSR;
- target minus median-random RSR;
- percentile rank of targeted RSR within random controls.

## Downstream metrics

### Multiple-choice tasks

- accuracy baseline/intervened;
- signed accuracy drift;
- prediction flip rate;
- correct->wrong rate;
- wrong->correct rate;
- paired bootstrap 95% CI;
- per-group macro drift where applicable.

### GSM8K

- exact-match baseline/intervened;
- signed EM drift;
- invalid-answer rate;
- correct->wrong and wrong->correct rates;
- generated token length change.

## Required run-level schema

Every run writes `run_config.json` with at least:

```json
{
  "experiment_id": "...",
  "stage": "parity|discovery|validation|test|downstream|sink_kd",
  "model_id": "...",
  "model_revision": "...",
  "tokenizer_id": "...",
  "tokenizer_revision": "...",
  "dtype": "float32|float16|bfloat16",
  "device": "cuda",
  "gpu_name": "...",
  "seed": 0,
  "dataset_id": "...",
  "dataset_config": "...",
  "dataset_split": "...",
  "manifest_sha256": "...",
  "seq_len": 40,
  "sink_target_position": 0,
  "sink_query_rule": "second_half",
  "sink_layers": [3, 4],
  "sink_heads": null,
  "neuron_definition": "mlp_intermediate_pre_output_projection",
  "selection_method": "abs_activation_x_sink_gradient",
  "neuron_fraction": 0.001,
  "k": 37,
  "alpha": 0.0,
  "control_type": "targeted|layer_random|activation_random|baseline",
  "control_seed": null
}
```

Exact field names may be extended, not removed.

## Required provenance schema

`provenance.json`:

```json
{
  "repo_commit": "...",
  "sink_repro_commit": "9ab67e914464b13863b67527d8ea14068ee9ff10",
  "sink_kd_commit": "db114c9c5eb6ffc5de13e444c783408ea7401c62",
  "python": "...",
  "torch": "...",
  "transformers": "...",
  "nnsight": "...",
  "cuda_runtime": "...",
  "gpu_name": "...",
  "command": "...",
  "started_at": "...",
  "finished_at": "...",
  "peak_memory_allocated_bytes": 0,
  "peak_memory_reserved_bytes": 0
}
```

## Per-example phenomenon schema

Recommended Parquet/CSV columns:

```text
experiment_id
model_id
stage
example_id
condition_id
control_type
control_seed
fraction
k
alpha
sink_baseline
sink_intervened
relative_sink_reduction
ce_baseline
ce_intervened
delta_ce
kl_baseline_to_intervened
top1_flip_rate
prompt_tokens
```

A separate `neuron_sets.json` maps each `condition_id` to layerwise neuron ids.

## Per-example downstream schema

For multiple choice:

```text
task
example_id
condition_id
gold
baseline_prediction
intervened_prediction
baseline_correct
intervened_correct
baseline_choice_scores
intervened_choice_scores
prompt_tokens
sink_baseline
sink_intervened
```

For GSM8K add:

```text
baseline_generation
intervened_generation
baseline_answer
intervened_answer
baseline_valid_answer
intervened_valid_answer
generation_tokens_baseline
generation_tokens_intervened
```

## Aggregated tables required for the paper

1. Baseline sink map by model/layer/head.
2. Targeted-vs-random sink reduction across neuron fractions.
3. Dose-response at `k*`.
4. Sink reduction vs CE drift scatter.
5. GPT-2-small vs GPT-2-medium replication table.
6. Qwen task-model preflight and causal gate.
7. Downstream task drift table B0/T1/T2/R1.
8. Optional Sink-KD teacher/student effect-profile comparison.
