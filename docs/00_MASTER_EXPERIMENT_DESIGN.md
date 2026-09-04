# Master Experiment Design

## Working title

**Same Sink, Different Neurons: Neuron-Level Causal Localization of Attention Sinks and Performance Drift**

## Core question

A model can display a strong first-position attention sink while the parameter-level route and functional importance of that sink differ across models. This project asks the next-level question:

> Is the sink supported by a sparse set of internal MLP neurons, and what happens to model function when those neurons are selectively suppressed?

The central distinction is between:

- **sink-forming units**: suppression strongly reduces the sink but produces little functional drift;
- **sink-functional units**: suppression reduces the sink and also changes language-model or task behavior;
- **general-purpose units**: suppression changes function without a sink-specific effect;
- **irrelevant units**: suppression materially changes neither.

This is not a project about "removing bad attention sinks." The sink may be useful. The project is about causal localization and the relationship between attention structure and function.

## Prior-code foundation

The design inherits, rather than redefines, the core attention-sink measurement from the pinned reproduction code:

- target position: position `0`;
- query positions: second half of the sequence;
- sink score: mean received attention to target position `0` over the registered attention-layer/head scope;
- GPT-2 depth band: preserve the upstream scaled/frozen band convention for parity runs.

The new work adds neuron-level attribution and MLP-intermediate suppression. It must not silently change the upstream sink definition.

## Models and why they are separated by role

### Mechanism-discovery models

1. `openai-community/gpt2` (GPT-2-small)
   - development and first falsification model;
   - small enough for the RTX 2060 SUPER;
   - directly comparable with the prior mechanistic sink work.

2. `openai-community/gpt2-medium`
   - confirmatory scale replication on the RTX 4080 SUPER;
   - also the teacher in the pinned GPT-2-medium -> GPT-2-small Sink-KD extension.

### Task-rich validation model

3. `Qwen/Qwen2.5-1.5B-Instruct`
   - used only after a sink-baseline preflight confirms the phenomenon is present in this checkpoint;
   - chosen because the prior codebase already contains Qwen2.5 sink instrumentation and because a 1.5B instruction model gives a more informative floor for MMLU/ARC/GSM8K/CulturalBench than GPT-2-small/medium.

Optional development analogue: `Qwen/Qwen2.5-0.5B-Instruct` on the RTX 2060 SUPER.

**Important:** neuron indices never transfer across models. Every checkpoint gets its own baseline sink map, attribution ranking, and causal validation before downstream evaluation.

### Sink-KD models

Use the existing `upstream/sink-kd` definitions, especially the primary GPT-2-medium -> random GPT-2-small arm:

- teacher: GPT-2-medium;
- student architecture: GPT-2-small;
- G0: CE-only/random-student training control;
- G1: logit KD;
- G2-aligned: logit + attention KD.

Analyze existing completed checkpoints if available. Do not retrain Sink-KD merely to unblock the first paper result. Sink-KD is a later mechanistic comparison after the basic neuron phenomenon passes.

## Data separation

The project must use three logically separate data roles.

### A. Upstream parity corpus

Use the original/frozen prior-paper fixtures or E1 corpus only to show that this repository reproduces the established sink metric and intervention outputs. The E1 mixture contains GSM8K, so it is **not allowed** for neuron selection.

### B. Neutral sink-discovery corpus

Primary choice: reuse/freeze the existing Sink-KD `openwebtext_validation_sink_300` corpus construction and manifest semantics.

Create three disjoint fixed partitions before inspecting neuron results:

- discovery: 100 examples;
- validation: 100 examples;
- test: 100 examples.

Primary sequence length: 40 tokens for direct comparability with the prior sink metric. A later robustness run may use 128 tokens, but the 40-token experiment remains the registered primary phenomenon test.

Roles:

- **discovery**: compute attribution scores and rank neurons;
- **validation**: select the registered causal operating point `k*` using sink and neutral-corpus CE only;
- **test**: one final unbiased estimate of targeted-vs-random sink reduction and neutral-corpus drift.

No MMLU/ARC/CulturalBench/GSM8K labels or examples may enter these stages.

### C. Downstream benchmark suite

Only after the phenomenon gate passes:

- MMLU;
- ARC-Challenge;
- CulturalBench;
- GSM8K.

Downstream data are evaluation-only.

## Primary unit and intervention

### MLP neuron definition

For each decoder layer, define the MLP neuron vector as the intermediate activation immediately before the final MLP output projection:

- GPT-2: input to `mlp.c_proj`;
- Qwen2.5: input to `mlp.down_proj`.

This definition gives one well-defined coordinate per MLP intermediate unit and works across GELU and SwiGLU architectures.

### Suppression

For selected neuron set `N_l` in layer `l`:

`a_l[..., N_l] <- alpha * a_l[..., N_l]`

with primary dose set:

`alpha in {1.00, 0.75, 0.50, 0.25, 0.00}`.

Apply suppression at all sequence positions. `alpha=1.00` is the identity condition.

Optional robustness after a positive primary result: replace selected units with their discovery-corpus mean rather than zeroing. Do not substitute mean ablation for the registered primary suppression.

## Sink-heavy attention scope

First compute baseline per-layer/per-head received attention to position 0 on discovery examples.

Define **sink-heavy attention layers** before neuron attribution as layers satisfying both:

1. layer sink score is in the top quartile of that model's layers; and
2. layer sink score is at least `0.15`.

If fewer than two layers satisfy both criteria, use the top two layers above `0.15`. If no layer exceeds `0.15`, the checkpoint fails the sink preflight and is not used for neuron claims.

Within those layers, record sink-heavy heads as the top quartile of heads by the same received-attention statistic. Head identity is diagnostic for the primary MLP-neuron experiment; the primary sink objective averages over the registered sink-heavy layers and all heads unless a head-restricted robustness run is explicitly registered.

## Causal-order-aware attribution objective

A decoder block computes attention before its MLP. Therefore an MLP at layer `l` cannot cause the attention matrix already produced at layer `l`.

For each MLP layer `l`, define a **future sink objective**:

`S_future(l) = mean sink attention over registered sink-heavy attention layers j where j > l`.

If no registered sink-heavy layer lies after `l`, that MLP layer is ineligible for the primary attribution scan.

For neuron `n` at layer `l`, primary discovery score:

`I(l,n) = mean_examples,tokens | a(l,n) * d S_future(l) / d a(l,n) |`.

Use absolute activation-times-gradient for ranking. Also save signed means for analysis, but do not rank by signed cancellation.

One backward pass may score all neurons in an eligible layer/example. Implement layer-wise or microbatched capture so the RTX 2060 SUPER can run the smoke version.

Attribution is a ranking heuristic, not causal evidence. Causality comes from held-out suppression.

## Neuron-set sizes

Use model-relative fractions to make cross-model comparisons meaningful:

`fraction in {0.01%, 0.05%, 0.10%, 0.25%, 0.50%, 1.00%}`

of all eligible MLP neurons. Round to the nearest valid positive integer and record the exact `k`.

The top-`k` set is formed globally across eligible `(layer, neuron)` pairs by discovery attribution score. Also retain per-layer rankings for diagnostics.

## Matched controls

Primary control: **layer-count matched random neurons**.

For a targeted top-`k` set, preserve the exact number of selected neurons in every layer and randomly sample the same number from non-target neurons of that layer.

Use 20 independent random-control draws for the full phenomenon confirmation; use 5 in the RTX 2060 SUPER smoke pilot.

Secondary confirmatory control after a positive effect: activation-matched random units, sampled from the same layer and nearest activation-magnitude bin.

Do not compare top-`k` only with unconstrained random neurons pooled across all layers.

## Phase structure

### Phase 0 — parity and hook correctness

Goal: prove the new harness has not changed the old phenomenon.

- run upstream tests/fixtures;
- reproduce GPT-2-small baseline sink on the frozen parity corpus within upstream tolerance;
- verify `alpha=1` identity;
- verify neuron hook semantics;
- verify discovery/validation/test manifests are disjoint.

No scientific claim yet.

### Phase 1 — RTX 2060 SUPER falsification pilot

Model: GPT-2-small.

Use small subsets only:

- 24 discovery examples;
- 24 validation examples;
- 24 test examples;
- 5 random controls;
- fractions `{0.05%, 0.10%, 0.25%}`;
- alphas `{1.0, 0.5, 0.0}`.

Purpose: determine whether implementation and signal are plausible, not produce paper statistics.

Proceed to full Phase 2 only if at least one non-identity targeted condition produces a held-out sink reduction larger than every one of the five matched-random controls and no parity test fails.

This pilot criterion is intentionally permissive; the full phenomenon gate is stricter.

### Phase 2 — full phenomenon confirmation on RTX 4080 SUPER

Models:

- GPT-2-small;
- GPT-2-medium.

Use the full 100/100/100 neutral splits, all registered fractions, all five alphas, 20 layer-matched random controls, and neutral-corpus CE/KL/top-1 drift.

The formal phenomenon gate is in `docs/01_PHENOMENON_GATE.md`.

### Phase 3 — task-rich checkpoint preflight and replication

Checkpoint: Qwen2.5-1.5B-Instruct.

Before downstream benchmarks:

1. confirm baseline sink > 0.15 on the neutral corpus;
2. localize its own sink-heavy layers;
3. run its own discovery ranking;
4. confirm targeted suppression beats matched random on held-out neutral test data.

If the Qwen checkpoint fails this gate, do not run it as if neuron indices from GPT-2 transfer. The task-rich evaluation can be marked blocked/model-negative.

### Phase 4 — downstream task drift

Freeze a small number of intervention operating points based only on neutral validation data. Then evaluate baseline, targeted suppression, and matched-random suppression on MMLU, ARC-Challenge, CulturalBench, and GSM8K as specified in `docs/02_DOWNSTREAM_TASKS.md`.

Primary comparison: targeted vs matched random at equal layerwise neuron count and equal `alpha`.

### Phase 5 — Sink-KD teacher/student comparison

After the basic phenomenon is established, apply the same neuron-localization/suppression protocol independently to the teacher and each available student checkpoint.

Key question:

> Does attention distillation produce sink-forming neurons that are less behaviorally load-bearing than the teacher's sink neurons?

Do not compare raw neuron indices across teacher/student. Compare effect profiles, normalized depth distributions, sparsity, dose-response, sink reduction, and functional drift.

## Registered operating point k*

Select `k*` on neutral validation data only.

For each model, among registered neuron fractions under `alpha=0`, choose the **smallest** fraction satisfying all of:

1. targeted relative sink reduction >= 10%;
2. targeted sink reduction exceeds the median layer-matched-random reduction by >= 10 percentage points of baseline sink;
3. targeted-vs-random difference has paired-bootstrap 95% CI lower bound > 0;
4. neutral-corpus CE increase <= 0.10 nats/token.

If no condition satisfies all four, do not tune on downstream tasks. Define `k_max_effect` as the registered fraction with the largest held-out sink reduction and label downstream use exploratory only.

For downstream evaluation, use at most three fixed operating points to control compute:

- `T1`: `k*`, `alpha=0`;
- `T2`: `k*`, `alpha=0.5`;
- `R1`: one preregistered layer-matched random set with the same per-layer counts as `k*`, `alpha=0`.

In addition, baseline/identity is always evaluated. The remaining random controls are used for mechanistic statistics, not full benchmark sweeps.

## Primary hypotheses

H1 — Sparse causal localization:

Targeted top-ranked neuron suppression reduces held-out sink strength more than layer-matched random suppression.

H2 — Dose response:

Within the targeted set, stronger suppression produces monotonically larger sink reduction.

H3 — Selectivity:

At some operating point, targeted suppression produces a larger sink change per unit of neutral-corpus CE drift than matched random suppression.

H4 — Cross-model recurrence, not coordinate identity:

The sparse-suppression phenomenon recurs in more than one checkpoint, but raw neuron ids and exact layer locations need not match.

H5 — Task-dependent functional drift:

At a fixed validated sink-reduction operating point, performance drift differs across knowledge, reasoning, culture, and arithmetic tasks.

H6 — Sink-KD dissociation:

A teacher and an attention-distilled student can show similar sink reduction under neuron suppression while exhibiting different functional drift, consistent with pattern/circuit/function dissociation.

## What would falsify the core idea?

The core sparse-neuron claim is not supported if, on the locked test split:

- targeted suppression does not outperform layer-matched random controls;
- apparent discovery effects disappear on held-out data;
- the effect requires suppressing such a large fraction of neurons that it is indistinguishable from generic capacity damage;
- or the result exists only under one brittle attribution/control definition and fails basic robustness.

A null downstream drift result does **not** falsify neuron-level sink causality; it instead supports sink-forming but weakly load-bearing units.
