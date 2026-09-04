# Coding-Agent Instructions: Neuron-Level Causal Analysis of Attention Sinks

This repository is a research codebase. Implement the experiment exactly as registered in `docs/00_MASTER_EXPERIMENT_DESIGN.md` unless a documented amendment is added before looking at the affected result.

## Scientific objective

Determine whether a sparse set of internal units causally supports the attention-sink phenomenon, and then measure how suppressing those units changes general language-model behavior and downstream task performance.

The project is deliberately gated:

1. reproduce the existing sink measurement and intervention semantics;
2. establish that neuron-level targeted suppression has a real held-out effect beyond matched random suppression;
3. only if that gate passes, run downstream task-drift experiments;
4. only after the basic phenomenon is established, run the Sink-KD teacher/student comparison.

Do not reverse this order.

## Upstream code is read-only reference code

Two exact source snapshots are pinned as submodules:

- `upstream/sink-repro`: Same Sink, Different Plumbing, `main`, commit `9ab67e914464b13863b67527d8ea14068ee9ff10`.
- `upstream/sink-kd`: A Sink Without the Plumbing / Sink-KD, `sink-inheritance-foundation`, commit `db114c9c5eb6ffc5de13e444c783408ea7401c62`.

Do not edit files inside either submodule. New code must live in this repository. Prefer thin adapters around upstream functions over copying entire scripts. Whenever a function is ported or semantics are reimplemented, record the upstream file/function in comments and provenance output.

## Hardware contract

Development/smoke hardware:

- NVIDIA RTX 2060 SUPER, 8 GB VRAM.
- Use it for unit tests, tiny synthetic tests, GPT-2-small smoke tests, and optionally Qwen2.5-0.5B-Instruct smoke tests.
- Do not design a smoke test that requires the full 8 GB budget. Batch size 1 and layer-wise/microbatched attribution are the default.
- Do not silently switch to a different scientific method merely to avoid OOM.

Full-run hardware:

- NVIDIA RTX 4080 SUPER, 16 GB VRAM.
- Use it for full GPT-2-small/medium phenomenon confirmation, Qwen2.5-1.5B-Instruct downstream-task runs, and any available Sink-KD checkpoint analysis.
- GPT-2 mechanistic confirmation should preserve the upstream float32 semantics where feasible. Qwen task experiments may use bfloat16, but the dtype must be recorded and baseline/intervention comparisons must use the same dtype.

If a registered run OOMs, stop and report the measured peak/mode. Do not change model, sequence length, layer set, neuron set, attribution definition, or benchmark protocol without an amendment.

## Non-negotiable anti-leakage rules

- Do not use MMLU, ARC, CulturalBench, or GSM8K examples to rank or select sink neurons.
- The existing E1 benchmark mixture includes GSM8K. It may be used only for parity/reproduction, never for neuron discovery when GSM8K is a downstream evaluation target.
- Neuron discovery/validation/test must use a neutral language-model corpus, preferably the existing pinned `openwebtext_validation_sink_300` machinery from Sink-KD, or a frozen replacement manifest registered before results.
- Downstream benchmark labels must never enter attribution, neuron ranking, `k` selection, or suppression-strength selection.
- The downstream benchmark suite is an evaluation set, not a hyperparameter-tuning set.

## Definition of a "neuron"

For the primary experiment, a neuron is one coordinate of the MLP intermediate activation immediately before the MLP output/down projection:

- GPT-2: input to `mlp.c_proj` after the GELU intermediate transformation.
- Qwen2.5: input to `mlp.down_proj` after the SwiGLU product.

Suppressing neuron `n` means multiplying that intermediate coordinate by `alpha` at all sequence positions for the selected layer(s). `alpha=1` is an exact identity control; `alpha=0` is full suppression.

Do not call residual-stream dimensions, attention-head channels, Q/K coordinates, or parameters "neurons" in primary results. They are separate unit types and belong to the optional dimension extension.

## Causal ordering constraint

A layer's MLP output cannot cause the attention weights already computed earlier in the same GPT-2/Qwen decoder block. Therefore neuron attribution for MLP layer `l` must target sink attention only in later attention layers. Use the future-sink objective specified in the master design. Do not attribute layer `l` neurons to same-layer pre-MLP attention.

## Implementation requirements

Every new experiment must emit:

- source commit hashes for this repo and both upstream submodules;
- exact model id and revision when available;
- tokenizer id/revision;
- device and GPU name;
- torch/transformers/nnsight versions;
- dtype;
- seed;
- dataset/config/split and a manifest hash;
- prompt/evaluation protocol version;
- sink metric definition and layer/head scope;
- neuron scoring method;
- selected neuron ids grouped by layer;
- control-selection seed and control neuron ids;
- suppression `alpha`;
- baseline and intervened sink metrics;
- baseline and intervened CE/KL/task metrics;
- peak VRAM and runtime.

Results must be append-only. Never overwrite a completed run with a different configuration.

## Required tests before real runs

At minimum implement tests for:

1. `alpha=1` reproduces the baseline logits and attention within the upstream numerical tolerance.
2. `alpha=0` changes only the requested MLP intermediate coordinates at the hook point.
3. layer indexing is explicit and zero-indexed internally.
4. selected neuron ids are in range for every layer/model.
5. fixed seeds reproduce random-control sets exactly.
6. ranking uses only discovery examples; validation/test manifests are rejected if passed to ranking code.
7. downstream benchmark code cannot be used by the ranking API.
8. sink metric parity with the upstream `compute_bos_attention_metric` on a frozen fixture.
9. causal-order target construction excludes same/earlier attention layers for an MLP neuron.
10. result schemas validate and include provenance.

## Do not optimize for a positive result

A null result is valid. Do not widen the candidate layer set, change the sink metric, change the discovery corpus, increase `k`, change the random-control definition, or switch attribution methods after seeing a failed gate unless the change is documented as a new experiment with a new id.

Read these files before implementation:

1. `docs/00_MASTER_EXPERIMENT_DESIGN.md`
2. `docs/01_PHENOMENON_GATE.md`
3. `docs/02_DOWNSTREAM_TASKS.md`
4. `docs/03_IMPLEMENTATION_SPEC.md`
5. `docs/04_HARDWARE_RUNBOOK.md`
6. `docs/05_METRICS_AND_SCHEMAS.md`
7. `docs/06_IMPLEMENTATION_PROMPTS.md`
8. `configs/experiment_plan.yaml`
