# Coding-Agent Execution Order

This file is the operational checklist. Follow it in order. Do **not** start downstream benchmarks until the neuron-level phenomenon gate passes on held-out neutral text.

## Stage A — RTX 2060 SUPER: implementation and falsification

Hardware target: RTX 2060 SUPER, 8 GB.

1. Initialize submodules and environment.
2. Run upstream parity tests and reproduce the registered GPT-2-small sink metric.
3. Implement the MLP-intermediate hook and verify `alpha=1.0` is numerically identical to baseline.
4. Build/freeze the neutral sink corpus manifest and disjoint discovery/validation/test splits.
5. Compute baseline per-layer/per-head sink maps on GPT-2-small.
6. Select sink-heavy attention layers using the registered rule in `00_MASTER_EXPERIMENT_DESIGN.md`.
7. Rank eligible MLP neurons with the future-sink activation-times-gradient objective.
8. Run the smoke suppression grid:
   - discovery/validation/test: 24/24/24 examples;
   - fractions: 0.05%, 0.10%, 0.25%;
   - alpha: 1.0, 0.5, 0.0;
   - five layer-count-matched random controls.
9. Stop if parity fails, the hook is not identity at alpha=1, or no targeted condition beats all five matched-random controls on held-out sink reduction.

This stage answers only: **is the proposed neuron-level sink effect plausible enough to justify a full experiment?**

## Stage B — RTX 4080 SUPER: confirm the phenomenon

Hardware target: RTX 4080 SUPER, 16 GB.

Run GPT-2-small and GPT-2-medium with the full registered design:

- neutral discovery/validation/test: 100/100/100;
- sequence length: 40 tokens primary;
- fractions: 0.01%, 0.05%, 0.10%, 0.25%, 0.50%, 1.00%;
- alpha: 1.00, 0.75, 0.50, 0.25, 0.00;
- 20 layer-count-matched random neuron sets;
- neutral functional measures: CE, PPL, KL, top-1 flip rate;
- mechanistic measure: received attention to position 0 under the frozen sink scope.

Use validation data only to freeze `k*`/operating point. Use the neutral test split once for the primary targeted-vs-random estimate.

Proceed only if the formal gate in `01_PHENOMENON_GATE.md` passes.

## Stage C — task-capable model replication

Primary checkpoint: `Qwen/Qwen2.5-1.5B-Instruct` on RTX 4080 SUPER.

Do **not** transfer GPT-2 neuron IDs or layers.

For Qwen independently:

1. measure its baseline sink map on neutral text;
2. require at least one layer sink score >= 0.15;
3. identify its own sink-heavy layers;
4. rank its own MLP intermediate neurons;
5. suppress targeted units and matched-random units;
6. require the held-out targeted-vs-random neuron effect to pass before downstream tasks.

If Qwen does not display the phenomenon, record that as a model-negative result and do not force benchmark experiments under a transferred mechanism.

## Stage D — downstream performance-drift suite

Only after Stage C passes, evaluate the frozen intervention conditions on four different task families:

- **MMLU** — broad academic/factual knowledge;
- **ARC-Challenge** — science/reasoning multiple choice;
- **CulturalBench** — cultural knowledge/judgment;
- **GSM8K** — multi-step arithmetic reasoning.

Use the exact protocols in `02_DOWNSTREAM_TASKS.md` and `configs/downstream_tasks.yaml`.

Conditions:

- B0: no suppression;
- T1: targeted `k*`, alpha=0.0;
- T2: targeted `k*`, alpha=0.5;
- R1: preregistered layer-matched random `k*`, alpha=0.0;
- identity-hook audit where needed.

Do not choose a different neuron count or alpha per benchmark. Benchmark data are evaluation-only and must never tune neuron ranking, `k*`, sink scope, or suppression strength.

Primary scientific readout per task is the paired relationship between:

1. sink reduction; and
2. task-performance drift.

A large sink reduction with little task drift supports sink-forming but weakly load-bearing units. Large sink reduction with broad task degradation supports functionally load-bearing sink units. Large task drift without sink reduction indicates generic model damage rather than sink-specific causality.

## Stage E — Sink-KD comparison

After the basic phenomenon is established, reuse the pinned Sink-KD checkpoints/definitions.

Run localization and suppression independently on:

- teacher;
- CE/random-student control when available;
- LogitKD student;
- attention-KD student;
- no-sink control when available.

Do not compare raw neuron indices across models. Compare normalized depth, sparsity, dose-response, sink reduction, CE/task drift, and targeted-vs-random effect size.

The key question is whether an attention-distilled student can reproduce the visible sink through neurons that are less behaviorally load-bearing than the teacher's sink neurons.

## Stop/continue rule

The project is intentionally falsifiable:

- if Stage A fails, debug implementation once using predefined tests; do not redesign toward a positive result;
- if Stage B fails after valid implementation, report a neuron-level null/weak result and do not claim localized causal neurons;
- if GPT-2 passes but Qwen fails, report architecture/model dependence and do not use GPT-2 neuron identities for Qwen tasks;
- downstream benchmarks are justified only after the phenomenon is real in the checkpoint being evaluated.

## Read before coding

1. `AGENTS.md`
2. `docs/00_MASTER_EXPERIMENT_DESIGN.md`
3. `docs/01_PHENOMENON_GATE.md`
4. `docs/02_DOWNSTREAM_TASKS.md`
5. `docs/03_IMPLEMENTATION_SPEC.md`
6. `docs/04_HARDWARE_RUNBOOK.md`
7. `docs/05_METRICS_AND_SCHEMAS.md`
8. `docs/06_IMPLEMENTATION_PROMPTS.md`
9. `configs/experiment_plan.yaml`
10. `configs/downstream_tasks.yaml`
