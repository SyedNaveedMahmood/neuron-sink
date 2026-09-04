# Experiment Design Index

The design is intentionally gated. The coding agent should read these in order:

1. [`00_MASTER_EXPERIMENT_DESIGN.md`](00_MASTER_EXPERIMENT_DESIGN.md) — scientific question, models, data separation, neuron definition, attribution, controls, stages, hypotheses, and falsification criteria.
2. [`01_PHENOMENON_GATE.md`](01_PHENOMENON_GATE.md) — exact go/no-go criteria for deciding whether targeted neuron suppression is a real held-out phenomenon.
3. [`02_DOWNSTREAM_TASKS.md`](02_DOWNSTREAM_TASKS.md) — MMLU, ARC-Challenge, CulturalBench, and GSM8K drift evaluation after the causal gate passes.
4. [`03_IMPLEMENTATION_SPEC.md`](03_IMPLEMENTATION_SPEC.md) — intended package/modules, hook points, interfaces, outputs, and milestone order.
5. [`04_HARDWARE_RUNBOOK.md`](04_HARDWARE_RUNBOOK.md) — RTX 2060 SUPER smoke-development policy and RTX 4080 SUPER full-run policy.
6. [`05_METRICS_AND_SCHEMAS.md`](05_METRICS_AND_SCHEMAS.md) — metrics, provenance, per-example schemas, and required paper tables.
7. [`06_IMPLEMENTATION_PROMPTS.md`](06_IMPLEMENTATION_PROMPTS.md) — sequential work packages for the coding agent.

Machine-readable registration files:

- [`../configs/experiment_plan.yaml`](../configs/experiment_plan.yaml)
- [`../configs/downstream_tasks.yaml`](../configs/downstream_tasks.yaml)

Root [`../AGENTS.md`](../AGENTS.md) contains non-negotiable implementation and anti-leakage rules.

No new scientific implementation is included in this design commit. The first code milestone is parity + GPT-2-small MLP-neuron suppression smoke testing.
