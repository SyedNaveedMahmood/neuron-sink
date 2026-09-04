# Coding-Agent Implementation Prompts

Use these as sequential work packages. Do not jump to later prompts before the previous prompt's acceptance criteria pass.

## Prompt 1 — provenance, manifests, and schemas

Implement project-local provenance and immutable dataset-manifest utilities plus typed result schemas described in `docs/05_METRICS_AND_SCHEMAS.md`.

Acceptance criteria:

- writes repo + submodule commit hashes;
- hashes manifests;
- refuses accidental overwrite of completed run directories;
- unit tests cover deterministic hashing and schema validation;
- no model inference required.

## Prompt 2 — GPT-2 sink parity adapter

Implement a GPT-2 model adapter and differentiable sink metric. Reuse the upstream sink metric semantics and add a test comparing the differentiable scalar against the pinned upstream `compute_bos_attention_metric` on the same fixed attention tensors.

Acceptance criteria:

- upstream parity test passes;
- same target position/query rule/layer scope;
- gradient exists through the differentiable score;
- no change to upstream submodule.

## Prompt 3 — MLP-neuron suppression hook

Implement GPT-2 MLP intermediate suppression at the input to `mlp.c_proj`.

Acceptance criteria:

- `alpha=1` is identical to baseline within tolerance;
- `alpha=0` zeros only requested intermediate coordinates at the hook point;
- model weights do not change;
- repeated contexts do not leak hooks;
- unit ids are range checked.

## Prompt 4 — sink map and causal-order-aware attribution

Implement baseline sink-heavy layer/head mapping and layer-wise activation-times-gradient ranking using `S_future(l)`.

Acceptance criteria:

- sink-heavy layers are frozen to JSON before ranking;
- MLP layer `l` only targets attention layers `j > l`;
- discovery manifest is mandatory;
- validation/test manifests are explicitly rejected by ranking API;
- scorer runs on 4 examples on RTX 2060 SUPER without OOM.

## Prompt 5 — selection and matched controls

Implement registered neuron fractions, global top-k selection, and layer-count-matched random controls.

Acceptance criteria:

- exact per-layer targeted counts are preserved in every random control;
- controls exclude targeted ids;
- 20 deterministic control sets can be generated from seeds;
- saved neuron-set file is stable across reruns.

## Prompt 6 — end-to-end 2060 SUPER phenomenon smoke

Implement the smoke pipeline from `configs/experiment_plan.yaml` for GPT-2-small.

Acceptance criteria:

- 24/24/24 disjoint neutral examples;
- fractions 0.05/0.10/0.25%;
- alpha 1/0.5/0;
- five matched random controls;
- per-example sink/CE/KL/top1 output;
- peak VRAM/runtime output;
- automatic smoke gate JSON.

Do not make full-run changes based on the sign of the smoke result except to fix a verified implementation defect.

## Prompt 7 — full phenomenon runner

Generalize the exact smoke implementation to GPT-2-small and GPT-2-medium on the RTX 4080 SUPER.

Acceptance criteria:

- full registered fraction/alpha grid;
- 20 random controls;
- 100/100/100 neutral split;
- separate discovery/validation/test outputs;
- automatic `k*` selection uses validation only;
- final test is one locked pass after `k*` is written.

## Prompt 8 — Qwen2.5 neuron adapter

Implement the architecture-specific adapter for Qwen2.5 with MLP neuron hook at `mlp.down_proj` input. Reuse the prior Qwen sink harness semantics where applicable, but do not assume GPT-2 positional mechanisms.

Acceptance criteria:

- Qwen2.5-0.5B-Instruct smoke runs on the 2060 SUPER if memory permits; otherwise use a synthetic/module-level unit test;
- Qwen2.5-1.5B-Instruct baseline sink preflight runs on 4080 SUPER;
- its neuron ranking is computed independently;
- GPT-2 neuron ids are never reused.

## Prompt 9 — downstream benchmark adapters

Implement deterministic evaluators for MMLU, ARC-Challenge, CulturalBench-Easy/Hard, and GSM8K exactly as `docs/02_DOWNSTREAM_TASKS.md` and `configs/downstream_tasks.yaml` specify.

Acceptance criteria:

- baseline can run twice with identical predictions;
- each adapter has a 20-example smoke set;
- intervention wrapper is shared across tasks;
- no task code imports or calls the neuron-ranking function;
- benchmark revision/prompt protocol is saved in manifests.

## Prompt 10 — full downstream drift

Run only B0/T1/T2/R1 after the Qwen checkpoint passes the causal gate and `k*` is frozen.

Acceptance criteria:

- baseline viability is checked before expensive intervention runs;
- paired per-example outputs are saved;
- accuracy/EM drift and bootstrap CIs are aggregated;
- MMLU/CulturalBench group macros are retained;
- GSM8K invalid answer and generation-length changes are reported.

## Prompt 11 — Sink-KD comparison

Add a checkpoint-analysis adapter around completed `upstream/sink-kd` GPT-2-medium -> GPT-2-small runs.

Acceptance criteria:

- no training unless explicitly requested in a later task;
- missing checkpoint produces `blocked_missing_checkpoint`, not fallback training;
- teacher and each student are ranked independently;
- compare normalized effect profiles rather than raw neuron ids;
- report sink reduction vs CE/task drift for each condition.

## Prompt 12 — final aggregation

Create machine-generated summary tables/figures from immutable result files.

Required primary figures:

1. sink-heavy layer/head map;
2. top-neuron vs random suppression by fraction;
3. dose response at `k*`;
4. sink reduction vs neutral CE drift;
5. downstream task drift;
6. optional teacher/student dissociation plot.

No figure script may hard-code paper numbers.
