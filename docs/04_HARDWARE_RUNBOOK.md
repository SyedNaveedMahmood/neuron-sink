# Hardware Runbook: RTX 2060 SUPER Development, RTX 4080 SUPER Full Runs

## Machine roles

### RTX 2060 SUPER — implementation machine

Assumed VRAM: 8 GB.

Use for:

- environment setup;
- unit tests;
- upstream parity tests;
- synthetic hook tests;
- GPT-2-small smoke attribution;
- tiny suppression sweeps;
- benchmark-adapter smoke subsets;
- optional Qwen2.5-0.5B-Instruct adapter smoke.

Do not use it to define final scientific thresholds from convenience subsets.

### RTX 4080 SUPER — full-run machine

Assumed VRAM: 16 GB.

Use for:

- 100/100/100 GPT-2-small/medium phenomenon runs;
- 20-control suppression sweeps;
- Qwen2.5-1.5B-Instruct sink preflight/localization;
- full downstream task evaluation;
- Sink-KD checkpoint analysis;
- aggregation where GPU is required.

## Dtype policy

### GPT-2 mechanistic experiments

Primary: float32, to stay close to the existing verified sink-reproduction semantics.

If GPT-2-medium attribution in float32 exceeds 16 GB because of gradient retention, the first permitted response is layer-wise capture + batch size 1 + graph release, not precision reduction. Only change dtype under a documented amendment after recording the OOM.

### Qwen2.5 task model

Primary: bfloat16 on RTX 4080 SUPER.

For Qwen smoke on RTX 2060 SUPER use float16 if bfloat16 is unsupported/inefficient. Smoke numeric values are not paper results; the 4080 baseline/intervention comparisons must use a single frozen dtype.

## Memory discipline

Attribution is the highest-risk path.

Required defaults:

- batch size 1 for initial attribution;
- one eligible MLP layer per backward pass;
- `model.eval()`;
- no optimizer state;
- zero gradients/set to `None` between examples;
- release saved activations/attention tensors after accumulation;
- do not retain all-layer attention graphs simultaneously if a layer-specific future-sink objective can be formed from only the necessary layers;
- log `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` per stage.

Suppression evaluation is inference-only and can increase batch size after measured preflight.

## RTX 2060 SUPER smoke sizes

Use the exact smoke policy from the master design:

- GPT-2-small;
- 24 discovery / 24 validation / 24 test examples;
- seq_len 40;
- neuron fractions 0.05%, 0.10%, 0.25%;
- alphas 1.0, 0.5, 0.0;
- 5 layer-matched random controls;
- batch size 1 initially.

For benchmark adapter smoke:

- 20 examples/task;
- B0 and T1 only are sufficient for code-path validation;
- no scientific interpretation.

## RTX 4080 SUPER full phenomenon matrix

For each of GPT-2-small and GPT-2-medium:

- 100 discovery / 100 validation / 100 test;
- seq_len 40 primary;
- fractions: 0.01%, 0.05%, 0.10%, 0.25%, 0.50%, 1.00%;
- alpha: 1.00, 0.75, 0.50, 0.25, 0.00;
- 20 layer-matched random sets;
- targeted sets: all fractions;
- random sets: full alpha grid may be expensive; minimum required random evaluation is alpha 0 for all fractions plus the full alpha grid at `k*`. If compute permits, run the full grid.

This asymmetric random policy must be recorded; do not pretend missing random-alpha cells were run.

## Qwen2.5-1.5B-Instruct path

Before task evaluation:

1. baseline neutral sink map;
2. sink-heavy layer selection;
3. discovery attribution;
4. validation operating point;
5. held-out test targeted-vs-random gate;
6. only then downstream tasks.

Use bfloat16, batch 1 for attribution, and conservative sequence lengths. Task prompts may exceed 40 tokens; do not truncate questions merely to match the sink-discovery length. Record prompt lengths and intervention sink scores separately.

## Full downstream scheduling

Order from cheapest/most deterministic to most expensive:

1. ARC-Challenge;
2. CulturalBench-Easy;
3. MMLU;
4. CulturalBench-Hard;
5. GSM8K generation.

Run baseline first for each benchmark and verify the baseline-viability rule before spending compute on all interventions. If a task is non-diagnostic at baseline, record it and skip expensive intervention sweeps for that task unless explicitly requested.

## Runtime estimation

Every full script must support a `--limit` or `--max-examples` dry run. Before launching a full benchmark or 20-control sweep:

- run at least 20 representative examples;
- record examples/sec or tokens/sec;
- record peak VRAM;
- estimate full wall time;
- record output size.

This estimate is operational only; it may not be used to alter the scientific condition silently.

## OOM policy

Allowed without scientific amendment:

- reduce batch size;
- process layers serially;
- reduce number of examples per microbatch while preserving total examples;
- enable inference-only no-grad paths where gradients are not required;
- stream results to disk.

Requires documented amendment:

- changing model/checkpoint;
- changing dtype for a paper run;
- changing sequence length;
- changing benchmark subset/full split;
- reducing number of random controls;
- changing neuron fractions or alphas;
- changing attribution method;
- changing sink layer/head selection rule.
