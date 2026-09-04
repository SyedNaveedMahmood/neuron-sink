# Downstream Task-Drift Design

## Purpose

Once neuron-level sink causality is established on neutral text, test whether suppressing validated sink neurons changes performance differently across task families.

The task suite intentionally spans different capabilities:

| Benchmark | Capability emphasis | Primary metric |
|---|---|---|
| MMLU | broad factual/academic knowledge | accuracy / macro subject accuracy |
| ARC-Challenge | grade-school science reasoning | accuracy |
| CulturalBench | cultural knowledge and culturally grounded judgment | accuracy; region/topic macro analysis |
| GSM8K | multi-step arithmetic reasoning | exact-match final answer |

This suite is for **functional drift**, not neuron selection.

## Important model-floor issue

GPT-2-small/medium can be near chance or near zero on modern instruction-style task protocols, especially GSM8K. A task with a floor-level baseline cannot provide a useful estimate of degradation.

Therefore:

- neutral LM CE/PPL is the primary functional metric for GPT-2 mechanistic runs;
- the full downstream suite is primarily run on `Qwen/Qwen2.5-1.5B-Instruct`, but only after that exact checkpoint independently passes the sink and neuron-causality preflight;
- GPT-2 task scores may be reported as diagnostics only when the baseline-viability rule below is satisfied.

## Benchmark identities

Pin the exact dataset revision in the generated manifest before full runs.

### MMLU

- Dataset family: `cais/mmlu`.
- Evaluate the standard 57-subject test set.
- Use a fixed 5-shot protocol for the primary run.
- Score multiple choice from model likelihood over answer choices rather than free-form letter parsing when the evaluation adapter supports it.
- Report overall micro accuracy and macro average over subjects; retain per-subject values.

### ARC

- Dataset: `allenai/ai2_arc`.
- Configuration: `ARC-Challenge`.
- Split: test.
- Primary protocol: fixed 0-shot multiple-choice likelihood scoring to reduce prompt-length confounding and keep the intervention comparison simple.
- Optional standard few-shot replication may be added after the primary result, but not substituted after inspecting results.

### CulturalBench

The requested "CultureBench" is registered here as **CulturalBench**:

- Hugging Face dataset: `kellycyy/CulturalBench`.
- Primary configuration: `CulturalBench-Easy`, test split, multiple-choice accuracy.
- Confirmatory configuration: `CulturalBench-Hard`, test split, using the benchmark's grouped hard-question scoring rather than treating each true/false row as an independent question.
- Report overall score and macro averages by country/region/topic when metadata permit.

Do not rename or silently substitute another cultural benchmark.

### GSM8K

- Dataset: `openai/gsm8k`.
- Configuration: `main`.
- Split: test.
- Primary protocol for the instruction model: fixed 5-shot chain-of-thought prompt, greedy decoding (`do_sample=False`), deterministic answer extraction, exact-match final numeric answer.
- Also record generation length and invalid-answer rate because suppression may change formatting without changing reasoning.

## Baseline-viability rule

A benchmark is considered diagnostic for drift only when baseline performance is meaningfully above its floor.

Before interpreting intervention drift:

- 4-choice tasks: baseline accuracy must be at least chance + 5 percentage points;
- variable-choice ARC: use empirical random-choice chance for the evaluated examples, then require baseline >= chance + 5 points;
- GSM8K: baseline exact match must be >= 5%;
- CulturalBench-Hard: use the benchmark-defined chance/floor for its grouped scoring and record it in the manifest.

If a checkpoint fails a benchmark's viability rule, still save the result but label that benchmark `non_diagnostic_floor=true` and do not claim preservation or degradation from it.

## Fixed intervention conditions for full benchmark runs

Do not sweep the entire mechanistic grid over every benchmark.

Freeze the following using only neutral validation data:

- `B0`: baseline/no suppression;
- `I1`: identity hook, selected `k*`, `alpha=1.0` (implementation control; may be run on a smaller audit subset once parity is proven);
- `T1`: targeted `k*`, `alpha=0.0`;
- `T2`: targeted `k*`, `alpha=0.5`;
- `R1`: one preregistered layer-matched random set with the same per-layer counts as `k*`, `alpha=0.0`.

For full benchmark compute, B0/T1/T2/R1 are sufficient. Do not choose a different `k` per benchmark.

## Evaluation pairing

Use exactly the same benchmark examples and prompt rendering for baseline and every intervention. Preserve example order and batch composition when possible.

For multiple-choice tasks, store per-example:

- answer choice;
- baseline choice scores;
- intervened choice scores;
- baseline prediction;
- intervened prediction;
- correctness before/after;
- prompt token length;
- sink score when captured.

For GSM8K, store:

- baseline output;
- intervention output;
- extracted final answers;
- correctness;
- invalid-format flag;
- generation token count.

This permits paired drift analysis rather than comparing aggregate accuracies only.

## Primary task-drift quantities

For each task and condition:

- `accuracy_or_em_baseline`;
- `accuracy_or_em_intervened`;
- signed drift `intervened - baseline`;
- absolute drift;
- paired bootstrap 95% CI of signed drift;
- fraction of examples whose correctness flips `correct -> wrong`;
- fraction `wrong -> correct`;
- prediction flip rate;
- mean/median change in sink score on task prompts when available.

For MMLU and CulturalBench additionally report macro group drift so one large category cannot dominate the aggregate.

## Cross-task interpretation

The key plot is not simply "accuracy after suppression." Use a two-axis view:

- x-axis: sink reduction;
- y-axis: task-performance drift.

Interpretation examples:

- large sink reduction + little task drift: sink-forming but weakly load-bearing neurons;
- large sink reduction + broad negative drift: functionally important sink neurons;
- large sink reduction + selective drift on one task family: task-specific coupling;
- little sink reduction + large drift: generic model damage, not a sink-specific effect.

## Task subset policy for development

On the RTX 2060 SUPER and early RTX 4080 SUPER debugging, use fixed small subsets only to validate adapters:

- 20 MMLU examples spread across subjects;
- 20 ARC-Challenge examples;
- 20 CulturalBench-Easy examples;
- 20 GSM8K examples.

These are **adapter smoke sets** and must not be used to change `k*`, alpha, neuron ranking, prompts, or scoring based on apparent performance.

## Full-run compute control

Run the full test split for B0/T1/T2/R1 only after all task adapters pass deterministic smoke tests. Cache tokenization and immutable prompt manifests. For multiple-choice benchmarks, batch choice scoring conservatively within 16 GB VRAM; for GSM8K generation, use batch size 1 initially and scale only after measured peak VRAM.
