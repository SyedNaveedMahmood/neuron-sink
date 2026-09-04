# Neuron-Sink Project Handover

Last updated: 2026-09-05 (Stage C complete; Qwen2.5-1.5B model-negative null)

## 1. Project in one paragraph

This repository studies the neuron-level causal substrate of attention sinks. The central question is not whether attention sinks are inherently good or bad, but whether a visible first-position sink is supported by a sparse set of internal MLP neurons, whether suppressing those neurons causally changes sink strength, and how much functional drift follows. The longer-term comparison asks whether the same macroscopic sink can rely on different microscopic neuron-level substrates across naturally learned GPT-2 models and attention-distilled Sink-KD students.

Working title:

**Same Sink, Different Neurons: Neuron-Level Causal Localization of Attention Sinks and Performance Drift**

## 2. Repository and source provenance

Repository:

`https://github.com/SyedNaveedMahmood/neuron-sink`

Project commit recorded by the Stage B and Stage C runs:

`5bfbf240eacfacfff078f08086f8eb93a0b62c3e`

The Stage B/C implementation, tests, reports, README, and this handover update are currently
uncommitted. The registered runs record `repo_dirty_at_run=true` and retain the exact dirty-path
list in provenance. No commit or push has been made.

Two upstream paper codebases are pinned as read-only Git submodules and must not be edited in place:

| Source | Path | Branch/source role | Pinned commit |
|---|---|---|---|
| Same Sink, Different Plumbing | `upstream/sink-repro` | original sink mechanism/reproduction code | `9ab67e914464b13863b67527d8ea14068ee9ff10` |
| A Sink Without the Plumbing / Sink-KD | `upstream/sink-kd` | attention-distillation teacher/student code | `db114c9c5eb6ffc5de13e444c783408ea7401c62` |

The upstream submodules are reference implementations. New code belongs in the root repository, preferably as thin adapters rather than copied upstream scripts.

## 3. Hardware plan

### Development / falsification machines

Two GPUs are registered for this role. Amendment `A001` in `docs/AMENDMENTS.md` added the
second; `neuron_sink.provenance.REGISTERED_GPUS["dev"]` is what the code enforces.

| Machine | GPU | VRAM | OS | Produced |
|---|---|---|---|---|
| 1 | NVIDIA GeForce RTX 2060 SUPER | 8 GB | Windows 10 | Tasks 1-3 |
| 2 | NVIDIA GeForce RTX 2060 | 12 GB | Windows 11 Pro | Task 2/3 reproduction, Tasks 4-7 |

Purpose: implementation, unit tests, GPT-2-small parity, hook validation, attribution smoke
work, small 24/24/24 falsification run.

The 12 GB card does **not** license a larger smoke experiment. The 8 GB smoke policy is
retained verbatim on both. Task 4 peaked at ~505 MiB, so headroom is irrelevant so far.

Environments used successfully:

- machine 1: `X:\project\neuron-sink\.venv`
- machine 2: `D:\0 Repositories\neuron-sink\.venv`, Hugging Face
  cache at `D:\.cache\huggingface` via `$NEURON_SINK_HF_CACHE` / `$HF_HOME`
  (`X:` does not exist on machine 2; the old hard-coded cache defaults were replaced)

Verified stack:

- Python 3.12.5 (machine 1) / 3.12.4 (machine 2)
- PyTorch 2.10.0+cu128
- Transformers 5.3.0
- NNsight 0.7.0
- datasets 4.8.4
- CUDA 12.8 through PyTorch

### Full-run machine

- GPU: NVIDIA RTX 4080 SUPER
- VRAM: 16 GB
- Purpose: full GPT-2-small/medium phenomenon confirmation, Qwen2.5-1.5B replication, downstream benchmarks, and later Sink-KD checkpoint analysis
- Produced: both Stage B GPT-2 full runs and Stage C Qwen2.5-1.5B preflight/full replication
- Environment: `F:\neuron-sink\.venv`; Hugging Face cache at
  `F:\.cache\huggingface\neuron-sink`
- Verified Stage B stack: Python 3.12.3, PyTorch 2.10.0+cu128, Transformers 5.3.0,
  NNsight 0.7.0, datasets 4.8.4, NumPy 2.4.3, pandas 3.0.1, CUDA 12.8

Do not weaken or change scientific settings merely to make them fit the 2060. Allowed OOM responses are smaller batches, serial layer processing, microbatching, graph release, and streaming outputs. Changes to model, dtype, sequence length, attribution method, fractions, alphas, control count, or benchmark protocol require a documented amendment.

## 4. Registered primary neuron definition

For GPT-2, one neuron is one coordinate of the post-activation MLP intermediate tensor entering:

`transformer.h[layer].mlp.c_proj`

For GPT-2-small this tensor has shape:

`[batch, sequence, 3072]`

For Qwen2.5, one neuron is one coordinate of the SwiGLU product
`SiLU(gate_proj(x)) * up_proj(x)` entering `model.layers[layer].mlp.down_proj`. For the registered
1.5B checkpoint this tensor has shape `[batch, sequence, 8960]`.

The suppression operation is:

`a_l[..., N_l] <- alpha * a_l[..., N_l]`

where `alpha=1.0` is identity and `alpha=0.0` is full suppression.

This project must not call residual-stream dimensions, Q/K/V coordinates, head channels, or parameters "neurons" in the primary experiment. Those are separate optional unit types.

## 5. Canonical sink definition

The new project preserves the pinned Sink-Repro definition rather than inventing a new metric.

For GPT-2-small parity:

- target key position: `0`
- query positions: second half of the sequence
- primary sequence length: 40 tokens
- at length 40, queries are positions `20..39`
- layer scope for the upstream scaled GPT-2-small parity metric: zero-indexed layers `3..10`, i.e. `[3,11)`
- all 12 heads in those layers
- mean received attention to position 0 over heads, queries, layers, then examples

Canonical upstream metric source:

`upstream/sink-repro/common/intervention_analysis_legacy.py`

The current pinned public wrapper in `common/intervention_analysis.py` has a GPT-2 recursion defect in its fallback path. The Task-2 root adapter therefore invokes the frozen legacy implementation directly. Do not "fix" the submodule; preserve it and document the adapter behavior.

## 6. Completed work

### Task 1 — Environment and source verification: PASS

Completed:

- exact upstream commits initialized and verified
- root requirements installed
- CUDA confirmed on RTX 2060 SUPER
- core Sink-Repro and Sink-KD modules import successfully
- no source substitution
- no scientific experiment performed during setup

Important operational detail: Git LFS smudge was skipped for large Sink-KD checkpoint pointers to avoid unnecessary downloads. Do not accidentally pull large checkpoints on the 2060 unless a later task explicitly needs them.

### Task 2 — GPT-2-small sink parity: PASS

Tracked report:

`reports/TASK2_GPT2_SINK_PARITY.md`

Runner:

`scripts/run_gpt2_sink_parity.py`

Result:

- upstream reference sink: `0.563683`
- reproduced sink: `0.5636834649182856`
- absolute difference: `4.649182855e-7`
- allowed difference: `6.63683e-5`
- evaluated examples: 300
- runtime: ~30.79 s warm cache
- peak allocated VRAM: ~519 MiB

Parity fixture and deterministic repeat were exact. Attention was finite, normalized, causal, and strongly concentrated at position 0.

The parity corpus is the original E1 mixture:

- 100 SST-2
- 100 GSM8K
- 100 HumanEval

**Critical leakage rule:** this E1 mixture contains GSM8K and is therefore parity-only. It must never be used for neuron discovery, ranking, operating-point selection, or any stage where GSM8K is a downstream evaluation target.

### Task 3 — GPT-2 MLP suppression hook: PASS

Current implementation commit:

`bfcb0201a7a46a3cad9be04b6ad55de685f15a0e`

Tracked report:

`reports/TASK3_GPT2_SUPPRESSION_HOOK.md`

Core implementation:

- `neuron_sink/model_adapters.py`
- `neuron_sink/suppression.py`
- `neuron_sink/__init__.py`
- `tests/test_gpt2_adapter.py`
- `tests/test_suppression.py`
- `scripts/verify_gpt2_suppression_hook.py`

Implemented behavior:

- `GPT2ModelAdapter` validates GPT-2 structure, layer ids, neuron ids, and MLP width
- `NeuronSet` is immutable and stores neurons by zero-indexed layer
- suppression uses scoped PyTorch forward-pre-hooks on `mlp.c_proj`
- selected coordinates are multiplied by alpha at all sequence positions
- `alpha=1.0` bypasses hook registration entirely
- hooks unwind cleanly
- model parameters are not mutated

Task-3 audit results:

- alpha=1 logits: exact equality
- alpha=1 attentions: exact equality
- sink difference: `0.0`
- alpha=0 selected-coordinate error: `0.0`
- alpha=0 unselected-coordinate difference: `0.0`
- alpha=0.5 scaling error: `0.0`
- multi-layer suppression: PASS
- invalid layer/neuron/alpha checks: PASS
- state leakage: none detected
- parameter SHA before/after: identical
- finite logits/attention: PASS
- automated tests: 16 passed, 0 failed
- peak allocated VRAM: ~494 MiB

The Task-3 neuron ids were arbitrary DEBUG coordinates only. They are not sink neurons and must not be cited as findings.

### Task 4 - Neutral corpus freeze and GPT-2-small sink map: PASS

Tracked report:

`reports/TASK4_NEUTRAL_CORPUS_AND_SINK_MAP.md`

Runners:

- `scripts/prepare_neutral_corpus.py`
- `scripts/map_sink_layers.py`

New modules:

- `neuron_sink/provenance.py` - shared provenance, registered-GPU gate, append-only output dirs,
  canonical hashing
- `neuron_sink/upstream_bridge.py` - isolated imports of the two pinned `common/` trees
- `neuron_sink/corpus.py` - frozen neutral corpus, split roles, anti-leakage guards
- `neuron_sink/sink_metrics.py` - sink map decomposition and the registered sink-heavy rule
- `tests/test_provenance.py`, `tests/test_corpus.py`, `tests/test_sink_metrics.py`

Corpus:

- registered source `openwebtext_validation_sink_300` from the pinned Sink-KD provider, at
  `block_size=40` for this project's registered sequence length
- OpenWebText document window `[400000, 408000)`, disjoint from the Sink-KD training window by
  construction
- 300 blocks -> disjoint discovery/validation/test of 100 each; smoke splits are the first 24 of
  each, so they nest inside the eventual Stage-B splits
- manifest SHA-256 `c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7`, reproduced
  exactly on a rebuild

Sink map, 24 neutral discovery examples, GPT-2-small:

- per-layer sink (zero-indexed 0..11): `0.038521, 0.123772, 0.161033, 0.410571, 0.449738, 0.650954,
  0.612202, 0.703484, 0.599864, 0.697401, 0.694212, 0.549964`
- sink-heavy layers `[7, 9, 10]` via the primary top-quartile-and-floor branch, no fallback
- sink-heavy heads: layer 7 `{2, 10, 11}`, layer 9 `{1, 6, 9}`, layer 10 `{1, 8, 10}`
- eligible MLP layers `[0..9]`
- scope SHA-256 `b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235`, identical on a
  repeat run
- map decomposition vs upstream `compute_bos_attention_metric`: max abs diff `3.01e-8`
- Hugging Face forward vs upstream manual baseline: max abs metric diff `4.84e-8`
- peak VRAM ~505 MiB, 18.1 s

The map describes where the sink is measured and constrains which MLP layers Task 5 may attribute.
No neuron has been ranked or suppressed, so nothing here is causal evidence.

### Task 5 - Future-sink activation-times-gradient attribution: PASS

Tracked report:

`reports/TASK5_NEURON_ATTRIBUTION.md`

Runner:

- `scripts/rank_neurons.py`

New/changed modules:

- `neuron_sink/attribution.py` - capture, objective, causal-order probe, ranking API
- `neuron_sink/sink_metrics.py` - added `differentiable_sink_score` and `load_frozen_sink_scope`;
  the Task-4 functions are unchanged
- `tests/test_attribution.py`

Objective, exactly as registered:

- `S_future(l)` = sink metric restricted to the frozen sink-heavy attention layers `j > l`
- `I(l,n) = mean over examples and token positions of | a(l,n) * dS_future(l)/da(l,n) |`
- `a` is the tensor entering `transformer.h[l].mlp.c_proj`
- ranking score is `mean_abs_attr`; `mean_signed_attr` is saved and never ranked by
- token scope is all 40 positions, matching `mean_over_examples_and_tokens` and
  `suppression_positions: all`
- discovery split only, 24-example smoke prefix; validation and test raise `LeakageError`

Results:

- 10 eligible MLP layers x 3072 neurons = 30,720 rows
- differentiable scorer vs the frozen metric and upstream: max abs diff `8.82e-8` (tolerance
  `7.02e-5`); the scalar actually differentiated vs the frozen metric: `6.31e-8`
- causal ordering measured, not assumed: for all ten eligible layers the future targets are
  reachable from `a(l)`, while same-layer and earlier-layer objectives carry no `grad_fn` at all
- gradients finite and non-zero for all 240 (layer, example) backward passes
- `S_future(9)` = `0.694212` reproduces Task 4's frozen layer-10 sink exactly; `S_future(0)` =
  `0.698366` reproduces the mean of the frozen layer-7/9/10 sinks
- attribution SHA-256 `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692`,
  identical on a repeat run
- peak VRAM `531.13 MiB`, 26.7 s; the required 4-example preflight passed first, no OOM
- automated tests: 127 passed, 142 subtests, 0 failed

The ranking is a heuristic. No neuron has been selected or suppressed, so nothing in Task 5 is
causal evidence.

### Task 6 - Global top-k selection and layer-count-matched controls: PASS

Tracked report:

`reports/TASK6_NEURON_SELECTION.md`

Runner and implementation:

- `scripts/select_neurons.py`
- `neuron_sink/selection.py`
- `tests/test_selection.py`

Implemented exactly as registered:

- reloaded all 30,720 Task-5 CSV rows with their original integer/float/string types and
  reproduced attribution SHA-256 `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692`
- explicit rounding rule: Decimal `ROUND_HALF_UP` on pool x percentage / 100, minimum 1
- all-six-fraction k check: 3, 15, 31, 77, 154, 307
- froze only smoke top-k sets: 0.05% -> 15, 0.10% -> 31, 0.25% -> 77
- global selection by Task-5 `rank_abs` / `mean_abs_attr`, never per layer or by signed score
- five layer-count-matched random controls per target, sampled without replacement from that
  layer's non-target ids
- composite RNG seed `np.random.default_rng([base_seed, draw_index, k])`, base seed 0; saved
  `control_seed` is the draw index 0..4
- zero-count layers are present in diagnostic count maps and omitted from `NeuronSet.by_layer`
- the same API generated 20 controls twice for all six fractions and reproduced them exactly

Frozen output:

- `configs/frozen/neuron_sets.json`
- 18 conditions: 3 targeted + 15 layer-random
- neuron-set SHA-256 `4fa22a2c68c8c3e56ed13b4f1c481b7b43d963b0190a619cacdc7c03c2672165`
- flat run-local table has 738 `(condition, layer, neuron)` rows
- runtime ~0.92 s on CPU; peak GPU memory 0
- automated suite: 157 passed, 155 subtests, 0 failed, 0 skipped with GPU integration enabled

Every control preserves its target's exact per-layer histogram and excludes that target's ids.
The frozen file round-trips through the existing immutable `NeuronSet` and regenerates every
control from its recorded composite seed. No model, corpus split, suppression, held-out metric, or
benchmark was run. These are intervention candidates, not causal neurons.

### Task 7 - GPT-2-small 24/24/24 suppression smoke: PASS

Tracked report:

`reports/TASK7_GPT2_SUPPRESSION_SMOKE.md`

Runner and implementation:

- `scripts/run_suppression_smoke.py`
- `neuron_sink/evaluation.py`
- `tests/test_evaluation.py`

Registered run:

`results/task7_gpt2_smoke/run_20260904T122521Z`

Results on the locked 24-example test split:

- baseline sink `0.718978450`
- all six targeted non-identity cells beat all five matching layer-random controls
- target RSR at alpha 0.5 / 0.0:
  - k=15: `0.058025` / `0.129197`
  - k=31: `0.042501` / `0.094772`
  - k=77: `0.056175` / `0.149656`
- all three targeted sets satisfy `RSR(alpha=1) <= RSR(alpha=0.5) <= RSR(alpha=0)`
- alpha=1 logits and attention were exact across every identity row
- finite/nondegenerate logits and finite, normalized, causal attention: PASS
- final baseline replay found no hook/state leakage
- 3,960 paired rows, 3,961 forwards including the final leakage probe
- runtime `77.165` s; peak allocated VRAM `539.69 MiB`
- automated project-local suite before the run: 167 passed, 155 subtests, 0 failed/skipped

This is held-out causal evidence at the deliberately permissive smoke scale. It clears Stage A and
justifies the registered RTX 4080 SUPER confirmation. It is not the formal 100-example/20-control
gate, does not select `k*`, and does not authorize Qwen or downstream benchmark runs.

### Stage B - Full GPT-2-small/medium phenomenon confirmation: PASS / PASS

Tracked analysis:

`reports/STAGE_B_FULL_PHENOMENON.md`

Runner and implementation:

- `scripts/run_full_phenomenon.py`
- `neuron_sink/stage_b.py`
- `neuron_sink/stats.py`
- generalized `neuron_sink/evaluation.py`, `neuron_sink/selection.py`, and
  `neuron_sink/provenance.py`
- `tests/test_stage_b.py`, `tests/test_stats.py`, and updated CUDA integration tests

Registered run directories:

- GPT-2-small: `results/stage_b_full/gpt2-small/run_20260904T131855Z`
- GPT-2-medium: `results/stage_b_full/gpt2-medium/run_20260904T140056Z`

Both models independently recomputed their discovery sink map, sink scope, future-sink ranking,
six targeted sets, and 120 deterministic layer-count-matched random sets. Each evaluated the full
126-condition x five-alpha grid on separate 100-example discovery, validation, and locked-test
splits. Test access was refused until the validation operating-point artifact existed and passed
its hash/schema checks.

Primary outcomes:

- both models passed the formal held-out causal gate at 0.05%, 0.10%, 0.25%, 0.50%, and 1.00%;
- all targeted test dose curves had Spearman `1.0`;
- GPT-2-small validation selected confirmatory `k*=15` (0.05%, `alpha=0`): validation RSR
  `0.113873`, Delta CE `0.086014`; held-out RSR `0.107774`;
- GPT-2-medium produced no confirmatory `k*`: 0.01% stayed under the CE budget but missed the
  effect thresholds, while every larger effective fraction exceeded the CE budget;
- GPT-2-medium therefore froze `k_max_effect=860` (1.00%, `alpha=0`) with
  `exploratory_only=true`; it must not be presented as a low-drift operating point;
- GPT-2-medium's held-out 0.05% set nevertheless demonstrated the formal causal effect: RSR
  `0.261207` versus random P95 `0.000316`, with bootstrap difference CI lower `0.231320`;
- identity, attention/logit validity, causal ordering, hook cleanup, model-state replay, split
  separation, and smoke-artifact preservation all passed.

Project-level interpretation is **strong support** because both GPT-2 models passed. The important
qualification is functional specificity: GPT-2-small has a validation-qualified operating point,
while GPT-2-medium's strong sink effects co-occur with large CE/KL/top-1 drift.

Final project-local suite with cached CUDA integration enabled: **182 passed, 155 subtests passed,
0 failed, 0 skipped**.

### Stage C - Qwen2.5-1.5B independent replication: NULL / MODEL-NEGATIVE

Tracked analysis:

`reports/STAGE_C_QWEN_REPLICATION.md`

Registered run:

`results/stage_c_full/qwen2.5-1.5b-instruct/run_20260904T160405Z`

Implementation added `Qwen2ModelAdapter`, a separate Qwen-tokenized frozen neutral manifest,
Stage-C-specific operating-point/gate schemas, real-checkpoint adapter preflight, and exact
after-validation resume verification. The run used Qwen revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, bfloat16/eager attention, and the same disjoint
100/100/100 neutral roles.

Primary outcomes:

- baseline sink preflight passed: maximum discovery-layer sink `0.672835` at layer 25;
- independent sink-heavy layers `[4, 6, 14, 23, 24, 25, 26]` and eligible MLP layers `[0..25]`;
- 232,960 Qwen neurons ranked from discovery only; no GPT-2 neuron/layer transfer;
- all 126 conditions and five alphas evaluated on each split; identity, validity, causal ordering,
  and state/hook cleanup passed;
- validation produced no confirmatory `k*`; exploratory `k_max_effect=2330` (1.00%) had RSR
  `0.066983` and Delta CE `1.025863`;
- on locked test, 1.00% targeted suppression beat all matched controls and had a positive
  bootstrap interval, but RSR was only `0.070714`, Spearman was `0.6`, and Delta CE was
  `1.093020`;
- no registered fraction met the 10% RSR or 0.8 dose-response requirements; formal status
  `NULL_OR_INVALID`, with no passing fraction;
- every one of the 189,300 scientific-grid forwards was finite and nondegenerate.
- final project-local suite with cached CUDA integration: 194 passed, 155 subtests, 0 failed/skipped.

The first process stopped after completed validation because of a runner function-alias NameError.
Test was still locked and no operating point existed. The corrected resume path reproduced the
discovery and validation row hashes, revalidated every discovery hash lock, froze and verified the
operating point, then accessed test once. Existing artifacts were not overwritten, and
`resume.json` records the event.

This is a scientific null, not an execution failure. Under the registered order it blocks Stage D
for this Qwen checkpoint.

## 7. Current code surface

The project currently has the implementation and completed evidence through Stage C. Downstream
adapters remain unimplemented because the Qwen causal gate did not pass.

Important files:

- `AGENTS.md` — non-negotiable coding/scientific rules
- `SOURCE_BRANCHES.md` — upstream provenance
- `docs/00_MASTER_EXPERIMENT_DESIGN.md` — master registered experiment
- `docs/01_PHENOMENON_GATE.md` — go/no-go criteria
- `docs/02_DOWNSTREAM_TASKS.md` — MMLU/ARC/CulturalBench/GSM8K protocols
- `docs/03_IMPLEMENTATION_SPEC.md` — intended architecture
- `docs/04_HARDWARE_RUNBOOK.md` — 2060/4080 operating policy
- `docs/05_METRICS_AND_SCHEMAS.md` — output requirements
- `docs/06_IMPLEMENTATION_PROMPTS.md` — coding-agent guidance
- `docs/07_EXECUTION_ORDER.md` — stage order
- `configs/experiment_plan.yaml`
- `configs/downstream_tasks.yaml`
- `configs/hardware_profiles.yaml`
- `docs/AMENDMENTS.md` - registered amendments A001-A003
- `reports/TASK2_GPT2_SINK_PARITY.md`
- `reports/TASK3_GPT2_SUPPRESSION_HOOK.md`
- `reports/TASK4_NEUTRAL_CORPUS_AND_SINK_MAP.md`
- `reports/TASK5_NEURON_ATTRIBUTION.md`
- `reports/TASK6_NEURON_SELECTION.md`
- `reports/TASK7_GPT2_SUPPRESSION_SMOKE.md`
- `reports/STAGE_B_FULL_PHENOMENON.md` - formal Stage B results and analysis
- `reports/STAGE_C_QWEN_REPLICATION.md` - Stage C Qwen replication and null analysis
- `configs/frozen/neutral_corpus_manifest.json` - frozen neutral corpus (tracked)
- `configs/frozen/qwen2_5_1_5b_instruct/neutral_corpus_manifest.json` - Qwen-tokenized neutral corpus
- `configs/frozen/sink_scope.json` - frozen sink-heavy scope (tracked)
- `configs/frozen/neuron_attribution.csv` - frozen discovery ranking, 30,720 rows (tracked)
- `configs/frozen/neuron_attribution_metadata.json` - its provenance and per-layer diagnostics
- `configs/frozen/neuron_sets.json` - frozen targeted and layer-count-matched random sets
- `neuron_sink/evaluation.py` - paired neutral metrics, aggregation, and held-out smoke gate
- `neuron_sink/stage_b.py` - full registered grids, operating-point freeze/unlock, and formal gate
- `neuron_sink/stage_c.py` - Qwen-specific schema, path, lock, and formal-gate boundary
- `neuron_sink/stats.py` - deterministic paired bootstrap, random percentiles, and dose response
- `scripts/run_suppression_smoke.py` - exact 24/24/24 Task-7 runner
- `scripts/run_full_phenomenon.py` - Stage B GPT-2 and Stage C Qwen preflight/full runner
- `tests/test_evaluation.py` - frozen-grid, metric, schema, aggregation, and gate tests
- `tests/test_stage_b.py`, `tests/test_stats.py` - Stage B boundaries and statistics
- `tests/test_qwen_stage_c.py` - Qwen adapter, causal hook, and Stage-C boundary tests

## 8. Immediate next work package: stop at the Stage C null

Stage C is complete and the exact registered Qwen checkpoint did not clear its independent causal
gate. The required next action is to preserve and report that model-negative result. Do not
implement or run MMLU, ARC-Challenge, CulturalBench, or GSM8K for this checkpoint.

A different Qwen checkpoint, unit type, attribution method, fraction, or intervention grid would
be a new experiment and must be registered prospectively. It cannot replace or be folded into the
completed Stage C result.

## 9. Stage B status

Stage B is formally complete. GPT-2-small and GPT-2-medium both pass the causal gate, giving strong
support for the registered neuron-level phenomenon. GPT-2-small has `k*=15`; GPT-2-medium has only
an exploratory `k_max_effect=860` because no effective validation fraction met the CE ceiling.

## 10. What happens after Stage C

1. Preserve the Qwen2.5-1.5B null and its exploratory fallback without retuning.
2. Do not enter Stage D because the required Qwen causal gate failed.
3. Any alternate checkpoint or mechanistic extension needs a new registered experiment before
   measurement.
4. Do not begin later Sink-KD work merely to bypass the failed Stage C-to-D gate.

Never transfer raw neuron ids from GPT-2 to Qwen or teacher to student. Compare normalized depth, sparsity, sink reduction, dose response, and functional drift instead.

## 11. Non-negotiable scientific rules

- Do not use downstream benchmark examples or labels for neuron ranking.
- Do not use the Task-2 E1 mixture for discovery because it contains GSM8K.
- Discovery, validation, and test must be disjoint.
- Attribution is a ranking heuristic, not causal evidence.
- Causal evidence comes from held-out suppression against matched controls.
- For MLP layer `l`, attribution target must contain only later sink-heavy attention layers `j > l`.
- Random controls must preserve the targeted set's per-layer neuron counts.
- Do not silently widen candidate layers, change sink metric, change `k`, or change controls after seeing a null result.
- Do not edit upstream submodules.
- Do not overwrite completed result directories with different configurations.
- Record model/version/dtype/seed/dataset manifests/neuron ids/control ids/alpha/runtime/VRAM/provenance for every real run.

## 12. Useful commands for a new teammate

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/SyedNaveedMahmood/neuron-sink.git
cd neuron-sink
```

If already cloned:

```bash
git pull --ff-only
git submodule update --init --recursive
```

Before scientific work, verify the submodule commits exactly:

```bash
git -C upstream/sink-repro rev-parse HEAD
git -C upstream/sink-kd rev-parse HEAD
```

Expected:

```text
9ab67e914464b13863b67527d8ea14068ee9ff10
db114c9c5eb6ffc5de13e444c783408ea7401c62
```

Run the full project-local suite, including cached CUDA integration tests on a registered GPU:

```powershell
$env:NEURON_SINK_RUN_GPU_INTEGRATION='1'
$env:NEURON_SINK_HF_CACHE='F:\.cache\huggingface\neuron-sink'
$env:HF_HOME='F:\.cache\huggingface'
.\.venv\Scripts\python.exe -m pytest tests -q
```

On the RTX 4080 SUPER machine, use `F:\neuron-sink\.venv` and the cache path above.

## 13. Known issues and cautions

1. **Pinned GPT-2 wrapper recursion bug:** `upstream/sink-repro/common/intervention_analysis.py` can recurse in the standard GPT-2 path. The Task-2 adapter deliberately calls the frozen legacy implementation. Do not change the submodule to hide this provenance issue.
2. **Windows Hugging Face symlink warning:** cache behavior may be less storage-efficient; it did not affect parity.
3. **Git LFS:** Sink-KD includes checkpoint pointers. Do not trigger large downloads accidentally.
4. **Result interpretation:** Stage B supplies formal held-out targeted-vs-random evidence in both
   GPT-2 models. GPT-2-medium has no confirmatory low-drift `k*`; its 1.00% fallback is exploratory.
5. **Architecture gate:** Qwen2.5-1.5B failed Stage C. Do not run the downstream task-drift suite
   for this checkpoint.
6. **Stage C resume:** the initial process stopped after validation on a function-alias NameError.
   The test split remained unopened; the resume path hash-verified completed artifacts before
   freezing the operating point and accessing test once. See the tracked report and `resume.json`.

## 14. Status at handover

- Environment/source setup: PASS
- GPT-2-small sink reproduction: PASS (reproduced exactly on both dev machines)
- GPT-2 MLP suppression hook: PASS (reproduced exactly on both dev machines)
- Neutral corpus freeze + per-layer/per-head sink map: PASS
- Future-sink activation-times-gradient attribution: PASS
- Neuron selection and matched controls: PASS
- GPT-2-small targeted-vs-random smoke result: PASS
- Full GPT-2-small phenomenon confirmation: PASS; confirmatory `k*=15`
- Full GPT-2-medium phenomenon confirmation: PASS; no `k*`, exploratory `k_max_effect=860`
- Project-level Stage B interpretation: STRONG SUPPORT
- Qwen2.5-1.5B baseline sink preflight: PASS
- Qwen2.5-1.5B independent causal replication: NULL / MODEL-NEGATIVE; no `k*`, exploratory `k_max_effect=2330`
- Downstream task drift: BLOCKED by Stage C gate
- Sink-KD neuron comparison: NOT STARTED

**Next action: stop at and preserve the registered Stage C null. Do not start downstream tasks for
this checkpoint. Any alternative checkpoint or method requires a new prospective experiment.**
