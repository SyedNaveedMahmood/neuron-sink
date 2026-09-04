# Neuron-Sink Project Handover

Last updated: 2026-09-04 (Task 5 complete)

## 1. Project in one paragraph

This repository studies the neuron-level causal substrate of attention sinks. The central question is not whether attention sinks are inherently good or bad, but whether a visible first-position sink is supported by a sparse set of internal MLP neurons, whether suppressing those neurons causally changes sink strength, and how much functional drift follows. The longer-term comparison asks whether the same macroscopic sink can rely on different microscopic neuron-level substrates across naturally learned GPT-2 models and attention-distilled Sink-KD students.

Working title:

**Same Sink, Different Neurons: Neuron-Level Causal Localization of Attention Sinks and Performance Drift**

## 2. Repository and source provenance

Repository:

`https://github.com/SyedNaveedMahmood/neuron-sink`

Current project state before this handover commit:

`c97b8204d3d5aca8d80fb8402f29da6874e003fa`

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
| 2 | NVIDIA GeForce RTX 2060 | 12 GB | Windows 11 Pro | Task 2/3 reproduction, Task 4 |

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

Do not weaken or change scientific settings merely to make them fit the 2060. Allowed OOM responses are smaller batches, serial layer processing, microbatching, graph release, and streaming outputs. Changes to model, dtype, sequence length, attribution method, fractions, alphas, control count, or benchmark protocol require a documented amendment.

## 4. Registered primary neuron definition

For GPT-2, one neuron is one coordinate of the post-activation MLP intermediate tensor entering:

`transformer.h[layer].mlp.c_proj`

For GPT-2-small this tensor has shape:

`[batch, sequence, 3072]`

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

## 7. Current code surface

The project currently has only the implementation needed through Task 4. Do not prematurely build all future modules.

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
- `docs/AMENDMENTS.md` - registered amendments (A001: second dev GPU)
- `reports/TASK2_GPT2_SINK_PARITY.md`
- `reports/TASK3_GPT2_SUPPRESSION_HOOK.md`
- `reports/TASK4_NEUTRAL_CORPUS_AND_SINK_MAP.md`
- `reports/TASK5_NEURON_ATTRIBUTION.md`
- `configs/frozen/neutral_corpus_manifest.json` - frozen neutral corpus (tracked)
- `configs/frozen/sink_scope.json` - frozen sink-heavy scope (tracked)
- `configs/frozen/neuron_attribution.csv` - frozen discovery ranking, 30,720 rows (tracked)
- `configs/frozen/neuron_attribution_metadata.json` - its provenance and per-layer diagnostics

## 8. Immediate next task: Task 6 of the RTX 2060 phase

Task 5 is complete; see `reports/TASK5_NEURON_ATTRIBUTION.md`.

The next task should be kept narrow:

**Global top-k selection over the frozen discovery ranking, plus five layer-count-matched random
control sets with fixed seeds.**

Do not run any suppression sweep, held-out evaluation, or gate in Task 6. That is Task 7.

What Tasks 4 and 5 froze, which Task 6 must consume rather than recompute:

| Artefact | Value / path |
|---|---|
| Neutral corpus (tracked) | `configs/frozen/neutral_corpus_manifest.json` |
| Corpus manifest SHA-256 | `c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7` |
| Sink scope (tracked) | `configs/frozen/sink_scope.json` |
| Sink scope SHA-256 | `b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235` |
| Discovery ranking (tracked) | `configs/frozen/neuron_attribution.csv` |
| Attribution SHA-256 | `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692` |
| Eligible pool | 10 layers x 3072 = 30,720 `(layer, neuron)` pairs |
| Ranking score | `mean_abs_attr` (never `mean_signed_attr`) |

Required scientific intent:

1. Derive exact `k` for each registered smoke fraction `{0.05%, 0.10%, 0.25%}` of the 30,720
   eligible neurons, rounding to the nearest positive integer, and record the exact `k`.
2. Select the global top-`k` by `mean_abs_attr` across eligible `(layer, neuron)` pairs. Retain the
   per-layer rankings for diagnostics; do not select per layer.
3. Record per-layer counts of each targeted set. The head of the ranking is depth-skewed - 124 of
   the top 307 are layer 9 - which is exactly why the primary control is layer-count matched.
4. Generate five layer-count-matched random control sets per condition with fixed seeds: the same
   number of neurons in every layer, sampled from that layer's non-targeted neurons. A control set
   may not contain a targeted neuron for that same condition.
5. Reruns with the same seed must reproduce the control ids exactly, and the saved neuron-set file
   must be stable across reruns.
6. Reuse `neuron_sink.suppression.NeuronSet`; it already validates layer/neuron ids and is what the
   suppression hook consumes.

Task 6 needs no GPU and no model forward. It is selection bookkeeping over a frozen CSV.

## 9. Remaining RTX 2060 task sequence

The development/falsification phase has seven total tasks. Tasks 1-5 are complete.

- Task 6: global top-k selection + layer-count-matched random controls
- Task 7: 24/24/24 end-to-end suppression smoke experiment and plausibility gate

Task 7 is the first real neuron-level scientific gate. At least one targeted non-identity condition
must reduce held-out sink more than all five layer-count-matched random controls. If valid
implementation remains null on the predefined retry, stop rather than tuning toward a positive
result.

## 10. What happens after the 2060 phase

Only if the 2060 phenomenon looks plausible:

1. move to RTX 4080 SUPER;
2. run full GPT-2-small and GPT-2-medium phenomenon confirmation with 100/100/100 neutral splits;
3. use 20 matched-random control sets;
4. independently preflight/localize a task-capable Qwen2.5-1.5B-Instruct checkpoint;
5. only if that exact checkpoint passes the sink/neuron gate, evaluate downstream drift on MMLU, ARC-Challenge, CulturalBench, and GSM8K;
6. later apply the same independent localization/suppression protocol to Sink-KD teacher/student checkpoints.

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

Run the existing Task-3 tests:

```bash
python -m pytest tests/test_gpt2_adapter.py tests/test_suppression.py
```

On the existing Windows development machine, use the repository `.venv` and keep Hugging Face/model caches on `X:` rather than filling the system drive.

## 13. Known issues and cautions

1. **Pinned GPT-2 wrapper recursion bug:** `upstream/sink-repro/common/intervention_analysis.py` can recurse in the standard GPT-2 path. The Task-2 adapter deliberately calls the frozen legacy implementation. Do not change the submodule to hide this provenance issue.
2. **Windows Hugging Face symlink warning:** cache behavior may be less storage-efficient; it did not affect parity.
3. **Git LFS:** Sink-KD includes checkpoint pointers. Do not trigger large downloads accidentally.
4. **Result interpretation:** Task 2 proves the established attention sink reproduces. Task 3 proves the suppression machinery is correct. Neither task demonstrates that any specific neurons cause the sink. That scientific claim begins only after attribution and held-out targeted-vs-random suppression.

## 14. Status at handover

- Environment/source setup: PASS
- GPT-2-small sink reproduction: PASS (reproduced exactly on both dev machines)
- GPT-2 MLP suppression hook: PASS (reproduced exactly on both dev machines)
- Neutral corpus freeze + per-layer/per-head sink map: PASS
- Future-sink activation-times-gradient attribution: PASS
- Neuron selection and matched controls: NOT STARTED
- Targeted-vs-random causal result: NOT STARTED
- Downstream task drift: NOT STARTED
- Sink-KD neuron comparison: NOT STARTED

**Next action: Task 6 only — global top-k selection over the frozen discovery ranking plus five layer-count-matched random control sets with fixed seeds. No suppression run, no gate.**
