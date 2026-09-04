# Neuron-Sink Project Handover

Last updated: 2026-09-04

## 1. Project in one paragraph

This repository studies the neuron-level causal substrate of attention sinks. The central question is not whether attention sinks are inherently good or bad, but whether a visible first-position sink is supported by a sparse set of internal MLP neurons, whether suppressing those neurons causally changes sink strength, and how much functional drift follows. The longer-term comparison asks whether the same macroscopic sink can rely on different microscopic neuron-level substrates across naturally learned GPT-2 models and attention-distilled Sink-KD students.

Working title:

**Same Sink, Different Neurons: Neuron-Level Causal Localization of Attention Sinks and Performance Drift**

## 2. Repository and source provenance

Repository:

`https://github.com/SyedNaveedMahmood/neuron-sink`

Current project state before this handover commit:

`bfcb0201a7a46a3cad9be04b6ad55de685f15a0e`

Two upstream paper codebases are pinned as read-only Git submodules and must not be edited in place:

| Source | Path | Branch/source role | Pinned commit |
|---|---|---|---|
| Same Sink, Different Plumbing | `upstream/sink-repro` | original sink mechanism/reproduction code | `9ab67e914464b13863b67527d8ea14068ee9ff10` |
| A Sink Without the Plumbing / Sink-KD | `upstream/sink-kd` | attention-distillation teacher/student code | `db114c9c5eb6ffc5de13e444c783408ea7401c62` |

The upstream submodules are reference implementations. New code belongs in the root repository, preferably as thin adapters rather than copied upstream scripts.

## 3. Hardware plan

### Development / falsification machine

- GPU: NVIDIA GeForce RTX 2060 SUPER
- VRAM: 8 GB
- OS used so far: Windows 10
- Purpose: implementation, unit tests, GPT-2-small parity, hook validation, attribution smoke work, small 24/24/24 falsification run

Environment used successfully:

`X:\project\neuron-sink\.venv`

Verified stack:

- Python 3.12.5
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

## 7. Current code surface

The project currently has only the implementation needed through Task 3. Do not prematurely build all future modules.

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
- `reports/TASK2_GPT2_SINK_PARITY.md`
- `reports/TASK3_GPT2_SUPPRESSION_HOOK.md`

## 8. Immediate next task: Task 4 of the RTX 2060 SUPER phase

The next task should be kept narrow:

**Freeze the neutral discovery/validation/test corpus and build the GPT-2-small baseline per-layer/per-head sink map.**

Do not implement neuron attribution in Task 4.

Required scientific intent:

1. Build/freeze a neutral sink corpus that does not overlap with downstream benchmark examples.
2. Prefer the existing Sink-KD `openwebtext_validation_sink_300` construction/manifest semantics if available and usable.
3. Create disjoint discovery/validation/test roles.
4. For the 2060 smoke phase, use 24/24/24 examples.
5. Preserve sequence length 40 for the primary sink phenomenon test.
6. Measure baseline per-layer/per-head attention received by position 0.
7. Apply the registered sink-heavy layer rule only after the map is computed.
8. Save the frozen corpus manifest and sink-scope artifact with hashes so later attribution cannot silently change them.

Registered sink-heavy layer rule:

- layer sink score in the top quartile of that model's layers; AND
- layer sink score at least `0.15`

If fewer than two layers satisfy both, use the top two layers above `0.15`. If no layer exceeds `0.15`, the checkpoint fails sink preflight.

The map is a prerequisite for attribution because an MLP neuron at layer `l` may only be attributed to sink attention in later layers. Same-layer attention occurs before that layer's MLP and cannot be causally downstream of it.

## 9. Remaining RTX 2060 SUPER task sequence

The development/falsification phase has seven total tasks. Tasks 1–3 are complete.

- Task 4: neutral corpus freeze + per-layer/per-head sink map
- Task 5: future-sink activation-times-gradient attribution
- Task 6: global top-k selection + layer-count-matched random controls
- Task 7: 24/24/24 end-to-end suppression smoke experiment and plausibility gate

Task 7 is the first real neuron-level scientific gate. At least one targeted non-identity condition must reduce held-out sink more than all five layer-count-matched random controls. If valid implementation remains null on the predefined retry, stop rather than tuning toward a positive result.

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
- GPT-2-small sink reproduction: PASS
- GPT-2 MLP suppression hook: PASS
- Neuron attribution: NOT STARTED
- Targeted-vs-random causal result: NOT STARTED
- Downstream task drift: NOT STARTED
- Sink-KD neuron comparison: NOT STARTED

**Next action: Task 4 only — freeze neutral corpus splits and build the GPT-2-small per-layer/per-head sink map on the RTX 2060 SUPER.**
