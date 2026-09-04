# Registered Amendments

`AGENTS.md` requires the experiment to be implemented exactly as registered in
`docs/00_MASTER_EXPERIMENT_DESIGN.md` "unless a documented amendment is added before looking at the
affected result". This file is that record. Each amendment states what changed, what did **not**
change, and why the change cannot bias a result.

Amendments are append-only. Do not edit a landed amendment; add a superseding one.

---

## A001 — Add a second registered development GPU (RTX 2060, 12 GB)

- **Date registered:** 2026-09-04
- **Registered before:** the Task-4 neutral corpus freeze and GPT-2-small sink map
- **Status:** active

### What changed

The development/falsification hardware role previously named exactly one GPU:

> NVIDIA RTX 2060 SUPER, 8 GB VRAM, Windows 10

Task-1 through Task-3 results were produced on that machine. Implementation has since moved to a
second machine whose GPU is:

> NVIDIA GeForce RTX 2060, 12 GB VRAM, Windows 11 Pro 10.0.26200

(`torch.cuda.get_device_name(0)` returns exactly `NVIDIA GeForce RTX 2060`; total memory is
12,884,443,136 bytes.) Both GPUs are now registered for the `dev` role.

Affected files:

- `docs/04_HARDWARE_RUNBOOK.md`
- `configs/hardware_profiles.yaml` (`dev_rtx2060_super` → `dev_rtx2060_family`)
- `neuron_sink/provenance.py` (`REGISTERED_GPUS`, which is what code actually enforces)
- `scripts/run_gpt2_sink_parity.py`, `scripts/verify_gpt2_suppression_hook.py`,
  `tests/test_suppression.py` — the hard-coded `"RTX 2060 SUPER"` assertion is replaced by
  `require_registered_gpu("dev")`. This is the only change made to those scripts; their inline
  provenance writers and all scientific logic are untouched.

The stale `X:\codex-cache\huggingface\neuron-sink` cache default, which pointed at a drive that does
not exist on the second machine, now reads the `NEURON_SINK_HF_CACHE` environment variable and
otherwise falls back to the Hugging Face default.

### What did not change

No scientific setting is altered by this amendment:

- model ids, revisions, and dtype (GPT-2 stays float32, eager attention);
- sequence length (40 tokens primary);
- the sink metric, its target position, query rule, and layer/head scope;
- neuron definition and the `mlp.c_proj` hook point;
- neuron fractions, alphas, `k` selection, or the matched-random control definition;
- corpus construction, split sizes, or the anti-leakage rules;
- the phenomenon gate thresholds in `docs/01_PHENOMENON_GATE.md`.

### Why this cannot bias a result

The second GPU has 12 GB rather than 8 GB. **The extra VRAM is not spent widening any registered
condition.** The RTX 2060 SUPER smoke policy in `docs/04_HARDWARE_RUNBOOK.md` — batch size 1,
one eligible MLP layer per backward pass, serial layer processing, 24/24/24 examples, 5 matched
controls — is retained verbatim as the `dev` policy regardless of which GPU is in use. Headroom is
recorded in provenance, not converted into a larger experiment. The full-run role remains the
RTX 4080 SUPER, unchanged.

Both GPUs are Turing-class consumer parts running the same pinned software stack (PyTorch
2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0, datasets 4.8.4, CUDA 12.8), so numerical semantics
are expected to be identical. That expectation is **verified rather than assumed**: Task 2 (sink
parity) and Task 3 (suppression-hook audit) were re-run on the RTX 2060 before Task 4 began, and
their results are recorded in `reports/TASK2_GPT2_SINK_PARITY.md` and
`reports/TASK3_GPT2_SUPPRESSION_HOOK.md` under "Reproduction on amended hardware". If that
re-verification had failed, Task 4 would not have proceeded.

---

## A002 - Freeze a tokenizer-specific neutral corpus for Stage C

- **Date registered:** 2026-09-04
- **Registered before:** any Stage-C Qwen sink measurement or neuron attribution
- **Status:** active

### What changed

Stage C uses `Qwen/Qwen2.5-1.5B-Instruct` at the exact resolved revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The existing frozen neutral manifest stores GPT-2
token ids and cannot be passed to Qwen without changing the text represented by those ids.
Stage C therefore gets a separate append-only manifest at
`configs/frozen/qwen2_5_1_5b_instruct/neutral_corpus_manifest.json`, produced before any Qwen
attention is measured.

The manifest is built by the existing `scripts/prepare_neutral_corpus.py` adapter using the
Qwen checkpoint's pinned tokenizer and the same registered upstream provider. The generated
manifest records its tokenizer revision and content hash, and the Stage-C runner refuses any
other hash.

### What did not change

- corpus provider: pinned Sink-KD `openwebtext_corpus`;
- source and document window: `stas/openwebtext-10k`, validation window;
- seed: 0;
- sequence length: 40 tokens;
- 300 packed blocks and disjoint 100/100/100 discovery, validation, and test roles;
- split assignment, anti-leakage checks, and test locking;
- Stage-C model, sink metric, ranking method, fractions, alphas, random controls, and gate.

### Why this cannot bias a result

Tokenization is a deterministic part of the model input, not a result-dependent choice. Qwen
cannot scientifically consume GPT-2 vocabulary ids. Registering the Qwen tokenizer before the
first Qwen forward preserves the same neutral source-text construction and split protocol while
making each 40-token block valid for the checkpoint under test. No sink value, attribution score,
validation outcome, or downstream label was inspected when this amendment was registered.

---

## A003 - Use dtype-derived numerical audit tolerances for Stage C

- **Date registered:** 2026-09-04
- **Registered before:** inspecting any Stage-C sink score or selecting Qwen sink-heavy layers
- **Status:** active

### What changed

The pinned upstream attention and metric tolerances were calibrated for float32 GPT-2. Stage C is
registered in bfloat16. For Qwen audit checks only, the absolute tolerance is now
`max(upstream_tolerance, 2 * torch.finfo(torch.bfloat16).eps)`, which is `0.015625`. This applies
to attention row-sum validity, decomposition of the sink map against the upstream scalar metric,
and equality of the differentiable future-sink objective to the frozen attention-map reduction.
Causal-mask leakage retains the upstream absolute tolerance because masked future entries are
expected to be exactly zero.

### What did not change

The model outputs are not rounded, corrected, or recast before sink measurement. The sink metric,
layer/head aggregation, `0.15` floor, sink-layer selection, neuron ranking, intervention grid,
random controls, CE/KL metrics, and causal-gate thresholds are unchanged. Baseline and
intervention forwards remain bfloat16 and use the same dtype.

### Why this cannot bias a result

This is a representation-precision audit bound derived only from the registered dtype, not from a
sink magnitude or causal effect. A first baseline attempt stopped at the audit boundary because
the float32 tolerance was inappropriately strict; no sink score, selected layer, attribution,
validation result, or test result was inspected. The new bound is registered before repeating
the preflight and applies symmetrically to every Qwen condition.

---

## A004 - Clarify Qwen tokenizer-revision provenance

- **Date recorded:** 2026-09-05
- **Status:** clarification only; no scientific setting changed

The pinned upstream corpus provider copies a tokenizer revision attribute that Transformers 5.3.0
leaves unset. Consequently, the frozen Qwen corpus JSON contains
`"tokenizer_revision": null`, just like the existing GPT-2 neutral manifest. A002's statement that
the manifest records the revision should be read as the complete corpus-freeze artifact set: the
freeze run's `run_config.json` records the resolved SHA
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, while the tracked manifest records the tokenizer
name and immutable token-id content hash.

The Stage-C runner hard-codes and records that exact tokenizer/model revision, enforces the Qwen
tokenizer name and registered manifest hash, and verifies the loaded model's `_commit_hash` before
measurement. This clarification changes no input IDs, split, model forward, sink value, ranking,
intervention, statistic, or gate result.
