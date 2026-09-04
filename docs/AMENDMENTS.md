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
