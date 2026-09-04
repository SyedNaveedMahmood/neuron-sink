# Upstream Source Provenance

This repository intentionally starts from two prior attention-sink codebases. Keep their provenance explicit so new neuron-level experiments can be traced back to the exact implementations they reuse.

## 1. Same Sink, Different Plumbing

- Upstream repository: `AlZubayer/MechanisticAccountofSinks`
- Branch: `main`
- Pinned source commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- Submodule path: `upstream/sink-repro`
- Local/source alias used in the prior project: `Sink-Repro`
- Intended reuse here: received-attention sink metric, GPT-2 anchor/route intervention semantics, NNsight execution, residual-sink analysis, matched controls, and loss/behavior evaluation.

An accessible collaborator fork also contains the reproduction code at `SyedNaveedMahmood/MechanisticAccountofSinks`; the submodule is pinned to the canonical source snapshot above rather than to a later merged fork state.

## 2. A Sink Without the Plumbing / Sink-KD

- Upstream repository: `AlZubayer/MechanisticAccountofSinks`
- Branch: `sink-inheritance-foundation`
- Pinned source commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Submodule path: `upstream/sink-kd`
- Local/source alias used in the prior project: `Sink-KD`
- Intended reuse here: teacher/student distillation setup, LogitKD/AttnKD controls, sink pattern/circuit/function comparisons, and causal delete/relocate evaluation.

The branch `sink-inheritance-slurm` in the collaborator fork is **not** substituted for `sink-inheritance-foundation`; it contains cluster-oriented changes and is a different lineage.

## Provenance source

The branch names and pinned source commits are recorded by the existing research integration manifest in `AlZubayer/ARCUS`:

- Sink-KD (`sink-inheritance-foundation`): `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Sink-Repro (`main`): `9ab67e914464b13863b67527d8ea14068ee9ff10`

When new code ports or rewrites an upstream intervention, record the upstream file/function and preserve its semantics. Prefer adapters or small extensions over silently altering the reference implementation.
