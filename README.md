# Neuron-Level Causal Analysis of Attention Sinks

Research repository for the next-stage attention-sink project: localize sink-associated neurons/dimensions, suppress them causally, and measure both sink reduction and downstream performance drift.

## Status

**Stages A-C are complete. Stage B passed for both GPT-2 checkpoints, while the independent
Qwen2.5-1.5B Stage C replication produced a valid model-specific null.** Amendment A005 registers
Stage C2 as a separate positive-signed replication on fresh neutral blocks; its 20-example runtime
preflight passes. Amendment A007 registers Stage C3, which additionally fixes the causal
*reachability* defect found by decomposing the Stage-C result per sink layer: 41.1% of the graded
metric was unreachable by the selected neurons, so a sign fix alone cannot clear the gate. Stage C
remains unchanged, and Stage D stays blocked unless a checkpoint passes its own locked held-out
gate.

| Task | Status |
|---|---|
| 1. Environment and source verification | PASS |
| 2. GPT-2-small sink parity | PASS |
| 3. GPT-2 MLP suppression hook | PASS |
| 4. Neutral corpus freeze + per-layer/per-head sink map | PASS |
| 5. Future-sink activation-times-gradient attribution | PASS |
| 6. Top-k selection + layer-matched random controls | PASS |
| 7. 24/24/24 suppression smoke experiment and plausibility gate | PASS |
| Stage B. Full 100/100/100 GPT-2-small and GPT-2-medium confirmation | PASS / PASS |
| Stage C. Qwen2.5-1.5B independent replication | NULL / MODEL-NEGATIVE |
| Stage C2. Fresh-corpus positive-signed Qwen replication | PREFLIGHT PASS / FULL RUN PENDING |
| Stage C3. Reachability-aware localization (A007) | REGISTERED / IMPLEMENTED / NOT RUN |

Stage B independently recomputed the discovery sink map, future-sink neuron ranking, and 20
layer-count-matched controls for each checkpoint, selected the operating point from validation
only, and opened each locked 100-example test split once. Five of six registered fractions passed
the formal test gate in each model. GPT-2-small selected confirmatory `k*=15` (0.05%); GPT-2-medium
passed the causal gate but had no validation fraction that combined the required effect with the
registered CE budget, so its `k_max_effect=860` fallback is exploratory only. See the full analysis
in [`reports/STAGE_B_FULL_PHENOMENON.md`](reports/STAGE_B_FULL_PHENOMENON.md).

The execution order is deliberately gated:

1. reproduce the existing sink measurement;
2. test on held-out neutral text whether targeted MLP-neuron suppression reduces the sink more than matched random suppression;
3. only if that effect is real, evaluate performance drift on MMLU, ARC-Challenge, CulturalBench, and GSM8K;
4. then compare natural teacher sinks with Sink-KD student sinks.

Stage C independently mapped Qwen's sink, ranked all 232,960 eligible `(layer, neuron)` pairs,
generated 20 matched controls per fraction, and ran the full locked 100/100/100 grid. At 1.00%,
targeted suppression beat matched random but achieved only 7.07% test RSR, Spearman `0.6`, and
Delta CE `1.093`; the formal gate therefore failed. See
[`reports/STAGE_C_QWEN_REPLICATION.md`](reports/STAGE_C_QWEN_REPLICATION.md).

Stage C2 changes only target ordering: it ranks strictly positive discovery
`mean_signed_attr` values so suppression has the preregistered first-order direction of reducing
the sink. It uses OpenWebText blocks 300-599, disjoint from Stage C's blocks 0-299, while retaining
the same checkpoint, hook, fractions, alphas, controls, metrics, and formal gate. See amendment
`A005` and [`configs/experiment_plan_c2.yaml`](configs/experiment_plan_c2.yaml).

Development/smoke implementation is registered for the RTX 2060 class (see amendment `A001` in
[`docs/AMENDMENTS.md`](docs/AMENDMENTS.md)); full runs are registered for an RTX 4080 SUPER.

## Experiment design

Start with [`AGENTS.md`](AGENTS.md) and [`docs/README.md`](docs/README.md).

The main design files are:

- `docs/00_MASTER_EXPERIMENT_DESIGN.md`
- `docs/01_PHENOMENON_GATE.md`
- `docs/02_DOWNSTREAM_TASKS.md`
- `docs/03_IMPLEMENTATION_SPEC.md`
- `docs/04_HARDWARE_RUNBOOK.md`
- `docs/05_METRICS_AND_SCHEMAS.md`
- `docs/06_IMPLEMENTATION_PROMPTS.md`
- `docs/AMENDMENTS.md`
- `configs/experiment_plan.yaml`
- `configs/experiment_plan_c2.yaml`
- `configs/downstream_tasks.yaml`

## Upstream codebases

This project starts from two existing attention-sink codebases:

1. **Same Sink, Different Plumbing** (`upstream/sink-repro`)
   - branch: `main`
   - pinned commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
   - reuse: sink metric, NNsight intervention harness, anchor/route interventions, residual-sink analysis, evaluation controls.

2. **A Sink Without the Plumbing / Sink-KD** (`upstream/sink-kd`)
   - branch: `sink-inheritance-foundation`
   - pinned commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
   - reuse: teacher/student setup, attention-distillation conditions, pattern-vs-circuit-vs-function controls, delete/relocate evaluation.

See `SOURCE_BRANCHES.md` for provenance.

## Research question

Can a sparse set of internal MLP neurons be causally linked to attention-sink formation, and how much neutral and downstream performance drifts when those units are suppressed?

The primary neuron is defined as an MLP intermediate coordinate immediately before the MLP output projection (`mlp.c_proj` input for GPT-2; `mlp.down_proj` input for Qwen2.5). Attribution ranks candidate units, but causal claims require held-out targeted suppression against layer-count-matched random controls.

## Repository layout

```text
neuron-sink/
├── AGENTS.md              # coding-agent rules
├── upstream/
│   ├── sink-repro/        # exact pinned submodule for Same Sink, Different Plumbing
│   └── sink-kd/           # exact pinned submodule for Sink-KD
├── docs/                  # registered scientific + implementation design
├── experiments/           # new experiment implementation/results structure
├── configs/               # design + run configs
│   └── frozen/            # immutable frozen artifacts (corpus manifest, sink scope)
├── scripts/               # project-level entry points
├── tests/                 # parity and intervention tests
├── results/               # generated outputs (ignored except .gitkeep)
├── SOURCE_BRANCHES.md
└── requirements.txt
```

## Clone

```bash
git clone --recurse-submodules https://github.com/SyedNaveedMahmood/neuron-sink.git
cd neuron-sink
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Environment

The root `requirements.txt` currently mirrors the verified Sink-Repro environment. Additional task-evaluation dependencies should be pinned only when the relevant adapter is implemented.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Next implementation milestone

Run the separately registered Stage C2 positive-signed replication and evaluate its unchanged
formal gate. Do not run MMLU, ARC-Challenge, CulturalBench, or GSM8K for this Qwen checkpoint unless
C2 passes. C2 must not reinterpret or overwrite the completed Stage C null.
