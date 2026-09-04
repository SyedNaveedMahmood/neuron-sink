# Neuron-Level Causal Analysis of Attention Sinks

Research repository for the next-stage attention-sink project: localize sink-associated neurons/dimensions, suppress them causally, and measure both sink reduction and downstream performance drift.

## Status

**Stage A (implementation and falsification) is in progress: Tasks 1-5 of 7 complete.** The two
prior-paper codebases are pinned as exact upstream submodules, and the experiment is specified in
`docs/` plus machine-readable configs in `configs/`.

| Task | Status |
|---|---|
| 1. Environment and source verification | PASS |
| 2. GPT-2-small sink parity | PASS |
| 3. GPT-2 MLP suppression hook | PASS |
| 4. Neutral corpus freeze + per-layer/per-head sink map | PASS |
| 5. Future-sink activation-times-gradient attribution | PASS |
| 6. Top-k selection + layer-matched random controls | not started |
| 7. 24/24/24 suppression smoke experiment and plausibility gate | not started |

No causal claim exists yet. Tasks 2-4 establish that the sink reproduces, that the suppression
machinery is exact, and where the sink is measured. Task 5 ranks all 30,720 eligible
`(layer, neuron)` pairs on the discovery split, but attribution is a ranking heuristic: no neuron
has been selected or suppressed, and nothing so far is causal evidence.

The execution order is deliberately gated:

1. reproduce the existing sink measurement;
2. test on held-out neutral text whether targeted MLP-neuron suppression reduces the sink more than matched random suppression;
3. only if that effect is real, evaluate performance drift on MMLU, ARC-Challenge, CulturalBench, and GSM8K;
4. then compare natural teacher sinks with Sink-KD student sinks.

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

Task 6: global top-k selection over the frozen discovery ranking at the registered smoke fractions,
plus five layer-count-matched random control sets with fixed seeds. The frozen inputs it must
consume — corpus manifest, sink scope, and the 30,720-row neuron ranking — are in `configs/frozen/`;
see `handover.md` section 8.

Full GPT-2-small/medium confirmation and Qwen/downstream runs remain reserved for the RTX 4080
SUPER.
