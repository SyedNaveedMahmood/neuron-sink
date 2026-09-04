# Neuron-Level Causal Analysis of Attention Sinks

Research repository for the next-stage attention-sink project: localize sink-associated neurons/dimensions, suppress them causally, and measure both sink reduction and downstream performance drift.

## Status

**Scaffold / source-integration stage.** The two prior-paper codebases are pinned as exact upstream submodules. No new neuron-localization or neuron-suppression implementation has been added yet.

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

Can a sparse set of neurons or attention dimensions be causally linked to attention-sink formation, and how much general performance drifts when those units are suppressed?

The initial experiment will:

1. identify sink-heavy layers/heads;
2. rank candidate neurons/dimensions by sink relevance;
3. suppress top-ranked units at graded strengths;
4. compare against size- and layer-matched random controls;
5. measure change in sink strength together with CE/perplexity/task-performance drift;
6. repeat the analysis for the natural teacher sink and Sink-KD students.

## Repository layout

```text
neuron-sink/
├── upstream/
│   ├── sink-repro/        # exact pinned submodule for Same Sink, Different Plumbing
│   └── sink-kd/           # exact pinned submodule for Sink-KD
├── experiments/           # new experiments go here
├── configs/               # project-level run configs
├── scripts/               # project-level entry points
├── tests/                 # parity and intervention tests
├── results/               # generated outputs (ignored except .gitkeep)
├── SOURCE_BRANCHES.md
└── requirements.txt       # baseline environment inherited from Sink-Repro for now
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

The root `requirements.txt` currently mirrors the verified Sink-Repro environment. Before the first Sink-KD training run, reconcile any additional dependencies from `upstream/sink-kd` rather than silently changing the pinned baseline.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Next implementation milestone

The first new code should be a small falsification pilot on GPT-2-small: reuse the existing sink metric and NNsight harness, add MLP-neuron suppression plus layer-matched random controls, and record `sink_before`, `sink_after`, `delta_sink`, `ce_before`, `ce_after`, and `delta_ce`.

No paper-specific result is claimed by this repository yet; this commit only prepares the reproducible code base and provenance.
