# Task 5 — Future-Sink Activation-times-Gradient Neuron Attribution

## Source provenance

- neuron-sink commit before Task 5: `c97b8204d3d5aca8d80fb8402f29da6874e003fa`
- sink-repro commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- sink-kd commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Both submodules were verified at their pinned commits and clean before and after every run.
- Sink metric cross-check: `compute_bos_attention_metric` and `compute_band` in
  `upstream/sink-repro/common/intervention_analysis_legacy.py`, and the tolerances
  `METRIC_ATOL`/`METRIC_RTOL` in `upstream/sink-repro/common/nnsight_engine.py` — the same frozen
  functions and tolerances Task 2 and Task 4 used. No upstream file was read through `sys.path`;
  all upstream imports go through `neuron_sink/upstream_bridge.py`.
- Hardware: `NVIDIA GeForce RTX 2060` (12 GB), Windows 11 Pro, registered by amendment `A001`
  (`docs/AMENDMENTS.md`), gated in code by `require_registered_gpu("dev")`.
- Runtime: Python 3.12.4, PyTorch 2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0, datasets 4.8.4,
  float32, eager attention, `model.eval()`, `torch.use_deterministic_algorithms(True)`, `cuda:0`.
- **No amendment was required.** Model, revision, dtype, sequence length, corpus, split roles, sink
  metric, sink scope, neuron definition, and the attribution objective are all consumed exactly as
  registered.

## Frozen inputs consumed, not recomputed

| Artefact | Value |
|---|---|
| Neutral corpus manifest | `configs/frozen/neutral_corpus_manifest.json` |
| Corpus manifest SHA-256 | `c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7` |
| Sink scope | `configs/frozen/sink_scope.json` |
| Sink scope SHA-256 | `b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235` |
| Sink-heavy attention layers | `[7, 9, 10]` |
| Eligible MLP layers | `[0 … 9]` |
| Split | discovery only, 24-example smoke prefix (`blk0` – `blk23`), seq_len 40 |

`neuron_sink.sink_metrics.load_frozen_sink_scope` refuses the scope unless it still hashes to its
recorded `sink_scope_sha256`, unless its `corpus_manifest_sha256` equals the manifest actually
loaded, and unless every `future_sink_layers[l]` entry is strictly later than `l` and is one of the
frozen sink-heavy layers. The Task-4 targets were read from that file; none was re-derived or
widened.

## Part A — Differentiable sink scorer

### Why it was needed

`neuron_sink/sink_metrics.py` computed the Task-4 map with `.detach()` into float64 NumPy. That is
correct for measurement and unusable for a backward pass. Task 5 adds
`differentiable_sink_score(attentions, layers, heads=None, target_pos=0)`, which returns a 0-dim
`torch.Tensor` and never detaches, converts to NumPy, or calls `.item()` inside the objective.

### Definition, and why it is the same scalar

Within a layer it takes the mean of `attention[heads, seq_len // 2:, target_pos]` over the selected
heads and second-half queries jointly, then means those per-layer values over the selected layers.
Because every head sees the same number of query positions, that is arithmetically identical to
`sink_scalar_from_map`, which means per-head first and then over heads — which Task 4 already showed
reproduces upstream `compute_bos_attention_metric` to `3.01e-8`. `heads=None` (all heads) is the
registered primary objective; the head parameter exists only for a separately registered
head-restricted robustness run and was not used here.

### Numerical check, on real attention tensors, before any gradient was trusted

Phase 1 of every run compares the differentiable scorer against the frozen metric and against
upstream on that run's own 24 examples. The run stops before Phase 2 if any comparison exceeds the
upstream tolerance.

| Comparison | Max absolute difference | Tolerance |
|---|---:|---:|
| Differentiable vs `sink_scalar_from_map`, all 12 layers | `7.820757952e-8` | `7.023e-5` |
| Differentiable vs upstream `compute_bos_attention_metric`, parity band `[3, 11)` | `6.332993507e-8` | `7.023e-5` |
| Differentiable vs `sink_scalar_from_map`, each layer's future-sink targets | `8.816520380e-8` | `7.023e-5` |
| The scalar actually differentiated vs the frozen metric, per eligible layer | `6.310572787e-8` | `7.023e-5` |

The last row is the one that matters for the gradient: it compares `S_future(l)` as evaluated inside
the backward pass against the frozen metric computed independently on a separate no-grad forward, for
all ten eligible layers. The tolerance is upstream's own `METRIC_ATOL + METRIC_RTOL * |value|`.

The unit tests repeat the same equality on fixed synthetic causal attention tensors across six
different layer subsets, plus the upstream parity band, plus a hand-computed head-restricted mean.

### Gradient existence

`neuron_sink/attribution.py` captures the tensor entering `transformer.h[l].mlp.c_proj` with a
forward-pre-hook that **detaches** it into a differentiable leaf and returns it in place. The value
the model computes with is unchanged (unit test: logits are bit-identical to a plain forward), and
`dS_future/da` is unchanged, because a partial derivative with respect to `a` does not depend on how
`a` was produced. The backward is a single `torch.autograd.grad(S_future, a)`.

Model parameters are set to `requires_grad_(False)` and the ranking API refuses to run if any
parameter still requires grad. With integer input ids and frozen parameters, nothing upstream of the
capture point requires grad, so no graph is built for layers `0 … l-1` at all and no embedding
backward ever runs. Gradients were finite and non-zero for every one of the 240 (layer, example)
backward passes.

### TASK5_SCORER

PASS

## Part B — Causal-order verification

`AGENTS.md` forbids attributing layer `l`'s neurons to same-layer pre-MLP attention. Task 5 turns
that constraint from an assumption into a measurement. `objective_depends_on_layer` builds a sink
objective from a chosen set of attention layers and reports whether the resulting score is a function
of layer `l`'s `c_proj` input at all.

For all ten eligible MLP layers, on a real discovery example:

| Probe | Result |
|---|---|
| Objective over the frozen future targets `j > l` | depends on `a(l)` — **True** for all 10 layers |
| Objective over the layer's own attention `j = l` | **False** for all 10 layers |
| Objective over earlier sink-heavy layers `j < l` (layers 8, 9 vs sink layer 7) | **False** |

"False" here is stronger than a numerically zero gradient: the score carries no `grad_fn`, so the
same-layer or earlier-layer objective is not a function of that MLP's activation in any part of the
computation. Three enforcement layers agree: `require_future_targets` rejects a non-causal target
list, `load_frozen_sink_scope` rejects a non-causal frozen scope, and this probe measures the model
itself.

## Part C — Neuron attribution

### Exact objective

For each eligible MLP layer `l`, with `a` the tensor entering `transformer.h[l].mlp.c_proj`:

- `S_future(l)` = the sink metric restricted to the frozen sink-heavy attention layers `j > l`;
- `I(l, n) = mean over examples and token positions of | a(l,n) * dS_future(l)/da(l,n) |`.

Ranking is by `mean_abs_attr`. `mean_signed_attr` is saved and is never ranked by; 46 of the top 100
neurons have a negative signed mean, so ranking on the signed score would be cancellation, not
importance.

Token scope is **all 40 positions**, matching the registered aggregation
(`configs/experiment_plan.yaml`: `aggregation: mean_over_examples_and_tokens`, unrestricted) and the
intervention this ranking predicts, which scales the neuron at all positions
(`suppression_positions: all`). `n_tokens = 24 x 40 = 960` per neuron.

### Setup

- Model `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, float32, eager
  attention, `model.eval()`, parameters frozen.
- Discovery split only, 24-example smoke prefix. `require_discovery_split` is called in the script
  before an output directory is even created, and again inside `rank_neurons`; every selected item's
  own frozen split role is checked as well.
- Batch size 1; one eligible MLP layer per backward pass; captured tensors released between examples;
  per-neuron sums accumulated in float64 on the CPU.
- The Hugging Face `output_attentions=True` forward — the path Task 4's map was built from and the
  path `neuron_sink.suppression` hooks — so attribution and every later measurement travel the same
  route.

### Results

10 eligible MLP layers x 3072 neurons = **30,720** `(layer, neuron)` rows.

| MLP layer | Future sink targets | `S_future(l)` | mean `\|a*g\|` | median | max | top neuron | max `\|grad\|` |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 0 | 7, 9, 10 | 0.698366 | 9.5816e-06 | 8.2186e-06 | 1.7828e-04 | 1115 | 4.2942e-03 |
| 1 | 7, 9, 10 | 0.698366 | 1.0668e-05 | 9.4062e-06 | 1.2808e-04 | 342 | 6.4706e-03 |
| 2 | 7, 9, 10 | 0.698366 | 1.1323e-05 | 1.0646e-05 | 3.3130e-04 | 2015 | 1.1612e-02 |
| 3 | 7, 9, 10 | 0.698366 | 1.0642e-05 | 1.0079e-05 | 1.9475e-04 | 2664 | 1.0069e-02 |
| 4 | 7, 9, 10 | 0.698366 | 1.2238e-05 | 1.1113e-05 | 3.7600e-04 | 787 | 5.6643e-03 |
| 5 | 7, 9, 10 | 0.698366 | 1.2433e-05 | 1.1053e-05 | 4.1840e-04 | 1790 | 5.6870e-03 |
| 6 | 7, 9, 10 | 0.698366 | 1.2872e-05 | 1.0556e-05 | 3.3260e-04 | 437 | 5.2123e-03 |
| 7 | 9, 10 | 0.695806 | 1.3678e-05 | 1.2006e-05 | 4.7473e-04 | 2367 | 3.6718e-03 |
| 8 | 9, 10 | 0.695806 | 1.6365e-05 | 1.1359e-05 | 2.4661e-03 | 1253 | 8.6980e-03 |
| 9 | 10 | 0.694212 | 1.8969e-05 | 1.3548e-05 | 1.2232e-03 | 840 | 8.1631e-03 |

Global: mean `1.2877e-05`, median `1.0516e-05`, max `2.4661e-03`, min `3.4210e-09`.

Two independent consistency ties to Task 4, neither of which was constructed to hold:

- `S_future(9)` averages the sink over layer 10 alone and equals `0.694212`, which is exactly Task
  4's frozen per-layer sink for layer 10 (`0.694212`).
- `S_future(0)` = `0.698366` = the mean of Task 4's frozen layer-7/9/10 sinks
  (`0.703484`, `0.697401`, `0.694212`).

### Top 20 by `mean_abs_attr` — a diagnostic, not a selection

No `k` has been chosen and no neuron set exists. Top-k selection and matched random controls are
Task 6; this table exists so the ranking is legible in the report.

| Rank | Layer | Neuron | `mean_abs_attr` | `mean_signed_attr` | `mean_abs_activation` | Targets |
|---:|---:|---:|---:|---:|---:|:--|
| 1 | 8 | 1253 | 2.466062e-03 | +2.381131e-03 | 3.6303 | 9, 10 |
| 2 | 9 | 840 | 1.223225e-03 | +8.646271e-04 | 3.4465 | 10 |
| 3 | 8 | 428 | 1.146708e-03 | -1.146600e-03 | 0.4415 | 9, 10 |
| 4 | 8 | 1961 | 8.573102e-04 | +8.051393e-04 | 0.3421 | 9, 10 |
| 5 | 9 | 1889 | 6.079284e-04 | +5.142841e-04 | 0.5557 | 10 |
| 6 | 8 | 2108 | 5.888591e-04 | +5.672751e-04 | 0.5506 | 9, 10 |
| 7 | 7 | 2367 | 4.747321e-04 | +1.199171e-04 | 2.5391 | 9, 10 |
| 8 | 5 | 1790 | 4.184011e-04 | -4.103728e-04 | 1.9010 | 7, 9, 10 |
| 9 | 4 | 787 | 3.759975e-04 | -3.211839e-04 | 1.6161 | 7, 9, 10 |
| 10 | 9 | 1329 | 3.638724e-04 | +3.500515e-04 | 0.3889 | 10 |
| 11 | 9 | 978 | 3.606015e-04 | +1.736392e-04 | 0.7157 | 10 |
| 12 | 6 | 437 | 3.325980e-04 | +2.837945e-04 | 1.3090 | 7, 9, 10 |
| 13 | 2 | 2015 | 3.312966e-04 | +1.086501e-04 | 0.5434 | 7, 9, 10 |
| 14 | 4 | 1894 | 3.296344e-04 | -2.256858e-04 | 1.7818 | 7, 9, 10 |
| 15 | 9 | 342 | 3.194714e-04 | +2.136799e-04 | 0.1995 | 10 |
| 16 | 5 | 1888 | 3.039019e-04 | -2.491227e-04 | 1.5779 | 7, 9, 10 |
| 17 | 9 | 1305 | 2.895795e-04 | +2.661300e-04 | 0.4359 | 10 |
| 18 | 9 | 253 | 2.745583e-04 | +8.120275e-05 | 0.1690 | 10 |
| 19 | 9 | 688 | 2.386625e-04 | +1.926317e-04 | 0.3387 | 10 |
| 20 | 7 | 2402 | 2.333423e-04 | -2.161195e-04 | 0.8813 | 9, 10 |

Layer composition of the head of the ranking, recorded now so Task 6's per-layer control matching
cannot be tuned after the fact:

| Prefix | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top 30 | 1 | 0 | 2 | 1 | 2 | 2 | 2 | 2 | 8 | 10 |
| top 100 | 4 | 2 | 6 | 4 | 5 | 3 | 19 | 7 | 25 | 25 |
| top 307 (1%) | 14 | 10 | 11 | 7 | 12 | 10 | 43 | 21 | 55 | 124 |

### Checks

| Check | Result |
|---|---|
| Differentiable scorer reproduces the frozen metric and upstream | PASS (`8.82e-8` <= `7.02e-5`) |
| The scalar actually differentiated is the frozen metric | PASS (`6.31e-8` <= `7.02e-5`) |
| Future targets are reachable from every eligible MLP layer | PASS (10/10) |
| Same-layer and earlier-layer objectives are unreachable | PASS (10/10, no `grad_fn`) |
| Every frozen target is strictly later than its MLP layer | PASS |
| Gradients finite for every (layer, example) | PASS (0 non-finite values) |
| No (layer, example) produced an all-zero gradient | PASS (0 of 240) |
| Row count | PASS (30,720 = 10 x 3072) |
| Neuron ids cover exactly `[0, 3072)` for every eligible layer | PASS |
| Only eligible layers appear | PASS (`[0 … 9]`; 10 and 11 absent) |
| `mean_abs_attr >= \|mean_signed_attr\|` for every row | PASS |
| `rank_abs` is a permutation of `1 … 30720` | PASS |
| Discovery split only, item ids equal the frozen smoke prefix | PASS |
| Validation split rejected by the ranking entry point | PASS (`LeakageError`, exit 1) |
| Test split rejected by the ranking entry point | PASS (`LeakageError`, exit 1) |
| Scope was frozen against the corpus actually loaded | PASS |
| Repeat run reproduces the attribution hash exactly | PASS (`9a87247b…b8d692`) |
| Submodules clean and pinned after the run | PASS |

- Attribution hash: `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692`
- Wall time: `26.68` s (full run), `11.00` s (warm repeat), `22.87` s (4-example preflight)
- Peak GPU memory allocated: `556,922,368` bytes (`531.13 MiB`), identical for every one of the ten
  layers
- Peak GPU memory reserved: `610,271,232` bytes (`582.00 MiB`)

Peak VRAM is `531 MiB` against a 12 GB card — `26 MiB` above Task 4's baseline-only `505 MiB`, which
is the cost of the single-layer backward graph. It would fit the 8 GB RTX 2060 SUPER with large
margin, so amendment `A001` again bought nothing scientific.

### Memory and OOM

The acceptance criterion in `docs/06_IMPLEMENTATION_PROMPTS.md` Prompt 4 requires the scorer to run
on 4 examples without OOM before the full run. It did: `results/task5_attribution/run_20260904T112037Z`,
peak `531.13 MiB`, PASS. There was no OOM at any point, so no OOM response was needed and the method
was not changed.

### TASK5_ATTRIBUTION

PASS

## Observations (not claims)

1. **Nothing here is causal evidence.** Attribution is a ranking heuristic. No neuron in this report
   has been shown to cause the sink, and none may be described as causal. Causal evidence requires
   held-out suppression against layer-count-matched random controls, which is Task 7.
2. Scores are not calibrated across layers. Later eligible layers have both a shorter gradient path
   to their targets and a smaller target set (layer 9 targets one attention layer; layer 0 targets
   three), and the score mass rises monotonically with depth (layer mean `9.58e-06` at layer 0 to
   `1.90e-05` at layer 9). The registered selection rule is nevertheless a *global* top-k across
   eligible pairs (`docs/00_MASTER_EXPERIMENT_DESIGN.md`, "Neuron-set sizes"), so Task 6 will inherit
   a depth-skewed set — 124 of the top 307 come from layer 9 alone. This is a property of the
   registered objective, surfaced rather than changed, and it is exactly why the primary control is
   layer-count matched.
3. The ranking is not strongly sparse under this heuristic: the top 1% of eligible neurons carry only
   8.65% of the total `|a*g|` mass. That is a description of the score distribution, not a prediction
   about suppression, and it is not a reason to change `k`.
4. Signed attributions are mixed in sign at the head of the ranking (46 of the top 100 are negative),
   which is why the registered ranking score is the absolute one.
5. The two exact agreements with Task 4's frozen per-layer sinks (layer-10 sink reproduced by
   `S_future(9)`, and the layer-7/9/10 mean by `S_future(0)`) are consistency observations across two
   independently written code paths, not results.

## Deviations and implementation notes

1. **Token scope is all 40 positions.** The registered aggregation is
   `mean_over_examples_and_tokens` with no restriction, and the suppression it predicts applies at
   all positions, so the attribution mean does too. Positions before the second-half query window
   carry non-zero gradient through the key/value route, which a unit test pins.
2. **The capture hook detaches.** This is a memory decision, not a scientific one: it makes the
   backward graph exactly the sub-network between the neuron and its targets. The captured value is
   unchanged (logits are bit-identical to a plain forward) and the partial derivative is unchanged.
3. **`torch.no_grad()` is deliberately overridden** inside `score_example`, but
   `torch.inference_mode()` cannot be, so it is reported as an explicit error rather than silently
   yielding no gradient.
4. **The ranking is frozen to `configs/frozen/`** (`neuron_attribution.csv`, 2.9 MB, plus
   `neuron_attribution_metadata.json`), for the same reason Task 4 froze the corpus and scope:
   `results/` is gitignored, so a freeze that lived only there would not be frozen. Re-running
   recomputes the ranking and **refuses to overwrite** a frozen file whose `attribution_sha256`
   differs.
5. **`rank_abs` / `rank_abs_in_layer` columns** are the ranking this task produces, ordered by
   `(-mean_abs_attr, layer, neuron)` so the order is total and a rerun is byte-comparable. They are
   not a selection: no `k`, no top-k set, no controls, no suppression exist in Task 5.
6. `future_sink_layers` is pipe-joined inside its CSV cell (`7|9|10`); the separator is recorded in
   the metadata JSON.
7. `torch.use_deterministic_algorithms(True)` was enabled for the whole run, including the backward
   passes. No PyTorch kernel on this path lacked a deterministic implementation.

## Rerun commands

```powershell
$env:NEURON_SINK_HF_CACHE="D:\.cache\huggingface\neuron-sink"
$env:HF_HOME="D:\.cache\huggingface"

# memory preflight required by docs/06_IMPLEMENTATION_PROMPTS.md Prompt 4
.venv\Scripts\python.exe scripts\rank_neurons.py --max-examples 4

# the registered run
.venv\Scripts\python.exe scripts\rank_neurons.py

# held-out splits must be refused
.venv\Scripts\python.exe scripts\rank_neurons.py --split validation
.venv\Scripts\python.exe scripts\rank_neurons.py --split test

$env:NEURON_SINK_RUN_GPU_INTEGRATION="1"
.venv\Scripts\python.exe -m pytest tests\
```

Automated suite: **127 passed, 142 subtests passed, 0 failed, 0 skipped** (with the CUDA integration
gate enabled), up from 83 passed / 35 subtests at the end of Task 4.

Run directories (gitignored): `results/task5_attribution/run_20260904T112037Z` (preflight),
`run_20260904T112207Z` (registered run), `run_20260904T112250Z` (determinism repeat).

## What Task 6 may now assume

- A frozen discovery-split ranking of all 30,720 eligible `(layer, neuron)` pairs exists at
  `configs/frozen/neuron_attribution.csv`, with hash
  `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692`.
- The ranking score is `mean_abs_attr`; `mean_signed_attr` is diagnostic only.
- Per-layer neuron counts at the head of the ranking are recorded above, so layer-count-matched
  random controls have a fixed target to match.
- Only layers `[0 … 9]` are present, and every row's `future_sink_layers` is strictly later than its
  layer.
- No neuron has been selected or suppressed. Nothing in Task 5 is causal evidence.

## TASK5

PASS
