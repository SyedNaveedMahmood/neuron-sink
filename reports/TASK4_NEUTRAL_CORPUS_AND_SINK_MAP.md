# Task 4 — Neutral Corpus Freeze and GPT-2-Small Sink Map

## Source provenance

- neuron-sink commit before Task 4: `31f8e56109f8db078d0514bf773294d611a4c0f0`
- sink-repro commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- sink-kd commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Both submodules were verified at their pinned commits and clean before and after every run.
- Corpus provider: `upstream/sink-kd/common/corpus_providers.py::openwebtext_corpus`, the
  registered `openwebtext_validation_sink_300` construction named by
  `configs/experiment_plan.yaml`.
- Sink metric cross-check: `compute_bos_attention_metric` and `intervention_a_baseline` in
  `upstream/sink-repro/common/intervention_analysis_legacy.py` — the same frozen functions Task 2
  used.
- Hardware: `NVIDIA GeForce RTX 2060` (12 GB), Windows 11 Pro, registered by amendment `A001`
  (`docs/AMENDMENTS.md`). Task 2 and Task 3 were re-run on this machine first and both reproduced
  their original results exactly; see their reports.
- Runtime: Python 3.12.4, PyTorch 2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0, datasets 4.8.4,
  float32, eager attention, `model.eval()`, deterministic algorithms, `cuda:0`.

## Part A — Frozen neutral corpus

### Source and construction

The registered source was used unchanged. `openwebtext_corpus` was called with
`split="validation"`, `purpose="sink"`, `n_blocks=300`, `seed=0`, which yields the corpus id
`openwebtext_validation_sink_300` exactly as registered.

- Dataset: `Skylion007/openwebtext` (streamed; now distributed as 80 native parquet shards, so the
  pinned `datasets==4.8.4` loads it without a dataset script).
- Document window: `[400000, 408000)` — upstream's `OPENWEBTEXT_SPLIT_WINDOWS["validation"]`. This
  window is **disjoint by construction** from the Sink-KD training window `[0, 400000)`, so a
  neutral example cannot contain a Sink-KD training token.
- Blocks are packed greedily with EOS separators from a `random.Random(0)` shuffle of the window's
  document order, then sliced `[0, 300)` from the reserved sink range
  (`SINK_BLOCKS_RESERVED = 300`).
- Tokenizer: `GPT2Tokenizer` from `gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` — the
  same revision the sink map and Task 2/3 use.

**One registered parameter differs from the Sink-KD default:** `block_size=40` rather than 128,
because `docs/00_MASTER_EXPERIMENT_DESIGN.md` registers 40 tokens as this project's primary sequence
length. `block_size` does not enter the upstream corpus id, so the corpus is still exactly
`openwebtext_validation_sink_300`. This is a parameter choice inside the registered construction,
not a change to it.

### Splits

The 300-block pool is partitioned into contiguous, disjoint windows of the frozen block order:

| Split | Size | Block indices | Smoke prefix (this phase) |
|---|---:|---|---:|
| discovery | 100 | `blk0` – `blk99` | first 24 (`blk0` – `blk23`) |
| validation | 100 | `blk100` – `blk199` | first 24 (`blk100` – `blk123`) |
| test | 100 | `blk200` – `blk299` | first 24 (`blk200` – `blk223`) |

Contiguous windows are how upstream itself guarantees disjointness, and the pool order is already a
seeded document shuffle, so a contiguous slice is not an ordered sample. The 24/24/24 smoke splits
used by the RTX 2060 phase are the deterministic **first 24 of each full split**, so they are nested
inside the eventual 100/100/100 Stage-B splits and no second freeze can contradict this one.

### Hashes and artefacts

- Project manifest SHA-256: `c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7`
  (covers item ids, split roles, token ids, and per-item metadata)
- Upstream corpus SHA-256: `8798eee8511245b16fe939dfa2d67eeb9f199443f11ea7842d1b5394dcf313ec`
  (upstream's own `manifest_sha256`, recorded verbatim so a later run can prove it saw
  byte-identical text)
- Tracked frozen manifest: `configs/frozen/neutral_corpus_manifest.json`
- Ignored run directory: `results/task4_neutral_corpus/run_20260904T104217Z`
- Build wall time: `186.08` s (dominated by streaming past 400,000 training documents to reach the
  validation window). Reruns hit the upstream block cache.

`configs/frozen/` is tracked because `results/` is gitignored; a freeze that only existed in an
ignored directory would not actually be frozen. Re-running the builder recomputes the manifest and
**refuses to overwrite** a frozen file whose hash differs.

### Part A checks

| Check | Result |
|---|---|
| Corpus id is the registered `openwebtext_validation_sink_300` | PASS |
| Pool size | 300 |
| All sequences exactly 40 tokens | PASS (`{"40": 300}`) |
| Split sizes 100/100/100 | PASS |
| Splits disjoint | PASS |
| Smoke splits are prefixes of the full splits | PASS |
| Item ids unique | PASS |
| Token ids within vocabulary range | PASS |
| Downstream-overlap guard | PASS — source is not MMLU/ARC/CulturalBench/GSM8K, and not the E1 mixture |
| Deterministic rebuild reproduces the manifest hash | PASS (`c6e07787…d480c7`) |

Recorded as a diagnostic, not a gate: decoding a block and re-encoding it reproduces the original
ids for **286/300** examples (95.3%). This is expected and is *not* a defect. Blocks are packed
spans of documents with EOS separators, so a block boundary can fall inside a byte-level BPE unit;
unlike a whole E1 example, a decoded mid-document chunk is not guaranteed to re-tokenize to itself.
The frozen `input_ids` are what every downstream stage consumes, so the model never sees the
re-encoded form.

### TASK4_CORPUS

PASS

## Part B — Baseline per-layer/per-head sink map

### Exact definition

- Target: key position `0`.
- Query positions: the second half of each sequence — at length 40, zero-indexed `20..39`.
- Per layer and head: mean of `attention[head, 20:40, 0]` over queries, then averaged over the 24
  discovery examples.
- Per-layer sink score: the mean of that layer's 12 head values.
- Split: **discovery only**, 24-example smoke prefix. The script calls `require_discovery_split`,
  which raises on a validation or test split, so a held-out split cannot reach this stage or the
  ranking stage that follows.

### Execution paths and why both were run

Every later stage measures interventions through the Hugging Face forward, because
`neuron_sink.suppression` registers a PyTorch pre-hook on a real `nn.Module`. The map is therefore
built from the Hugging Face `output_attentions=True` forward, so the frozen scope matches the
execution path that will measure suppression.

To prove that choice did not move the metric, each example was **also** run through the pinned
upstream manual baseline (`intervention_a_baseline`) that produced the Task-2 number, and both were
compared against upstream `compute_bos_attention_metric`:

| Comparison | Max absolute difference | Tolerance |
|---|---:|---:|
| This module's map decomposition vs upstream metric (same attentions) | `3.012950178e-8` | `7.023e-5` |
| Hugging Face forward vs upstream manual forward, sink metric | `4.842877388e-8` | `7.023e-5` |
| Hugging Face forward vs upstream manual forward, raw attention | `5.602836609e-6` | `1.1e-5` |

Mean sink over the upstream parity band `[3, 11)`: `0.6023032290` from the map decomposition versus
`0.6023032319` from the upstream manual path. The two execution paths and the two metric
implementations agree to within 5e-8 on the metric.

The `[3, 11)` band is **parity-only**. The sink-heavy rule below is applied over all 12 layers, as
`docs/00_MASTER_EXPERIMENT_DESIGN.md` requires.

### Results

Baseline received attention at position 0, GPT-2-small, 24 neutral discovery examples:

| Zero-indexed layer | Paper layer | Mean sink | Sink-heavy | Top-quartile heads |
|---:|---:|---:|:--:|:--|
| 0 | 1 | 0.038521 | | |
| 1 | 2 | 0.123772 | | |
| 2 | 3 | 0.161033 | | |
| 3 | 4 | 0.410571 | | |
| 4 | 5 | 0.449738 | | |
| 5 | 6 | 0.650954 | | |
| 6 | 7 | 0.612202 | | |
| 7 | 8 | **0.703484** | **yes** | 2, 10, 11 |
| 8 | 9 | 0.599864 | | |
| 9 | 10 | **0.697401** | **yes** | 1, 6, 9 |
| 10 | 11 | **0.694212** | **yes** | 1, 8, 10 |
| 11 | 12 | 0.549964 | | |

The full 12x12 per-layer/per-head matrix is in `configs/frozen/sink_scope.json` and the run's
`sink_map.json`.

### Frozen sink scope

The registered rule — top quartile **and** score >= 0.15; fall back to the top two above the floor
if fewer than two qualify; fail preflight if none clears the floor — was applied **after** the map
was computed.

- Quartile size: `ceil(12 / 4) = 3` layers, ties broken by ascending layer index.
- Layers above the 0.15 floor: 10 of 12.
- **Rule applied: `top_quartile_and_floor`** — the primary branch, no fallback.
- **Sink-heavy layers: `[7, 9, 10]`** (all three well above the floor).
- Sink-heavy heads (top quartile within each selected layer, `ceil(12/4) = 3` heads): layer 7 →
  `{2, 10, 11}`, layer 9 → `{1, 6, 9}`, layer 10 → `{1, 8, 10}`. Head identity is diagnostic; the
  primary objective averages over the selected layers and all heads.
- **Eligible MLP layers for Task-5 attribution: `[0 … 9]`** — every layer strictly before the last
  sink-heavy layer. Layers 10 and 11 are ineligible because no sink-heavy attention layer follows
  them.
- Per-MLP-layer future sink targets are frozen alongside, e.g. layer 0 → `(7, 9, 10)`, layer 7 →
  `(9, 10)`, layer 9 → `(10,)`. Every target is strictly later than its MLP layer, enforcing the
  causal-ordering constraint in `AGENTS.md`.
- Frozen scope SHA-256: `b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235`
- Tracked artefact: `configs/frozen/sink_scope.json`
- Ignored run directory: `results/task4_sink_map/run_20260904T104809Z`

### Part B checks

| Check | Result |
|---|---|
| Map decomposition reproduces the upstream metric | PASS (`3.01e-8` <= `7.02e-5`) |
| Hugging Face and upstream manual forwards agree | PASS (`4.84e-8` <= `7.02e-5`) |
| All attention values finite | PASS (0 non-finite) |
| Attention rows sum to one | PASS (max error `3.576278687e-7`) |
| Causal — no attention to future positions | PASS (max `0.0`) |
| Map finite | PASS |
| Sink-heavy layers clear the registered floor | PASS |
| At least one eligible MLP layer exists | PASS (10 layers) |
| Rule applied without an incomplete fallback | PASS |
| Repeat run reproduces the scope hash exactly | PASS (`b8b4c623…f0235`) |
| Submodules clean and pinned after the run | PASS |

- Wall time: `18.05` s (first run), `6.07` s (warm repeat)
- Peak GPU memory allocated: `529,351,680` bytes (`504.83 MiB`)
- Peak GPU memory reserved: `585,105,408` bytes (`558.00 MiB`)

Peak VRAM is ~0.5 GB against a 12 GB card, and would fit the 8 GB RTX 2060 SUPER with large margin,
so amendment `A001` did not relax any memory constraint.

### TASK4_SINK_MAP

PASS

## Observations (not claims)

1. The neutral-corpus layer profile has the same shape as the Task-2 E1 profile, with values
   systematically slightly higher (e.g. layer 7: `0.7035` neutral vs `0.6716` on E1). The three
   sink-heavy layers selected here — 7, 9, 10 — are also the three highest-scoring layers in the
   Task-2 E1 table. This is a consistency observation across two different corpora, not a result.
2. **Nothing here is causal evidence.** The map describes where the sink is measured, and the frozen
   scope constrains which MLP layers Task 5 may attribute. No neuron has been ranked or suppressed.

## Deviations and implementation notes

1. **`block_size=40` instead of the Sink-KD default 128.** Required by this project's registered
   primary sequence length. Discussed under Part A; the corpus id is unchanged.
2. **Upstream import isolation.** Both submodules ship same-named top-level modules in `common/`
   (`datasets_loader`, `nnsight_engine`, `provenance`, …), and they mix import styles: sink-kd's
   `corpus_providers` uses a `from . import x` fallback while sink-repro's
   `intervention_analysis_legacy` uses an absolute `from datasets_loader import …`. Putting either
   directory on `sys.path` permanently, as `scripts/run_gpt2_sink_parity.py` does, means the first
   snapshot imported wins for the process and could shadow `neuron_sink.provenance`.
   `neuron_sink/upstream_bridge.py` therefore scopes the `sys.path` entry to the import and harvests
   each snapshot's modules into a private cache. No upstream file was modified. The corpus manifest
   hash was verified identical before and after this mechanism was introduced.
3. **The registered fallback branch is nearly unreachable.** Any layer above the floor outscores
   every layer below it, so when at most a quartile of layers clear the floor, those layers *are*
   the top quartile. Consequently "fewer than two satisfy both criteria" can only happen when
   exactly one layer is above the floor, and the "top two above the floor" fallback can then only
   return that one layer. The implementation records this case as `fallback_incomplete` rather than
   silently returning a short scope. This is a property of the registered rule, surfaced rather than
   changed; it did not arise here, since the primary branch applied.
4. **Decode/re-encode round-trip is 286/300**, by construction rather than by defect. See Part A.
5. **`pytest` is not pinned in `requirements.txt`.** It was not pinned on the RTX 2060 SUPER either;
   `requirements.txt` deliberately mirrors the verified Sink-Repro runtime environment. It is
   installed separately in the virtual environment.
6. Hugging Face emitted the known Windows symlink warning and an unauthenticated-request warning.
   Neither affects dataset or model content.

## Rerun commands

```powershell
$env:NEURON_SINK_HF_CACHE="D:\.cache\huggingface\neuron-sink"
$env:HF_HOME="D:\.cache\huggingface"

.venv\Scripts\python.exe scripts\prepare_neutral_corpus.py
.venv\Scripts\python.exe scripts\map_sink_layers.py

$env:NEURON_SINK_RUN_GPU_INTEGRATION="1"
.venv\Scripts\python.exe -m pytest tests\
```

Automated suite: **83 passed, 35 subtests passed, 0 failed, 0 skipped** (with the CUDA integration
gate enabled).

## What Task 5 may now assume

- The neutral corpus, its split roles, and their hashes are frozen and tracked.
- Attribution may read **only** the discovery split; `require_discovery_split` enforces this.
- The sink-heavy attention scope is `[7, 9, 10]`, frozen with a hash, and must not be recomputed
  per condition.
- Eligible MLP layers are `[0 … 9]`, with per-layer future-sink targets already frozen.

## TASK4

PASS
