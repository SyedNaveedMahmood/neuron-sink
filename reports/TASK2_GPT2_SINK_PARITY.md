# Task 2 — GPT-2-Small Sink Parity

## Source provenance

- neuron-sink commit before Task 2: `beb79f1a6dfc5b96cfa6c2a690a0905835c0c24f`
- sink-repro commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- sink-kd commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Canonical metric: `compute_bos_attention_metric` in
  `upstream/sink-repro/common/intervention_analysis_legacy.py`, publicly re-exported by
  `common/intervention_analysis.py`.
- Canonical GPT-2-small Table-1 entry point:
  `python common/intervention_analysis.py --mode dataset --model gpt2` (manual engine by
  default). Its CLI runs all interventions, so the Task-2 baseline-only adapter invokes
  `intervention_a_baseline` and the metric directly.
- The standalone upstream frozen-file audit passed: all 12 files in
  `BASELINE_HASHES.json` matched their normalized SHA-256 values.

## Exact sink definition

- Target: key position `0`.
- Source/query positions: the second half of each sequence. At length 40 these are
  zero-indexed positions `20..39`.
- Layer scope: the upstream scaled GPT-2-small band `[3, 11)`, i.e. zero-indexed layers
  `3..10` (paper layers 4–11). `compute_band(12, "scaled")` returns this band.
- Head scope: all 12 attention heads in every included layer.
- Aggregation: for each example and included layer, mean
  `attention[head, query=20:40, key=0]` over heads and queries; mean those eight layer
  values; then mean the resulting scalar over all 300 examples.
- Sequence length: exactly 40 GPT-2 tokens, with `add_special_tokens=False`.

This is the existing upstream received-attention/BOS-position-0 definition; it does not
differ from the registered project definition.

## Model

- Requested model/tokenizer: `gpt2`
- Resolved repository: `openai-community/gpt2`
- Resolved model/tokenizer revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Dtype: float32
- Attention implementation configured on the Hugging Face model: `eager`
- Measurement engine: pinned Sink-Repro manual baseline
- Mode: `model.eval()`, deterministic algorithms enabled, no gradients, no training, no
  optimizer, and no quantization
- Device: `cuda:0`, NVIDIA GeForce RTX 2060 SUPER

## Dataset / fixture

- Corpus: upstream `corpus_providers.frozen_e1_corpus`, which delegates verbatim to
  `datasets_loader.sample_benchmark_datasets`.
- Components:
  - `stanfordnlp/sst2`, no config, train split, 100 examples
  - `openai/gsm8k`, `main` config, train split, 100 examples
  - `openai/openai_humaneval`, no config, test split, 100 examples
- Construction: discard examples shorter than 40 GPT-2 tokens, truncate retained examples
  to 40 tokens, shuffle deterministically within the upstream sampler, and take 100 per
  dataset.
- Seed: `0`, matching the registered seed-000 reference artifact.
- Evaluated examples: 300 total.
- Manifest SHA-256: `b7caf6b666b502790542a962b428a0b22dfa2d08bf77f51c383e249e7ce78c64`
- Saved ignored manifest:
  `results/task2_gpt2_sink_parity/run_20260904T090907Z/sample_manifest.json`.
- Frozen small fixture: the GPT-2 baseline (`int_a`) portion of
  `tests/fixtures/run_intervention_golden.pt` was rerun on its three fixed synthetic
  inputs. Maximum sink-metric difference and maximum raw-attention difference were both
  exactly `0.0`. The upstream tolerances are metric `atol=1e-5`, `rtol=1e-4` and attention
  `atol=1e-6`, `rtol=1e-5`.

## Results

- Machine-readable upstream reference: `0.563683` (standard error `0.002170`, `n=300`),
  transcribed in `BASELINE_HASHES.json` from the registered seed-000
  `bos_attention_stats_overall.csv` artifact.
- Upstream comparison tolerance: `atol + rtol*|reference| = 0.0000663683`.
- Reproduced baseline sink: `0.5636834649182856`.
- Absolute difference: `0.0000004649182855` (`4.649182855e-7`).
- Tolerance result: PASS.

Per-layer baseline received attention (all heads; second-half queries; key position 0):

| Zero-indexed layer | Paper layer | Mean sink |
|---:|---:|---:|
| 0 | 1 | 0.035294837 |
| 1 | 2 | 0.122190104 |
| 2 | 3 | 0.157175187 |
| 3 | 4 | 0.380206295 |
| 4 | 5 | 0.431022011 |
| 5 | 6 | 0.608165514 |
| 6 | 7 | 0.569377407 |
| 7 | 8 | 0.671649346 |
| 8 | 9 | 0.573061562 |
| 9 | 10 | 0.659879205 |
| 10 | 11 | 0.616106376 |
| 11 | 12 | 0.481964356 |

The full 12-layer × 12-head summary is stored in the ignored run's `summary.json`.

- Repeatability: the first three fixed corpus examples were rerun; maximum sink difference
  was `0.0` and maximum attention-map difference was `0.0`.
- Successful run wall time: `30.7884175` seconds (warm Hugging Face cache).
- Peak PyTorch GPU memory allocated: `544,522,240` bytes (`519.30 MiB`).
- Peak PyTorch GPU memory reserved: `595,591,168` bytes (`568.00 MiB`).

## Sanity checks

- PASS — canonical upstream sink metric executed successfully.
- PASS — all 300 tokenized sequences had exactly 40 tokens.
- PASS — all attention values were finite and within `[0, 1]`; NaN/Inf count was zero.
- PASS — attention rows summed to one; maximum absolute row-sum error was
  `4.172325134277344e-7`.
- PASS — causal future-position attention was zero; measured maximum was `0.0`.
- PASS — position 0 ranked first among the always-visible first-half key positions. Its
  received attention was `0.563683465`, versus `0.008036122` averaged over positions
  1–19, a ratio of `70.14`.
- PASS — three-example deterministic repeat was exact.
- PASS — frozen GPT-2 baseline fixture was exact.
- PASS — both upstream submodules remained clean and at their pinned commits.

## Deviations

1. The upstream Table-1 CLI was not invoked end-to-end because it always runs all ten
   interventions; Task 2 explicitly permits baseline only. The root adapter reuses its
   frozen corpus, baseline forward, band rule, and sink metric without copying them.
2. At the pinned commit, `common/intervention_analysis.py` rebinds
   `_legacy.manual_self_attention_new` to its wrapper and then calls that same rebound name
   in the OpenAI-GPT-2 fallback, causing `RecursionError: maximum recursion depth exceeded`.
   The adapter therefore imports the frozen `intervention_analysis_legacy.py` directly.
   For OpenAI GPT-2 the wrapper documents this path as semantically unchanged because both
   configuration flags it adds support for are false.
3. Deterministic PyTorch algorithms were explicitly enabled. The baseline has no stochastic
   model operation, and the result agrees with the upstream reference well inside its
   registered tolerance.
4. The successful runtime used the already downloaded `X:` Hugging Face cache. Initial
   acquisition emitted unauthenticated-request and unsupported-Windows-symlink warnings;
   these affect rate limits/storage efficiency, not model or dataset contents.
5. Two earlier ignored result directories are retained append-only:
   `run_20260904T090552Z` stopped on the pinned recursion defect before a sink value, and
   `run_20260904T090803Z` completed baseline computation but failed while serializing a
   NumPy boolean. The completed authoritative run is `run_20260904T090907Z`.

Rerun command (with cache/temp environment variables directed to `X:`):

```powershell
.venv\Scripts\python.exe scripts\run_gpt2_sink_parity.py --model-id gpt2 --revision main --sample-size 100 --cut-length 40 --seed 0 --repeat-size 3 --cache-dir X:\codex-cache\huggingface\neuron-sink
```

## TASK2_PARITY

PASS

## Reproduction on amended hardware (RTX 2060, 12 GB)

Amendment `A001` (`docs/AMENDMENTS.md`) registered a second development GPU. Task 2 was re-run
unchanged on that machine before Task 4 began, to re-establish the Phase-0 parity gate there rather
than assume it transfers. The original RTX 2060 SUPER result above is untouched.

- Repo commit at re-run: `31f8e56109f8db078d0514bf773294d611a4c0f0`
- GPU: `NVIDIA GeForce RTX 2060` (12 GB), Windows 11 Pro 10.0.26200
- Stack: Python 3.12.4, PyTorch 2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0, datasets 4.8.4
- Only code change: the hard-coded `"RTX 2060 SUPER"` assertion became
  `neuron_sink.provenance.require_registered_gpu("dev")`. Scientific logic and the inline
  provenance writer are unchanged.
- Ignored run directory: `results/task2_gpt2_sink_parity/run_20260904T103640Z`

Results were **bit-identical** to the RTX 2060 SUPER run:

| Quantity | RTX 2060 SUPER | RTX 2060 (12 GB) |
|---|---|---|
| Manifest SHA-256 | `b7caf6b6…78c64` | `b7caf6b6…78c64` |
| Reproduced sink | `0.5636834649182856` | `0.5636834649182856` |
| Absolute difference | `4.649182855e-7` | `4.649182855e-7` |
| Allowed difference | `0.0000663683` | `0.0000663683` |
| Max attention row-sum error | `4.172325134277344e-7` | `4.172325134277344e-7` |
| Max causal future attention | `0.0` | `0.0` |
| Position-0 concentration ratio | `70.14` | `70.14372010721634` |
| Evaluated examples | 300 | 300 |

All twelve per-layer received-attention values matched the table above to nine decimals. Every
sanity check passed, the deterministic three-example repeat was exact, and both submodules were
clean and at their pinned commits before and after.

- Wall time: `95.163` s (cold dataset download; the Task-2 corpus was fetched fresh on this machine)
- Peak GPU memory allocated: `520,142,848` bytes (`496.05 MiB`)
- Peak GPU memory reserved: `562,036,736` bytes (`536.00 MiB`)

Rerun command on this machine:

```powershell
$env:NEURON_SINK_HF_CACHE="D:\.cache\huggingface\neuron-sink"
.venv\Scripts\python.exe scripts\run_gpt2_sink_parity.py --model-id gpt2 --revision main --sample-size 100 --cut-length 40 --seed 0 --repeat-size 3 --cache-dir D:\.cache\huggingface\neuron-sink
```

### TASK2_PARITY (amended hardware)

PASS
