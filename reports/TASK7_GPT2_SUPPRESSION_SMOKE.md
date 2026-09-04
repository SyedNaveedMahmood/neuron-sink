# Task 7 — GPT-2-Small End-to-End Suppression Smoke

## Outcome

**Execution: COMPLETE. Registered Phase-1 smoke gate: PASS.**

On the locked 24-example test split, every targeted non-identity condition reduced the
frozen sink metric more than all five of its layer-count-matched random controls.  All
three targeted sets also had the registered dose direction
`RSR(alpha=1) <= RSR(alpha=0.5) <= RSR(alpha=0)`.

This is the permissive RTX-2060 plausibility gate. It justifies moving to the registered
RTX 4080 SUPER full confirmation; it is not the formal 100-example/20-control phenomenon
result, does not select `k*`, and does not authorize downstream benchmarks.

## Registered inputs

- Model: `openai-community/gpt2`, revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Runtime: float32, eager attention, `model.eval()`, deterministic algorithms, batch 1
- Hardware: NVIDIA GeForce RTX 2060, 12 GB, registered by amendment A001
- Neutral corpus: `openwebtext_validation_sink_300`, sequence length 40
- Splits: first frozen 24 discovery, 24 validation, and 24 test examples; disjoint
- Sink scope: zero-indexed attention layers `[7, 9, 10]`, all heads, key position 0,
  second-half query positions
- Target fractions / exact `k`: 0.05% / 15, 0.10% / 31, 0.25% / 77
- Conditions: three targeted sets and five frozen layer-count-matched random controls
  for each target
- Alpha order: `[1.0, 0.5, 0.0]`
- Evaluation protocol: `neutral_next_token_sink_ce_kl_top1_v1`; CE, KL, and top-1 use
  next-token positions `0..38`, with KL direction `p_baseline || p_intervened`

Frozen hashes consumed and reverified:

| Artefact | SHA-256 |
|---|---|
| Neutral corpus manifest | `c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7` |
| Sink scope | `b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235` |
| Discovery attribution | `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692` |
| Frozen neuron sets | `4fa22a2c68c8c3e56ed13b4f1c481b7b43d963b0190a619cacdc7c03c2672165` |

No ranking, reselection, control redraw, sink-scope recomputation, or benchmark data path
was invoked by Task 7.

## Held-out test result

The test baseline sink was `0.718978450`; baseline CE was `4.274522` nats/token.
Relative sink reduction (RSR) is `(S0 - S1) / S0`; positive values mean the sink weakened.

| Target | k | alpha | Target RSR | Maximum RSR among 5 random controls | Target − max random | ΔCE | KL | Top-1 flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05% | 15 | 0.5 | 0.058025 | 0.001215 | 0.056810 | 0.034297 | 0.029206 | 0.095085 |
| 0.05% | 15 | 0.0 | 0.129197 | 0.002465 | 0.126732 | 0.126444 | 0.138438 | 0.216880 |
| 0.10% | 31 | 0.5 | 0.042501 | -0.000195 | 0.042695 | 0.054855 | 0.040861 | 0.118590 |
| 0.10% | 31 | 0.0 | 0.094772 | -0.000371 | 0.095143 | 0.203137 | 0.201048 | 0.261752 |
| 0.25% | 77 | 0.5 | 0.056175 | -0.001146 | 0.057321 | 0.100390 | 0.069443 | 0.139957 |
| 0.25% | 77 | 0.0 | 0.149656 | -0.002228 | 0.151884 | 0.403734 | 0.375917 | 0.330128 |

All six strict targeted-vs-maximum-random comparisons passed.  The strongest registered
smoke sink reduction was 14.97% for `k=77`, `alpha=0`; the smallest set (`k=15`) reduced
the held-out sink by 12.92% under full suppression.  Functional drift increased under
stronger suppression, and no Task-7 result should be used to choose `k*`: operating-point
selection belongs to the full validation protocol on the RTX 4080 SUPER.

For transparency, the five-control RSR ranges were:

| Fraction | alpha | Minimum random RSR | Median random RSR | Maximum random RSR |
|---|---:|---:|---:|---:|
| 0.05% | 0.5 | -0.000531 | -0.000143 | 0.001215 |
| 0.05% | 0.0 | -0.001015 | -0.000273 | 0.002465 |
| 0.10% | 0.5 | -0.001971 | -0.001233 | -0.000195 |
| 0.10% | 0.0 | -0.003935 | -0.002461 | -0.000371 |
| 0.25% | 0.5 | -0.002535 | -0.001899 | -0.001146 |
| 0.25% | 0.0 | -0.005026 | -0.003788 | -0.002228 |

## End-to-end correctness checks

| Check | Result |
|---|---|
| Frozen artefact hashes and cross-links | PASS |
| Exact 24/24/24 split membership and order | PASS |
| Exact frozen 18-condition order | PASS |
| Exact alpha order `[1.0, 0.5, 0.0]` | PASS |
| Alpha=1 logits exact for all identity rows | PASS |
| Alpha=1 attentions exact for all identity rows | PASS |
| All logits and attentions finite | PASS |
| No all-zero logits | PASS |
| Maximum attention row-sum error | `3.57627868652344e-7` |
| Maximum causal future attention | `0.0` |
| Hooks restored after every condition | PASS |
| Final baseline replay / state leakage | exact / PASS |
| Held-out target beats all five matched controls | PASS, 6 of 6 non-identity cells |
| Dose direction for at least one fraction | PASS, 3 of 3 fractions |

The saved table contains 3,960 rows: 1,320 for each split. It has 165
`(split, condition, alpha)` cells, including one baseline cell per split, with exactly 24
examples in every cell. There were 3,960 registered grid forwards plus one final baseline
state-leakage probe.

## Runtime and provenance

- Registered run directory:
  `results/task7_gpt2_smoke/run_20260904T122521Z`
- Wall time: `77.1648095` seconds
- Split evaluation times: discovery `24.516` s, validation `24.111` s, test `24.690` s
- Peak GPU allocated: `565,910,528` bytes (`539.69 MiB`)
- Peak GPU reserved: `624,951,296` bytes (`596.00 MiB`)
- Python 3.12.4; PyTorch 2.10.0+cu128; Transformers 5.3.0; NNsight 0.7.0;
  datasets 4.8.4; NumPy 2.4.3; CUDA runtime 12.8
- Root `HEAD` recorded by the run:
  `e1d20e4228ae8428996215648197b0aa004f188a`
- Sink-Repro: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- Sink-KD: `db114c9c5eb6ffc5de13e444c783408ea7401c62`

The run correctly records `repo_dirty_at_run=true`: completed Task-6 files and the new
Task-7 implementation were present in the working tree but had not yet been committed.
The exact dirty-file list is retained in `run_config.json` and `provenance.json`; both
upstream submodules were pinned and clean.

Before the registered run, a full project-local test pass with GPU integration reported
`167 passed, 155 subtests passed` with no failures or skips. A 1/1/1 preflight exercised
all 18 conditions and three alphas, passed identity/validity/leakage checks, peaked at
`565,910,528` allocated bytes, and correctly emitted `NOT_EVALUATED_DRY_RUN` rather than a
scientific conclusion.

## Machine-readable outputs

The ignored run directory contains:

- `run_config.json` and `provenance.json`
- an exact run-local `neuron_sets.json`
- `per_example_discovery.csv`, `per_example_validation.csv`, `per_example_test.csv`
- combined `per_example.csv`
- `aggregate.csv` and `aggregate.json`
- `smoke_gate.json`
- `summary.json`

## Interpretation boundary and next step

Task 7 provides held-out targeted-vs-matched-random causal evidence at smoke scale and
passes the predefined plausibility gate. It does not establish the formal model-level
phenomenon gate because the registered full confirmation requires 100/100/100 examples,
20 controls, the full fraction/alpha grid, paired uncertainty, and GPT-2-small plus
GPT-2-medium on an RTX 4080 SUPER.

The next work package is Prompt 7: generalize this exact implementation for the registered
full GPT-2-small/medium experiment on the RTX 4080 SUPER. Do not run Qwen or downstream
benchmarks yet.

## TASK7

**SMOKE GATE PASS**
