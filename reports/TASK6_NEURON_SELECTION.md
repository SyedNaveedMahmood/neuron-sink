# Task 6 — Global Top-k Neuron Selection and Matched Random Controls

## Source provenance

- neuron-sink commit before Task 6: `e1d20e4228ae8428996215648197b0aa004f188a`
- sink-repro commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- sink-kd commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Both upstream submodules were verified at their pinned commits and clean.
- Runtime: Python 3.12.4, PyTorch 2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0,
  datasets 4.8.4.
- Device: CPU. Task 6 loaded no model, ran no forward pass, called no GPU gate, and read no
  corpus split or downstream benchmark.
- No amendment was required. Fractions, ranking score, rounding, control definition, draw count,
  and seeds follow the registered design unchanged.

## Frozen inputs consumed

| Artefact | Verified value |
|---|---|
| Neutral corpus manifest | `configs/frozen/neutral_corpus_manifest.json` |
| Corpus manifest SHA-256 | `c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7` |
| Frozen sink scope | `configs/frozen/sink_scope.json` |
| Sink scope SHA-256 | `b8b4c623cb50d078b1e62c5a5bece1b24abab48933b45babd0e76856baaf0235` |
| Task-5 attribution CSV | `configs/frozen/neuron_attribution.csv` |
| Attribution SHA-256 | `9a87247bd8925c107da2e860b57cdebc0586f6404e8028b69cacab96ceb8d692` |
| Ranking score | `mean_abs_attr` |
| Eligible layers | zero-indexed `[0..9]` |
| MLP width | 3,072 in every eligible layer |
| Eligible pool | 30,720 `(layer, neuron)` pairs |

`neuron_sink.selection.load_frozen_attribution` restored the CSV fields to the exact integer,
float, and string types emitted by Task 5 and reproduced the attribution hash above. It then
verified all 30,720 rows, full neuron-id coverage per layer, both rank permutations, the global
`(-mean_abs_attr, layer, neuron)` total order, and every row's frozen future-sink targets. The
metadata's corpus and scope hashes were checked against the frozen Task-4 artefacts. The ranking
was not regenerated.

## Exact k and rounding

The registered rule is recorded as:

> nearest integer by Decimal ROUND_HALF_UP on eligible_pool_size*fraction_percent/100; minimum 1

At a pool size of 30,720, the reusable API gives:

| Fraction | Exact k | Phase |
|---:|---:|:--|
| 0.01% | 3 | Stage-B API check |
| 0.05% | 15 | frozen smoke condition |
| 0.10% | 31 | frozen smoke condition |
| 0.25% | 77 | frozen smoke condition |
| 0.50% | 154 | Stage-B API check |
| 1.00% | 307 | Stage-B API check |

Only the three registered smoke fractions were frozen. Selection is the global `rank_abs <= k`
cut, not a per-layer selection and never a sort by `mean_signed_attr`. The three targeted sets are
strictly nested.

## Targeted sets

Per-layer counts, including diagnostic zeros:

| Condition | Fraction | k | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `targeted_f0p05` | 0.05% | 15 | 0 | 0 | 1 | 0 | 2 | 1 | 1 | 1 | 4 | 5 |
| `targeted_f0p10` | 0.10% | 31 | 1 | 0 | 2 | 1 | 2 | 2 | 2 | 2 | 9 | 10 |
| `targeted_f0p25` | 0.25% | 77 | 4 | 1 | 5 | 4 | 3 | 3 | 15 | 4 | 20 | 18 |

Zero-count layers are retained in `per_layer_counts` and omitted from `NeuronSet.by_layer`, because
the Task-3 adapter correctly rejects empty layer entries. Exact neuron ids are stored in the frozen
artefact; they are attribution-ranked intervention candidates, not causal neurons.

## Layer-count-matched controls

Five controls were generated for each targeted fraction, giving 15 random conditions and 18 total
conditions. Stable ids are:

- `targeted_f0p05`; `layer_random_f0p05_s0` through `layer_random_f0p05_s4`;
- `targeted_f0p10`; `layer_random_f0p10_s0` through `layer_random_f0p10_s4`;
- `targeted_f0p25`; `layer_random_f0p25_s0` through `layer_random_f0p25_s4`.

For each target and draw, layers are visited in ascending order and `c` distinct ids are sampled
uniformly without replacement from that layer's `[0, 3072)` ids minus that condition's targeted
ids. No constraint was added between different control draws.

The RNG and composite seed are recorded as:

`np.random.default_rng([registered_base_seed, control_seed_draw_index, k])`

with base seed 0 and `control_seed` equal to the draw index. This separates the RNG streams for
different fractions while preserving the registered `control_seed: 0` meaning for the first draw.
The API generated all 20 Stage-B draws twice for every one of the six registered fractions; all 126
conditions (6 targets + 120 controls) were byte-identical between calls. Only five draws per smoke
target were frozen.

## Frozen outputs

- Frozen neuron sets: `configs/frozen/neuron_sets.json`
- Neuron-set SHA-256: `4fa22a2c68c8c3e56ed13b4f1c481b7b43d963b0190a619cacdc7c03c2672165`
- Run-local flat table: `neuron_sets.csv`, 738 rows, one per `(condition_id, layer, neuron)`
- First run: `results/task6_selection/run_20260904T115826Z`
- Determinism repeat: `results/task6_selection/run_20260904T115836Z`

The document hash covers the whole JSON except its own `neuron_sets_sha256` field. A saved-document
loader reconstructs every condition through the existing immutable `NeuronSet`, verifies exact k,
layer counts, ranges, target exclusion, draw-index coverage, and regenerates each control from its
recorded composite seed. A rerun refuses to overwrite a different frozen artefact; the second run
reported that the existing file already matched.

## Checks

| Check | Result |
|---|---|
| Task-5 typed CSV reload reproduces attribution SHA-256 | PASS |
| Metadata corpus/scope hashes match frozen Task-4 artefacts | PASS |
| 30,720 rows; layers exactly `[0..9]`; ids exactly `[0..3072)` per layer | PASS |
| `rank_abs` is `1..30720` and matches independent absolute-score sort | PASS |
| Exact k for all six registered fractions | PASS |
| Global targeted cuts are nested at k=15, 31, 77 | PASS |
| Every random control preserves its target's per-layer histogram | PASS |
| Every random control excludes its own target ids | PASS |
| Every set has unique in-range ids and only eligible layers | PASS |
| Five smoke draws are distinct per fraction | PASS |
| Twenty-draw API is deterministic for all six fractions | PASS |
| Every set round-trips through `NeuronSet` and saved JSON | PASS |
| Repeated runner produces the same neuron-set SHA-256 | PASS |
| Model/GPU/forward/suppression/held-out/downstream paths absent | PASS |

Runtime was 0.923 s for the first run and 0.907 s for the repeat. Peak GPU allocated and reserved
memory were both zero, as required for this CPU-only task.

Automated suite after Task 6: **157 passed, 155 subtests passed, 0 failed, 0 skipped** with the GPU
integration gate enabled, up from 127 passed / 142 subtests before Task 6. The 30 new Task-6 tests
instantiate no model.

## Interpretation boundary

Nothing in Task 6 is causal evidence. The frozen target ids were selected by a discovery attribution
heuristic and have not been suppressed. The random sets are controls prepared for Task 7. No sink,
CE, KL, top-1, held-out, or phenomenon-gate result was computed here.

## TASK6

PASS
