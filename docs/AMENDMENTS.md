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

---

## A002 - Freeze a tokenizer-specific neutral corpus for Stage C

- **Date registered:** 2026-09-04
- **Registered before:** any Stage-C Qwen sink measurement or neuron attribution
- **Status:** active

### What changed

Stage C uses `Qwen/Qwen2.5-1.5B-Instruct` at the exact resolved revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The existing frozen neutral manifest stores GPT-2
token ids and cannot be passed to Qwen without changing the text represented by those ids.
Stage C therefore gets a separate append-only manifest at
`configs/frozen/qwen2_5_1_5b_instruct/neutral_corpus_manifest.json`, produced before any Qwen
attention is measured.

The manifest is built by the existing `scripts/prepare_neutral_corpus.py` adapter using the
Qwen checkpoint's pinned tokenizer and the same registered upstream provider. The generated
manifest records its tokenizer revision and content hash, and the Stage-C runner refuses any
other hash.

### What did not change

- corpus provider: pinned Sink-KD `openwebtext_corpus`;
- source and document window: `stas/openwebtext-10k`, validation window;
- seed: 0;
- sequence length: 40 tokens;
- 300 packed blocks and disjoint 100/100/100 discovery, validation, and test roles;
- split assignment, anti-leakage checks, and test locking;
- Stage-C model, sink metric, ranking method, fractions, alphas, random controls, and gate.

### Why this cannot bias a result

Tokenization is a deterministic part of the model input, not a result-dependent choice. Qwen
cannot scientifically consume GPT-2 vocabulary ids. Registering the Qwen tokenizer before the
first Qwen forward preserves the same neutral source-text construction and split protocol while
making each 40-token block valid for the checkpoint under test. No sink value, attribution score,
validation outcome, or downstream label was inspected when this amendment was registered.

---

## A003 - Use dtype-derived numerical audit tolerances for Stage C

- **Date registered:** 2026-09-04
- **Registered before:** inspecting any Stage-C sink score or selecting Qwen sink-heavy layers
- **Status:** active

### What changed

The pinned upstream attention and metric tolerances were calibrated for float32 GPT-2. Stage C is
registered in bfloat16. For Qwen audit checks only, the absolute tolerance is now
`max(upstream_tolerance, 2 * torch.finfo(torch.bfloat16).eps)`, which is `0.015625`. This applies
to attention row-sum validity, decomposition of the sink map against the upstream scalar metric,
and equality of the differentiable future-sink objective to the frozen attention-map reduction.
Causal-mask leakage retains the upstream absolute tolerance because masked future entries are
expected to be exactly zero.

### What did not change

The model outputs are not rounded, corrected, or recast before sink measurement. The sink metric,
layer/head aggregation, `0.15` floor, sink-layer selection, neuron ranking, intervention grid,
random controls, CE/KL metrics, and causal-gate thresholds are unchanged. Baseline and
intervention forwards remain bfloat16 and use the same dtype.

### Why this cannot bias a result

This is a representation-precision audit bound derived only from the registered dtype, not from a
sink magnitude or causal effect. A first baseline attempt stopped at the audit boundary because
the float32 tolerance was inappropriately strict; no sink score, selected layer, attribution,
validation result, or test result was inspected. The new bound is registered before repeating
the preflight and applies symmetrically to every Qwen condition.

---

## A004 - Clarify Qwen tokenizer-revision provenance

- **Date recorded:** 2026-09-05
- **Status:** clarification only; no scientific setting changed

The pinned upstream corpus provider copies a tokenizer revision attribute that Transformers 5.3.0
leaves unset. Consequently, the frozen Qwen corpus JSON contains
`"tokenizer_revision": null`, just like the existing GPT-2 neutral manifest. A002's statement that
the manifest records the revision should be read as the complete corpus-freeze artifact set: the
freeze run's `run_config.json` records the resolved SHA
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, while the tracked manifest records the tokenizer
name and immutable token-id content hash.

The Stage-C runner hard-codes and records that exact tokenizer/model revision, enforces the Qwen
tokenizer name and registered manifest hash, and verifies the loaded model's `_commit_hash` before
measurement. This clarification changes no input IDs, split, model forward, sink value, ranking,
intervention, statistic, or gate result.

---

## A005 - Stage C2 positive-signed Qwen replication on fresh neutral blocks

- **Date registered:** 2026-09-05
- **Registered before:** constructing the Stage-C2 corpus or running any Stage-C2 model forward
- **Status:** active; defines a new experiment and does not supersede the completed Stage-C result

### Motivation and status of the prior result

Stage C is complete and remains a valid model-specific null under its registered absolute
activation-times-gradient ranking. Post-result analysis found that 18 of its smallest 23-neuron
target set had negative `mean_signed_attr`. For suppression by `alpha < 1`, the first-order change
is

`delta S_future ~= -(1 - alpha) * (a * dS_future/da)`.

Negative signed attribution therefore predicts that suppression will *increase* the sink. That is
the direction observed for the Stage-C 0.01%-0.10% target sets. This observation motivates a new,
explicitly post-hoc Stage-C2 replication; it does not change, repair, or reinterpret Stage C as a
pass.

### What changed

Stage C2 changes the discovery selection statistic from descending `mean_abs_attr` to descending
positive `mean_signed_attr`, with deterministic `(layer, neuron)` tie-breaking. Every registered
top-k set must contain only neurons whose discovery `mean_signed_attr` is strictly greater than
zero. If the discovery ranking contains fewer positive-score neurons than the largest registered
`k`, Stage C2 stops as invalid before validation or test access.

Because the Stage-C validation and test outcomes have already been inspected, Stage C2 uses a new
Qwen-tokenized neutral manifest at
`configs/frozen/qwen2_5_1_5b_instruct_c2/neutral_corpus_manifest.json`. It is built by the same pinned
`upstream/sink-kd/common/corpus_providers.py::openwebtext_corpus` provider from the same OpenWebText
validation document window, seed 0, and 40-token packing. The provider purpose is `ppl`, which
selects packed block indices 300-599; Stage C used purpose `sink`, block indices 0-299. Stage C2
then assigns its own frozen 100/100/100 discovery, validation, and locked-test roles. Here `ppl` is
only the upstream provider's offset label; the corpus remains neutral text and contains no
downstream benchmark examples or labels.

The Stage-C2 experiment id is `stage_c2_qwen_signed_replication_v1`. Its outputs use separate
append-only `results/stage_c2_preflight/` and `results/stage_c2_full/` roots and Stage-C2-specific
operating-point and formal-gate schemas.

### What did not change

- checkpoint and revision: `Qwen/Qwen2.5-1.5B-Instruct` at
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- tokenizer/revision, bfloat16 dtype, eager attention, RTX 4080 SUPER full-run role, and batch size
  1;
- position-0/second-half-query sink metric and discovery-only sink-heavy-layer rule;
- the causal-order-aware `S_future(l)` objective, activation-times-gradient calculation,
  all-position aggregation, and pre-`down_proj` SwiGLU neuron definition;
- global selection across eligible `(layer, neuron)` pairs; no layer balancing, layer
  normalization, sign-consistency threshold, or head restriction is introduced;
- registered fractions, five alpha values, 20 layer-count-matched random controls, control RNG,
  neutral CE/KL/top-1 metrics, validation operating-point rule, formal held-out gate, bootstrap,
  and dose-response threshold;
- downstream benchmark data remain forbidden for discovery or selection, and Stage D remains
  blocked unless this exact checkpoint clears the independent Stage-C2 formal gate.

### Why this is a valid new test rather than post-hoc gate tuning

The new hypothesis and its failure condition are frozen before any Stage-C2 model result is
observed. The signed score is selected for a directional causal reason, not because a Stage-C2
validation cell performed well. Fresh discovery, validation, and test blocks prevent the
already-inspected Stage-C examples from serving as confirmatory evidence. Keeping all other
scientific settings unchanged makes Stage C2 a single-factor test of whether direction-blind
selection caused the Qwen null. A Stage-C2 null remains a null; no additional ranking or layer
selection change may be made under this experiment id.

---

## A006 - Descriptive per-layer decomposition of the completed Stage-C result

- **Date registered:** 2026-09-05
- **Registered before:** running any new model forward for this diagnostic
- **Status:** post-hoc descriptive diagnostic; it cannot change a gate or operating point

### Motivation and inferential status

The completed Stage-C files report the registered sink score averaged over attention layers
4, 6, 14, 23, 24, 25, and 26. They do not retain intervened scores for each layer separately.
Because the Stage-C validation and test results have already been inspected, a per-layer
decomposition run is explicitly post-hoc. Its purpose is to locate the already-observed aggregate
effect, not to provide new confirmatory evidence, select a neuron set, tune alpha, or alter the
Stage-C null. It also cannot alter the independently registered Stage-C2 experiment.

### Frozen diagnostic protocol

The diagnostic reuses, without reranking, the completed Stage-C run's exact checkpoint/revision,
bfloat16/eager forward semantics, frozen neutral-corpus test examples, sink scope, absolute-score
attribution artifact, targeted neuron sets, and five registered alpha values. For each test
example it performs one baseline forward and one forward for every targeted set/alpha pair. It
records the position-0/second-half-query/all-head sink score independently for every frozen sink
layer, then reports

`RSR_layer = 1 - mean(S_intervened,layer) / mean(S_baseline,layer)`.

The 20 matched random draws remain frozen and are referenced by hash and selection metadata, but
are not rerun: the completed Stage-C gate already compared targeted and random suppression, while
this diagnostic asks only where the targeted aggregate effect occurred. Alpha 1 remains an exact
identity check, hook state is checked after every intervention, and a final reference forward
checks state leakage. Outputs live under the separate append-only
`results/stage_c_posthoc_per_layer/` root with complete provenance.

### Arithmetic check available without a model forward

For every selected set and alpha, the diagnostic also records the first-order aggregate
prediction from the frozen discovery attribution:

`predicted delta S = -(1 - alpha) * L * sum_i(mean_signed_attr_i * |future(i)| / |sink_layers|)`,

where `L=40` is the registered sequence length. The length factor reverses the discovery
implementation's mean over token positions; the future-layer fraction converts each neuron's
causal future-sink objective to the registered all-sink-layer average. This check predicts only
the aggregate change: the saved attribution averaged a source layer's future attention targets,
so it cannot identify individual downstream attention-layer contributions.

---

## A007 - Stage C3 reachability-aware, direction-aware, measurement-based localization

- **Date registered:** 2026-09-05
- **Registered before:** constructing any Stage-C3 corpus, and before any Stage-C3 model forward
- **Status:** active; defines a new experiment. It does **not** supersede, repair, or reinterpret
  Stage C, and it does not depend on the outcome of the still-running Stage C2.

### Motivation and status of the prior results

Stage C is a completed, valid model-specific null under its registered absolute
activation-times-gradient ranking. Amendment A005 already identified one defect in it: 18 of the 23
neurons in its smallest target set had negative `mean_signed_attr`, so suppression was predicted -
and observed - to *increase* the sink.

A per-sink-layer decomposition of the completed Stage-C test result identifies a second, independent
defect. At the 0.01% condition the measured relative sink reduction per registered sink layer was:

| Sink layer | RSR | Weight in the graded metric | Contribution |
|---:|---:|---:|---:|
| 4 | 0.00% | 13.2% | 0.00 pp |
| 6 | 0.00% | 13.4% | 0.00 pp |
| 14 | 0.00% | 14.5% | 0.00 pp |
| 23 | -2.92% | 15.5% | -0.45 pp |
| 24 | -1.38% | 14.1% | -0.19 pp |
| 25 | -0.48% | 15.6% | -0.07 pp |
| 26 | -13.75% | 13.7% | -1.89 pp |

These reconstruct the reported aggregate of -2.598% to within 0.012 pp, so the decomposition is
consistent with the frozen result rather than a re-measurement of it.

Layers 4, 6 and 14 moved by *exactly* zero because the selected neurons were concentrated in MLP
layer 25, and a decoder block computes attention before its MLP: attention at layer `j <= l` is
already fixed when layer `l`'s MLP runs. 41.1% of the graded metric was therefore not merely
unmoved but causally unreachable by the selected set, at any sparsity and any alpha. Under global
top-k selection this is not a tuning accident; it follows from comparing scores across depths that
have different path lengths and different target sets.

The two defects are separable and both must be fixed for the registered gate to be a fair test on
this checkpoint. Fixing only the sign - Stage C2 - would at best flip layer 26 from -13.75% to
+13.75%, which is an aggregate of +1.9% against a 10% threshold.

### What changed

1. **Per-target-layer attribution.** `S_j` is differentiated separately for each reachable
   sink-heavy attention layer `j > l` instead of once for their mean. The registered scalar is
   unchanged and remains the mean of these terms; `neuron_sink/sink_metrics.py` exposes the terms
   through `differentiable_sink_scores_per_layer`, and the identity `terms.mean() ==
   differentiable_sink_score(...)` is pinned by test.
2. **Per-sink-layer budgeted selection.** Each registered sink layer `j` receives a share of `k`
   proportional to its weight in the graded metric, `w_j = S0_j / sum_j S0_j`, allocated by largest
   remainder so the shares total exactly `k`. Each share is filled only from MLP layers `l < j`.
   Fill order is ascending `j` - the earliest sink layer has the smallest upstream pool and so picks
   first - neurons already taken are skipped, and any shortfall is redistributed from the global
   measured order.
3. **Strictly positive direction requirement**, inherited from A005's reasoning and applied per
   target: a candidate must reduce the sink *at the layer it was drawn for*.
4. **Measured-ablation rerank.** The gradient score only shortlists. The final ranking is the effect
   directly measured under the registered intervention (`alpha=0`) on discovery examples only. A
   first-order score around the current activation is not a reliable predictor of ablating that
   activation to zero, and one ablated forward yields the sink at every registered layer at once, so
   this is affordable.
5. **float32 metric and attribution arithmetic.** The sink reduction and the
   `activation * gradient` product are computed in float32 even when the registered forward is
   bfloat16. `neuron_sink/evaluation.py` already does this for CE and KL. A bfloat16 reduction
   returns a bfloat16 scalar, so at a sink of ~0.6 the metric carries a resolution of about 0.0023 -
   the same order as an entire matched-random control effect in Stage C (its random-control 95th
   percentile RSR values were ~2e-4). **The backward pass still runs in the registered dtype**; this
   amendment recovers precision in the metric and the product, and does not claim to turn a bfloat16
   backward into a float32 one.
6. **Per-sink-layer reporting is mandatory.** Every Stage-C3 condition writes the sink at every
   registered layer to `<stage>/suppression/per_layer_sink.csv`. The shared 41-column
   `PHENOMENON_ROW_FIELDS` schema is *not* extended, so Stage B, C and C2 artefacts remain readable
   by the existing loader.
7. **A direction guard** runs before the intervention grid and aborts the run if any targeted
   condition's own discovery measurements point the wrong way. The quantity checked is the sum of
   the selected neurons' *measured* marginal sink reductions, which is a stronger check than the
   first-order sign, and is still a sum of marginals rather than a claim about the joint effect.
   The joint effect of the largest targeted set is measured separately and reported alongside the
   sum of its marginals, so redundancy is stated rather than assumed.
8. **Registered comparison arms.** Alongside the C3 primary arm, Stage C3 registers: the whole-MLP
   layer-attenuation baseline from `docs/01_PHENOMENON_GATE.md` ("Layer-level baseline"), extended
   with an all-eligible-layers condition; the Stage-C absolute-ranked arm and the Stage-C2
   signed-only arm re-run on the same fresh blocks at `alpha=0`, so an improvement can be
   attributed to the ranking change rather than to the change of corpus; and GPT-2-small as a
   method-validation control, because it is a checkpoint already known to pass and a method that
   cannot reproduce its Stage-B effect is at fault.

   **These attenuation conditions are maximal interventions, not upper bounds.** Suppression is not
   monotone in the sink - Stage C is itself a case where suppressing neurons *increased* it - so a
   sparse subset can exceed a whole-layer or all-layer effect, and a set spanning several MLP layers
   can exceed any single layer. A dev check on GPT-2-small showed exactly that. The one rigorous
   bound in this design is causal reachability: a set drawn from MLP layers `L` cannot move any sink
   layer `j <= min(L)`, so its achievable relative reduction is at most the baseline sink mass of
   the reachable layers over the total. That arithmetic bound is what identified the 41.1%
   unreachable share above, and it is recorded per condition as `reachable_metric_weight`.

### Corpus windows

Stage C used the pinned provider's `sink` window (global blocks 0-299) and Stage C2 uses its `ppl`
window (300-599). The provider exposes only those two offsets, but `purpose="ppl"` has no cap on
`n_blocks`, so requesting 600 blocks and dropping the first 300 reaches blocks **600-899**. Packing
is a prefix operation, so the dropped prefix is byte-identical to the Stage-C2 corpus, and Stage C2
remains exactly reproducible; this equality is asserted by test. Qwen Stage C3 therefore uses
`openwebtext_validation_ppl_600_skip300`, blocks 600-899. GPT-2-small has only ever consumed the
GPT-2-tokenized `sink` window, so its Stage-C3 corpus is `openwebtext_validation_ppl_300`, blocks
300-599, with no skip. Both manifests are frozen and hash-pinned before any run, and the runner
refuses to start against an unpinned manifest.

Two superficially similar routes were considered and rejected, because both give **false**
disjointness: a different `seed` reshuffles the same 8,000 validation documents, and
`train_documents`/`validation_documents` change the document window without being encoded in the
upstream corpus id.

### What did not change

- checkpoint and revision: `Qwen/Qwen2.5-1.5B-Instruct` at
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, and `openai-community/gpt2` at
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- tokenizer/revision, bfloat16 forward dtype for Qwen and float32 for GPT-2, eager attention, the
  RTX 4080 SUPER full-run role, and batch size 1;
- the position-0 / second-half-query sink metric, its target position and query rule, and the
  discovery-only sink-heavy-layer rule (top quartile and a 0.15 floor);
- the causal-ordering constraint, the pre-`down_proj` / pre-`c_proj` neuron definition, and
  all-position suppression;
- registered fractions, the five alpha values, 20 layer-count-matched random controls, the control
  RNG and its seed derivation, and the control re-derivation check;
- the neutral CE/KL/top-1 metrics, the validation operating-point rule, the paired bootstrap, the
  dose-response threshold, and **the formal held-out gate exactly as registered** - RSR >= 0.10 on
  the mean over all sink-heavy layers, above the random 95th percentile, bootstrap lower bound above
  zero, Spearman >= 0.8, valid outputs. Change 2 is what makes that metric reachable; the threshold
  itself is untouched;
- downstream benchmark data remain forbidden for discovery or selection, and Stage D remains blocked
  unless a checkpoint clears an independent formal gate.

### Why this is a valid new test rather than post-hoc gate tuning

The failure condition is frozen before any Stage-C3 measurement, on blocks no Stage-C3 model has
seen. The changes are motivated by two mechanical defects that are visible in the *already
completed* Stage-C artefacts - a sign error and a causal-reachability error - not by any Stage-C3
cell performing well. Neither the gate nor any threshold is relaxed; the gate metric is left
identical so C3 stays directly comparable with Stage B, Stage C and Stage C2. The registered
comparison arms exist precisely so that a positive C3 result can be attributed to a specific change
rather than to the new corpus. A Stage-C3 null remains a null, and no further ranking, budget, or
selection change may be made under this experiment id.

If attenuating every eligible MLP layer at once barely moves a given sink layer, that is strong
evidence - not proof, because suppression is not monotone - that the sink is not MLP-mediated at
that depth. The registered response is to report exactly that, and to consider the
attention-dimension extension already described in `docs/01_PHENOMENON_GATE.md` as a separate
registered experiment, rather than to make a further change to Stage C3.

### Correction to A002

Amendment A002 records the Stage-C corpus source as `stas/openwebtext-10k`. That line is wrong. The
pinned loader (`upstream/sink-kd/common/datasets_loader.py`, `OPENWEBTEXT_HF_PATH`) and all three
frozen manifests record `Skylion007/openwebtext`, which is the dataset that was actually used.
Amendments are append-only, so A002 is left as written and this note is the correction of record.
The data, the manifests and their hashes are unaffected.
