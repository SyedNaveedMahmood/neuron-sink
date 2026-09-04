# Phenomenon Gate: Is Neuron-Level Sink Suppression Real?

This file defines the go/no-go logic before expensive downstream evaluation.

## Why a gate is required

A large model has many neurons. If we rank thousands of units on the same examples used for evaluation, some set will appear important by selection bias. The project therefore separates discovery, validation, and final test data and compares targeted suppression against matched random controls.

The downstream benchmark suite must not run until the basic causal effect is demonstrated on neutral held-out data.

## Phase 0 gate — implementation parity

All must pass:

- upstream sink metric fixture/parity test passes;
- baseline GPT-2-small attention maps and sink metric match the reference implementation within the existing upstream tolerance;
- `alpha=1` suppression is numerically identical to baseline within tolerance for logits, CE, and sink score;
- no validation/test example is used by the ranking code;
- layer/neuron ids are deterministic and correctly recorded;
- causal-order target construction excludes same/earlier attention layers;
- two repeated runs with the same seed produce identical selected neurons and controls.

Failure: fix implementation before any scientific run.

## Phase 1 smoke gate — RTX 2060 SUPER

This is only a plausibility gate.

On GPT-2-small, with the small smoke subsets:

- at least one targeted non-identity condition must reduce held-out sink more than all 5 matched-random draws;
- the targeted intervention must not produce NaNs, degenerate all-zero logits, or malformed attention;
- `alpha=0.5` should lie between identity and full suppression directionally for at least one tested fraction, although exact monotonicity is not required at smoke scale.

If no condition clears this gate, inspect implementation. If implementation is valid and a second fixed smoke seed is also null, stop before full runs and report that the proposed sparse MLP-neuron phenomenon was not detected under this definition.

## Full causal gate — RTX 4080 SUPER

Apply separately to GPT-2-small and GPT-2-medium on the locked 100-example neutral test split.

For each registered targeted condition calculate:

- baseline sink `S0`;
- intervened sink `S1`;
- relative sink reduction `RSR = (S0 - S1) / S0`;
- matched-random `RSR` distribution across 20 control sets;
- neutral CE drift;
- KL drift;
- top-1 flip rate.

### A model passes the primary phenomenon gate if all are true

1. **Effect size:** at least one registered top-neuron fraction at `alpha=0` has `RSR >= 0.10` on test.
2. **Matched-control superiority:** for that same fraction, targeted `RSR` is greater than the 95th percentile of the 20 layer-matched-random `RSR` values.
3. **Paired uncertainty:** paired bootstrap over test examples gives a 95% CI with lower bound > 0 for `RSR_target - median(RSR_random)`.
4. **Dose direction:** for the selected fraction, Spearman correlation between suppression dose `(1-alpha)` and test `RSR` is >= 0.8.
5. **No catastrophic degeneration:** CE/logit outputs remain finite for every evaluated example.

The model may pass even if CE drift is large; that would indicate sink-functional units rather than a selective removable sink. CE <= 0.10 nats is required only for the `k*` selective operating point, not for causal existence.

### Project-level interpretation

- **Strong support:** both GPT-2-small and GPT-2-medium pass.
- **Model-specific support:** exactly one passes; downstream/cross-architecture work may continue but claims must be model-specific.
- **Core null:** neither passes after implementation parity; do not run the full downstream suite merely searching for a positive result.

## Stability checks after a positive gate

These are confirmatory, not selection criteria:

- rerank from two halves of discovery data and measure top-k Jaccard/overlap by layer;
- activation-matched random control;
- 128-token neutral-corpus replication;
- mean-replacement ablation instead of zeroing for `k*`;
- head-restricted sink objective using only sink-heavy heads.

A result that disappears under every stability check should be described as brittle.

## Layer-level baseline

To answer the original coarse question about layers, include a simple layer attenuation baseline after the neuron effect passes:

- attenuate the complete MLP intermediate vector in each eligible layer by the same alpha grid;
- compare sink reduction and CE drift with sparse top-neuron suppression;
- do **not** zero entire transformer blocks.

The useful comparison is:

> Can a small selected neuron set achieve a meaningful fraction of whole-MLP sink reduction with substantially less functional drift?

This layer baseline is not used to discover neurons.

## Optional attention-dimension extension

Only after the MLP-neuron result is established, add a separate unit type:

- Q/K or head-output dimensions inside sink-heavy heads;
- separately named and separately analyzed;
- matched random dimensions within the same head/layer;
- never pool them with MLP neurons in one ranking.
