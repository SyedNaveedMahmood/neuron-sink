# Stage B - Full GPT-2 Phenomenon Confirmation

## Outcome

**Execution: COMPLETE. GPT-2-small: PASS. GPT-2-medium: PASS. Project-level interpretation:
STRONG SUPPORT.**

Both independently ranked GPT-2 checkpoints passed the preregistered formal causal gate on their
locked 100-example neutral test splits. At full suppression, targeted neurons reduced the sink more
than the 95th percentile of 20 layer-count-matched random controls, the paired-bootstrap confidence
intervals excluded zero, suppression dose was monotonic, and every evaluated output remained valid.

The operating-point result is more qualified:

- GPT-2-small selected a confirmatory validation operating point, `k*=15` (0.05% of its eligible
  pool), at `alpha=0`.
- GPT-2-medium did not produce a validation-qualified `k*`. Its 0.01% condition stayed within the
  CE budget but missed the effect thresholds; every larger effective condition exceeded the
  `0.10` nats/token CE budget. The frozen fallback is therefore `k_max_effect=860` (1.00%) and is
  marked `exploratory_only=true`.

Thus Stage B establishes sparse, targeted neuron-level control of the measured sink in both GPT-2
models. It does not establish that a low-functional-drift operating point exists in both models,
and it does not license transfer of GPT-2 neuron identities to Qwen or another checkpoint.

## Registered design and execution

| Item | Value |
|---|---|
| Experiment | `stage_b_full_phenomenon_v1` |
| Models | `openai-community/gpt2`; `openai-community/gpt2-medium` |
| Neutral corpus | `openwebtext_validation_sink_300` |
| Split sizes | 100 discovery / 100 validation / 100 locked test |
| Sequence length | 40 tokens |
| Attribution | mean absolute activation-times-gradient for the future-sink objective |
| Neuron hook | GPT-2 `mlp.c_proj` input, all token positions |
| Fractions | 0.01%, 0.05%, 0.10%, 0.25%, 0.50%, 1.00% |
| Alphas | 1.00, 0.75, 0.50, 0.25, 0.00 |
| Controls | 20 deterministic layer-count-matched random sets per fraction |
| Statistics | 10,000 paired-bootstrap resamples, seed 0; random-control 95th percentile; Spearman dose response |
| Runtime mode | float32, eager attention, `model.eval()`, deterministic algorithms, batch size 1 |
| Hardware | NVIDIA GeForce RTX 4080 SUPER, 17,170,956,288 total bytes reported by PyTorch |

The model revisions were resolved before scientific outputs were examined:

| Model | Model revision | Tokenizer revision |
|---|---|---|
| GPT-2-small | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| GPT-2-medium | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |

No Qwen model, downstream benchmark, robustness extension, layer baseline, or Sink-KD checkpoint
was accessed during Stage B.

## Preflight results

Each registered run was preceded by a 20-example-per-executed-split preflight. Preflights ran
discovery and validation only, never opened the test split, and emitted
`NOT_EVALUATED_DRY_RUN` rather than a scientific gate conclusion.

| Model | Preflight path | Runtime | Estimated full-grid evaluation | Estimated output | Peak allocated / reserved |
|---|---|---:|---:|---:|---:|
| GPT-2-small | `results/stage_b_preflight/gpt2-small/run_20260904T131354Z` | 278.817 s | 1,831.526 s | 123,128,917 bytes | 574,430,208 / 641,728,512 bytes |
| GPT-2-medium | `results/stage_b_preflight/gpt2-medium/run_20260904T135245Z` | 487.529 s | 3,340.380 s | 179,495,955 bytes | 1,530,371,584 / 1,553,989,632 bytes |

Preflight `summary.json` file SHA-256 values are, respectively,
`80d1b4a2ac94ff10d9747be6f238712f5802073ac0baea47c5caa39efb70d358` and
`791050e79417e062388370a02953542db45b52f6f29867cb2e23ad3dd32e9c8d`.

## Independent discovery and localization

The full ranking, scope, and controls were recomputed independently for each checkpoint using only
the 100 discovery examples.

| Model | Discovery sink-heavy attention layers | Eligible MLP layers | Eligible pool | Target k values |
|---|---|---|---:|---|
| GPT-2-small | `[7, 9, 10]` | `[0..9]` | 30,720 | 3, 15, 31, 77, 154, 307 |
| GPT-2-medium | `[9, 16, 18, 19, 20, 21]` | `[0..20]` | 86,016 | 9, 43, 86, 215, 430, 860 |

The rankings favor later eligible MLP layers, but they are not confined to the immediately
preceding block. For GPT-2-small, 10 of the 15 `k*` neurons are in layers 7-9. For GPT-2-medium,
all nine neurons in the smallest set are in layers 13, 14, 16, and 18-20; 27 of the 43 neurons in
the 0.05% set are in layers 13-20. This is consistent with the future-sink objective and the late
sink-heavy scopes. It is descriptive localization, not evidence that raw neuron IDs correspond
across models.

Discovery per-layer sink values were:

- GPT-2-small, layers 0-11: `0.038346, 0.125278, 0.156884, 0.403917, 0.444471,
  0.655603, 0.614068, 0.715439, 0.616776, 0.707960, 0.705318, 0.562547`.
- GPT-2-medium, layers 0-23: `0.066625, 0.157812, 0.089694, 0.111595, 0.347462,
  0.481047, 0.676612, 0.669888, 0.544332, 0.719224, 0.594898, 0.621456,
  0.655483, 0.578501, 0.665330, 0.591850, 0.715552, 0.694875, 0.741003,
  0.776818, 0.783209, 0.759325, 0.680910, 0.446097`.

## Validation-only operating-point selection

The target-minus-random interval is the paired-bootstrap 95% interval computed over examples.
Operating-point selection used validation only and was written and hash-verified before the locked
test split could be accessed.

### GPT-2-small validation

| Fraction | k | Target RSR | Target - median random RSR | CI lower | Delta CE | Qualifies `k*` |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.01% | 3 | 0.075028 | 0.075301 | 0.072023 | 0.016511 | No |
| 0.05% | 15 | 0.113873 | 0.114330 | 0.106684 | 0.086014 | **Yes** |
| 0.10% | 31 | 0.106964 | 0.107307 | 0.100137 | 0.118772 | No |
| 0.25% | 77 | 0.165055 | 0.167454 | 0.156356 | 0.288037 | No |
| 0.50% | 154 | 0.168298 | 0.172206 | 0.158426 | 0.360923 | No |
| 1.00% | 307 | 0.237394 | 0.246159 | 0.228273 | 0.476153 | No |

The smallest qualifying fraction was 0.05%, so the frozen confirmatory operating point is
`k*=15`, `alpha=0`. The 0.10% condition narrowly exceeded the CE ceiling and therefore could not
replace it.

### GPT-2-medium validation

| Fraction | k | Target RSR | Target - median random RSR | CI lower | Delta CE | Qualifies `k*` |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.01% | 9 | 0.060764 | 0.060813 | 0.057785 | 0.042428 | No |
| 0.05% | 43 | 0.249902 | 0.250114 | 0.218237 | 0.675727 | No |
| 0.10% | 86 | 0.572475 | 0.573232 | 0.529827 | 1.905451 | No |
| 0.25% | 215 | 0.761048 | 0.762671 | 0.721627 | 3.274435 | No |
| 0.50% | 430 | 0.944400 | 0.946494 | 0.941466 | 5.437511 | No |
| 1.00% | 860 | 0.947300 | 0.951915 | 0.949305 | 6.162816 | No |

No registered fraction met all four operating-point conditions. The frozen fallback is the
largest-effect registered fraction, 1.00% (`k=860`, `alpha=0`), explicitly marked exploratory.
The sharp jump between 0.01% and 0.05% shows that a strong mechanistic effect exists, but it is
coupled to large neutral-language-model damage under full suppression.

## Locked-test formal gate

The formal gate is independent of the validation CE-budget rule. A fraction passes when its
full-suppression test RSR is at least 0.10, exceeds the 95th percentile of its 20 random controls,
has a target-minus-median-random bootstrap interval with lower bound above zero, has Spearman dose
correlation at least 0.8, and has valid outputs.

### GPT-2-small test

Baseline sink was `0.712447`; baseline CE was `4.113561` nats/token and baseline PPL was
`61.164155`.

| Fraction | k | Target RSR | Random P95 RSR | Difference CI lower | Spearman | Delta CE | KL | Top-1 flip | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.01% | 3 | 0.073195 | 0.000324 | 0.069548 | 1.000 | 0.018675 | 0.036364 | 0.115128 | No |
| 0.05% | 15 | 0.107774 | 0.000706 | 0.101329 | 1.000 | 0.120271 | 0.158010 | 0.236667 | **Yes** |
| 0.10% | 31 | 0.100371 | 0.001344 | 0.093809 | 1.000 | 0.154296 | 0.186903 | 0.257179 | **Yes** |
| 0.25% | 77 | 0.163756 | 0.001096 | 0.156325 | 1.000 | 0.371316 | 0.350578 | 0.331538 | **Yes** |
| 0.50% | 154 | 0.167841 | 0.000334 | 0.161195 | 1.000 | 0.450132 | 0.430823 | 0.368974 | **Yes** |
| 1.00% | 307 | 0.239857 | -0.001251 | 0.233450 | 1.000 | 0.547851 | 0.549721 | 0.405128 | **Yes** |

Passing fractions were `[0.05, 0.10, 0.25, 0.50, 1.00]`. At the validation-selected `k*`, the
held-out sink effect replicated: RSR was 10.78%, versus a random-control P95 of 0.071%. Its test
Delta CE was `0.120271`, slightly above the validation selection ceiling even though validation
Delta CE was `0.086014`. This does not invalidate the preregistered selection or formal causal gate,
but it warns that the functional-drift budget is not perfectly stable across 100-example splits.

### GPT-2-medium test

Baseline sink was `0.749382`; baseline CE was `3.835774` nats/token and baseline PPL was
`46.329256`.

| Fraction | k | Target RSR | Random P95 RSR | Difference CI lower | Spearman | Delta CE | KL | Top-1 flip | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.01% | 9 | 0.058543 | 0.000164 | 0.055292 | 1.000 | 0.039518 | 0.066090 | 0.153333 | No |
| 0.05% | 43 | 0.261207 | 0.000316 | 0.231320 | 1.000 | 0.770421 | 0.802078 | 0.429231 | **Yes** |
| 0.10% | 86 | 0.627516 | -0.000138 | 0.582556 | 1.000 | 2.194389 | 2.285832 | 0.673590 | **Yes** |
| 0.25% | 215 | 0.777869 | -0.000440 | 0.741446 | 1.000 | 3.362618 | 3.466994 | 0.811026 | **Yes** |
| 0.50% | 430 | 0.946420 | -0.000965 | 0.945436 | 1.000 | 5.531537 | 5.617948 | 0.890513 | **Yes** |
| 1.00% | 860 | 0.949199 | -0.002272 | 0.952206 | 1.000 | 6.354784 | 6.514408 | 0.904872 | **Yes** |

Passing fractions were `[0.05, 0.10, 0.25, 0.50, 1.00]`. The 0.05% set already removed 26.12%
of the held-out sink while matched random suppression remained effectively zero. Larger fractions
approached complete sink removal, but functional drift grew rapidly: the 0.05% condition flipped
42.9% of next-token top-1 predictions and the 1.00% fallback flipped 90.5%. The strongest sink
effects therefore cannot be interpreted as selective, low-cost interventions.

## Dose response

All twelve targeted fraction/model combinations had Spearman correlation `1.0` between suppression
dose and test RSR. Identity (`alpha=1`) had RSR 0 exactly. The remaining test RSR curves were:

| Model | Fraction | alpha=0.75 | alpha=0.50 | alpha=0.25 | alpha=0.00 |
|---|---:|---:|---:|---:|---:|
| GPT-2-small | 0.01% | 0.016529 | 0.034431 | 0.053455 | 0.073195 |
| GPT-2-small | 0.05% | 0.023229 | 0.049920 | 0.079102 | 0.107774 |
| GPT-2-small | 0.10% | 0.020956 | 0.044767 | 0.071965 | 0.100371 |
| GPT-2-small | 0.25% | 0.029886 | 0.064782 | 0.108058 | 0.163756 |
| GPT-2-small | 0.50% | 0.026373 | 0.058999 | 0.104052 | 0.167841 |
| GPT-2-small | 1.00% | 0.031455 | 0.074327 | 0.139708 | 0.239857 |
| GPT-2-medium | 0.01% | 0.013461 | 0.027512 | 0.042467 | 0.058543 |
| GPT-2-medium | 0.05% | 0.019963 | 0.043231 | 0.098165 | 0.261207 |
| GPT-2-medium | 0.10% | 0.022635 | 0.066778 | 0.235650 | 0.627516 |
| GPT-2-medium | 0.25% | 0.023510 | 0.067723 | 0.299490 | 0.777869 |
| GPT-2-medium | 0.50% | 0.023452 | 0.152140 | 0.738542 | 0.946420 |
| GPT-2-medium | 1.00% | 0.021678 | 0.088989 | 0.905967 | 0.949199 |

The clean within-fraction dose response strengthens the causal interpretation. The response is not
linear, especially for GPT-2-medium: its larger sets retain little effect at mild suppression and
then collapse the sink between `alpha=0.5` and `alpha=0.25` or 0. This is evidence of a nonlinear
mechanism or compensatory regime, not a reason to change the registered alpha grid.

## Cross-model interpretation

The result supports three claims and rejects a stronger one:

1. **Sparse targeted causality is reproducible.** Both checkpoints independently produced a
   held-out targeted effect far outside their layer-matched random distributions.
2. **The phenomenon is not an artifact of a single GPT-2 scale.** Both models pass, which meets the
   registered project-level definition of strong support.
3. **Causal concentration differs by scale.** GPT-2-medium's effect rises much more sharply with
   fraction and reaches approximately 95% sink reduction, while GPT-2-small reaches approximately
   24% at 1.00%.
4. **Selective sink control is not universally demonstrated.** GPT-2-small has a validation-qualified
   low-drift operating point, but GPT-2-medium does not. For the medium model, strong sink reduction
   co-occurs with large CE, KL, and prediction drift.

The matched-random controls typically changed RSR by only a few tenths of a percentage point, often
in the negative direction. Equal neuron count and equal per-layer distribution are therefore
insufficient to explain the targeted effect. This argues for attribution-specific neuron identity,
while the functional-drift results caution against describing those units as sink-only neurons.

## Correctness and leakage audit

| Check | GPT-2-small | GPT-2-medium |
|---|:---:|:---:|
| Exact registered 100/100/100 split sizes | PASS | PASS |
| Full 126-condition grid and five alphas | PASS | PASS |
| 20 deterministic matched controls per fraction | PASS | PASS |
| Ranking consumed discovery examples only | PASS | PASS |
| Validation operating point frozen before test | PASS | PASS |
| Locked test accessed exactly after artifact verification | PASS | PASS |
| Alpha=1 logits and attention exact | PASS | PASS |
| All forwards finite and nondegenerate | PASS | PASS |
| Attention normalized and causal | PASS | PASS |
| Same/earlier attention excluded from attribution objective | PASS | PASS |
| Final baseline replay / hook-state leakage | exact / PASS | exact / PASS |
| Separate model paths and no smoke overwrite | PASS | PASS |

Each split contains 63,100 paired rows/forwards: one 100-example baseline plus 126 conditions x five
alphas x 100 examples. Each full run recorded 189,301 forwards including the final state-leakage
probe. No vocabulary-sized logits were retained on GPU across the experiment.

The final project-local suite, including real cached CUDA integration tests, reported:
**182 passed, 155 subtests passed, 0 failed, 0 skipped in 7.54 seconds**.

## Run paths, runtime, and memory

| Model | Registered run path | Runtime | Peak allocated | Peak reserved | Files / bytes |
|---|---|---:|---:|---:|---:|
| GPT-2-small | `results/stage_b_full/gpt2-small/run_20260904T131855Z` | 1,737.014 s | 574,430,208 (547.82 MiB) | 641,728,512 (612.00 MiB) | 24 / 99,084,407 |
| GPT-2-medium | `results/stage_b_full/gpt2-medium/run_20260904T140056Z` | 3,195.937 s | 1,530,371,584 (1,459.48 MiB) | 1,553,989,632 (1,482.00 MiB) | 24 / 108,557,698 |

Both result directories are append-only and gitignored. Each contains run configuration and
provenance, the discovery sink map/scope/ranking/neuron sets, separate per-example and aggregate
suppression outputs for discovery/validation/test, the frozen operating point, the formal gate,
and the final summary.

## Content-hash chain

The common neutral corpus manifest SHA-256 is
`c6e077871003e29e12aaaf4c7f24d8e17eb5a1c919a20f3693d069742dd480c7`.

| Artifact | GPT-2-small | GPT-2-medium |
|---|---|---|
| Sink scope content hash | `122d55365767111ec911e1e7d4988ac8c8340771447840e356d4737122cb8ed4` | `ef7f6755e52f68d6e7c1afcc06b22bcddfd178edddb61129ebb2243061d7d079` |
| Attribution content hash | `8dd97d7be79336ba8fe02c37ad2fbb98f6522222aaa42ba9312c82fbad7f0374` | `1f07c74750781afc26f341a8689e47b8560f483d7ef5d32453caf985d041b460` |
| Neuron-set content hash | `acc3af2f2df3f35ea98d8edce4d0463251fd8b82c57f7bbe48eec05c1b78eded` | `9cdcb2016470d51c2b7200d327b4d7f36bac9a68de58ec770e79633638b29e25` |
| Discovery rows hash | `f7150abfec06972199e1143a03151105f97bad7193c27c62399d907cfac3a571` | `fe9f4e68a49b9d711d24332c56a74e607d0085f2b1a658d2fb3f21194de3e8ae` |
| Validation rows hash | `8fc4f12af1c7dc23806495defe10dbe4dceae24c7c0315a9c3de3ad4ef3bf29d` | `06ccd9ca671573feeba4de2039fce1b929a5becce7f67764ae98862984b710de` |
| Test rows hash | `82b46858b95ce295885f78e08d52c181bd0c78a146a801cddfb2ee94bb425b17` | `74f8334bb2959bdc54af5ecc952c3f825ace5d18e0b55eb3783fb9d508eaf8e6` |
| Operating-point content hash | `26defd357da8bd789a22189b83640f2492f20c491d3010cc6f87cf1524068ead` | `343a6549a8efc6cf42cc14ff0c6ad02df8f7aab1dd3e87af996a0090ed024c95` |
| Formal-gate content hash | `48f5eb176188ed89c90a7678afb59567fb043a7f68f71c90093ce7c2ee581a21` | `a2c3b5bb4f6ec2b2a89ae6c62352f53404c136b6885e37a14c0e149ed60beba4` |

For file-level verification of the five root result documents:

| File | GPT-2-small file SHA-256 | GPT-2-medium file SHA-256 |
|---|---|---|
| `run_config.json` | `18e20e2917a3dbec3a6e1df69fba8d102db6af31bfd999db11fd90dca58aa61d` | `8beefe3b787a0df66f80b8dc5bc77b17b866e552f8a8ea7f3f3859106e252f79` |
| `provenance.json` | `24cf4efd8b338a3b00d46eed3660813afa019a6378d7cef63cc8c227b9f9f384` | `9a1b5af09a55f16540e3a7536f818b8fb967a33ac213ceddab1a1969ea0d7baa` |
| `summary.json` | `8d68711268e526273b22d9df4f2384ed0d05310497a260448534c8577823d205` | `10dad27c0d4975e9cc688a9de9e2d7397ae9e612e32605d3ed71e0d62717b636` |
| `operating_point.json` | `874680e452241f4ebe27476720132f9ddf34bd86f2962be4a2a2b914a04f4dc6` | `28a28671ac859addab68163c9b1cfd04f18dff942c668c0d3f19eb98e20cd4d8` |
| `formal_gate.json` | `d97f50a8af22abee19b666de0988338249d213b0b2304521f42454846a7822a7` | `98239960c62f0aa6974b4d8bb004e44cd8e7aab18c294e6bea6e58788fd21a47` |

## Provenance and reproducibility boundary

Both runs recorded root commit `5bfbf240eacfacfff078f08086f8eb93a0b62c3e`, Sink-Repro commit
`9ab67e914464b13863b67527d8ea14068ee9ff10`, and Sink-KD commit
`db114c9c5eb6ffc5de13e444c783408ea7401c62`. The runtime stack was Python 3.12.3,
PyTorch 2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0, datasets 4.8.4, NumPy 2.4.3,
pandas 3.0.1, and CUDA runtime 12.8 on Windows 11.

`repo_dirty_at_run=true` is expected and transparent: the Stage B implementation and tests were in
the working tree but had not been committed. The exact dirty paths are retained in each
`provenance.json`. Both upstream submodules remained clean and pinned. No commit or push was made.

## Limitations

- The formal gate measures causal control of the registered attention-sink metric; it does not by
  itself show that the intervened neurons are dedicated exclusively to sink formation.
- GPT-2-medium has no confirmatory low-drift `k*`. Its fallback is not suitable for confirmatory
  downstream claims.
- GPT-2-small's validation-qualified `k*` exceeded the CE ceiling on the independent test split by
  `0.020271` nats/token, highlighting sampling uncertainty in functional cost.
- Only one neutral corpus manifest, sequence length, attribution method, and seed family were used,
  as preregistered. Robustness extensions must be separate experiments rather than retroactive
  tuning.
- GPT-2 task performance is generally floor-limited; Stage B neutral CE/KL/top-1 drift should not
  be confused with the later task-rich Qwen evaluation.
- The nonlinear GPT-2-medium response leaves open whether the large effects reflect a compact
  coordinated circuit, thresholded compensation failure, or broader functional collapse.

## Next permitted stage

Stage B's positive result permits Stage C, not immediate downstream evaluation. The next checkpoint
is `Qwen/Qwen2.5-1.5B-Instruct`, which must receive its own baseline sink map, sink-heavy scope,
future-sink neuron ranking, matched controls, validation operating point, and held-out causal gate.
GPT-2 neuron IDs and layers must not transfer. Only if that exact Qwen checkpoint passes may Stage D
run MMLU, ARC-Challenge, CulturalBench, and GSM8K under frozen B0/T1/T2/R1 conditions.

## STAGE B

**FORMAL PHENOMENON GATE: PASS FOR BOTH MODELS - STRONG SUPPORT**
