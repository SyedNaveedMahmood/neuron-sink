# Stage C - Qwen2.5-1.5B Independent Replication

## Outcome

**Execution: COMPLETE. Baseline sink preflight: PASS. Formal held-out causal gate:
NULL / MODEL-NEGATIVE.**

`Qwen/Qwen2.5-1.5B-Instruct` has a strong baseline position-0 attention sink, but the
independently attributed MLP neurons did not reproduce the registered GPT-2 causal result. No
registered fraction achieved the required 10% held-out sink reduction, and no fraction achieved
the required Spearman dose correlation of at least 0.8. The 1.00% set significantly outperformed
its layer-matched random controls, but reduced the test sink by only 7.07%, had Spearman `0.6`, and
increased neutral CE by `1.093` nats/token.

Validation produced no confirmatory `k*`. The frozen `k_max_effect=2330` (1.00%) operating point
is explicitly exploratory only. Under the registered execution order, Stage D downstream
benchmarks are therefore **blocked for this checkpoint**. This null must not be bypassed by
transferring GPT-2 neurons, changing fractions, weakening the gate, or tuning on test.

## Registered execution

| Item | Value |
|---|---|
| Experiment | `stage_c_qwen_replication_v1` |
| Model/tokenizer | `Qwen/Qwen2.5-1.5B-Instruct` |
| Exact revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Neuron | SwiGLU product entering `mlp.down_proj` |
| Neutral corpus | Qwen-tokenized `openwebtext_validation_sink_300` |
| Splits | 100 discovery / 100 validation / 100 locked test |
| Sequence length | 40 tokens |
| Attribution | mean absolute activation-times-gradient for `S_future(l)` |
| Fractions | 0.01%, 0.05%, 0.10%, 0.25%, 0.50%, 1.00% |
| Alphas | 1.00, 0.75, 0.50, 0.25, 0.00 |
| Controls | 20 deterministic layer-count-matched random sets per fraction |
| Runtime mode | bfloat16, eager attention, deterministic, batch size 1 |
| Hardware | NVIDIA GeForce RTX 4080 SUPER; 17,170,956,288 bytes |

Amendment A002 registered the tokenizer-specific manifest before Qwen measurement. Its SHA-256 is
`e38f7d3e21ef13287228ef5bb661995f0d628f1a61c99475c73ce3649ceb7426`; the same upstream
provider, validation document window, seed, sequence length, and split roles were retained.
Amendment A003 registered bfloat16 audit tolerances derived from dtype epsilon. Neither amendment
changes the scientific sink metric or causal gate. A004 clarifies that the upstream corpus object's
tokenizer-revision field is null; the exact revision is retained in freeze/run configuration and
enforced alongside tokenizer name and manifest hash.

## Implementation and preflight

Stage C adds a validated `Qwen2ModelAdapter` while retaining the GPT-2 adapter API. It hooks the
8960-coordinate tensor `SiLU(gate_proj(x)) * up_proj(x)` entering `down_proj`. Synthetic tests and
the real-checkpoint audit established:

- `alpha=1` installs no hook and reproduces logits and all attentions exactly;
- `alpha=0` zeros only requested coordinates at every token position;
- the real hook tensor has shape `[1, 40, 8960]` and dtype bfloat16;
- hooks and model state are restored exactly;
- layer 25's MLP can affect sink attention in layer 26 but not layer 25;
- Qwen GQA attentions expose all 12 query heads to the sink metric.

The required 100-example baseline-only preflight passed at
`results/stage_c_preflight/qwen2.5-1.5b-instruct/run_20260904T154839Z`. Maximum layer sink was
`0.672835025` at layer 25, above the registered `0.15` floor. The 20-example discovery/validation
runtime preflight at `run_20260904T154945Z` emitted no gate conclusion, passed every identity,
gradient, causal-order, validity, and leakage check, and measured a 3,299,403,776-byte peak.

## Independent discovery

Qwen discovery used all 100 discovery blocks and did not reuse any GPT-2 layer or neuron ID.

- sink-heavy attention layers: `[4, 6, 14, 23, 24, 25, 26]`;
- eligible MLP layers by causal ordering: `[0..25]`;
- eligible neuron pool: `26 * 8960 = 232,960`;
- target sizes: `23, 116, 233, 582, 1165, 2330`;
- ranking rows: 232,960, using only discovery examples;
- sink-scope hash: `ee24bdc33056701c1017dcbbbcfa13e8e89799dadf176930c691d68e9ad17d09`;
- attribution hash: `f3728902050118bfd34fe883cfb017f126a5fc5c1215906626a8e66c85a72fb4`;
- neuron-set hash: `bf15d8ec72bb37fb911a71b0d8f8ac9458087a0c4035919d5cb75f7a59409a0a`.

Per-layer discovery sink scores, layers 0-27, were:

`0.007289, 0.017744, 0.469839, 0.445775, 0.570990, 0.492083, 0.580959,
0.287383, 0.512960, 0.462556, 0.393850, 0.297080, 0.521429, 0.278026,
0.627029, 0.300740, 0.412532, 0.485527, 0.350870, 0.570152, 0.455334,
0.450601, 0.536248, 0.668493, 0.609233, 0.672835, 0.594128, 0.011134`.

The smallest target set was heavily concentrated in late eligible layers: layer 18 had 1 neuron,
layers 22-24 had 2 each, and layer 25 had 16. Its complete grouped IDs are preserved in
`discovery/neuron_sets.json`; all 120 matched-control sets and seeds are stored there as well.

## Validation-only operating point

No fraction met all four preregistered validation conditions.

| Fraction | k | Target RSR | Target - median random | CI lower | Delta CE | Qualifies |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.01% | 23 | -0.025842 | -0.025796 | -0.027653 | 0.011392 | No |
| 0.05% | 116 | -0.019846 | -0.019418 | -0.022056 | 0.158013 | No |
| 0.10% | 233 | -0.021628 | -0.021093 | -0.023964 | 0.198907 | No |
| 0.25% | 582 | -0.002231 | -0.001477 | -0.005058 | 0.275879 | No |
| 0.50% | 1165 | 0.010400 | 0.011546 | 0.007043 | 0.448458 | No |
| 1.00% | 2330 | 0.066983 | 0.068042 | 0.059570 | 1.025863 | No |

At small fractions, targeted suppression strengthened rather than reduced the sink. The two
largest fractions had positive effects beyond random, but neither reached 10% RSR and both greatly
exceeded the `0.10` CE budget. The validation artifact therefore froze 1.00% as
`k_max_effect`, with `exploratory_only=true`; it did not manufacture a confirmatory `k*`.

## Locked-test result

Test baseline sink was `0.627054138`; baseline CE was `3.778829217` nats/token and baseline PPL
was `43.764773`.

| Fraction | k | Target RSR | Random P95 | Difference CI lower | Spearman | Delta CE | KL | Top-1 flip | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.01% | 23 | -0.025975 | 0.000374 | -0.027580 | -1.0 | 0.010211 | 0.010447 | 0.054615 | No |
| 0.05% | 116 | -0.019700 | 0.000061 | -0.022041 | -1.0 | 0.148284 | 0.216652 | 0.113590 | No |
| 0.10% | 233 | -0.020446 | 0.000137 | -0.023418 | -1.0 | 0.187279 | 0.257067 | 0.155641 | No |
| 0.25% | 582 | -0.000178 | 0.000078 | -0.003112 | -0.3 | 0.273353 | 0.333910 | 0.236154 | No |
| 0.50% | 1165 | 0.011286 | 0.000678 | 0.007927 | 0.3 | 0.484449 | 0.553919 | 0.334103 | No |
| 1.00% | 2330 | 0.070714 | 0.000196 | 0.061658 | 0.6 | 1.093020 | 1.146877 | 0.495641 | No |

The 0.50% and 1.00% targets beat the random-control 95th percentile and had positive bootstrap
intervals. That is evidence that attribution found Qwen-specific units with non-random influence,
but it is insufficient for the registered causal claim: their effects were only 1.13% and 7.07%,
and their dose curves failed monotonicity. All smaller fractions increased or barely changed sink.

The targeted RSR curves (`alpha=0.75, 0.50, 0.25, 0.00`) were:

| Fraction | 0.75 | 0.50 | 0.25 | 0.00 | Spearman |
|---:|---:|---:|---:|---:|---:|
| 0.01% | -0.008473 | -0.015762 | -0.020930 | -0.025975 | -1.0 |
| 0.05% | -0.008788 | -0.014707 | -0.018267 | -0.019700 | -1.0 |
| 0.10% | -0.009099 | -0.014398 | -0.018386 | -0.020446 | -1.0 |
| 0.25% | -0.006305 | -0.008558 | -0.008451 | -0.000178 | -0.3 |
| 0.50% | -0.007056 | -0.008705 | -0.003873 | 0.011286 | 0.3 |
| 1.00% | -0.006944 | -0.009163 | 0.001115 | 0.070714 | 0.6 |

This non-monotonicity is qualitatively unlike Stage B, where all twelve GPT-2 dose curves had
Spearman `1.0`. It suggests Qwen's ranked coordinates are not a sparse, smoothly suppressible
substrate of the measured sink under this intervention, even though very broad full suppression
can move the sink at substantial language-model cost.

## Correctness and execution audit

| Check | Result |
|---|:---:|
| Qwen-specific tokenizer manifest and IDs | PASS |
| Independent 100-example discovery ranking | PASS |
| Full 126-condition, five-alpha grid | PASS |
| 20 layer-count-matched controls per fraction | PASS |
| Validation operating point frozen before test | PASS |
| Test accessed only after schema/hash verification | PASS |
| `alpha=1` logits and attention exact | PASS |
| All 189,300 scientific-grid forwards finite/nondegenerate | PASS |
| Attention normalized and causal | PASS |
| Future-layer causal ordering | PASS |
| Hook and model-state leakage absent | PASS |

The first process completed discovery and validation, then stopped at a runner function-alias
`NameError` before writing an operating point or reading test. The corrected resume path reloaded
and reproduced both completed row hashes, revalidated the scope/ranking/neuron-set hash chain,
froze and verified the operating point, and only then accessed test. No completed artifact was
overwritten. `resume.json` records this boundary explicitly.

## Runtime, artifacts, and hashes

- registered run: `results/stage_c_full/qwen2.5-1.5b-instruct/run_20260904T160405Z`;
- discovery / validation / test grid runtimes: `1905.263 / 1782.398 / 2173.605` seconds;
- attribution runtime: `141.559` seconds;
- measured component runtime across both processes: `6019.090` seconds (100.32 minutes);
- resumed-process peak allocated / reserved: `3,290,587,648 / 3,508,535,296` bytes;
- output size: 25 files, 113,706,953 bytes;
- operating-point canonical hash: `941af8f385efef08ab44fd02e9f3da291c2b3547bed272de4c3591129d073c1e`;
- formal-gate canonical hash: `5dd1d185115b3f78be6c9b9293a3f07dbb11fba229af882e5e943f2846e2777f`;
- test-row hash: `09cf7656ec71129e3aa78fbff8799a0be155b9fafbee3e8228ce9cab32eef262`.

The run records root commit `5bfbf240eacfacfff078f08086f8eb93a0b62c3e`, Sink-Repro commit
`9ab67e914464b13863b67527d8ea14068ee9ff10`, and Sink-KD commit
`db114c9c5eb6ffc5de13e444c783408ea7401c62`. The implementation and report were uncommitted, so
provenance correctly records a dirty root worktree and its complete path list.

The final project-local suite with cached CUDA integration enabled reported **194 passed, 155
subtests passed, 0 failed, 0 skipped in 7.54 seconds**.

## Interpretation and next permitted action

Stage C demonstrates that the baseline sink itself transfers to Qwen2.5-1.5B, but the registered
sparse neuron-level causal result does not. The strongest registered intervention remains below
the effect threshold, lacks a monotonic dose response, and incurs severe neutral-language drift.
This is a valid architecture/checkpoint null, not evidence that the sink is absent.

Per the master design, **do not implement or run Stage D MMLU, ARC-Challenge, CulturalBench, or
GSM8K experiments for this Qwen checkpoint.** The next action is to stop and preserve this
model-negative result. Any alternative checkpoint, attribution method, fraction, alpha, or unit
type would require a prospectively registered new experiment or amendment and must not be folded
into Stage C.
