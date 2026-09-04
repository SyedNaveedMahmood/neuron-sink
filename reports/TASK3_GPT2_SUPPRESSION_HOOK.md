# Task 3 — GPT-2 MLP Suppression Hook

## Source provenance

- neuron-sink starting commit: `b302e2d030555b4fa95507b6c5c2369ac2475093`
- sink-repro commit: `9ab67e914464b13863b67527d8ea14068ee9ff10`
- sink-kd commit: `db114c9c5eb6ffc5de13e444c783408ea7401c62`
- Model: `openai-community/gpt2` (requested as its equivalent short id, `gpt2`)
- Model revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Runtime: Transformers 5.3.0, PyTorch 2.10.0+cu128, float32, eager attention,
  `model.eval()`, deterministic algorithms, `cuda:0`
- GPU: NVIDIA GeForce RTX 2060 SUPER
- Input: first fixed 40-token example (`sst2:38619`) from the passing Task-2
  manifest, SHA-256
  `b7caf6b666b502790542a962b428a0b22dfa2d08bf77f51c383e249e7ce78c64`

## Hook definition

- Exact module path: `transformer.h[layer].mlp.c_proj`.
- Tensor location: the forward-pre-hook input to `c_proj`, after `c_fc` and
  `NewGELUActivation` and before the final MLP projection.
- Observed GPT-2-small tensor shape for the audit: `[1, 40, 3072]` =
  `[batch, sequence, MLP intermediate]`.
- GPT-2-small layers: 12, zero-indexed `0..11`.
- MLP intermediate width: 3072 in every layer, derived from
  `c_proj.weight.shape[0]` and checked against `c_fc.weight.shape[1]`; it is not
  hard-coded by the adapter.

Transformers 5.3.0's installed `GPT2MLP.forward` executes `c_fc`, activation,
`c_proj`, then dropout. Therefore the pre-hook input is precisely the registered
post-activation MLP neuron vector, rather than a residual dimension, projection output,
weight, head channel, or Q/K/V coordinate.

## Implementation

- `neuron_sink/__init__.py`: minimal public Task-3 package surface.
- `neuron_sink/model_adapters.py`: GPT-2-only structure, layer, width, neuron-id,
  and hook-input validation.
- `neuron_sink/suppression.py`: immutable `NeuronSet` and scoped suppression
  context.
- `tests/test_gpt2_adapter.py`: synthetic GPT-2 adapter and failure-path tests.
- `tests/test_suppression.py`: synthetic semantic/state tests plus a gated cached-model
  CUDA integration test.
- `scripts/verify_gpt2_suppression_hook.py`: repeatable real GPT-2 Task-3 audit.

The implementation uses a PyTorch forward-pre-hook because `c_proj` directly receives
the registered post-GELU tensor. For `alpha < 1`, it clones that tensor, multiplies only
the selected coordinates at every sequence position, and returns the replacement input.
Hooks are context-scoped and removed in `finally`-equivalent context cleanup. Audit
callbacks receive detached clones so they cannot mutate the live forward. `alpha=1`
validates all inputs but bypasses hook registration and tensor cloning entirely.

The real-model audit used only arbitrary DEBUG coordinates:
layer 2 `{0, 17, 31}` and layer 7 `{1, 19, 47}`. They were not ranked and are not
labelled sink or causal neurons.

## Alpha=1 identity

- `torch.equal` logits: `True`.
- `torch.equal` for every attention tensor: `True`.
- Maximum absolute logits difference: `0.0`.
- Maximum absolute attention difference: `0.0`.
- Baseline Task-2 sink score on the fixed example: `0.5729276798665524`.
- Identity Task-2 sink score: `0.5729276798665524`.
- Absolute sink-score difference: `0.0`.
- Active suppression hooks in the identity context: `0`.
- Result: **PASS**.

## Alpha=0 audit

- Every selected coordinate was exactly zero at both selected `c_proj` inputs.
- Maximum selected-coordinate zero/scaling error: `0.0`.
- Maximum unselected-coordinate difference: `0.0`.
- Shape, dtype (`torch.float32`), and device (`cuda:0`) were preserved.
- Result: **PASS**.

## Alpha=0.5 audit

- Maximum error versus `0.5 * baseline_selected`: `0.0`.
- Maximum unselected-coordinate difference: `0.0`.
- Shape, dtype (`torch.float32`), and device (`cuda:0`) were preserved.
- Result: **PASS**.

## Multi-layer audit

- Selected zero-indexed layers: `[2, 7]` in one `NeuronSet` and one context.
- Both layer hooks ran once; each changed only its own requested coordinates.
- Result: **PASS**.

## Input-validation tests

- Invalid layer 12: rejected with `IndexError` and the valid range `[0, 12)`.
- Negative neuron -1: rejected with `IndexError`.
- Out-of-range neuron 3072: rejected with `IndexError` and range `[0, 3072)`.
- NaN alpha: rejected with `ValueError` as non-finite.
- Alpha -0.1 and 1.1: rejected with `ValueError` as outside `[0, 1]`.
- Unit tests also reject non-integer/bool layer and neuron ids, nonnumeric alpha,
  duplicate/empty neuron sets, wrong model structure, and wrong hook tensor shape.
- Result: **PASS**.

## State-leakage audit

- Sequence on the same model object: baseline, alpha=0, baseline.
- Baseline-before/after maximum logits difference: `0.0`.
- Baseline-before/after maximum attention difference: `0.0`; every tensor was
  `torch.equal`.
- Suppression hooks remaining: `0`; pre-hook registries matched their initial state.
- Full parameter-value SHA-256 before and after:
  `2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527`.
- Parameter mutation: none detected.
- Result: **PASS**.

## Output-validity audit

- Non-finite logits: `0`.
- Non-finite attention values: `0`.
- Maximum absolute attention row-sum error: `2.980232238769531e-7`, within the
  upstream attention tolerance.
- Result: **PASS**.

## Runtime

- Authoritative real-model audit wall time: `3.202699000001303` seconds with a warm,
  local `X:` Hugging Face cache.
- Peak PyTorch GPU memory allocated: `517,869,568` bytes (`493.88 MiB`).
- Peak PyTorch GPU memory reserved: `559,939,584` bytes (`534.00 MiB`).
- Complete automated suite with the CUDA integration gate enabled: 16 passed,
  0 failed, 0 skipped in `0.799` seconds.
- Ignored machine-readable audit:
  `results/task3_gpt2_suppression_hook/run_20260904T092546Z/summary.json`.

## Deviations / problems

NONE. A forward-pre-hook was selected from the mechanisms explicitly allowed for Task 3.
The model and Task-2 fixture were already cached on `X:`, so no model, dataset, or cache
content was written to the low-space `C:` drive. No upstream file was changed.

## TASK3_HOOK

PASS

## Reproduction on amended hardware (RTX 2060, 12 GB)

Amendment `A001` (`docs/AMENDMENTS.md`) registered a second development GPU. The Task-3 audit was
re-run unchanged on that machine before Task 4 began. The original RTX 2060 SUPER result above is
untouched.

- Repo commit at re-run: `31f8e56109f8db078d0514bf773294d611a4c0f0`
- GPU: `NVIDIA GeForce RTX 2060` (12 GB), Windows 11 Pro 10.0.26200
- Stack: Python 3.12.4, PyTorch 2.10.0+cu128, Transformers 5.3.0, NNsight 0.7.0, datasets 4.8.4
- Fixture: first 40-token example (`sst2:38619`) of the re-run Task-2 manifest, SHA-256
  `b7caf6b666b502790542a962b428a0b22dfa2d08bf77f51c383e249e7ce78c64` — the same manifest hash the
  RTX 2060 SUPER produced
- Code changes: the `"RTX 2060 SUPER"` assertion became
  `neuron_sink.provenance.require_registered_gpu("dev")`, and the `--cache-dir` default, which
  pointed at the nonexistent `X:` drive, now reads `$NEURON_SINK_HF_CACHE`. The hook implementation
  in `neuron_sink/suppression.py` is unchanged.
- Ignored run directory: `results/task3_gpt2_suppression_hook/run_20260904T103834Z`

| Quantity | RTX 2060 SUPER | RTX 2060 (12 GB) |
|---|---|---|
| Hook tensor shape | `[1, 40, 3072]` | `[1, 40, 3072]` |
| MLP intermediate width | 3072 | 3072 |
| `alpha=1` logits equal | True | True |
| `alpha=1` max logits difference | `0.0` | `0.0` |
| `alpha=1` max attention difference | `0.0` | `0.0` |
| `alpha=1` sink difference | `0.0` | `0.0` |
| `alpha=0` max unselected difference | `0.0` | `0.0` |
| `alpha=0.5` max scaling error | `0.0` | `0.0` |
| Baseline before/after logits difference | `0.0` | `0.0` |
| Hooks removed / parameters unchanged | True / True | True / True |
| Max attention row-sum error | `2.980232238769531e-7` | `2.980232238769531e-7` |
| Peak GPU memory allocated | `517,869,568` B | `517,869,568` B |
| Peak GPU memory reserved | `559,939,584` B | `559,939,584` B |

Peak memory matched to the byte. Wall time was `2.039` s with a warm local cache.

Automated suite on this machine, with the CUDA integration gate enabled
(`NEURON_SINK_RUN_GPU_INTEGRATION=1`): **16 passed, 0 failed, 0 skipped**, plus 17 subtests — the
same counts recorded on the RTX 2060 SUPER. The gated integration test was renamed
`test_real_gpt2_c_proj_input_on_registered_dev_gpu` since it is no longer SUPER-specific.

The Task-3 neuron ids remain arbitrary DEBUG coordinates. They are not sink neurons and must not be
cited as findings.

### TASK3_HOOK (amended hardware)

PASS
