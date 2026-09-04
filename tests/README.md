# Tests

Run the project-local suite explicitly so pytest does not try to collect the two read-only upstream
test trees, which contain duplicate module basenames:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Set `NEURON_SINK_RUN_GPU_INTEGRATION=1` to include the cached real-GPT-2 CUDA tests. The suite
covers upstream sink parity, alpha=1 identity, zero/graded suppression, causal ordering,
anti-leakage, frozen attribution/selection, matched controls, Task-7 neutral metrics and schemas,
and the automatic held-out smoke gate.
