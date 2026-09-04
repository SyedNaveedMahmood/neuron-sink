# Scripts

Project-local entry points through the completed RTX-2060 Stage A live here. The Task-7
suppression command is:

```powershell
.venv\Scripts\python.exe scripts\run_suppression_smoke.py
```

Use `--max-examples N` only for a runtime/VRAM preflight. Limited runs deliberately emit
`NOT_EVALUATED_DRY_RUN`; only the exact 24/24/24 run evaluates the scientific smoke gate.

All runners consume pinned upstream code or frozen root-repository artefacts. Do not edit the
upstream submodules or replace these entry points with copied upstream scripts.
