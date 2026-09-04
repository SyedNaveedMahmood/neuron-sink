# Targeted Suppression

The first pilot is complete. `scripts/run_suppression_smoke.py` suppresses the frozen GPT-2 MLP
neuron sets at alpha 1.0/0.5/0.0, compares them with five frozen layer-count-matched controls, and
saves paired sink/CE/KL/top-1 metrics on the disjoint 24/24/24 neutral splits.

The registered run passed the smoke gate; see `reports/TASK7_GPT2_SUPPRESSION_SMOKE.md`. This is
not the formal Stage-B confirmation and must not be used to bypass the RTX-4080 GPT-2-small/medium
experiment.
