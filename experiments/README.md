# Experiments

New experiments belong here. Reference implementations remain pinned under `upstream/` and should not be edited in place.

## Planned sequence

### 1. `neuron_localization/`
Localize candidate sink-associated MLP neurons and attention dimensions inside already identified sink-heavy layers/heads. Start with GPT-2-small. Attribution is a ranking heuristic only; causal claims require intervention.

### 2. `suppression/`
Run graded activation suppression on top-ranked units and matched random controls. Minimum output schema:

- model/checkpoint
- seed
- layer/head/module
- neuron or dimension set
- `k`
- suppression coefficient `alpha`
- `sink_before`, `sink_after`, `delta_sink`
- `ce_before`, `ce_after`, `delta_ce`
- optional KL / top-1 flips / downstream task metrics

### 3. `teacher_student/`
Apply the same localization and suppression protocol to the natural teacher sink and Sink-KD conditions (LogitKD, AttnKD, and available no-sink controls) without changing their upstream training definitions.

## Controls

At minimum use size- and layer-matched random units. Add activation-matched controls if the pilot shows a nontrivial effect. Preserve the existing paper harness's frozen dataset and sink-measurement conventions wherever possible.
