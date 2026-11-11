# Simulations

With a micron sink SWC and synpts in hand, build the model (`TSModel`), restore loops at `# CYCLE_BREAK` / `# MULTI_NECK` sites (passive runs use the loop-capable reference solver; see [Software](software.md#solvers-and-loops)), generate synaptic event streams, and record voltages.

Every spine runs through the same entry points under `simulations/`, with the spine as an argument.

Run from the repository root as modules.

```bash
uv run python -m simulations.axons TS1
uv run python -m simulations.axons all
uv run python -m simulations.axons_dash TS76 --scenario synapse-poisson
uv run python -m simulations.loop_attenuation
uv run python -m simulations.ts2.ts2_integration --verbose
```

Passive studies run on the scipy reference solver and studies with HH on jaxley (`TSSimulator(solver="auto")`; see [Software](software.md#solvers-and-loops)). Both close the spine's loops exactly. The TS1 axon PDF study (about 430 runs) takes about 7 minutes on a laptop CPU.

Precomputed morphologies already in `data/` are `data/swc/microns/TS{id}_wsink_r10um.swc` (also `r5um`). Synapse coordinates are `data/pointsets/microns/TS{id}_synpts.txt`. Expected inputs for an axon study also include `data/ts_axons/ts{id}_axons.txt`. PDF / HTML outputs go under `simulations/ts{id}/axons/results/`.

```
simulations/
  README.md
  axons.py                axon PDF study:        python -m simulations.axons TS1 [TS3 ...|all]
  axons_dash.py           3D Dash dashboard:     python -m simulations.axons_dash TS1 --scenario sequential
  axons_animation.py      HTML voltage movie:    python -m simulations.axons_animation TS1
  loop_attenuation.py     loops closed vs cut:   python -m simulations.loop_attenuation [TS1 ...]
  common/
    spines.py             spine registry: ids, inputs, parameter bank, per-spine overrides
    scenarios.py          named input scenarios (simultaneous, sequential, periodic, ...)
    axons_report.py       pairwise / sequential / synchrony PDF study
    axons_dash.py, axons_animation.py, inputs.py, model.py, utils.py
  ts1/neurosignature/     TS1 descriptor pipeline
  ts2/                    TS2 Poisson integration + frusta animation
  ts{id}/axons/results/   axon-study outputs (gitignored)
  attenuation/results/    loop-attenuation outputs
```

Per-spine choices live in **one place**, `simulations/common/spines.py`: the parameter bank (`default_parameter_bank()`, the in-vivo ICx bank) and `_OVERRIDES` (for example TS1's sequential axon order). Scenario defaults (which axons fire, rates, pulse timing) are keyword arguments in `simulations/common/scenarios.py`; the CLIs pick a scenario by name. Leave the rest of `common/` alone unless a change should apply to every spine.

## Axon maps

Format (`data/ts_axons/ts{id}_axons.txt`): one axon per line, comma-separated **1-based** synapse indices covering `1..N` uniquely (`N` = lines in the matching synpts file). Empty lines are ignored; comments are not supported.

`ts1_axons.txt` is the original TS1 map (10 axons, 25 synapses). The other files are **placeholders** (round-robin into `min(10, N)` axons) so the shared pipeline can run; replace them when real axon assignments are known.

| File | Synapses | Axons | Notes |
|------|----------|-------|-------|
| `ts1_axons.txt` | 25 | 10 | original map |
| `ts2_axons.txt` | 6 | 6 | placeholder |
| `ts3_axons.txt` | 46 | 10 | placeholder |
| `ts4_axons.txt` | 23 | 10 | placeholder |
| `ts48_axons.txt` | 18 | 10 | placeholder |
| `ts67_axons.txt` | 8 | 8 | placeholder |
| `ts76_axons.txt` | 5 | 5 | placeholder |

TS21 and TS24 have sink SWCs but no `TS*_synpts.txt`, so they have no axon studies; the loop-attenuation run stimulates their farthest spine point instead.

## Runners

| Path | Role |
|------|------|
| `simulations/axons.py` | Axon PDF study for any spine with an axon map (`python -m simulations.axons TS1`) |
| `simulations/axons_dash.py`, `axons_animation.py` | Dash dashboard and HTML voltage animation (`--scenario`, `--port`, HH options) |
| `simulations/loop_attenuation.py` | Peak depolarization vs distance, loops closed vs cut, all nine spines |
| `ts1/neurosignature/ts1_neurosignature.py` | Neurosignature descriptor pipeline (`uv sync --group neurosignature`; [design](neurosignature.md)) |
| `ts2/ts2_integration.py`, `ts2_integration_viz.py` | TS2 Poisson integration + PDF; HTML frusta animation |
| `common/spines.py`, `common/scenarios.py` | Spine registry and named input scenarios |
| `common/axons_report.py`, `common/inputs.py`, `common/utils.py` | Report, scenario builders, axon topology helpers |

All of the above use `TSSimulator` / `TSModel` and `toric_spines_sim.paths`.

## Archived

Superseded scripts live under `archive/simulations/` (`ts1/original/`, standalone `ts2_sim.py`, empty `ts3`/`ts4` placeholders). Do not run them.
