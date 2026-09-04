# Simulations

Run from the repository root as modules. Shared axon analysis lives in
[`common/`](common/); each spine has thin files you can edit
(`params.py`, `inputs.py`) plus runners.

```bash
uv run python -m simulations.ts1.axons
uv run python -m simulations.ts2.axons
uv run python -m simulations.ts2.ts2_integration
uv run python -m simulations.ts2.ts2_integration --verbose
```

Build the NMODL catalogue first if `toric_spines_sim/mechanisms/custom-catalogue.so` is missing:

```bash
uv run bash scripts/make_custom_catalogue.sh
```

Blessed morphologies are `data/swc/microns/TS{id}_wsink_r10um.swc` (also `r5um`). Synapse coordinates are `data/pointsets/microns/TS{id}_synpts.txt`.

## Shared vs per-model

| Path | Role |
|------|------|
| [common/utils.py](common/utils.py) | Axon colors, pulse times, topology load (`n_axons` from the assignment file) |
| [common/inputs.py](common/inputs.py) | `InputScenario` builders (pulse / periodic / Poisson) |
| [common/axons_report.py](common/axons_report.py) | Pairwise / sequential / synchrony PDF study |
| [common/axons_dash.py](common/axons_dash.py), [common/axons_animation.py](common/axons_animation.py) | Dash and HTML animation helpers |
| `simulations/ts{id}/params.py` | Parameter bank: starts from `make_default_parameter_bank()`; uncomment `.value` lines to override |
| `simulations/ts{id}/inputs.py` | Axon map path, scenario knobs (`ACTIVE_AXONS`, rates, sequential order) |
| `simulations/ts{id}/axons.py` | Thin `run_axons_study(MODEL)` entry point |

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

TS21 and TS24 have sink SWCs but no `TS*_synpts.txt`, so they are not wired up yet.

## Active

| Path | Role |
|------|------|
| [ts1/axons.py](ts1/axons.py) (and ts2, ts3, ts4, ts48, ts67, ts76) | Axon PDF study (`python -m simulations.ts{id}.axons`) |
| `ts{id}/axons_dash.py`, `axons_animation.py` | Dash dashboard and HTML voltage animation |
| [ts1/neurosignature/ts1_neurosignature.py](ts1/neurosignature/ts1_neurosignature.py) | Neurosignature descriptor pipeline |
| [ts1/hypergrid/ts1_hypergrid.py](ts1/hypergrid/ts1_hypergrid.py) | TS1 HyperGrid parameter sweep |
| [ts2/ts2_integration.py](ts2/ts2_integration.py) | TS2 Poisson integration + PDF |
| [ts2/ts2_integration_viz.py](ts2/ts2_integration_viz.py) | TS2 HTML frusta animation |

All of the above use `TSSimulator` and `toric_spines_sim.paths`.

## Archived

Superseded scripts live under [`archive/simulations/`](../archive/README.md) (`ts1/original/`, standalone `ts2_sim.py`, empty `ts3`/`ts4` placeholders). Do not run them.
