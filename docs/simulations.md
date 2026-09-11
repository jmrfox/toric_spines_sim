# Simulations

With a micron sink SWC and synpts in hand, instantiate the Arbor cell (`TSModel` / `TSRecipe`), restore loops with gap junctions at `# CYCLE_BREAK` / `# MULTI_NECK` sites, generate synaptic event streams, and record voltages.

Each spine with synpts has a package under `simulations/`. Shared analysis lives in `simulations/common/`; you normally edit only the per-spine files.

Run from the repository root as modules.

```bash
uv run python -m simulations.ts1.axons
uv run python -m simulations.ts1.axons_dash
uv run python -m simulations.ts76.axons_dash
uv run python -m simulations.ts2.ts2_integration
uv run python -m simulations.ts2.ts2_integration --verbose
```

Build the NMODL catalogue first if `toric_spines_sim/mechanisms/custom-catalogue.so` is missing or stale (see [Getting started](getting-started.md)). Confirm with `check_catalogue()`:

```python
from toric_spines_sim.model import check_catalogue
check_catalogue()
```

If that raises, rebuild from the repository root:

```bash
uv run bash scripts/make_custom_catalogue.sh
```

Precomputed morphologies already in `data/` are `data/swc/microns/TS{id}_wsink_r10um.swc` (also `r5um`). Synapse coordinates are `data/pointsets/microns/TS{id}_synpts.txt`. Expected inputs for an axon study also include `data/ts_axons/ts{id}_axons.txt`. PDF / HTML outputs go under `simulations/ts{id}/axons/results/`.

```
simulations/
  README.md
  common/                 shared axon PDF / Dash / animation helpers
  ts1/                    (same pattern for ts2, ts3, ts4, ts48, ts67, ts76)
    params.py             returns make_icx_parameter_bank_invivo(); switch factory or set .value to tweak
    inputs.py             SWC / synpts / axon map + scenario options
    axons.py              PDF study
    axons_dash.py         3D Dash dashboard
    axons_animation.py    HTML voltage animation
    neurosignature/       TS1-only descriptor pipeline
    hypergrid/            TS1-only parameter sweep
  ts2/
    ts2_integration.py    extra Poisson integration + PDF
    ts2_integration_viz.py
```

Tweak biophysics in `params.py`. Each spine returns `make_icx_parameter_bank_invivo()`; uncomment `make_default_parameter_bank()` or `make_icx_parameter_bank_invitro()`, or set `.value` on sampled keys, for a per-spine change. Tutorials use `make_default_parameter_bank()`. Tweak which axons fire, rates, and pulse timing in `inputs.py`. Leave `common/` alone unless a change should apply to every spine.

## Shared vs per-model

| Path | Role |
|------|------|
| `simulations/common/utils.py` | Axon colors, pulse times, topology load (`n_axons` from the assignment file) |
| `simulations/common/inputs.py` | `InputScenario` builders (pulse / periodic / Poisson) |
| `simulations/common/axons_report.py` | Pairwise / sequential / synchrony PDF study |
| `simulations/common/axons_dash.py`, `axons_animation.py` | Dash and HTML animation helpers |
| `simulations/ts{id}/params.py` | Returns `make_icx_parameter_bank_invivo()`; uncomment another factory or set `.value` to tweak |
| `simulations/ts{id}/inputs.py` | Axon map path, scenario options (`ACTIVE_AXONS`, rates, sequential order) |
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

## Active runners

| Path | Role |
|------|------|
| `simulations/ts{id}/axons.py` (ts1, ts2, ts3, ts4, ts48, ts67, ts76) | Axon PDF study (`python -m simulations.ts{id}.axons`) |
| `ts{id}/axons_dash.py`, `axons_animation.py` | Dash dashboard and HTML voltage animation |
| `ts1/neurosignature/ts1_neurosignature.py` | Neurosignature descriptor pipeline (`uv sync --group neurosignature`; [design](neurosignature.md)) |
| `ts1/hypergrid/ts1_hypergrid.py` | TS1 HyperGrid parameter sweep |
| `ts2/ts2_integration.py` | TS2 Poisson integration + PDF |
| `ts2/ts2_integration_viz.py` | TS2 HTML frusta animation |

All of the above use `TSSimulator` and `toric_spines_sim.paths`.

## Archived

Superseded scripts live under `archive/simulations/` (`ts1/original/`, standalone `ts2_sim.py`, empty `ts3`/`ts4` placeholders). Do not run them.
