# toric_spines_sim

Arbor simulations of barn-owl **toric spines** (Sanculi et al., 2020). These are large, hole-containing (genus > 0) spines on space-specific neurons in ICx. The goal is to characterize how an isolated spine integrates synaptic input, including electrical leak into the rest of the cell.

Work in this repo is: turn an EM mesh into a cable (SWC) model, attach a cylindrical **sink**, drive synapses from mapped axons, and analyze sink / compartment voltages.

High-level path: mesh → skeleton → SWC (`# CYCLE_BREAK` / `# MULTI_NECK`) → sink → `TSModel` / `TSRecipe` → event generators → `TSSimulator.run()` → traces, k-matrix, Dash / PDF.

## Getting started

1. Install [uv](https://docs.astral.sh/uv/) if needed, clone the repo, then `uv sync` (needs git access to `swctools`, `jscip`, and `neurosignature`).
2. Build custom NMODL mechanisms (once, and again after editing `.mod` files):

```bash
uv run bash scripts/make_custom_catalogue.sh
```

3. Open `notebooks/examples/params_demo` and `events_demo`, then a morphology notebook (`view_wsink_swcs`) and an integration notebook (`ts1_wsink_sim` or `cylinder_wsink_integrate`).
4. Canonical scripted experiment: `uv run python -m simulations.ts1.axons` (or `ts2`, `ts3`, …). See [simulations/README.md](simulations/README.md).
5. Tests: `uv run pytest`

Private git dependencies: without `swctools` you cannot load SWCs; without `jscip` you cannot build parameter banks; without `neurosignature` only the TS1 neurosignature script fails. Mesh skeletonization / SWC fitting need `uv sync --extra mesh` (below).

## Project layout

```
toric_spines_sim/
  data/                 morphologies and pointsets used by sims
  notebooks/            examples, morphology review, integration
  scripts/              CLI for mesh → SWC → sink (and catalogue build)
  simulations/          per-spine experiments (edit params here)
  toric_spines_sim/     Python package (geometry, model, simulator, viz)
  tests/
  archive/              superseded scripts and old SWC names — do not run
```

| Path | Role |
|------|------|
| `data/` | Meshes, skeletons, SWCs, synapse / neck coordinates, axon maps |
| `scripts/` | Supported pipeline CLIs (`skeletonize_meshes.py`, `fit_swc.py`, `append_sink.py`, …) |
| `toric_spines_sim/` | Library: SWC I/O, sink append, `TSModel`, `TSSimulator`, events, viz |
| `simulations/` | One folder per spine plus shared axon-study code |
| `notebooks/examples/` | Parameter / event / cylinder demos |
| `notebooks/ts_morphology/` | Review skeletons, fitted SWCs, and sink models |
| `notebooks/ts_integration/` | Per-spine integration notebooks |
| `archive/` | Historical scripts and morphologies |

Use `toric_spines_sim.paths` instead of `../../data/...` (see [Path helpers](#path-helpers)).

## Data layout

Put **new model files** in `data/` using the `TS{id}` stem (e.g. `TS1`, `TS48`). Pixel files are EM voxels; micron files are the same geometry scaled by **0.005 µm/pixel (5 nm/pixel)** at conversion time (`--um-per-px` on `append_sink.py` / `active_zones_from_nff.py`). Simulations should load **micron** SWCs and synpts.

| Path | What belongs there |
|------|--------------------|
| `data/mesh/` | Closed triangle meshes: `TS{id}.obj`. Full cell: `cell_wrapped_simplified.obj` (neckpoints) |
| `data/skeletons/` | Mean-curvature skeletons: `TS{id}.polylines.txt` |
| `data/swc/pixels/` | Fitted cables `TS{id}.swc` and sink models `TS{id}_wsink_r{R}um.swc` |
| `data/swc/microns/` | Same SWCs in µm — **use these for simulation** |
| `data/nff/` | Active-zone NFF markers `TS{id}_AZ.nff` (pixel coords) |
| `data/pointsets/pixels/` | Neckpoints `TS{id}_neckpoint.txt`, AZ XYZ `TS{id}_AZ.txt` |
| `data/pointsets/microns/` | Neckpoints plus synapse sites `TS{id}_synpts.txt` |
| `data/ts_axons/` | Axon→synapse maps `ts{id}_axons.txt` (1-based synapse indices) |
| `data/events/` | Demo event streams |

Blessed experiment morphologies are `data/swc/microns/TS{id}_wsink_r10um.swc` (also `r5um`). Older `TS*_s50_*` / `TS*_s200_*` names live under `archive/data/swc/`.

To add a spine, drop `TS{id}.obj` in `data/mesh/` (and optionally `TS{id}_AZ.nff` in `data/nff/`), then run the pipeline below. Existing meshes: TS1, TS2, TS3, TS4, TS21, TS24, TS48, TS67, TS76.

## Morphology pipeline

Skeletonization and SWC fitting need the optional mesh extra ([pymcfs](https://github.com/jmrfox/pymcfs), [mascaf](https://github.com/jmrfox/mascaf)):

```bash
# system dependency for pymcfs CHOLMOD (Linux/WSL)
sudo apt install libsuitesparse-dev

uv sync --extra mesh
```

You can process one stem (`TS1.obj`) or every `TS*.obj` (`--all`). Review notebooks live under `notebooks/ts_morphology/`.

### 1. Organize meshes

Place closed TS meshes at `data/mesh/TS{id}.obj`. Keep the cell mesh at `data/mesh/cell_wrapped_simplified.obj` if you will compute neckpoints.

### 2. Skeletonize

Mean-curvature flow → `data/skeletons/TS{id}.polylines.txt`. Defaults match pymcfs `profile="auto"`, `branching="sparse"`, tip extension on, 500 iterations, 300s timeout.

```bash
uv run python scripts/skeletonize_meshes.py TS1.obj
uv run python scripts/skeletonize_meshes.py --all
```

Review mesh vs skeleton in `notebooks/ts_morphology/view_skeletons`.

### 3. Fit SWC cables

mascaf fit → `data/swc/pixels/TS{id}.swc`. SWC is a tree, so cycle-forming node pairs are stored as `# CYCLE_BREAK` (and `# MULTI_NECK` when needed) comments.

```bash
uv run python scripts/fit_swc.py TS1.obj
uv run python scripts/fit_swc.py --all --basis-optimize --verbose
```

Review mesh vs SWC in `notebooks/ts_morphology/view_fitted_swc`.

### 4. Neckpoints (recommended before the sink)

Compare each isolated mesh to the full cell mesh → `data/pointsets/pixels/TS{id}_neckpoint.txt`. `append_sink.py` uses this file when present; otherwise it attaches at the SWC root.

```bash
# Single-neck spines: keep the largest detected cap
uv run python scripts/compute_neckpoints.py TS1 TS2 TS4 TS24 TS48 TS76 --max-necks 1
# Multi-neck / review without writing
uv run python scripts/compute_neckpoints.py TS3 --dry-run
uv run python scripts/compute_neckpoints.py TS21 --max-necks 2
uv run python scripts/compute_neckpoints.py TS67 --max-necks 1
```

### 5. Append a cylindrical sink

The isolated spine must leak into a stand-in for the dendrite/soma. The default sink is a 10 µm-radius cylinder. The script writes **both** pixel and micron SWCs:

- `data/swc/pixels/TS{id}_wsink_r10um.swc`
- `data/swc/microns/TS{id}_wsink_r10um.swc`

```bash
uv run python scripts/append_sink.py TS1 TS2
uv run python scripts/append_sink.py --all
uv run python scripts/append_sink.py TS1 --radius-um 20
```

Review in `notebooks/ts_morphology/view_wsink_swcs`. SWC tags used throughout: spine=3, sink=5, sink tip=6.

### 6. Synapse coordinates (for simulation)

If you have NFF active-zone files, convert them and project onto the cable:

```bash
uv run python scripts/active_zones_from_nff.py
```

That writes pixel AZ files and `data/pointsets/microns/TS{id}_synpts.txt`. Axon maps in `data/ts_axons/` are one axon per line, comma-separated **1-based** synapse indices covering `1..N`. `ts1_axons.txt` is the real TS1 map; the other `ts*_axons.txt` files are placeholders. TS21 and TS24 have sink SWCs but no synpts yet, so they are not wired into `simulations/`.

Library equivalents (same defaults as the CLIs):

```python
from toric_spines_sim.geometry import skeletonize_mesh, fit_swc
from toric_spines_sim.geometry.mesh_pipeline import (
    default_polylines_path,
    default_swc_path,
    resolve_mesh_path,
)

mesh = resolve_mesh_path("TS1.obj")
skeletonize_mesh(mesh, default_polylines_path(mesh))
fit_swc(mesh, default_polylines_path(mesh), default_swc_path(mesh))
```

## Simulations

Each spine with synpts has a package under `simulations/`. Shared analysis lives in `simulations/common/`; you normally edit only the per-spine files.

```
simulations/
  README.md
  common/                 shared axon PDF / Dash / animation helpers
  ts1/                    (same pattern for ts2, ts3, ts4, ts48, ts67, ts76)
    params.py             parameter bank (uncomment .value lines to override)
    inputs.py             SWC / synpts / axon map + scenario knobs
    axons.py              PDF study
    axons_dash.py         3D Dash dashboard
    axons_animation.py    HTML voltage animation
    neurosignature/       TS1-only descriptor pipeline
    hypergrid/            TS1-only parameter sweep
  ts2/
    ts2_integration.py    extra Poisson integration + PDF
    ts2_integration_viz.py
```

Tweak biophysics in `params.py` (starts from `make_default_parameter_bank()`). Tweak which axons fire, rates, and pulse timing in `inputs.py`. Leave `common/` alone unless a change should apply to every spine.

```bash
uv run python -m simulations.ts1.axons
uv run python -m simulations.ts1.axons_dash
uv run python -m simulations.ts76.axons_dash
uv run python -m simulations.ts2.ts2_integration
```

Expected inputs for an axon study: `data/swc/microns/TS{id}_wsink_r10um.swc`, `data/pointsets/microns/TS{id}_synpts.txt`, `data/ts_axons/ts{id}_axons.txt`. PDF / HTML outputs go under `simulations/ts{id}/axons/results/`.

Details, axon-map table, and TS1-only scripts: [simulations/README.md](simulations/README.md).

### Package map (`toric_spines_sim/`)

- `geometry/` — SWC I/O, cycle breaks, mesh→skeleton→SWC, sink append, dendrite helpers
- `model/` — `TSModel`, synapses, gap junctions, `TSRecipe`
- `simulation/` — parameters, `TSSimulator`, inputs, `SimulationResults`
- `events/` — event generators / rate curves
- `viz/` — Plotly / Dash / animation helpers
- `kmatrix.py`, `report.py` — pairwise integration analysis / PDF reporting
- `paths.py` — repo-root data paths (`get_swc_path`, …)

Temperature is Kelvin (in vitro ~280 K; barn owl in vivo ~313 K). Capacitance is µF/cm²; leak is S/cm²; axial resistivity is Ω·cm.

## Environment (uv)

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository-url>
cd toric_spines_sim
uv sync
```

Run scripts and modules from the **repository root**:

```bash
uv run python scripts/append_sink.py TS1
uv run python -m simulations.ts1.axons
uv run jupyter lab
```

```bash
uv sync --upgrade
uv sync --upgrade --refresh          # e.g. after swctools git updates
uv add package_name
uv add --dev package_name
```

## Path helpers

```python
from toric_spines_sim.paths import (
    get_swc_path,
    get_pointset_path,
    get_mesh_path,
    get_skeleton_path,
    get_simulation_path,
    DATA_DIR,
)

swc_file = get_swc_path("TS1_wsink_r10um.swc", units="microns")
pointset = get_pointset_path("TS1_synpts.txt", units="microns")
mesh = get_mesh_path("TS1.obj")
skeleton = get_skeleton_path("TS1.polylines.txt")
output_file = get_simulation_path("ts1", "axons", "results")
```
