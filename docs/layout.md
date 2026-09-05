# Project and data layout

Use `toric_spines_sim.paths` instead of `../../data/...` (see [Path helpers](package.md#path-helpers)).

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

To add a spine, drop `TS{id}.obj` in `data/mesh/` (and optionally `TS{id}_AZ.nff` in `data/nff/`), then run the [morphology pipeline](morphology.md). From IMOD: [From IMOD to mesh](reconstruction.md). Existing meshes: TS1, TS2, TS3, TS4, TS21, TS24, TS48, TS67, TS76.
