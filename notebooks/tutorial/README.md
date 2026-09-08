# Tutorial notebooks

Each notebook covers one package concept. Work through them in order, or open a single topic. By default they load precomputed files from `data/`. Set `RECOMPUTE = True` only when you want to re-run an expensive step; output then goes to `notebooks/tutorial/_artifacts/` so `data/` is not overwritten.

Open Jupyter from the **repository root** (`uv run jupyter lab`). Pair `.py` sources with notebooks:

```bash
uv run jupytext --sync notebooks/tutorial/*.py
```

`.ipynb` files are not tracked by git. Expensive pymcfs / mascaf / sink rewrite cells default to `RECOMPUTE = False`. Optional all-spine galleries stay behind `SHOW_ALL = False`. Notebooks **11–16** need the local NMODL catalogue (`uv run bash scripts/make_custom_catalogue.sh`).

IMOD reconstruction and CGALLab are not in this series — see [From IMOD to mesh](../../docs/reconstruction.md) and [Skeletons](../../docs/skeletons.md). A raw Arbor example (not `TSSimulator`) is `notebooks/misc/arbor_cable_cell.py`.

### Morphology

| Notebook | Topic |
|----------|--------|
| `01_meshes` | Mesh paths, input/output, watertight / Euler quality checks, `load_and_repair` |
| `02_skeletonization_basic` | `skeletonize_mesh` with package defaults |
| `03_skeletonization_advanced` | pymcfs overrides, `mesh_to_swc` |
| `04_swc_cable_fit` | `fit_swc`, `# CYCLE_BREAK` |
| `05_neck_points` | `NeckpointParams`, TS vs cell mesh |
| `06_sink_attachment` | `SinkGeometry`, append sink, tags 5/6 |
| `07_synapses` | synpts, axon maps, `SynapsePopulation` |

### Inputs

| Notebook | Topic |
|----------|--------|
| `08_parameters` | `ParameterBank` / `ParameterSet`, Sanculi $C_m$, EPSC $\tau$ |
| `09_events_basic` | Deterministic vs stochastic generators, `FlatRateCurve` |
| `10_events_advanced` | Other rate curves, shared routing, axon remap |
| `11_mechanisms` | NMODL catalogue, `MODEL_REGISTRY` |

### Model and simulation

| Notebook | Topic |
|----------|--------|
| `12_tsmodel_and_tsrecipe` | `TSModel.build_cell()`, gap junctions, `TSRecipe` |
| `13_tssimulator` | `TSSimulator` high-level interface and `run()` |
| `14_simulation_results` | traces, `integrate_voltages_by_tag`, save/load |

### Analysis and visualization

| Notebook | Topic |
|----------|--------|
| `15_pdf_reports` | `PdfReport` |
| `16_kmatrix` | Pairwise $k$ (TS1, then cylinder / TS2) |
| `17_visualization` | Mesh/SWC/Arbor figures; animation and Dash pointers |
| `18_geodesics_and_compartments` | Graph distances and compartment labels |
| `19_spiny_dendrite` | Synthetic classical-spine comparison morphology |

Extra demos stay under `notebooks/misc/` (including an earlier spiny-dendrite notebook and a raw Arbor example). Full axon PDF study: `uv run python -m simulations.ts1.axons`.
