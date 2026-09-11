# Tutorial notebooks

Each notebook covers one package concept. 
I use the `jupytext` package to write notebooks to the repo as `.py` files, rather than writing ipynb files. This is because ipynb files with 3D plotly objects tend to be huge. You can "build" the `.ipynb` notebooks from the `.py` files using `uv run jupytext --sync *.py`.  
`jupytext` is not included in the default dependency list, so you can either add it to your venv yourself or sync to the dev dependency group.

```bash
uv add jupytext
OR
    uv sync --dev
THEN
uv run jupytext --sync notebooks/tutorial/*.py
```

By default they load precomputed data files from `data/`. Set `RECOMPUTE = True` if you want to re-run the calculation; output then goes to `notebooks/tutorial/_artifacts/` so `data/` is not overwritten.
Expensive pymcfs / mascaf / sink cells default to `RECOMPUTE = False`. Optional all-spine galleries stay behind `SHOW_ALL = False`. The NMODL catalogue is needed from `mechanisms` through `kmatrix`, except `pdf_reports` (`uv run bash scripts/make_custom_catalogue.sh`). Refer to the docs if you have not built the catalogue yet. 

IMOD reconstruction and CGALLab are not in this series — see [From IMOD to mesh](../../docs/reconstruction.md) and [Skeletons](../../docs/skeletons.md). A raw Arbor example (not `TSSimulator`) is `notebooks/misc/arbor_cable_cell.py`.

### Morphology

| Notebook | Topic |
|----------|--------|
| `01_meshes` | Mesh paths, input/output, watertight / Euler quality checks, `load_and_repair` |
| `02_pymcfs_basic` | `skeletonize_mesh` with package defaults |
| `03_pymcfs_advanced` | pymcfs options on one TS mesh |
| `04_mascaf_basic` | `fit_swc`, `# CYCLE_BREAK`, `mesh_to_swc` |
| `05_mascaf_advanced` | `fit_swc` arguments (`max_edge_length_frac`, radii, basis optimizer) |
| `06_neck_points` | `NeckpointParams`, TS vs cell mesh |
| `07_sink_attachment` | `SinkGeometry`, append sink, tags 5/6 |

### Inputs

| Notebook | Topic |
|----------|--------|
| `08_parameters` | `ParameterBank` / `ParameterSet`, Sanculi $C_m$, EPSC $\tau$ |
| `09_synapses_basic` | synpts row order, `SynapsePoint`, default AMPA path |
| `10_synapses_advanced` | `MODEL_REGISTRY`, mixed populations, labels vs indices, axon maps |
| `11_events_basic` | Bring-your-own times (lists and labeled timestamp files), then generators |
| `12_events_advanced` | Other rate curves, shared routing, axon-order remap |
| `13_mechanisms` | NMODL catalogue, `MODEL_REGISTRY` |

### Model and simulation

| Notebook | Topic |
|----------|--------|
| `14_tsmodel_and_tsrecipe` | `TSModel.build_cell()`, gap junctions, `TSRecipe` |
| `15_tssimulator` | `TSSimulator` high-level interface and `run()` |
| `16_simulation_results` | traces, `integrate_voltages_by_tag`, save/load |

### Analysis and visualization

| Notebook | Topic |
|----------|--------|
| `17_pdf_reports` | `PdfReport` |
| `18_kmatrix` | Pairwise $k$ (TS1, then cylinder / TS2) |
| `19_visualization` | Mesh/SWC/Arbor figures; animation and Dash pointers |
| `20_geodesics_and_compartments` | Graph distances and compartment labels |
| `21_spiny_dendrite` | Synthetic classical-spine comparison morphology |

Extra demos stay under `notebooks/misc/` (including an earlier spiny-dendrite notebook and a raw Arbor example). Full axon PDF study: `uv run python -m simulations.ts1.axons`.
