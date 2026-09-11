# Package tour

The library lives in `toric_spines_sim/`. Full signatures and docstrings are in the [API reference](api/index.md).

- [`geometry/`](api/geometry.md) — SWC I/O, cycle breaks, mesh→skeleton→SWC, sink append, dendrite helpers
- [`model/`](api/model.md) — `TSModel`, synapses, gap junctions, `TSRecipe`, `check_catalogue`
- [`simulation/`](api/simulation.md) — parameters, `TSSimulator`, inputs, `SimulationResults`
- [`events/`](api/events.md) — event generators / rate curves
- [`viz/`](api/viz.md) — Plotly / Dash / animation helpers
- [`kmatrix.py`](api/kmatrix.md), [`report.py`](api/report.md) — pairwise integration analysis / PDF reporting
- [`paths.py`](api/paths.md) — repo-root data paths (`get_swc_path`, …)

Temperature is Kelvin (in vitro ~297 K; barn owl in vivo ~313 K). Capacitance is µF/cm²; leak is S/cm²; axial resistivity is Ω·cm. See [Software](software.md#biophysics-notes) for ICx in-vitro / in-vivo banks.

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
