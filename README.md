# toric_spines_sim

In this project, we carry out simulations of toric spines (Sanculi, et al., 2020) using Arbor (arbor-sim.org). 
Our primary goal is to characterize the integration properties of isolated toric spines in response to synaptic inputs.

Toric spines (TSs) are structures found in space-specific neurons (SSNs) in the external nucleus of the inferior colliculus (ICx) of the barn owl. They differ significantly in morphology from more commonly studied dendritic spines, being characterized by holes (topological genus > 0, hence the name "toric"), being much larger in size, and other particular attributes.

Workflow:
1. The morphology is presumed important to their function, so we want the simulation to accurately represent that. To do this, I have used a custom pipeline which converts the TS mesh to a SWC model using a mean-curvature flow algorithm to determine the skeleton and then a fitting algorithm to find corresponding radii. Since SWC describes directed trees (no cycles), each SWC file is annotated with which pairs of nodes should be connected to form cycles. 
2. The morphology is loaded using a separate package of mine, `swctools` (https://github.com/jmrfox/swctools), which also provides visualizations. 
2.5. Since each TS is isolated, we need to extend the morphology with a "sink" to correctly account for electrical diffusion into the rest of the cell. At this stage, we use a simple cylindrical geometry to represent the sink. We can then observe the membrane potential in the sink and study the effects of different sized sinks on the results. 
3. The simulation is carried out using Arbor. We load in the morphology, add mechanisms such as synapses and ion channels, and simulate synaptic inputs. Each synaptic input is specified as a stream of events (time stamps indicating when a synaptic event occurs), which are then applied to the simulation. So, for $N_s$ synapses, we have a set $S = \{S_i | i=1,2,\dots,N_s\}$, where $S_i = \{t_{ij}\}$. The results of the simulation are the membrane potential time-series at a point, such as $V_\text{sink}(t)$, or multiple points; $V(t) = \{V_i(t) | i=1,2,\dots,N_p\}$.
4.  We can study integrative properties of the TS by designing $S$ and analyzing $V(t)$. A primary analysis is to compute pairwise phenomena: for a pair of synapses $i,j$ with corresponding event rates $\lambda_i,\lambda_j \in \{ \lambda_\text{min}, \lambda_\text{max}\}$, we can run the simulation over the outer product of $S_i$ and $S_j$ and analyze the resulting $V_\text{sink}(t)$. 

## Environment (uv)

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

### Installation

If you don't have uv installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

Clone the repository and sync dependencies:
```bash
git clone <repository-url>
cd toric_spines_sim
uv sync
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

### Running Scripts

Use `uv run` to execute Python scripts with the project environment:
```bash
uv run python scripts/script_name.py
uv run python -m module_name
```

For Jupyter notebooks:
```bash
uv run jupyter lab
```

### Updating Dependencies

To update all packages to their latest compatible versions:
```bash
uv sync --upgrade
```

To refresh and update specific packages (e.g., `swctools` from git):
```bash
uv sync --upgrade --refresh
```

### Adding New Dependencies

```bash
uv add package_name
uv add --dev package_name  # for development dependencies
```

## Path Management

The project uses a centralized path management system (`toric_spines_sim.paths`) to avoid issues with relative paths when moving scripts around.

### Usage

Instead of using relative paths like `../../data/swc/pixels/file.swc`, use the path utilities:

```python
from toric_spines_sim.paths import (
    get_swc_path,
    get_pointset_path,
    get_simulation_path,
    PROJECT_ROOT,
)

# Get paths to data files
swc_file = get_swc_path("TS1_s200.swc", units="microns")
pointset = get_pointset_path("TS1_synpts.txt", units="pixels")

# Get paths within simulation directories
output_file = get_simulation_path("ts1", "results.html")

# Access project root or data directories directly
from toric_spines_sim.paths import DATA_DIR, SWC_MICRONS_DIR
custom_path = DATA_DIR / "custom" / "file.txt"
```

### Benefits

- **Location-independent**: Scripts work regardless of where they're located in the project
- **No relative path headaches**: No need to update `../../` paths when moving files
- **Centralized**: All path logic in one place, easy to modify if directory structure changes
- **Type-safe**: Returns `Path` objects for modern path handling
