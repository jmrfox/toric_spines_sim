# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # pypet + Arbor demo
#
# This notebook demonstrates how to use the [`pypet`](https://pypet.readthedocs.io/) parameter exploration toolkit
# to drive a small [Arbor](https://arbor-sim.org) single-cell simulation.
#
# We will:
# - define a simple cable cell and recipe in Arbor,
# - define a parameter space for current amplitude and duration,
# - use `pypet.Environment` to manage a trajectory and run simulations, and
# - collect and plot the maximum somatic voltage for each parameter set.
#

# %%
import arbor as A
from arbor import units as U

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pypet_rebuild import (
    Environment,
    cartesian_product,
    HDF5StorageService,
    Trajectory,
    Parameter,
    Result,
)


# %% [markdown]
# ## Define a simple Arbor single-cell model
#
# We follow the same basic pattern as in `arbor_tutorial_single_cell_detailed_recipe.ipynb`,
# but keep the model minimal so we can run many simulations quickly.
#


# %%
def build_simple_cell():
    """Return a cable_cell and label_dict for a simple branched morphology."""
    tree = A.segment_tree()
    root = A.mnpos

    # Soma (tag 1)
    soma = tree.append(root, (0.0, 0.0, 0.0, 5.0), (20.0, 0.0, 0.0, 5.0), tag=1)

    # Single dendrite (tag 3)
    dend = tree.append(soma, (20.0, 0.0, 0.0, 2.0), (200.0, 0.0, 0.0, 1.0), tag=3)

    morph = A.morphology(tree)

    labels = A.label_dict(
        {
            "all": "(all)",
            "soma": "(tag 1)",
            "dend": "(tag 3)",
            "root": "(root)",
            "terminal": "(terminal)",
        }
    )

    decor = (
        A.decor()
        .paint("(all)", A.density("pas"))
        .paint("(all)", A.density("hh"))
        .place("(root)", A.iclamp(10 * U.ms, 1 * U.ms, current=2 * U.nA), "iclamp0")
        .place("(root)", A.threshold_detector(-10 * U.mV), "detector")
    )

    # Simple CV policy: fixed max extent
    cvp = A.cv_policy_max_extent(20.0)

    cell = A.cable_cell(morph, decor, labels, cvp)
    return cell, labels, morph


class SimpleRecipe(A.recipe):
    """Recipe wrapping a single cable_cell with a voltage probe at the soma."""

    def __init__(self, cell):
        A.recipe.__init__(self)
        self._cell = cell
        self._props = A.neuron_cable_properties()

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self._cell

    def probes(self, gid):
        # Probe membrane voltage in the soma region
        return [A.cable_probe_membrane_voltage("(tag 1)", "Um")]

    def global_properties(self, kind):
        return self._props


# %% [markdown]
# ## pypet integration: define a simulation function
#
# `pypet` expects a function that takes a trajectory object as its only argument.
# We pull parameter values from the trajectory, run Arbor, and store results back into the trajectory.
#


# %%
def run_arbor_sim(traj: Trajectory) -> Trajectory:
    """Run a single Arbor simulation for the current trajectory parameter set.

    Parameters read from `traj`:
    - stimulus.current_nA
    - stimulus.duration_s
    - sim.tfinal_s
    - sim.dt_s
    """
    # Build cell and recipe
    cell, labels, morph = build_simple_cell()

    # Update the stimulus in the decor by re-instantiating with the requested current/duration.
    # For simplicity, we rebuild the cell with a modified decor here.
    current = traj.parameters["stimulus.current_nA"].value * U.nA
    duration = (
        traj.parameters["stimulus.duration_s"].value * 1000 * U.ms
    )  # Convert s to ms for Arbor

    decor = (
        A.decor()
        .paint("(all)", A.density("pas"))
        .paint("(all)", A.density("hh"))
        .place("(root)", A.iclamp(10 * U.ms, duration, current=current), "iclamp0")
        .place("(root)", A.threshold_detector(-10 * U.mV), "detector")
    )
    cvp = A.cv_policy_max_extent(20.0)
    cell = A.cable_cell(morph, decor, labels, cvp)

    recipe = SimpleRecipe(cell)

    # Create context and domain decomposition (single-cell, serial)
    ctx = A.context()
    decomp = A.partition_load_balance(recipe, ctx)
    sim = A.simulation(recipe, ctx, decomp)

    handle = sim.sample(
        (0, "Um"), A.regular_schedule(traj.parameters["sim.dt_s"].value * 1000 * U.ms)
    )
    sim.run(
        traj.parameters["sim.tfinal_s"].value * 1000 * U.ms,
        traj.parameters["sim.dt_s"].value * 1000 * U.ms,
    )

    samples = sim.samples(handle)
    if not samples:
        raise RuntimeError("No samples returned; check probe configuration.")
    data, meta = samples[0]
    t = data[:, 0]
    v = data[:, 1]

    # Store summary statistics and (optionally) the full trace in the trajectory.
    traj.results.max_voltage_mV = float(np.max(v))
    traj.results.min_voltage_mV = float(np.min(v))
    traj.results.spike_like = bool(np.max(v) > 0.0)

    # To keep the HDF5 file small, we only store downsampled traces by default.
    stride = max(1, len(t) // 200)
    traj.results.t_s = t[::stride] / 1000.0  # Convert from ms to s
    traj.results.v_mV = v[::stride]

    return traj


# %% [markdown]
# ## Configure the pypet environment and parameters
#
# We create a trajectory with a small Cartesian product of current amplitude and duration,
# and a single set of simulation parameters (`tfinal`, `dt`).
#

# %%
traj = Trajectory(name="arbor_pypet_demo")
traj.add_parameter(Parameter("stimulus.current_nA", 2.0))
traj.add_parameter(Parameter("stimulus.duration_s", 0.001))
traj.add_parameter(Parameter("sim.tfinal_s", 0.05))
traj.add_parameter(Parameter("sim.dt_s", 0.000025))

storage = HDF5StorageService(file_path="arbor_pypet_demo.hdf5")
env = Environment(trajectory=traj, storage=storage)

# Define a simple 2D grid of parameters to explore.
current_values = [0.5, 1.0, 2.0]
duration_values = [0.5, 1.0, 2.0]

space = {
    "stimulus.current_nA": current_values,
    "stimulus.duration_s": duration_values,
}
env.run_exploration(run_arbor_sim, space)


# %% [markdown]
# ## Run the parameter exploration
#
# `pypet` now iterates over all parameter combinations and calls `run_arbor_sim`.
#

# %%
env.f_run(run_arbor_sim)


# %% [markdown]
# ## Inspect results
#
# We read back the results from the trajectory and assemble them into a small DataFrame
# for analysis and visualization.
#

# %%
rows = []
for run_name in traj.f_iter_runs():
    traj.v_idx = run_name
    rows.append(
        {
            "run": run_name,
            "current_nA": traj.stimulus.current_nA,
            "duration_s": traj.stimulus.duration_s,
            "max_voltage_mV": traj.results.max_voltage_mV,
            "min_voltage_mV": traj.results.min_voltage_mV,
            "spike_like": bool(traj.results.spike_like),
        }
    )

df_results = pd.DataFrame(rows)
df_results.sort_values(["current_nA", "duration_s"], inplace=True)
df_results


# %% [markdown]
# ### Example visualization: heatmap of max voltage
#

# %%
pivot = df_results.pivot(
    index="duration_s", columns="current_nA", values="max_voltage_mV"
)
plt.figure(figsize=(5, 4))
im = plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel("current (nA)")
plt.ylabel("duration (s)")
plt.colorbar(im, label="max V (mV)")
plt.title("Max somatic voltage vs. stimulus parameters")
plt.tight_layout()
plt.show()
