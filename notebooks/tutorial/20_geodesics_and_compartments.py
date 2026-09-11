# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 20 — Geodesics and compartments
#
# Graph distances on a micron SWC from `data/` (with `# CYCLE_BREAK`
# reconnections restored) use `compute_geodesic_distances`. Compartment
# labels relative to a source→target path use `classify_compartments`.
# Probe XYZ maps back onto SWC nodes with `map_probes_to_nodes`. Units
# match the SWC (microns for simulation files).

# %%
from toric_spines_sim.geometry import (
    classify_compartments,
    compute_geodesic_distances,
    geodesic_distances_from_probe,
    map_probes_to_nodes,
    map_xyz_to_nearest_probes,
    neck_point_from_swc_file,
    sink_endpoint_location_from_swc_file,
)
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.utils import load_xyz_points

swc_path = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts = load_xyz_points(get_pointset_path("TS1_synpts.txt", units="microns"))
neck = neck_point_from_swc_file(swc_path)
sink = sink_endpoint_location_from_swc_file(swc_path)
print("SWC:", swc_path)
print("neck:", neck)
print("sink:", sink)

# %% [markdown]
# ## Distances from the neck
#
# Nearest SWC node to `neck` is the source. Unreachable nodes are omitted.

# %%
dist_from_neck = compute_geodesic_distances(swc_path, neck)
print("compute_geodesic_distances →", type(dist_from_neck).__name__, "n nodes:", len(dist_from_neck))
print(f"max geodesic from neck: {max(dist_from_neck.values()):.2f} µm")
print(f"min geodesic from neck: {min(dist_from_neck.values()):.2f} µm")

# %% [markdown]
# ## Compartment classes
#
# `classify_compartments(source, target)` labels each node:
# `main_path`, `branch`, `lateral`, or `sink` (see docstring for
# `path_based` vs `distance_based`).

# %%
classes = classify_compartments(swc_path, neck, sink, mode="path_based")
print("classify_compartments →", type(classes).__name__, "n nodes:", len(classes))
counts: dict[str, int] = {}
for label in classes.values():
    counts[label] = counts.get(label, 0) + 1
print("path_based counts:", dict(sorted(counts.items())))

# %% [markdown]
# ## Probe maps
#
# `map_probes_to_nodes` : probe label → nearest SWC node.
# `map_xyz_to_nearest_probes` : arbitrary XYZ → nearest probe label
# (used by the Dash app to color synapses).
# `geodesic_distances_from_probe` : distances from one named probe.

# %%
record_points = {"sink": sink, "neck": neck}
probe_nodes = map_probes_to_nodes(swc_path, record_points)
print("map_probes_to_nodes →", type(probe_nodes).__name__, probe_nodes)

named_syn = {f"syn_{i}": xyz for i, xyz in enumerate(synpts[:5])}
syn_to_probe = map_xyz_to_nearest_probes(record_points, named_syn)
print("first 5 synpts → nearest probe:", syn_to_probe)

from_sink = geodesic_distances_from_probe(
    swc_path, record_points, "sink", ["neck"]
)
print("geodesic sink → neck:", from_sink)
