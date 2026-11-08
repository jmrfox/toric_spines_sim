"""Discretization / time-step convergence of the reference solver.

Five synapses on a spine fire together; reports the peak EPSP at the
synapse sites and at the neck for each (dx, dt), relative to the finest run.

Run: uv run python scripts/diagnostics/convergence.py [TS1] [in_vitro|in_vivo]
"""

import logging
import sys

import numpy as np

logging.disable(logging.WARNING)

from toric_spines_sim.geometry.sink import neck_point_from_swc_file
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_icx_parameter_bank
from toric_spines_sim.utils import load_xyz_points

name = sys.argv[1] if len(sys.argv) > 1 else "TS1"
condition = sys.argv[2] if len(sys.argv) > 2 else "in_vitro"
swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
synpts = get_pointset_path(f"{name}_synpts.txt", units="microns")
pts = load_xyz_points(synpts)
record = {f"syn{i}": tuple(pts[i]) for i in range(min(5, len(pts)))}
record["neck"] = tuple(neck_point_from_swc_file(swc))
events = {f"syn_{i}": ([5.0] if i < 5 else []) for i in range(len(pts))}

rows = {}
for dx in (1.0, 0.5, 0.25, 0.1, 0.05):
    for dt in (0.05, 0.025, 0.01):
        b = make_icx_parameter_bank(condition)
        b["T_ms"].value = 40.0
        b["dt_sim_ms"].value = dt
        b["dt_record_ms"].value = dt
        b["discretization_um"].value = dx
        p = b.sample()
        p["hh_tags"] = []
        tr = TSSimulator(swc, synpts, events, p, record_points=record, solver="reference").run().voltage_traces
        rows[(dx, dt)] = {c: float((tr[c].values - p["Vrest_mV"]).max()) for c in tr.columns}

ref = rows[(0.05, 0.01)]
print(f"{name} {condition}: peak EPSP (mV) and max relative error vs dx=0.05, dt=0.01")
for (dx, dt), pk in rows.items():
    err = max(abs(pk[c] / ref[c] - 1) for c in ref)
    print(f"  dx={dx:<5} dt={dt:<6} syn0={pk['syn0']:.4f} neck={pk['neck']:.4f}  max rel err={100 * err:.2f}%")
