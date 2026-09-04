"""
TS1 Voltage Propagation Figure

Simulates a single alpha-synapse event on TS1 (passive membrane) and produces
two publication figures:
  1. Peak membrane potential vs geodesic distance from the synapse site
  2. Voltage traces over time for all compartments

Both plots are colour-coded into four compartment categories:
  A  main_path  — on the shortest path from synapse → neck
  B  branch     — off the main path but structurally related (see CLASSIFICATION_MODE)
  C  lateral    — all other spine compartments
  D  sink       — boundary-condition sink compartments

Set CLASSIFICATION_MODE below to switch between "path_based" and
"distance_based" definitions for the branch / lateral split.
"""

import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulations.ts1.params import make_parameter_bank as make_ts1_parameter_bank
from toric_spines_sim.events.generators import DeterministicEventGenerator
from toric_spines_sim.events.rate_curves import FlatRateCurve
from toric_spines_sim.geometry import (
    classify_compartments,
    compute_geodesic_distances,
    map_probes_to_nodes,
)
from toric_spines_sim.paths import get_pointset_path, get_simulation_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.geometry.sink import neck_point_from_swc_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How to split "branch" vs "lateral" compartments:
#   "path_based"     — a spine node is "branch" if its shortest path back to the
#                      synapse passes through an interior node of the main
#                      synapse→neck path (i.e. it forks off the main trunk).
#   "distance_based" — a spine node is "branch" if its geodesic distance to the
#                      neck is less than the synapse's distance to the neck
#                      (i.e. it sits closer to the neck regardless of route).
CLASSIFICATION_MODE = "distance_based"  # "path_based"
SYNAPSE_INDEX = 9  # which synapse point to activate (index into synpts file)
SHOW_SINK = False  # set False to hide sink compartments from both plots

# Paths
swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
results_dir = get_simulation_path("ts1", "original", "results")
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Biophysical parameters (matching ts1_integration_passive.py)
# ---------------------------------------------------------------------------

parameter_bank = make_ts1_parameter_bank()
parameter_bank["T_ms"].value = 100
parameter_bank["delay_ms"].value = 10
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.02
parameter_bank["dt_record_ms"].value = 0.1
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["neck_radius_scale"].value = 1.0
parameter_bank["cm_uF_per_cm2"].value = 2.0
parameter_bank["rL_ohm_cm"].value = 150
# HH off (passive only)
parameter_bank["K_gbar_S_per_cm2"].value = 0.0
parameter_bank["Na_gbar_S_per_cm2"].value = 0.0
parameter_bank["hh_leak_g_S_per_cm2"].value = 0.0
# Passive leak
parameter_bank["pas_leak_g_S_per_cm2"].value = 0.001
# Synapse
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()

# ---------------------------------------------------------------------------
# Prepare single-synapse input
# ---------------------------------------------------------------------------

all_synpts = load_xyz_points(synpts_filepath)
synapse_xyz = all_synpts[SYNAPSE_INDEX]
logger.info("Activating synapse %d at (%.4f, %.4f, %.4f)", SYNAPSE_INDEX, *synapse_xyz)

# Write a temporary synpts file containing only the chosen synapse point
single_synpts_file = tempfile.NamedTemporaryFile(
    mode="w", suffix="_single_synpt.txt", delete=False
)
single_synpts_file.write(f"{synapse_xyz[0]} {synapse_xyz[1]} {synapse_xyz[2]}\n")
single_synpts_file.flush()
single_synpts_path = Path(single_synpts_file.name)
single_synpts_file.close()

# One synapse, one periodic event (rate chosen so exactly one event fires)
events_generator = DeterministicEventGenerator(
    rate_curves=[
        FlatRateCurve(10.0)
    ],  # period=100ms; one event at t=delay_ms within the window
    n_synapses_per_axon=[1],
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
)
events_tsgroup = events_generator.generate()

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

logger.info("Starting simulation...")
sim = TSSimulator(
    swc_filepath,
    single_synpts_path,
    events_tsgroup,
    parameters,
    record_points="all",
)
results = sim.run()
logger.info(
    "Simulation complete. %d probes recorded.", len(results.voltage_traces.columns)
)

# Clean up temp file
single_synpts_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Extract peak voltages from TsdFrame
# ---------------------------------------------------------------------------

peak_voltages = {}  # probe_label → peak V (mV)
voltage_traces = {}  # probe_label → (time_array, voltage_array)

for probe_label in results.voltage_traces.columns:
    tsd = results.voltage_traces[probe_label]
    times_ms = tsd.index.values * 1000.0  # convert from seconds
    voltages = tsd.values
    peak_voltages[probe_label] = float(np.max(voltages))
    voltage_traces[probe_label] = (times_ms, voltages)

logger.info("Extracted peak voltages from %d probes.", len(peak_voltages))

# ---------------------------------------------------------------------------
# Graph analysis: geodesic distances + compartment classification
# ---------------------------------------------------------------------------

geodesic_distances = compute_geodesic_distances(swc_filepath, synapse_xyz)
neck_xyz = neck_point_from_swc_file(swc_filepath)
classifications = classify_compartments(
    swc_filepath,
    source_xyz=synapse_xyz,
    target_xyz=neck_xyz,
    mode=CLASSIFICATION_MODE,
)
probe_to_node = map_probes_to_nodes(swc_filepath, results.record_points)

# Build per-probe data arrays
probe_labels = sorted(peak_voltages.keys())
distances_arr = []
peaks_arr = []
categories_arr = []

for probe_label in probe_labels:
    node_id = probe_to_node.get(probe_label)
    if node_id is None:
        continue
    dist = geodesic_distances.get(node_id)
    if dist is None:
        continue
    distances_arr.append(dist)
    peaks_arr.append(peak_voltages[probe_label])
    categories_arr.append(classifications.get(node_id, "lateral"))

distances_arr = np.array(distances_arr)
peaks_arr = np.array(peaks_arr)
categories_arr = np.array(categories_arr)

# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

# Publication-quality rcParams
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }
)

CATEGORY_COLORS = {
    "main_path": "green",
    "branch": "blue",
    "lateral": "red",
    "sink": "brown",
}
CATEGORY_LABELS = {
    "main_path": "Main path (syn → neck)",
    "branch": "Branch (off main path)",
    "lateral": "Lateral / loop",
    "sink": "Sink",
}
CATEGORY_ORDER = ["main_path", "branch", "lateral"]
if SHOW_SINK:
    CATEGORY_ORDER.append("sink")

# ---------------------------------------------------------------------------
# Plot 1: Peak voltage vs geodesic distance
# ---------------------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(8, 5))
for cat in CATEGORY_ORDER:
    mask = categories_arr == cat
    if not np.any(mask):
        continue
    ax1.scatter(
        distances_arr[mask],
        peaks_arr[mask],
        c=CATEGORY_COLORS[cat],
        label=CATEGORY_LABELS[cat],
        s=50,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )
ax1.set_xlabel("Geodesic distance from synapse (µm)")
ax1.set_ylabel("Peak membrane potential (mV)")
ax1.set_title("Voltage attenuation along TS1 morphology")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.grid(True, alpha=0.3, linewidth=0.8)
fig1.tight_layout()

fig1_path = (
    results_dir / f"TS1_voltage_prop_{CLASSIFICATION_MODE}_{SYNAPSE_INDEX}.pdf"
)
fig1.savefig(fig1_path, dpi=300)
logger.info("Saved scatter plot to %s", fig1_path)

# ---------------------------------------------------------------------------
# Plot 2: Voltage traces over time
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 5))

# Build a mapping from probe → category for the traces plot
probe_category = {}
for probe_label in probe_labels:
    node_id = probe_to_node.get(probe_label)
    if node_id is not None:
        probe_category[probe_label] = classifications.get(node_id, "lateral")

# Plot in reverse category order so main_path traces are on top
plotted_labels = set()
for cat in CATEGORY_ORDER:
    for probe_label in probe_labels:
        if probe_category.get(probe_label) != cat:
            continue
        if probe_label not in voltage_traces:
            continue
        times, voltages = voltage_traces[probe_label]
        label = CATEGORY_LABELS[cat] if cat not in plotted_labels else None
        plotted_labels.add(cat)
        ax2.plot(
            times,
            voltages,
            color=CATEGORY_COLORS[cat],
            alpha=0.3,
            linewidth=1.0,
            label=label,
        )

ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Membrane potential (mV)")
ax2.set_title("Voltage traces — single synapse activation on TS1")
ax2.legend(loc="upper right", framealpha=0.9)
ax2.grid(True, alpha=0.3, linewidth=0.8)
fig2.tight_layout()

fig2_path = (
    results_dir / f"TS1_voltage_traces_{CLASSIFICATION_MODE}_{SYNAPSE_INDEX}.pdf"
)
fig2.savefig(fig2_path, dpi=300)
logger.info("Saved traces plot to %s", fig2_path)

# ---------------------------------------------------------------------------
# Print parameter summary for caption
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SIMULATION PARAMETERS (for figure caption)")
print("=" * 60)
print(f"  SWC file:            {swc_filepath.name}")
print(f"  Synapse index:       {SYNAPSE_INDEX}")
print(
    f"  Synapse location:    ({synapse_xyz[0]:.4f}, {synapse_xyz[1]:.4f}, {synapse_xyz[2]:.4f}) µm"
)
print(f"  Classification mode: {CLASSIFICATION_MODE}")
print(f"  V_rest:              {parameters['Vrest_mV']} mV")
print(f"  c_m:                 {parameters['cm_uF_per_cm2']} µF/cm²")
print(f"  g_pas:               {parameters['pas_leak_g_S_per_cm2']} S/cm²")
print(f"  R_a:                 {parameters['rL_ohm_cm']} Ω·cm")
print(f"  AMPA g_max:          {parameters['ampa_gmax_uS']} µS")
print(f"  AMPA τ:              {parameters['ampa_tau_ms']} ms")
print(f"  T_sim:               {parameters['T_ms']} ms")
print(f"  dt_sim:              {parameters['dt_sim_ms']} ms")
print(f"  dt_record:           {parameters['dt_record_ms']} ms")
print(f"  Discretization:      {parameters['discretization_um']} µm")
print(f"  Figures saved to:    {results_dir}")
print("=" * 60)
