"""Neurosignature analysis pipeline for TS1 morphology.

Inputs
    data/swc/microns/TS1_wsink_r10um.swc
    data/pointsets/microns/TS1_synpts.txt
    data/pointsets/microns/TS1_neckpoint.txt

Run from the repository root::

    uv sync --group neurosignature
    uv run python -m simulations.ts1.neurosignature.ts1_neurosignature

Success: descriptor matrices and figures under ``simulations/ts1/results/``.
Requires the ``neurosignature`` extra: ``uv sync --group neurosignature``.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import neurosignature as ns

from toric_spines_sim.paths import get_swc_path, get_pointset_path, get_simulation_path
from toric_spines_sim.simulation import TSSimulator, make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
n_synapses = len(load_xyz_points(synpts_filepath))

parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 2000  # 2 seconds for neurosignature analysis
parameter_bank["delay_ms"].value = 20
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.02
parameter_bank["dt_record_ms"].value = 1.0
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["neck_radius_scale"].value = 1.0
# Arbor's "default" membrane capacitance = 0.01 uF/cm^2
# but analysis from Sanculi gives O(1 uF/cm^2)
parameter_bank["cm_uF_per_cm2"].value = 2.0
parameter_bank["rL_ohm_cm"].value = 150
# hh_tags empty + hh_scale 0 keeps the sink passive
hh_on = False
parameter_bank["hh_leak_e_mV"].value = -54.3
parameter_bank["hh_scale"].value = 1.0 if hh_on else 0.0
# passive leak
# Arbor's "default" leak conductance = 0.001 S/cm^2
leak_on = True
if leak_on:
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.001
else:
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.0
# synapse parameters
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()
parameters["hh_tags"] = [5] if hh_on else []

# Load neck point coordinates for reference channel
neckpoint_filepath = get_pointset_path("TS1_neckpoint.txt", units="microns")
neck_point = np.loadtxt(neckpoint_filepath)
logger.info(f"Neck point coordinates: {neck_point}")

def find_nearest_probe(record_points: dict, target_point: np.ndarray) -> str:
    """Find the probe label nearest to a target point."""
    min_dist = float("inf")
    nearest_label = None
    for label, coords in record_points.items():
        dist = np.linalg.norm(np.array(coords) - target_point)
        if dist < min_dist:
            min_dist = dist
            nearest_label = label
    return nearest_label


def run_simulation(input_events):
    """Run simulation and return full SimulationResults object."""
    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        input_events,
        parameters,
        record_points="all",
    )
    return sim.run()


def fn(input_events):
    """Simulation function for neurosignature pipeline - returns TsdFrame."""
    results = run_simulation(input_events)
    return results.voltage_traces


# Initial run to find reference channel
logger.info("Running initial trial to find reference channel...")
input_gen = ns.InputGenerator(
    n_channels=n_synapses,  # Use actual number of synapses as input channels
    rate_hz=50.0,
    dt_ms=1.0,
    seed=int(parameters["seed"]),
)
initial_events = input_gen.generate(duration_ms=2000.0)
initial_results = run_simulation(initial_events)

# Find nearest probe to neck point
nearest_probe_label = find_nearest_probe(initial_results.record_points, neck_point)
reference_channel = initial_results.voltage_traces.columns.get_loc(nearest_probe_label)
logger.info(f"Reference probe: {nearest_probe_label} (index {reference_channel})")

# Set up VectorDescriptor with all 12 components from notebook
descriptor_components = [
    ns.ReferenceStd(),
    ns.ReferenceSpectralCentroid(dt_ms=1.0),
    ns.ResidualMean(),
    ns.ResidualStd(),
    ns.ResidualEnergy(dt_ms=1.0),
    ns.ResidualSpectralCentroidMean(dt_ms=1.0),
    ns.ResidualSpectralCentroidStd(dt_ms=1.0),
    ns.ResidualParticipationRatio(),
    ns.ResidualMaxEigenvalue(),
    ns.ResidualEigenvalueEntropy(),
    ns.CrossCorrelationMean(),
    ns.TransmissionEfficiency(),
]

vd = ns.VectorDescriptor(descriptor_components, reference_channel=reference_channel)
component_labels = vd.labels()
logger.info(f"Descriptor dimension: {len(component_labels)}")

# Set up simulator and experiment
simulator = ns.Simulator(fn=fn)
experiment = ns.Experiment(
    input_gen=input_gen,
    simulator=simulator,
    descriptor=vd,
)

# Run experiment with 50 trials at 2000ms duration
logger.info("Running experiment: 50 trials, 2000ms duration...")
n_trials = 50
duration_ms = 2000.0
descriptor_matrix = experiment.run(n_trials=n_trials, duration_ms=duration_ms)
normed_matrix, desc_mean, desc_std = experiment.normalize(descriptor_matrix)

logger.info(f"Descriptor matrix shape: {descriptor_matrix.shape}")

# Save results
results_dir = get_simulation_path("ts1", "results")
results_dir.mkdir(parents=True, exist_ok=True)
np.save(results_dir / "neurosignature_descriptor_matrix.npy", descriptor_matrix)
np.save(results_dir / "neurosignature_descriptor_mean.npy", desc_mean)
np.save(results_dir / "neurosignature_descriptor_std.npy", desc_std)
logger.info(f"Saved descriptor data to {results_dir}")

# Generate plots
fig_dir = results_dir

# Plot 1: Single trial example (input raster + output traces)
logger.info("Generating single trial plot...")
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Input raster
ax = axes[0]
time_ms = np.asarray(initial_results.voltage_traces.index) * 1000.0
for key in sorted(initial_results.input_events.keys())[:10]:  # Show first 10 inputs
    spike_times_ms = initial_results.input_events[key].index.values * 1000.0
    ax.vlines(spike_times_ms, key - 0.4, key + 0.4, linewidth=1.2, alpha=0.7)
ax.set_ylabel("Input channel")
ax.set_title("Poisson Input Spike Trains (first 10 channels)")
ax.set_xlim(0, duration_ms)

# Output traces (first 5 probes)
ax = axes[1]
outputs_arr = np.asarray(initial_results.voltage_traces)
for i in range(min(5, outputs_arr.shape[1])):
    ax.plot(time_ms, outputs_arr[:, i], label=f"Probe {i}", alpha=0.85)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.set_title("Voltage Traces (first 5 probes)")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, duration_ms)

plt.tight_layout()
fig.savefig(fig_dir / "neurosignature_single_trial.png", dpi=150)
plt.close(fig)

# Plot 2: Single trial descriptor bar chart
logger.info("Generating descriptor bar chart...")
single_desc = vd.compute(initial_results.voltage_traces)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(single_desc)), single_desc, alpha=0.8, color="steelblue")
ax.set_xticks(range(len(single_desc)))
ax.set_xticklabels(component_labels, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
ax.set_ylabel("Value")
ax.set_title(f"Descriptor Vector — Single Trial (dim={len(single_desc)})")
plt.tight_layout()
fig.savefig(fig_dir / "neurosignature_descriptor_single.png", dpi=150)
plt.close(fig)

# Plot 3: Multi-trial descriptor matrix heatmap + statistics
logger.info("Generating multi-trial descriptor plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
ax = axes[0]
im = ax.imshow(descriptor_matrix.T, aspect="auto", cmap="viridis")
ax.set_xlabel("Trial")
ax.set_ylabel("Descriptor component")
ax.set_yticks(range(len(component_labels)))
ax.set_yticklabels(component_labels, fontsize=7)
ax.set_title(
    f"Raw Descriptor Matrix ({n_trials} trials × {len(component_labels)} components)"
)
plt.colorbar(im, ax=ax)

# Mean ± Std
ax = axes[1]
x = np.arange(len(desc_mean))
ax.bar(
    x, desc_mean, yerr=desc_std, capsize=3, alpha=0.8, color="steelblue", ecolor="black"
)
ax.set_xticks(x)
ax.set_xticklabels(component_labels, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
ax.set_ylabel("Value")
ax.set_title(f"Mean ± Std across {n_trials} trials (raw)")

plt.tight_layout()
fig.savefig(fig_dir / "neurosignature_descriptor_multi.png", dpi=150)
plt.close(fig)

logger.info(f"All plots saved to {fig_dir}")
logger.info("Neurosignature analysis complete!")
