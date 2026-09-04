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
# Canonical TS1 integration notebook (sink voltage / k-matrix).
#
# Inputs: ``TS1_wsink_r10um.swc``, ``TS1_synpts.txt`` (microns).
# Uses ``TSSimulator`` via ``kmatrix.simulation_probe_dict``.
# For the scripted PDF path see ``simulations/ts1/axons.py``.

# %%
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.kmatrix import simulation_probe_dict as simulation
from toric_spines_sim.events import (
    DeterministicEventGenerator,
    StochasticEventGenerator,
    FlatRateCurve,
)
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter
from jscip import ParameterBank, ParameterSet
from toric_spines_sim.viz import (
    plot_morphology_3d,
    plot_morphology_frusta_3d,
    VizConfig,
)
from toric_spines_sim.geometry.sink import sink_endpoint_location_from_swc_file
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.paths import get_swc_path, get_pointset_path

import matplotlib.pyplot as plt

from typing import Tuple

# %% [markdown]
# # Integration study of TS1 with sink
#
# First, we set up a function to do the simulation for a given set of input rates and parameters.

# %%
swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
total_synapses = len(load_xyz_points(synpts_filepath))
model_name = "ts1"

logger.info(f"{model_name} has {total_synapses} synapses")

sink_endpoint = sink_endpoint_location_from_swc_file(swc_filepath)
logger.info(f"Sink endpoint: {sink_endpoint}")


# Simulation helper: ``simulation`` is ``toric_spines_sim.kmatrix.simulation_probe_dict``.


# %%
parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 1000.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["cm_uF_per_cm2"].value = (
    2.0  # default = 0.01 uF/cm^2, analysis from Sanculi gives O(1 uF/cm^2)
)
parameter_bank["rL_ohm_cm"].value = 150
# hh
hh_on = False
hh_scale = 0.05
parameter_bank["hh_leak_e_mV"].value = -54.3
parameter_bank["hh_scale"].value = hh_scale if hh_on else 0.0
# passive leak
leak_on = True
if leak_on:
    parameter_bank["pas_leak_g_S_per_cm2"].value = (
        0.001  # default = 0.001 ( 1 / 1000 Ohms * cm^2 )
    )
else:
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.0
# synapse parameters
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()
parameters["hh_tags"] = [5] if hh_on else []

print("Parameter set:\n", parameter_bank)

# %%
input_rates_hz = [0.0] * total_synapses
input_rates_hz[0] = input_rates_hz[1] = 50.0

results = simulation(
    swc_filepath,
    synpts_filepath,
    input_rates_hz,
    parameters,
    record_point=sink_endpoint,
    event_type="poisson",
)
plotter_ts = TimeSeriesPlotter(
    title="Membrane potential in sink", xlim=(0, 1000), figsize=(12, 4)
)
plotter_ts.add_time_series_from_arbor(results["probe"])
plotter_ts.show()
plotter_rp = RasterPlotter(title="Input event streams", xlim=(0, 1000), figsize=(12, 4))
plotter_rp.add_streams(results["events"])
plotter_rp.show()

# %%
config = VizConfig(width=900, height=700)

fig = plot_morphology_frusta_3d(
    results["segment_tree"],
    backend="plotly",  # force Plotly
    overlays={"syn": [syn.location for syn in results["synapses"].values()]},
    n_sides=20,
    alpha=0.8,
    config=config,
)
fig.show()


# %% [markdown]
# ### Integration nonlinearity measure $k$
#
# From [Li et al. 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004014), the integration nonlinearity measure $k$ for two inputs is defined as this ratio.
#
# $$ k(t) = \frac{V(t|1+2) - V(t|1) - V(t|2)}{V(t|1) V(t|2)}$$
#
# where $V(t|1)$, $V(t|2)$, and $V(t|1+2)$ are the baseline-subtracted "somatic" voltages at time $t$ for the three possible input combinations: input 1 alone, input 2 alone, and both inputs together. In our case, somatic voltage is replaced by sink voltage.
#
# However, computing this directly as a time-series is not generally possible, since if either input is 0, the denominator will be 0, and if both inputs are 0, the ratio will be undefined.
# Instead, we can write the equation like this,
#
# $$  k \times V(t|1) V(t|2) = V(t|1+2) - V(t|1) - V(t|2)$$
#
# and solve for $k$ as a constant slope by linear regression, concatenating data over many trials.
#
# In the present model, $V(t|1+2)$ is generally far less than $V(t|1)+V(t|2)$, so $k$ will be negative.


# %%
def compute_pairwise_voltages(
    total_synapses: int,
    active_synapse_pair: Tuple[int, int],
    rate_pair: Tuple[float, float],
    swc_filepath: str,
    synpts_filepath: str,
    parameter_bank: ParameterBank,
    record_point: Tuple[float, float, float],
    seeds: list[int],
):
    """For a given pair of synapses (1,2), compute the baseline-subtracted voltages V(1), V(2), and V(1+2).
    Do this for a given range of seeds: each seed produces a different input stream."""
    logger = logging.getLogger(__name__)
    total_time = parameter_bank["T_ms"].value
    # run simulation for short time and read out baseline voltage
    input_rates_hz_0 = [0.0] * total_synapses
    parameter_bank["T_ms"].value = 100.0
    results_0 = simulation(
        swc_filepath=swc_filepath,
        synpts_filepath=synpts_filepath,
        input_rates_hz=input_rates_hz_0,
        parameters=parameters,
        record_point=record_point,
    )
    baseline = results_0["probe"][0][0][-1, 1]
    logger.debug("Baseline voltage: %f", baseline)

    # loop over samples and fill m lists
    v_1 = []
    v_2 = []
    v_12 = []
    parameter_bank["T_ms"].value = total_time
    logger.debug("Total time reset in parameter_bank: %f", parameter_bank["T_ms"].value)
    for seed in seeds:
        seed = int(seed)
        logger.debug(f"Seed: {seed}")
        parameter_bank["seed"].value = seed
        parameters = parameter_bank.sample()
        # run 1
        input_rates_hz_1 = [0.0] * total_synapses
        input_rates_hz_1[active_synapse_pair[0]] = rate_pair[0]
        results_1 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_1,
            parameters=parameters,
            record_point=record_point,
        )
        v_1.append(results_1["probe"][0][0][:, 1] - baseline)
        # run 2
        input_rates_hz_2 = [0.0] * total_synapses
        input_rates_hz_2[active_synapse_pair[1]] = rate_pair[1]
        results_2 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_2,
            parameters=parameters,
            record_point=record_point,
        )
        v_2.append(results_2["probe"][0][0][:, 1] - baseline)
        # run 1+2
        input_rates_hz_12 = [0.0] * total_synapses
        input_rates_hz_12[active_synapse_pair[0]] = rate_pair[0]
        input_rates_hz_12[active_synapse_pair[1]] = rate_pair[1]
        results_12 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_12,
            parameters=parameters,
            record_point=record_point,
        )
        v_12.append(results_12["probe"][0][0][:, 1] - baseline)
    time = results_12["probe"][0][0][:, 0]
    v_1 = np.array(v_1)
    v_2 = np.array(v_2)
    v_12 = np.array(v_12)
    # compute distance between synapse locations
    synapse_locations = np.loadtxt(synpts_filepath)
    synapse_1_location = synapse_locations[active_synapse_pair[0]]
    synapse_2_location = synapse_locations[active_synapse_pair[1]]
    distance = np.linalg.norm(synapse_1_location - synapse_2_location)
    # done
    results = {
        "time": time,
        "v_1": v_1,
        "v_2": v_2,
        "v_12": v_12,
        "intersynapse_distance": distance,
    }
    return results


active_synapse_pair = [0, 1]
rate_pair = [50, 50]
seeds = np.arange(3)
results = compute_pairwise_voltages(
    total_synapses=total_synapses,
    active_synapse_pair=active_synapse_pair,
    rate_pair=rate_pair,
    swc_filepath=swc_filepath,
    synpts_filepath=synpts_filepath,
    parameters=parameters,
    record_point=sink_endpoint,
    seeds=seeds,
)
v_1, v_2, v_12, time = results["v_1"], results["v_2"], results["v_12"], results["time"]

# Make 3 vertically stacked subplots sharing x
plotter = TimeSeriesPlotter(
    # title=f"Pairwise Voltages: {active_synapse_pair} @ {rate_pair} Hz\nintersynapse distance: {results['intersynapse_distance']:.2f} um",
    xlabel="t (s)",
    ylabels=["V(1) (mV)", "V(2) (mV)", "V(1)+V(2) (mV)", "V(1+2) (mV)"],
    nrows=4,
    sharex=True,
    figsize=(8, 8),
    dpi=150,
    ylim=[-72, -40],
)

# Add data to each row
plotter.add_time_series(
    time,
    v_1[0] - 70,
    label="V(1)",
    color="tab:blue",
    row=0,
    summary="max",
    summary_color="tab:green",
)
plotter.add_time_series(
    time,
    v_2[0] - 70,
    label="V(2)",
    color="tab:blue",
    row=1,
    summary="max",
    summary_color="tab:green",
)
plotter.add_time_series(
    time,
    v_1[0] + v_2[0] - 70,
    label="V(1)+V(2)",
    color="tab:blue",
    row=2,
    summary="max",
    summary_color="tab:green",
)
plotter.add_time_series(
    time,
    v_12[0] - 70,
    label="V(1+2)",
    color="tab:blue",
    row=3,
    summary="max",
    summary_color="tab:green",
)


# %%
def solve_k_linreg(v_1, v_2, v_12):
    """Solve for k using linear regression, combining trials.
    a = v_1 * v_2
    b = v_12 - v_1 - v_2
    solve ak=b for k using linear fit."""
    a = v_1 * v_2
    b = v_12 - v_1 - v_2
    a = a.flatten()
    b = b.flatten()
    coeffs, cov_matrix = np.polyfit(a, b, 1, cov="unscaled")
    results = {
        "coeffs": coeffs,
        "uncertainty": np.sqrt(np.diag(cov_matrix)),
        "a": a,
        "b": b,
    }
    return results


results = solve_k_linreg(v_1, v_2, v_12)
k = results["coeffs"][0]
intercept = results["coeffs"][1]

# scatter plot of a and b with line of slope k
plt.figure(figsize=None, dpi=100)
plt.scatter(results["a"], results["b"], marker="o", s=1, alpha=0.2)
plt.xlabel("$V(1)V(2)$")
plt.ylabel("$V(1+2) - V(1) - V(2)$")
plt.plot(results["a"], k * results["a"] + intercept, c="k")
plt.title(
    f"Linear fit to determine $k$\n$k = {k:.5f} \pm {results['uncertainty'][0]:.5f}$"
)
plt.savefig("../out/linreg_k.png")
plt.show()


# %%
def compute_k_matrix(
    total_synapses: int,
    active_synapse_pair: tuple[int, int],
    rate_list: list[float],
    swc_filepath: str,
    synpts_filepath: str,
    parameter_bank: dict[str, float],
    record_location: tuple[float, float, float],
    seeds: list[int],
):

    n_rates = len(rate_list)
    k_matrix = np.zeros((n_rates, n_rates))
    k_unc_matrix = np.zeros((n_rates, n_rates))
    sim_results = {}
    rate_pair_list = []
    for i in range(n_rates):
        for j in range(i + 1):
            rate_i = rate_list[i]
            rate_j = rate_list[j]
            rate_pair = (rate_i, rate_j)
            rate_pair_list.append(rate_pair)
            results = compute_pairwise_voltages(
                total_synapses,
                active_synapse_pair,
                rate_pair,
                swc_filepath,
                synpts_filepath,
                parameter_bank,
                record_location,
                seeds,
            )
            sim_results[rate_pair] = {
                "v_1": results["v_1"],
                "v_2": results["v_2"],
                "v_12": results["v_12"],
                "time": results["time"],
            }
            linreg_results = solve_k_linreg(
                results["v_1"], results["v_2"], results["v_12"]
            )
            k = linreg_results["coeffs"][0]
            intercept = linreg_results["coeffs"][1]
            k_matrix[i, j] = k
            k_matrix[j, i] = k
            k_unc_matrix[i, j] = linreg_results["uncertainty"][0]
            k_unc_matrix[j, i] = linreg_results["uncertainty"][0]

    results = {
        "k_matrix": k_matrix,
        "sim_results": sim_results,
        "rate_pair_list": rate_pair_list,
        "intersynapse_distance": results["intersynapse_distance"],
    }
    return results


def plot_k_matrix(K, rate_list, active_synapse_pair, savefile=None, dpi=100):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=dpi)
    tick_labels = [str(i) for i in rate_list]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(f"Synapse {active_synapse_pair[0]} rate (Hz)")
    ax.set_ylabel(f"Synapse {active_synapse_pair[1]} rate (Hz)")
    im = ax.imshow(K, cmap="viridis", aspect="auto", origin="lower")
    ax.set_title("K (1/mV)\nactive synapses = " + str(active_synapse_pair))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    if savefile:
        fig.savefig(savefile)
    return fig


make_single_k_plot = True
savefile = f"../out/{model_name}_k_matrix_example.png"
if make_single_k_plot:
    active_synapse_pair = (0, 1)
    rate_list = [25, 50, 75, 100, 125]
    results = compute_k_matrix(
        total_synapses=total_synapses,
        active_synapse_pair=active_synapse_pair,
        rate_list=rate_list,
        swc_filepath=swc_filepath,
        synpts_filepath=synpts_filepath,
        parameters=parameters,
        record_location=sink_endpoint,
        seeds=seeds,
    )
    plot_k_matrix(
        results["k_matrix"], rate_list, active_synapse_pair, savefile=savefile, dpi=100
    )
    plt.show()

# %% [markdown]
# ## Loop over K matrix calculations and plot histogram grid
#
# Collect k matrix data from many pairs of synapses and plot distributions in a grid.

# %%
import random

all_synapse_pairs = []
for i in range(total_synapses):
    for j in range(i + 1, total_synapses):
        all_synapse_pairs.append((i, j))

# choose random subset of all pairs to calculate
n_pairs = len(all_synapse_pairs)
synapse_pairs = all_synapse_pairs[:n_pairs]
rate_list = np.arange(10, 130, 10)

compute_all_k_matrices = True  # WARNING: This may take a long time!

k_matrices_by_synapse_pair = {}
distance_by_synapse_pair = {}
if compute_all_k_matrices:
    for active_synapse_pair in synapse_pairs:
        results = compute_k_matrix(
            total_synapses=total_synapses,
            active_synapse_pair=active_synapse_pair,
            rate_list=rate_list,
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            parameters=parameters,
            record_location=sink_endpoint,
            seeds=seeds,
        )
        k = results["k_matrix"]
        savefile = f"../out/{model_name}_kmatrix_s{active_synapse_pair[0]}_s{active_synapse_pair[1]}.png"
        # plot_k_matrix(k, rate_list, active_synapse_pair, savefile)
        k_matrices_by_synapse_pair[active_synapse_pair] = k
        # k_matrices_by_synapse_pair[active_synapse_pair[::-1]] = k
        # distance_by_synapse_pair[active_synapse_pair] = results["intersynapse_distance"]
        # distance_by_synapse_pair[active_synapse_pair[::-1]] = results["intersynapse_distance"]

# %%
# save k_matrices_by_synapse_pair to file
import pickle as pkl

with open(f"../out/{model_name}k_matrices_by_synapse_pair.pkl", "wb") as f:
    pkl.dump(k_matrices_by_synapse_pair, f)

# %%
# order keys in k_matrices_by_synapse_pair for lower triangular grid
if (
    list(k_matrices_by_synapse_pair.keys())[0][0]
    < list(k_matrices_by_synapse_pair.keys())[0][1]
):
    k_matrices_by_synapse_pair = {
        ij[::-1]: k_matrices_by_synapse_pair[ij] for ij in k_matrices_by_synapse_pair
    }

# create bins from range of k_matrices_by_synapse pair values
all_k_values = np.concatenate(
    [np.asarray(M).ravel() for M in k_matrices_by_synapse_pair.values()]
)
bins = np.linspace(all_k_values.min(), all_k_values.max(), 16)
print(k_matrices_by_synapse_pair.keys())
hgp = HistogramGridPlotter(
    mats_by_ij=k_matrices_by_synapse_pair,
    bins=bins,
    dpi=150,
    subplot_xlabel="k (1/mV)",
    subplot_ylabel=None,
    major_pad=0.05,
    outer_tick_labelsize=20,
    outer_xlabel="Synapse index (j)",
    outer_ylabel="Synapse index (i)",
)
hgp.show()


# %% [markdown]
# # Make report PDF

# %%
from swctools import SWCModel, PointSet, FrustaSet, plot_model


all_synapse_pairs = []
for i in range(total_synapses):
    for j in range(i + 1, total_synapses):
        all_synapse_pairs.append((i, j))

# choose random subset of all pairs to calculate
n_pairs = 1
synapse_pairs = all_synapse_pairs[:n_pairs]
rate_list = [25, 50]
seeds = np.arange(1)
plot_seed_idx = 0
n_pairwise_signal_plots = 0

make_report = False  # WARNING: This may take a long time!
include_3d_plot_in_report = False

if make_report:
    logger.info("Building report...")
    report = PdfReport(f"../out/{model_name}_integration_report.pdf")
    report.add_title(f"{model_name.capitalize()} with sink integration")
    report.add_paragraph(
        f"""This report aggregates multiple simulation runs of {model_name}with their parameters, metrics, and figures.
        {model_name.capitalize()} has {total_synapses} synapses.
        All simulations are run with the same parameters, shown in the table."""
    )
    # parameter table
    params = {
        key: float(value) for key, value in parameter_bank.get_default_values().items()
    }
    report.add_dict_table(params, column_names=("Parameter", "Value"), columns=4)
    report.add_page_break()
    # 3d plot
    if include_3d_plot_in_report:
        logger.info("Adding 3D plot...")
        swc_model = SWCModel.from_swc_file(swc_filepath)
        ps = PointSet.from_txt_file(synpts_filepath)
        frusta = FrustaSet.from_swc_model(swc_model, sides=10, end_caps=False)
        ps_projected = ps.project_onto_frusta(frusta)
        fig = plot_model(
            swc_model=swc_model,
            frusta=frusta,
            show_frusta=True,
            show_centroid=True,
            slider=False,
            point_set=ps_projected,
            point_size=0.1,
            point_color="crimson",
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )
        report.add_figure(fig)
        logger.info("3D plot added.")
        report.add_page_break()
    # pairwise voltage and k matrix plots
    for active_synapse_pair in synapse_pairs:
        logger.info(
            f"Adding pairwise voltage and k matrix plots for {active_synapse_pair}..."
        )
        figures = []
        results = compute_k_matrix(
            total_synapses=total_synapses,
            active_synapse_pair=active_synapse_pair,
            rate_list=rate_list,
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            parameters=parameters,
            record_location=sink_endpoint,
            seeds=seeds,
        )
        current_rate_pairs = results["rate_pair_list"]
        for rate_pair in current_rate_pairs[:n_pairwise_signal_plots]:
            time = results["sim_results"][rate_pair]["time"]
            v_1 = results["sim_results"][rate_pair]["v_1"][plot_seed_idx]
            v_2 = results["sim_results"][rate_pair]["v_2"][plot_seed_idx]
            v_12 = results["sim_results"][rate_pair]["v_12"][plot_seed_idx]
            plotter = TimeSeriesPlotter(
                title=f"Pairwise Voltages: {active_synapse_pair} @ {rate_pair} Hz\nintersynapse distance: {results['intersynapse_distance']:.2f} um",
                xlabel="t (s)",
                ylabels=["v(1)", "v(2)", "v(1+2)"],
                nrows=3,
                sharex=True,
                figsize=(8, 6),
            )
            # Add data to each row
            plotter.add_time_series(time, v_1, label="v(1)", color="tab:blue", row=0)
            plotter.add_time_series(time, v_2, label="v(2)", color="tab:blue", row=1)
            plotter.add_time_series(time, v_12, label="v(1+2)", color="tab:blue", row=2)
            figures.append(plotter.figure)
        # k matrix plot
        k_fig = plot_k_matrix(results["k_matrix"], rate_list, active_synapse_pair)
        figures.append(k_fig)
        report.add_simulation(
            f"Pairwise Voltages: {active_synapse_pair}",
            parameters={"Active Synapse Pair": active_synapse_pair},
            figures=figures,
            page_break_after=True,
        )
    logger.info("Report built.")
    report.build()
