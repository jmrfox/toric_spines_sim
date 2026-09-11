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
# # 18 — Pairwise integration (k-matrix)
#
# Pairwise integration lives in `toric_spines_sim.kmatrix`
# (`simulation`, `compute_pairwise_voltages`, `solve_k_linreg`). For two
# synapses $i$ and $j$, three simulations (i only, j only, both) give
#
# $$ V_{ij} - V_i - V_j = k \, V_i V_j $$
#
# on baseline-subtracted **sink** voltage. $k < 0$ means sublinear summation.
#
# Needs the NMODL catalogue (`mechanisms`). `T_ms` and `seeds` stay small
# here; full rate-grid sweeps belong in `simulations/ts{id}/`.

# %%
from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.kmatrix import (
    compute_k_matrix,
    compute_pairwise_voltages,
    simulation,
    solve_k_linreg,
)
from toric_spines_sim.model import check_catalogue
from toric_spines_sim.paths import (
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter, VizConfig
from toric_spines_sim.viz import plot_morphology_frusta_3d

check_catalogue()

# Full outer-product k-matrix (many simulations). Leave False for a short demo.
RUN_K_GRID = False
SEEDS = [0]


def make_tutorial_bank():
    parameter_bank = make_default_parameter_bank()
    parameter_bank["T_ms"].value = 150.0
    parameter_bank["delay_ms"].value = 20.0
    parameter_bank["discretization_um"].value = 1.0
    parameter_bank["dt_sim_ms"].value = 0.05
    parameter_bank["cm_uF_per_cm2"].value = 2.0
    parameter_bank["rL_ohm_cm"].value = 150.0
    parameter_bank["ampa_gmax_uS"].value = 0.1
    parameter_bank["hh_scale"].value = 0.0
    return parameter_bank


def load_morphology(swc_name: str, synpts_name: str):
    swc_path = get_swc_path(swc_name, units="microns")
    synpts_path = get_pointset_path(synpts_name, units="microns")
    n_syn = len(load_xyz_points(synpts_path))
    sink_xyz = sink_endpoint_location_from_swc_file(swc_path)
    return swc_path, synpts_path, n_syn, sink_xyz


# %% [markdown]
# ## TS1 — one pair, one rate
#
# Synapses 0 and 1 at 50 Hz. First a single combined run (same idea as
# `tssimulator`), then the three-way pairwise voltages and a linear fit for $k$.

# %%
swc_path, synpts_path, n_syn, sink_xyz = load_morphology(
    "TS1_wsink_r10um.swc", "TS1_synpts.txt"
)
print("SWC:", swc_path)
print("synapses:", n_syn, "sink:", sink_xyz)

parameter_bank = make_tutorial_bank()
parameters = parameter_bank.sample()
parameters["hh_tags"] = []

rates = [0.0] * n_syn
rates[0] = rates[1] = 50.0
single = simulation(
    swc_path,
    synpts_path,
    rates,
    parameters,
    record_point=sink_xyz,
    event_type="poisson",
    probe_label="sink",
)

print("simulation() →", type(single).__name__)
print("  voltage_traces columns:", list(single.voltage_traces.columns))
print("  n event streams:", len(single.input_events))

print("simulation() →", type(single).__name__)
print("  voltage_traces columns:", list(single.voltage_traces.columns))
print("  n event streams:", len(single.input_events))

plotter = TimeSeriesPlotter(
    title="TS1 sink voltage (syn 0+1 at 50 Hz)",
    xlim=(0.0, float(parameters["T_ms"])),
    figsize=(12, 4),
)
plotter.add_time_series(single.voltage_traces["sink"], label="sink")
plotter.show()

raster = RasterPlotter(
    title="Input events",
    xlim=(0.0, float(parameters["T_ms"])),
    figsize=(12, 4),
)
raster.add_streams(single.input_events, linelength=0.8)
raster.show()

config = VizConfig(width=900, height=700)
fig = plot_morphology_frusta_3d(
    single.segment_tree,
    backend="plotly",
    overlays={"syn": [syn.location for syn in single.synapses.values()]},
    n_sides=10,
    alpha=0.8,
    config=config,
)
fig.show()

# %%
pair = compute_pairwise_voltages(
    n_syn,
    (0, 1),
    (50.0, 50.0),
    str(swc_path),
    str(synpts_path),
    parameter_bank,
    sink_xyz,
    SEEDS,
    probe_label="sink",
)
print("compute_pairwise_voltages keys:", sorted(pair))
fit = solve_k_linreg(pair["v_1"], pair["v_2"], pair["v_12"])
k = float(fit["coeffs"][0])
print("solve_k_linreg keys:", sorted(fit))
print(f"intersynapse distance: {pair['intersynapse_distance']:.3f} µm")
print(f"k (slope) = {k:.4g}  ± {fit['uncertainty'][0]:.4g}")

# %%
if RUN_K_GRID:
    grid = compute_k_matrix(
        n_syn,
        (0, 1),
        [25.0, 50.0],
        str(swc_path),
        str(synpts_path),
        parameter_bank,
        sink_xyz,
        SEEDS,
        probe_label="sink",
    )
    print("k-matrix:\n", grid["k_matrix"])
else:
    print("RUN_K_GRID is False; skip the rate outer product.")

# %% [markdown]
# ## Cylinder and TS2
#
# Same helpers, different files. Skip a morphology if its SWC is missing.

# %%
others = [
    ("cylinder", "cylinder_wsink_r20um.swc", "cylinder_synpts.txt", (0, 1)),
    ("TS2", "TS2_wsink_r10um.swc", "TS2_synpts.txt", (0, 1)),
]
for name, swc_name, syn_name, active in others:
    try:
        other_swc, other_syn, other_n, other_sink = load_morphology(swc_name, syn_name)
    except FileNotFoundError as exc:
        print(f"skip {name}: {exc}")
        continue
    print(f"\n{name}: {other_n} synapses, sink {other_sink}")
    other_parameter_bank = make_tutorial_bank()
    other_pair = compute_pairwise_voltages(
        other_n,
        active,
        (50.0, 50.0),
        str(other_swc),
        str(other_syn),
        other_parameter_bank,
        other_sink,
        SEEDS,
        probe_label="sink",
    )
    other_fit = solve_k_linreg(
        other_pair["v_1"], other_pair["v_2"], other_pair["v_12"]
    )
    print(
        f"  k = {float(other_fit['coeffs'][0]):.4g}  "
        f"distance = {other_pair['intersynapse_distance']:.3f}"
    )
