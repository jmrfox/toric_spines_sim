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
# # 17 — PDF reports
#
# To assemble a letter-size PDF of titles, tables, and figures, use
# `PdfReport` (a thin wrapper around ReportLab): `add_*` methods, then
# `build()`. Production axon PDFs live in `simulations/ts{id}/`
# (`python -m simulations.ts1.axons`). This notebook writes a short example
# under `_artifacts/` and does not run Arbor.

# %%
from toric_spines_sim.events import DeterministicEventGenerator, FlatRateCurve
from toric_spines_sim.paths import NOTEBOOKS_DIR
from toric_spines_sim.report import PdfReport
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.viz import RasterPlotter

tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

T_MS = 80.0
events = DeterministicEventGenerator(
    rate_curves=[FlatRateCurve(40.0), FlatRateCurve(80.0)],
    n_synapses_per_axon=[1, 1],
    T_ms=T_MS,
    labels=["syn_0", "syn_1"],
).generate()

raster = RasterPlotter(title="Demo raster", xlim=(0.0, T_MS), figsize=(8, 3))
raster.add_streams(events)
fig_raster = raster.figure

params = make_default_parameter_bank().sample()
sampled_parameters = {
    "T_ms": params["T_ms"],
    "cm_uF_per_cm2": params["cm_uF_per_cm2"],
    "ampa_gmax_uS": params["ampa_gmax_uS"],
    "ampa_tau_ms": params["ampa_tau_ms"],
}

# %%
pdf_path = tutorial_output_dir / "tutorial_report.pdf"
report = PdfReport(str(pdf_path))
print("PdfReport:", type(report).__name__)
print("  output:", pdf_path)
print("  content width (in):", report.content_width_in)
print("  content max height (in):", report.content_max_height_in)
report.add_title("Tutorial PDF example")
report.add_heading("What this class does", level=1)
report.add_paragraph(
    "PdfReport accumulates a letter-size document. Call add_title, "
    "add_heading, add_table, add_figure, or add_simulation, then build()."
)
report.add_heading("Default bank (subset)", level=2)
report.add_dict_table(sampled_parameters)
report.add_simulation(
    "Two-channel periodic input",
    parameters={"T_ms": T_MS, "rates_hz": "40, 80"},
    metrics={"n_streams": len(events)},
    figures=[fig_raster],
)
report.build()
print("wrote", pdf_path)
