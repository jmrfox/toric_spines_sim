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

# %%
import pandas as pd
from toric_spines_sim.paths import get_data_path

data = pd.read_csv(get_data_path("toric_spines_properties_imputed.csv"))
data

# %%
# print average values

data["RMP (mV)"].mean()

# %%
avg_area = 1895 + 500 # um^2
cm_nF_um2 = data["Capacitance (nF)"]/avg_area
# 1 nf/um^2 = 10^3 F/m^2 = 10^5 uF/cm^2
cm_uF_cm2 = cm_nF_um2 * 10**5
print(cm_uF_cm2)
print(f"Cm (uF/cm^2)= {cm_uF_cm2.median():0.3f} +/- {cm_uF_cm2.std():0.3f}")

