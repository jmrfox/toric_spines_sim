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
# Canonical example: ``jscip.ParameterBank`` sampling (no Arbor run).

# %%
from jscip import ParameterBank, IndependentScalarParameter, DerivedParameter
import logging

logging.basicConfig(level=logging.INFO)

pb = ParameterBank(
    {
        "N": IndependentScalarParameter(3),
        "Z": IndependentScalarParameter(3),
        "Force": IndependentScalarParameter(5.0, is_sampled=True, range=(1.0, 10.0)),
        "A": DerivedParameter(lambda params: params["N"] + params["Z"]),
    }
)

pset = pb.sample()
print("Sampled parameter set:\n", pset)
print("Derived A =", pset["A"])
