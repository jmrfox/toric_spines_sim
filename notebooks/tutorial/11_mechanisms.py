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
# # 11 — Mechanisms (NMODL catalogue)
#
# `TSModel.build_cell()` loads
# `toric_spines_sim/mechanisms/custom-catalogue.so`. That file is not tracked
# by git and must be built on this machine:
#
# ```bash
# uv run bash scripts/make_custom_catalogue.sh
# ```
#
# This notebook maps package synapse keys (`ampa`) to NMODL names (`ampasyn`).
# It does not teach NMODL. Sources live in `mechanisms/my_catalogue/`;
# `mechanisms/other/` is unused reference.

# %%
from pathlib import Path

from toric_spines_sim.model.model import CUSTOM_CATALOGUE_PATH
from toric_spines_sim.model.synapse import MODEL_REGISTRY
from toric_spines_sim.paths import PROJECT_ROOT

mech_root = PROJECT_ROOT / "toric_spines_sim" / "mechanisms"
catalogue_src = mech_root / "my_catalogue"
other_src = mech_root / "other"

print("catalogue .so:", CUSTOM_CATALOGUE_PATH)
print("  exists:" , CUSTOM_CATALOGUE_PATH.is_file())
print("NMODL sources:")
for path in sorted(catalogue_src.glob("*.mod")):
    print(f"  {path.name}")
print("unused / reference:")
for path in sorted(other_src.glob("*.mod")):
    print(f"  {path.name}")

if not CUSTOM_CATALOGUE_PATH.is_file():
    raise FileNotFoundError(
        f"Missing {CUSTOM_CATALOGUE_PATH}. From the repo root run:\n"
        "  uv run bash scripts/make_custom_catalogue.sh"
    )

# %% [markdown]
# ## Registry → catalogue
#
# `MODEL_REGISTRY` is how `SynapsePopulation` chooses a mechanism and which
# `ParameterSet` keys fill NMODL RANGE variables (`gmax`, `tau`, …).

# %%
print(f"{'key':8s}  {'NMODL':12s}  global ParameterSet keys")
for key, spec in MODEL_REGISTRY.items():
    print(f"{key:8s}  {spec['mechanism']:12s}  {spec['global_keys']}")

# %% [markdown]
# Density mechanism `hhnotemp` is **not** in the synapse registry. `TSModel`
# applies it on `hh_tags` when `hh_scale != 0` (notebook 08 / 12).

# %%
hh = (catalogue_src / "hhnotemp.mod").read_text().splitlines()[:12]
print("hhnotemp.mod (head):")
for line in hh:
    print(" ", line)

ampa = (catalogue_src / "ampasyn.mod").read_text()
print("\nampasyn POINT_PROCESS / PARAMETER:")
for line in ampa.splitlines():
    if line.strip().startswith(("POINT_PROCESS", "RANGE", "tau", "e ", "gmax")):
        print(" ", line)

# %% [markdown]
# ## Load the catalogue
#
# Same call `TSModel` makes. After changing a `.mod` file, upgrading Arbor, or
# switching OS/compiler: delete the `.so` and rebuild. Do not copy a catalogue
# between machines.

# %%
import arbor as A

catalogue = A.load_catalogue(str(CUSTOM_CATALOGUE_PATH))
print("loaded catalogue:", type(catalogue).__name__)
# Mechanism names present in this catalogue:
try:
    names = list(catalogue)
except TypeError:
    names = [p.stem for p in catalogue_src.glob("*.mod")]
print("mechanisms:", names)

# %% [markdown]
# Notebooks 12–16 need this `.so`. Density vs point-process placement happens
# in `TSModel.build_cell()`, not here.
