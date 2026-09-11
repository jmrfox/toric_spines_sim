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
# # 13 — Mechanisms (NMODL catalogue)
#
# `TSModel` and `TSSimulator` load the compiled NMODL catalogue
# (`custom-catalogue.so`). To build it:
#
# ```bash
# uv run bash scripts/make_custom_catalogue.sh
# ```
#
# Package synapse keys (`ampa`) map to NMODL names (`ampasyn`) via
# `MODEL_REGISTRY`. This notebook does not teach NMODL. Sources live in
# `mechanisms/my_catalogue/`; `mechanisms/other/` is unused reference.

# %%
from toric_spines_sim.model import CUSTOM_CATALOGUE_PATH, check_catalogue
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

# %% [markdown]
# ## Load the catalogue
#
# Same call `TSModel` makes. After changing a `.mod` file, upgrading Arbor, or
# switching OS/compiler: delete the `.so` and rebuild. Do not copy a catalogue
# between machines.

# %%
catalogue = check_catalogue()
print("loaded catalogue:", type(catalogue).__name__)
try:
    names = list(catalogue)
except TypeError:
    names = [p.stem for p in catalogue_src.glob("*.mod")]
print("mechanisms:", names)

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
# applies it on `hh_tags` when `hh_scale != 0` (`parameters` and `tsmodel_and_tsrecipe`).

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
# `tsmodel_and_tsrecipe` through `kmatrix` need this `.so`. Density vs
# point-process placement happens in `TSModel.build_cell()`, not here.
