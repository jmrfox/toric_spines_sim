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
# # View spine + sink SWCs
#
# Plot every `*_wsink_*.swc` under `data/swc/microns/` (or pixels).
# Generate them with:
#
# ```bash
# uv run python scripts/append_sink.py --all
# ```

# %%
from pathlib import Path

from swctools import PointSet, SWCModel, plot_model

from toric_spines_sim.paths import SWC_MICRONS_DIR, SWC_PIXELS_DIR, get_pointset_path

# "microns" or "pixels"
UNITS: str = "microns"
# Optional: restrict to stems, e.g. ["TS1", "TS3"]. Empty = all wsink SWCs.
STEMS: list[str] = []
# Prefer this sink radius label when multiple exist for one stem (e.g. "10").
PREFERRED_RADIUS_UM: str = "5"

# %%
swc_dir = SWC_MICRONS_DIR if UNITS == "microns" else SWC_PIXELS_DIR
candidates = sorted(swc_dir.glob("TS*_wsink_*.swc"))

by_stem: dict[str, list[Path]] = {}
for path in candidates:
    stem = path.name.split("_wsink_", 1)[0]
    if STEMS and stem not in STEMS:
        continue
    by_stem.setdefault(stem, []).append(path)

swc_paths: list[Path] = []
for stem in sorted(by_stem, key=lambda s: (len(s), s)):
    paths = by_stem[stem]
    preferred = [p for p in paths if f"_r{PREFERRED_RADIUS_UM}um" in p.name]
    swc_paths.append(preferred[0] if preferred else paths[0])

print(f"Found {len(swc_paths)} spine+sink SWC(s) under {swc_dir}")
for path in swc_paths:
    print(f"  {path.name}")

if not swc_paths:
    raise FileNotFoundError(
        f"No TS*_wsink_*.swc files under {swc_dir}. Run scripts/append_sink.py --all."
    )

# %%
for swc_path in swc_paths:
    stem = swc_path.name.split("_wsink_", 1)[0]
    model = SWCModel.from_swc_file(str(swc_path), validate_reconnections=False)

    fig = plot_model(
        swc_model=model,
        slider=False,
        title=f"{swc_path.stem} ({UNITS})",
        show_axes=False,
        show_frusta=True,
        show_centroid=False,
        width=1200,
        height=900,
    )
    fig.show()
