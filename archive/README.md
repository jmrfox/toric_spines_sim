# Archive — do not use

Historical scripts, notebooks, morphologies, and figures kept for reference only.

Nothing in this directory is a supported experiment path. Use the package APIs
(`TSSimulator`, `scripts/append_sink.py`) and current files under `data/swc/`
(`TS*_wsink_r{5,10}um.swc`) instead.

| Location | What it was |
|----------|-------------|
| `simulations/ts1/original/` | Early TS1 scripts, superseded by `simulations/ts1/axons.py` |
| `simulations/ts2/ts2_sim.py` | Standalone Arbor CLI that duplicated package SWC/GJ helpers |
| `simulations/ts3/`, `ts4/` | Empty placeholders |
| `notebooks/ts_morphology/ts*_prepare_with_sink*` | Sink append notebooks; use `scripts/append_sink.py` |
| `notebooks/ts_morphology/ts*_view_model*` | Per-spine model viewers; use `view_wsink_swcs` |
| `data/swc/**/{old,hold,raw}` | Pre-`wsink` SWC names (`TS*_s50_*`, `TS*_s200_*`) |
| `outputs/outdated/` | Old k-matrix / integration figures |
