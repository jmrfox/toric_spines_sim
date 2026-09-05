# Software

This project is a small stack of Python packages plus Arbor, not a single monolith. `toric_spines_sim` is the simulation/analysis repo collaborators clone; the others are installed by `uv sync` from GitHub.

## Environment

Use **[uv](https://docs.astral.sh/uv/)** (not conda) for Python. Each project has a `.venv` at the repo root. You do not activate it: prefix commands with `uv run`.

```bash
uv add numpy                          # PyPI
uv add "git+https://github.com/jmrfox/mascaf.git"
uv run python scripts/append_sink.py TS1
uv run pytest
```

`pyproject.toml` is the source of truth for dependencies. Optional extras:

| Group | Install | What it is for |
|-------|---------|----------------|
| (default) | `uv sync` | simulation, pymcfs, mascaf, tests (`dev` is included) |
| `neurosignature` | `uv sync --group neurosignature` | TS1 descriptor pipeline and example notebook |
| `docs` | `uv sync --group docs` | MkDocs preview (`mkdocs serve`) |

If you truly need conda-forge packages alongside PyPI, [Pixi](https://pixi.prefix.dev/latest/) can pull both; this repo does not require that.

**OS.** Arbor does not ship native Windows wheels. The supported path is Linux or macOS, or **WSL2 Ubuntu** on Windows, with the clone on the Linux filesystem (`~/...`, not `/mnt/c/...`). Cursor / VS Code can edit the same tree from Windows. macOS works the same as Linux once CMake and the NMODL catalogue toolchain are installed — see [Getting started](getting-started.md).

**Interactivity.** Jupyter notebooks (jupytext `.py` sources) are the lightweight lab notebook. The axon studies also ship [Dash](https://dash.plotly.com/) apps (terminal server + browser UI). [marimo](https://marimo.io/) is an option for new notebooks; nothing here depends on it.

## Packages developed for this work

| Package | Role |
|---------|------|
| **[toric_spines_sim](https://github.com/jmrfox/toric_spines_sim)** | This repo: morphology pipeline CLIs, `TSModel` / `TSSimulator`, axon studies, Dash / PDF |
| **[mascaf](https://github.com/jmrfox/mascaf)** | Mesh And Skeleton CAble model Fitting. Fits an SWC cable to a mesh + skeleton. This project pins the `release` branch. Docs: [mascaf.readthedocs.io](https://mascaf.readthedocs.io/en/latest/). Methods: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.10.721501v1) |
| **[pymcfs](https://github.com/jmrfox/pymcfs)** | Python mean-curvature flow skeletonization (MCFS). **This is the supported skeletonizer** (`scripts/skeletonize_meshes.py`) |
| **[swctools](https://github.com/jmrfox/swctools)** | SWC I/O and morphology helpers used by mascaf and the simulator |
| **[jscip](https://github.com/jmrfox/jscip)** | Parameter banks (sampling, derived quantities, constraints). Used heavily; you do not need to extend it to run sims |
| **[neurosignature](https://github.com/jmrfox/neurosignature)** | Optional. Descriptors for multi-channel event-in / multi-channel signal-out systems. Design: [Neurosignature](neurosignature.md) |

mascaf’s GitHub `main` branch still contains extra toric-spine research scripts; `release` is the installable package this repo uses.

## Other libraries

| Tool | Role |
|------|------|
| **[Arbor](https://docs.arbor-sim.org/)** | Cable-cell simulator. Custom NMODL catalogue is built locally (see [Getting started](getting-started.md)) |
| **[pynapple](https://pynapple.org/)** | Event and time-series types (`Ts`, `Tsd`, `TsGroup`) for inputs and results |
| **[CGAL](https://www.cgal.org/)** / **[CGALLab](https://www.cgal.org/demo/6.2/CGALlab.zip)** | Computational geometry. Original MCFS implementation used here; still useful for **interactive** mesh repair and one-off skeletonization. Not required for the automated pipeline. Details: [Skeletons](skeletons.md) |

## Biophysics notes

Temperature is Kelvin (in vitro ~280 K; barn owl in vivo ~313 K). Capacitance is µF/cm²; leak is S/cm²; axial resistivity is Ω·cm. Default banks in `make_default_parameter_bank()` annotate Sanculi-derived leak / Cm / τ_m; in-vivo leak is often taken as ~3× the in-vitro Sanculi value. Per-spine experiments override values in `simulations/ts{id}/params.py`.
