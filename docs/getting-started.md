# Getting started

Install this repo and run a first simulation. For the surrounding packages, OS notes, and Dash vs Jupyter: [Software](software.md). For what the project is asking scientifically: [Overview](index.md).

You need **git**, **Python 3.12+**, and **[uv](https://docs.astral.sh/uv/)**. `uv sync` installs everything, including [jaxley](https://jaxley.readthedocs.io/) (pinned to 0.14) and the GitHub packages `swctools` and `jscip` (and `pymcfs` / `mascaf` for the mesh pipeline). Without `swctools` you cannot load SWCs; without `jscip` you cannot build parameter banks. There is nothing to compile: synaptic kinetics and channels are Python.

The TS1 neurosignature pipeline is optional. Install it with `uv sync --group neurosignature`.

## Linux and macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/jmrfox/toric_spines_sim.git
cd toric_spines_sim
uv sync
uv run pytest
```

jaxley runs on the CPU by default. A CUDA build of jaxlib gives GPU support, but these models need float64 (set automatically by the jaxley backend), which is slow on most consumer GPUs; passive studies run on the scipy reference solver anyway (see [Software](software.md#solvers-and-loops)).

## Windows

Use **[WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)** with Ubuntu and clone **inside** the Linux filesystem (`~/...`, not `/mnt/c/...`); the mesh tools (pymcfs, mascaf) are only tested there. Authenticate to GitHub from the WSL shell, then follow the Linux steps.

## First run

Work through `notebooks/tutorial/` in order (`01_meshes` … `21_spiny_dendrite`). Pair `.py` sources with notebooks via jupytext (`uv run jupytext --sync notebooks/tutorial/*.py`). `13_mechanisms` and `14_tsmodel_and_solvers` explain the synapse kinetics, the cable graph with closed loops, and the two solvers. Extra demos live under `notebooks/misc/`. All-spine morphology galleries use `SHOW_ALL = True` in the morphology notebooks.

Full axon PDF study: `uv run python -m simulations.ts1.axons` (or `ts2`, `ts3`, …). See [Simulations](simulations.md). Tests: `uv run pytest`.

Mesh skeletonization / SWC fitting use pymcfs and mascaf (installed with `uv sync`; see [Morphology pipeline](morphology.md)).

## Arbor evidence (optional)

The project used Arbor until November 2026. Arbor cannot close cable loops: its gap junctions use the peer voltage from the start of each step, which acts like a fake capacitance of `g·dt` on each side. The scripts that show this, and the frozen Arbor model they use, live in `scripts/arbor_evidence/`. To run them you need a C++ toolchain for Arbor's NMODL catalogue:

```bash
sudo apt install cmake g++ make            # macOS: xcode-select --install && brew install cmake
uv sync --group arbor-evidence
uv run --group arbor-evidence bash scripts/arbor_evidence/make_custom_catalogue.sh
uv run --group arbor-evidence python scripts/arbor_evidence/gj_fake_capacitance.py
```

The catalogue `.so` is specific to OS, compiler and Arbor version; rebuild it after any of those change. On macOS, if `arbor-build-catalogue` fails in `make` on generated C++, the toolchain is mismatched: reinstall the Command Line Tools and build with `--cxx "$(xcrun --find c++)"`. With this group installed, `uv run pytest` also checks that Arbor still reproduces the frozen baselines in `tests/data/golden/`.

## Environment (uv)

Run scripts and modules from the **repository root**:

```bash
uv run python scripts/append_sink.py TS1
uv run python -m simulations.ts1.axons
uv run jupyter lab
```

```bash
uv sync --upgrade
uv sync --upgrade --refresh          # e.g. after swctools git updates
uv sync --group neurosignature       # TS1 neurosignature script / example notebook
uv add package_name
uv add --dev package_name
```

## Building these docs

Collaborators do not need this. To preview locally:

```bash
uv sync --group docs
uv run mkdocs serve
```
