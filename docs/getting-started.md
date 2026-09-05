# Getting started

Install this repo and run a first simulation. For the surrounding packages, OS notes, and Dash vs Jupyter: [Software](software.md). For what the project is asking scientifically: [Overview](index.md).

You need **git**, **Python 3.12+**, and **[uv](https://docs.astral.sh/uv/)**. `uv sync` clones the GitHub packages `swctools` and `jscip` (and `pymcfs` / `mascaf` for the mesh pipeline). Without `swctools` you cannot load SWCs; without `jscip` you cannot build parameter banks.

The TS1 neurosignature pipeline is optional. Install it with `uv sync --group neurosignature`.

`uv sync` installs a binary [Arbor](https://docs.arbor-sim.org/en/latest/install/python.html) wheel (`arbor>=0.11.0`). `import arbor` can succeed while simulations still fail until you build the **local NMODL catalogue**. That step compiles `toric_spines_sim/mechanisms/my_catalogue/*.mod` to C++ (`modcc` + CMake + `make`) and writes `toric_spines_sim/mechanisms/custom-catalogue.so` (gitignored). The `.so` is specific to OS, compiler, and Arbor version — do not copy one machine’s catalogue onto another. Rebuild after changing `.mod` files, upgrading Arbor, or changing OS/compiler.

## Linux

```bash
sudo apt install git cmake g++ python3-dev make
# Fedora/RHEL: sudo dnf install git cmake gcc-c++ python3-devel make
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/jmrfox/toric_spines_sim.git
cd toric_spines_sim
uv sync
uv run bash scripts/make_custom_catalogue.sh
uv run python -c "import arbor as A; A.print_config()"
```

## macOS

Xcode Command Line Tools provide `clang`, `make`, and the SDK. Homebrew provides CMake (Apple does not ship it):

```bash
xcode-select --install
brew install cmake
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then the same clone / `uv sync` / catalogue commands as Linux. `uv sync` succeeding is **not** enough: `arbor-build-catalogue` compiles generated C++ with the **system** `c++` because pip wheels often omit `arbor.config()['CXX']` ([arbor PR 2051](https://github.com/arbor-sim/arbor/pull/2051)).

**Known failure:** the venv is fine, but `arbor-build-catalogue` dies in `make` with a syntax error in **generated** C++ (not in the `.mod` files). That almost always means a broken or mismatched toolchain — missing CLT after a macOS upgrade, Homebrew GCC mixed with Apple headers, or `make` still the CLT stub. Do not edit the generated C++.

1. Reinstall CLT and select them: `sudo xcode-select -s /Library/Developer/CommandLineTools`
2. Confirm `cmake`, `make`, and `c++` exist (`which cmake make c++`). `c++ --version` should be Apple clang ≥ 15 (Arbor’s documented minimum).
3. Rebuild verbose with an explicit compiler:

```bash
cd toric_spines_sim/mechanisms
uv run arbor-build-catalogue custom my_catalogue -v --cxx "$(xcrun --find c++)"
```

4. Do not compile the catalogue with Homebrew `g++` unless Arbor itself was built with that same compiler.
5. After a macOS or Arbor upgrade, delete `custom-catalogue.so` and rebuild.

## Windows

Arbor does not ship native Windows wheels. Use **[WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)** with Ubuntu, install the Linux packages above, and clone **inside** the Linux filesystem (`~/...`, not `/mnt/c/...`). Authenticate to GitHub from the WSL shell, then follow the Linux steps. Native `scripts/make_custom_catalogue.bat` is not the supported path.

## First run

Open `notebooks/examples/params_demo` and `events_demo`, then a morphology notebook (`view_wsink_swcs`) and an integration notebook (`ts1_wsink_sim` or `cylinder_wsink_integrate`). Pair `.py` sources with notebooks via jupytext (`uv run jupytext --to notebook path/*.py` if the `.ipynb` is missing).

Canonical scripted experiment: `uv run python -m simulations.ts1.axons` (or `ts2`, `ts3`, …). See [Simulations](simulations.md). Tests: `uv run pytest`.

Mesh skeletonization / SWC fitting use pymcfs and mascaf (installed with `uv sync`; see [Morphology pipeline](morphology.md)).

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
