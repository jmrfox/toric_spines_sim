# Handoff TODO

This repo grew organically while developing Arbor simulations of barn-owl toric spines (Sanculi et al., 2020). The goal of this list is to make the codebase understandable and usable for students and collaborators without building a separate documentation site.

Prefer improving **README sections, module/docstring clarity, logging, and example notebooks** over adding Sphinx/pdoc package docs.

---

## Priority legend

- **P0** — blockers for new users getting a first simulation running
- **P1** — clarity / correctness / cleanup that will save collaborators time
- **P2** — polish, archival, nice-to-have consistency

---

## P0 — Onboarding and “how do I run this?”

- [ ] Expand `README.md` beyond env/paths into a short collaborator guide:
  - Scientific goal in 1–2 paragraphs (keep existing TS background; tighten if needed)
  - High-level workflow: SWC (+ cycle breaks) → sink → model/recipe → events → simulate → analyze/viz
  - Recommended starting path (e.g. `notebooks/examples/` then `notebooks/ts_morphology/` / `notebooks/ts_integration/`)
  - How to run package tests: `uv run pytest`
  - How to run a canonical simulation (prefer `uv run -m …` from repo root; document the intended entry points under `simulations/`)
  - Pointers to private/git deps (`swctools`, `jscip`, `neurosignature`) and what breaks if they are missing
- [ ] Document the **data layout** (`data/swc/{pixels,microns}`, `pointsets/`, `nff/`, `events/`, `ts_axons/`): what each folder is for, units convention, and which files are “current” vs historical
- [ ] Document the **package map** briefly in the README (one line each):
  - `geometry/` — SWC I/O, cycle breaks, sink append, dendrite helpers
  - `model/` — `TSModel`, synapses, gap junctions, recipe
  - `simulation/` — parameters, `TSSimulator`, inputs, results
  - `events/` — event generators / rate curves
  - `viz/` — Plotly/Dash/animation helpers
  - `kmatrix.py`, `report.py` — pairwise integration analysis / reporting
- [ ] Clarify mechanism / catalogue build steps (`toric_spines_sim/mechanisms/`, `nmodl/`, `scripts/make_custom_catalogue.bat`): when a rebuild is needed, and provide a Linux/WSL-friendly command if the `.bat` is the only documented path
- [ ] Make `simulations/README.md` useful: list each active experiment directory, intended entry module, expected inputs/outputs, and which are superseded

---

## P1 — Code that is outdated, duplicated, or should be removed/archived

### Likely legacy / superseded

- [ ] Audit `simulations/ts2/ts2_sim.py` — early standalone Arbor script that reimplements helpers now in the package (`parse_cycle_breaks`, SWC reading, etc.). Decide: rewrite against `TSModel`/`TSSimulator`, move to `notebooks/examples/`, or archive/delete
- [ ] Audit `simulations/ts1/original/` vs current `ts1_*.py` / axons / neurosignature / hypergrid flows. Label clearly in README or remove if fully superseded
- [ ] Decide fate of empty `simulations/ts3/` and `simulations/ts4/` (placeholder for future work vs delete until needed)
- [ ] Triage `outputs/outdated/` and large PDF/pickle artifacts under `simulations/ts2/` — keep only what is needed for reproducibility; move the rest out of the working tree or document why they stay
- [ ] Triage `data/swc/**/old`, `hold`, `raw` (and any duplicate pixel/micron `*_wsink_*` files): document the precomputed morphologies already in `data/` or relocate archives
- [ ] Remove or finish deprecation of `prepare_ampa_synapses` / `prepare_nmda_synapses` in `model/synapse.py` (already marked deprecated in favor of `SynapsePopulation.from_file`); update call sites and `__init__` exports

### Duplication and drift

- [ ] Search for copied utilities still living outside the package (e.g. local `parse_cycle_breaks` / SWC parsers in simulation scripts) and route them through `toric_spines_sim.geometry`
- [ ] Fix stale references in comments/docstrings (example: `scripts/append_sink.py` points at `parse_cycle_breaks` in `toric_spines_sim/model.py`; the real home is `geometry/swc.py`)
- [ ] Align notebook scripts with package APIs after any cleanup; run `uv run jupytext --sync` when editing paired `.py`/`.ipynb` files
- [ ] Review `scripts/`: keep CLI tools that collaborators need (`append_sink.py`); relocate demos (`animation_example.py`) or smoke tests (`_smoke_test_dash.py`) and document which are supported

### Dependencies / project metadata

- [ ] Clean `pyproject.toml` metadata (`description` is still a placeholder)
- [ ] Revisit unused / questionable dependencies if not needed for day-to-day work (e.g. Sphinx/Furo/breathe/myst/pdoc if no docs site; `logging` on PyPI is not stdlib `logging` and is likely accidental)
- [ ] Confirm Windows-only or one-off tooling is either replaced or clearly marked optional

---

## P1 — Make the code self-explanatory (docstrings, comments, logging)

### Docstrings and module headers

- [ ] Ensure every public module has a short module docstring stating purpose and typical entry points
- [ ] Fill or tighten class/function docstrings on the collaborator-facing surface:
  - `TSModel`, `TSRecipe`, `TSSimulator`, `SimulationResults`
  - `SynapsePopulation` / synapse + GJ preparation
  - sink append API (`SinkGeometry`, `append_sink_to_swc*`)
  - event generators and axon-event remapping
  - k-matrix / pairwise integration helpers
- [ ] Prefer “why / units / invariants” in comments over narrating obvious code; call out non-obvious SWC conventions (`# CYCLE_BREAK reconnect i j`, sink headers, tag meanings)

### Logging

- [ ] Standardize on `logging.getLogger(__name__)` (avoid ad-hoc `basicConfig` in library modules such as `kmatrix.py`; leave configuration to scripts/CLIs)
- [ ] Replace remaining debug `print`s in library code with leveled logs where useful
- [ ] Make scripts/sim entry points set a sensible default log level (`INFO` for normal runs, `--verbose` → `DEBUG`)
- [ ] Keep logs actionable for long runs (what morphology, n synapses, T, seed, output path)

### README / examples as living docs

- [ ] Curate a small set of **canonical examples** that still run end-to-end; mark exploratory notebooks (`arbor_examples/`, old demos) as such at the top of the file
- [ ] For each canonical notebook/script, add a 5–10 line header: goal, inputs, how to run, what “success” looks like
- [ ] Ensure path usage goes through `toric_spines_sim.paths` (no brittle `../../data/...` in anything we still recommend)

---

## P1 — Simulations and analysis workflows

- [ ] Document the intended TS1 storyline: morphology prep → params/inputs → integration / axons / neurosignature / hypergrid Dash — which are publication paths vs experiments
- [ ] Document the intended TS2 storyline: `ts2_integration.py` / viz vs older `ts2_sim.py`
- [ ] Write down parameter conventions (in vivo vs in vitro temperature, passive vs HH sink, units) in one place (README subsection or `simulations/` notes) so students do not inherit silent magic numbers
- [ ] Note reproducibility expectations: seeds, event file provenance, where results should be written

---

## P2 — Tests, quality bar, and repo hygiene

- [ ] Map existing tests (`tests/test_*.py`) to package modules; note gaps for recipe/simulator, k-matrix, and viz (viz may stay lightly tested)
- [ ] Add or extend tests when touching legacy rewrites (especially sink, SWC cycle breaks, synapse prep migration)
- [ ] Decide a simple contribution checklist for collaborators: sync deps with `uv sync`, run `uv run pytest`, keep notebooks synced via jupytext, prefer package APIs over copy-paste
- [ ] `.gitignore` / large artifacts: ensure generated HTML/PDF/pkl and catalogue build products are intentional if tracked
- [ ] Optional: add a short “Archived / do not use” list at the bottom of this file as decisions are made

---

## Suggested first week for a new collaborator

1. `uv sync` and open `notebooks/examples/params_demo` + `events_demo`
2. Run a morphology+sink prep notebook for TS1 or the cylinder example
3. Run one integration notebook/simulation and produce a voltage trace or simple figure
4. Skim `toric_spines_sim/model/model.py` and `simulation/core.py` with the README package map
5. Pick one P1 cleanup item above rather than starting a new analysis fork of outdated scripts

---

## Decision log (fill in as you go)

| Item | Decision | Date |
|------|----------|------|
| `ts2_sim.py` | archive | 2026-09-04 |
| `simulations/ts1/original/` | archive | 2026-09-04 |
| `simulations/ts3`, `ts4` | archive (were empty) | 2026-09-04 |
| `outputs/outdated/` | archive | 2026-09-04 |
| Deprecated synapse helpers | dropped from public `__init__`; wrappers remain in `model/synapse.py` | 2026-09-04 |
| Doc tooling deps (Sphinx/pdoc/…) | drop | 2026-09-04 |
