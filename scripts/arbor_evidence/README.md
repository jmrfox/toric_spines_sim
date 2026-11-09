# Arbor evidence (branch `arbor-evidence`)

This branch keeps the evidence that Arbor cannot close MASCAF's cable loops,
next to the Jaxley / reference-solver package it was compared against. `main`
no longer contains Arbor; the pre-migration Arbor code is on `arbor-legacy`.

Arbor needs a tree. Its gap junctions compute the current from the peer
voltage at the start of each step, which acts like a fake capacitance of
`g·dt` on each side of the junction, so no weight reproduces a cable joint.

## Setup

```bash
git checkout arbor-evidence
uv sync --group arbor-evidence
uv run --group arbor-evidence bash scripts/arbor_evidence/make_custom_catalogue.sh
```

## Scripts

| Script | Shows |
|---|---|
| `gj_cable_closure_test.py` | A cable cut and rejoined by a junction: no weight works |
| `gj_fake_capacitance.py` | Junction = closed loop + `2·g·dt` capacitance, at two dt |
| `reference_vs_arbor.py` | Per spine: tree vs closed loops vs Arbor junctions |
| `fig12_regimes.py` | MASCAF Fig 12 under tree / misplaced / correct regimes |
| `ts1_gj_sweep.py` | TS1 response vs junction weight |
| `check_gj_placement.py` | The original (pre-fix) junction placement bug |
| `ring_animation/` | Data and page for the loop-closure animation |
| `make_golden.py` | Regenerates the frozen baselines in `tests/data/golden/` |

`arbor_model/` is a frozen copy of the Arbor model (TSModel, TSRecipe,
gap junctions, simulator, NMODL catalogue) as of November 2026.
