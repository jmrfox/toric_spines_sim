"""Freeze Arbor baseline traces before Arbor is removed (migration phase 0).

Arbor is correct on loop-free morphologies, and it is the only HH reference
we have (the reference solver is passive). Each case writes
``tests/data/golden/<name>.npz`` with:

- ``t_ms``: sample times (every simulation step)
- ``V``: membrane potential (mV), shape (n_times, n_probes)
- ``probe_xyz``: probe coordinates (µm), so any backend can record at the same points
- ``syn_xyz``, ``event_ms``: synapse locations and the event times per synapse
- ``meta``: JSON with the case definition, all scalar parameters, Arbor
  version and git commit

Cases:
- passive_<spine>_<condition>_<receptor>: TS48 / TS76 (no loops), ICx in-vitro
  and in-vivo banks, AMPA and NMDA
- hh_<spine>_<sub|spike>: HH on the (geometric) sink, below and above threshold
- tree_TS1_invitro_ampa: TS1 with its loops cut (gj_weight = 0); the "tree"
  arm of the loop comparisons

Uses the frozen Arbor model in ``scripts/arbor_evidence/arbor_model``. Run from
the repository root (needs the ``arbor-evidence`` group and its catalogue):
    uv sync --group arbor-evidence
    uv run --group arbor-evidence bash scripts/arbor_evidence/make_custom_catalogue.sh
    uv run --group arbor-evidence python scripts/arbor_evidence/make_golden.py          # write
    uv run --group arbor-evidence python scripts/arbor_evidence/make_golden.py --check  # compare
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

import arbor as A
from arbor import units as U

from toric_spines_sim.geometry.sink import neck_point_from_swc_file
from toric_spines_sim.geometry.swc import iter_swc_samples
from arbor_model import TSModel, TSRecipe, prepare_gap_junctions
from toric_spines_sim.model import SynapsePopulation
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_icx_parameter_bank
from toric_spines_sim.utils import load_xyz_points

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "tests" / "data" / "golden"
DT_MS = 0.025
DX_UM = 0.1


def _probes(swc: Path, syn_xyz: np.ndarray) -> np.ndarray:
    """Synapse sites, neck, sink middle and tip, and spread spine samples."""
    samples = iter_swc_samples(swc)
    spine = np.array([s[2:5] for s in samples if s[1] == 3])
    sink = np.array([s[2:5] for s in samples if s[1] in (5, 6)])
    picks = spine[np.linspace(0, len(spine) - 1, 8).round().astype(int)]
    pts = [*syn_xyz[:5], neck_point_from_swc_file(swc), sink[len(sink) // 2], sink[-1], *picks]
    return np.array(pts, dtype=float)


def _params(condition: str, overrides: dict):
    bank = make_icx_parameter_bank(condition)
    base = dict(dt_sim_ms=DT_MS, dt_record_ms=DT_MS, discretization_um=DX_UM)
    for key, value in {**base, **overrides}.items():
        if key not in ("hh_tags", "gj_weight"):
            bank[key].value = value
    p = bank.sample()
    p["hh_tags"] = list(overrides.get("hh_tags", []))
    return p


def run_case(case: dict) -> dict:
    """Simulate one case in Arbor; returns arrays + metadata."""
    swc = get_swc_path(f"{case['spine']}_wsink_r10um.swc", units="microns")
    synpts = get_pointset_path(f"{case['spine']}_synpts.txt", units="microns")
    p = _params(case["condition"], case["overrides"])
    syn_xyz = np.array(load_xyz_points(synpts), dtype=float)
    synapses = SynapsePopulation.from_file(synpts, model=case["receptor"], global_parameters=p).synapses
    labels = list(synapses)
    if isinstance(case["events"], dict):
        by_index = {int(k): v for k, v in case["events"].items()}  # JSON makes keys str
        events = {lab: list(by_index.get(i, [])) for i, lab in enumerate(labels)}
    else:  # same event list on every synapse
        events = {lab: list(case["events"]) for lab in labels}
    gjs = prepare_gap_junctions(swc, p, weight=case["overrides"].get("gj_weight"))
    probe_xyz = _probes(swc, syn_xyz)
    record = {f"p{k:02d}": tuple(xyz) for k, xyz in enumerate(probe_xyz)}

    built = TSModel(swc, synapses, gjs, record, p).build_cell()
    recipe = TSRecipe(built["cell"], synapses, gjs, record, events=events, parameters=p,
                      custom_catalogue=built["custom_catalogue"])
    sim = A.simulation(recipe)
    handles = {lab: sim.sample((0, f"v_{lab}"), A.regular_schedule(DT_MS * U.ms)) for lab in record}
    sim.run(float(p["T_ms"]) * U.ms, DT_MS * U.ms)
    traces = []
    t = None
    for lab in record:
        data = sim.samples(handles[lab])[0][0]
        t = data[:, 0]
        traces.append(data[:, 1])
    V = np.column_stack(traces)

    scalars = {}
    for key in p.index:
        value = p[key]
        if np.ndim(value) == 0:
            scalars[key] = float(value)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    event_ms = np.array([events[lab] for lab in labels], dtype=object)
    meta = {
        "case": case,
        "parameters": scalars,
        "hh_tags": list(p["hh_tags"]),
        "n_gap_junctions": len(gjs),
        "arbor_version": A.__version__,
        "git_commit": commit,
        "dt_ms": DT_MS,
        "discretization_um": DX_UM,
    }
    return {"t_ms": t, "V": V, "probe_xyz": probe_xyz, "syn_xyz": syn_xyz,
            "event_ms": event_ms, "meta": meta}


def cases() -> list[dict]:
    out = []
    first5 = {i: [5.0 + 2.0 * i] for i in range(5)}
    for spine in ("TS48", "TS76"):
        for condition in ("in_vitro", "in_vivo"):
            for receptor, T in (("ampa", 60.0), ("nmda", 200.0)):
                out.append({
                    "name": f"passive_{spine}_{condition}_{receptor}",
                    "spine": spine, "condition": condition, "receptor": receptor,
                    "events": first5, "overrides": {"T_ms": T},
                })
        # HH on the geometric sink (lumped sink would scale channel totals).
        hh = {"T_ms": 60.0, "sink_lumped": 0, "hh_scale": 1.0, "hh_tags": [5]}
        out.append({"name": f"hh_{spine}_sub", "spine": spine, "condition": "in_vitro",
                    "receptor": "ampa", "events": first5,
                    "overrides": {**hh, "ampa_gmax_uS": 0.0004}})  # peak ~ -61.8 mV, below threshold
        out.append({"name": f"hh_{spine}_spike", "spine": spine, "condition": "in_vitro",
                    "receptor": "ampa", "events": [5.0, 25.0],
                    "overrides": {**hh, "ampa_gmax_uS": "SPIKE"}})
    out.append({"name": "tree_TS1_invitro_ampa", "spine": "TS1", "condition": "in_vitro",
                "receptor": "ampa", "events": first5,
                "overrides": {"T_ms": 60.0, "gj_weight": 0.0}})
    return out


# From --tune: smallest tested per-synapse AMPA gmax (µS) at which every synapse
# firing once makes the geometric HH sink spike (both spines peak ~ +39 mV).
SPIKE_GMAX_US = {"TS48": 0.002, "TS76": 0.002}


def _resolve(case: dict) -> dict:
    case = json.loads(json.dumps(case))
    if case["overrides"].get("ampa_gmax_uS") == "SPIKE":
        g = SPIKE_GMAX_US[case["spine"]]
        if g is None:
            raise SystemExit(f"run --tune first: no spiking gmax for {case['spine']}")
        case["overrides"]["ampa_gmax_uS"] = g
    return case


def tune() -> None:
    """Smallest per-synapse AMPA gmax (µS) that makes the HH sink spike."""
    for spine in SPIKE_GMAX_US:
        for g in (0.002, 0.004, 0.008, 0.016, 0.032, 0.064):
            case = {"name": "tune", "spine": spine, "condition": "in_vitro", "receptor": "ampa",
                    "events": [5.0], "overrides": {"T_ms": 30.0, "sink_lumped": 0, "hh_scale": 1.0,
                                                    "hh_tags": [5], "ampa_gmax_uS": g}}
            r = run_case(case)
            peak = r["V"][:, 7].max()  # sink tip
            print(f"{spine} gmax {g}: sink peak {peak:.1f} mV")
            if peak > 0:
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="regenerate and compare to disk")
    ap.add_argument("--tune", action="store_true", help="find spiking gmax for the HH cases")
    args = ap.parse_args()
    if args.tune:
        tune()
        return
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for case in cases():
        case = _resolve(case)
        r = run_case(case)
        n_events = sum(len(e) for e in r["event_ms"])
        if n_events == 0:
            raise SystemExit(f"{case['name']}: no synaptic events were scheduled")
        path = GOLDEN_DIR / f"{case['name']}.npz"
        rest = r["meta"]["parameters"]["Vrest_mV"]
        if args.check:
            old = np.load(path, allow_pickle=True)
            diff = float(np.abs(old["V"] - r["V"]).max())
            worst = max(worst, diff)
            print(f"{case['name']:32} max |diff| vs disk {diff:.2e} mV")
        else:
            np.savez_compressed(path, t_ms=r["t_ms"], V=r["V"], probe_xyz=r["probe_xyz"],
                                syn_xyz=r["syn_xyz"], event_ms=r["event_ms"],
                                meta=json.dumps(r["meta"]))
            print(f"{case['name']:32} peak {(r['V'] - rest).max():7.2f} mV  "
                  f"min {(r['V'] - rest).min():7.2f}  -> {path.relative_to(ROOT)}")
    if args.check and worst > 1e-9:
        sys.exit(1)


if __name__ == "__main__":
    main()
