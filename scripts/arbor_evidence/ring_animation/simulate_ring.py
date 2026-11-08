"""Simulate a toric-spine-like ring three ways for the loop-closure animation.

Geometry: ring (radius 2 um, cable radius 0.25 um), cut at the top
(# CYCLE_BREAK), thin neck at the bottom, lumped sink (Sanculi whole-cell C/Rin).
A single AMPA synapse sits just clockwise of the cut.

Runs: reference (loop merged exactly), Arbor with the cut joined by a gap
junction at several weights, and Arbor with the loop cut (weight 0).
Every run records every simulation step (dt = 0.025 ms).

Writes ring_data.json next to this script.
Run: uv run --group arbor-evidence python scripts/arbor_evidence/ring_animation/simulate_ring.py
"""

import json
import sys
import logging
import math
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arbor_model import ArborSimulator  # noqa: E402

from toric_spines_sim.simulation import TSSimulator, make_icx_parameter_bank  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ring_geometry import (  # noqa: E402
    CUT_DEG, DT, N, R_CABLE, R_RING, SYN_DEG, T, T_SYN, WEIGHTS, write_geometry, xy,
)


def main():
    tmp = HERE / "_geometry"
    tmp.mkdir(exist_ok=True)
    swc, syn, (bx, by) = write_geometry(tmp)
    K = 96
    ring_angles = [CUT_DEG - 360.0 * (k + 0.5) / K for k in range(K)]  # clockwise from cut
    probes = {f"r{k:03d}": (*xy(a), 0.0) for k, a in enumerate(ring_angles)}
    # the two compartments on either side of the cut
    probes["cut_cw"] = (*xy(CUT_DEG - 1.0), 0.0)
    probes["cut_ccw"] = (*xy(CUT_DEG + 1.0), 0.0)
    probes["neck"] = (bx, by - 1.0, 0.0)
    probes["sink"] = (bx, by - 6.5, 0.0)

    def params(w):
        b = make_icx_parameter_bank("in_vitro")
        for k, v in dict(T_ms=T, dt_sim_ms=DT, dt_record_ms=DT, discretization_um=0.1,
                         ampa_gmax_uS=0.004).items():
            b[k].value = v
        p = b.sample()
        p["hh_tags"] = []
        return p

    events = {"syn_0": [T_SYN]}
    runs = {}
    t = None
    for name, solver, w in [("true", "reference", 0.0), ("cut", "arbor", 0.0)] + [
        (f"gj_{w:g}", "arbor", w) for w in WEIGHTS
    ]:
        if solver == "arbor":
            r = ArborSimulator(swc, syn, events, params(w), record_points=probes, gj_weight=w).run()
        else:
            r = TSSimulator(swc, syn, events, params(w), record_points=probes, solver=solver).run()
        vt = r.voltage_traces
        t = vt.index.values * 1000.0  # pynapple stores seconds
        runs[name] = {c: vt[c].values for c in probes}
        rest = params(0)["Vrest_mV"]
        ring = np.array([runs[name][f"r{k:03d}"] for k in range(K)]) - rest
        jump = np.abs(runs[name]["cut_cw"] - runs[name]["cut_ccw"]).max()
        flips = int((np.diff(np.sign(np.diff(runs[name]["cut_ccw"]))) != 0).sum())
        print(f"{name:9} ring peak {ring.max():6.2f} mV  far side of cut peak "
              f"{(runs[name]['cut_ccw'] - rest).max():6.2f}  synapse side {(runs[name]['cut_cw'] - rest).max():6.2f}  "
              f"max jump across cut {jump:6.2f} mV  slope sign changes {flips}")

    rest = params(0)["Vrest_mV"]
    out = {
        "dt_ms": DT,
        "t_syn_ms": T_SYN,
        "rest_mV": rest,
        "n_ring": K,
        "ring_angles_deg": ring_angles,
        "cut_deg": CUT_DEG,
        "syn_deg": SYN_DEG,
        "weights_uS": WEIGHTS,
        "t_ms": [round(float(x), 4) for x in t],
        "runs": {
            name: {
                "ring": [[round(float(v) - rest, 3) for v in d[f"r{k:03d}"]] for k in range(K)],
                "cut_cw": [round(float(v) - rest, 4) for v in d["cut_cw"]],
                "cut_ccw": [round(float(v) - rest, 4) for v in d["cut_ccw"]],
                "neck": [round(float(v) - rest, 4) for v in d["neck"]],
                "sink": [round(float(v) - rest, 4) for v in d["sink"]],
            }
            for name, d in runs.items()
        },
    }
    (HERE / "ring_data.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", HERE / "ring_data.json", f"{(HERE / 'ring_data.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
