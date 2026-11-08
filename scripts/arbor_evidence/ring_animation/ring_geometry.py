"""Toric-spine-like ring geometry used by the loop-closure animation and tests (no Arbor)."""

import math
from pathlib import Path

HERE = Path(__file__).parent
N = 48  # ring samples
R_RING, R_CABLE = 2.0, 0.25
CUT_DEG = 90.0  # top
SYN_DEG = 70.0  # just clockwise of the cut (angles decrease clockwise)
DT, T, T_SYN = 0.025, 8.0, 1.0
WEIGHTS = [0.001, 0.015, 0.1, 1.0, 10.0]


def xy(deg):
    a = math.radians(deg)
    return R_RING * math.cos(a), R_RING * math.sin(a)


def write_geometry(tmp: Path):
    lines = [f"# CYCLE_BREAK reconnect {N + 1} 1"]
    # ring goes clockwise from the cut: sample i at CUT_DEG - 360*(i-1)/N
    for i in range(1, N + 1):
        x, y = xy(CUT_DEG - 360.0 * (i - 1) / N)
        lines.append(f"{i} 3 {x:.6f} {y:.6f} 0 {R_CABLE} {-1 if i == 1 else i - 1}")
    x, y = xy(CUT_DEG)
    lines.append(f"{N + 1} 3 {x:.6f} {y:.6f} 0 {R_CABLE} {N}")
    bottom = N // 2 + 1  # sample at CUT_DEG - 180 = bottom
    bx, by = xy(CUT_DEG - 180.0)
    nid = N + 2
    lines.append(f"{nid} 3 {bx:.6f} {by - 0.75:.6f} 0 0.12 {bottom}")
    lines.append(f"{nid + 1} 3 {bx:.6f} {by - 1.5:.6f} 0 0.12 {nid}")
    lines.append(f"{nid + 2} 5 {bx:.6f} {by - 2.5:.6f} 0 3.0 {nid + 1}")
    lines.append(f"{nid + 3} 5 {bx:.6f} {by - 6.5:.6f} 0 3.0 {nid + 2}")
    lines.append(f"{nid + 4} 6 {bx:.6f} {by - 10.5:.6f} 0 3.0 {nid + 3}")
    swc = tmp / "ring_spine.swc"
    swc.write_text("\n".join(lines) + "\n")
    sx, sy = xy(SYN_DEG)
    syn = tmp / "ring_synpts.txt"
    syn.write_text(f"{sx:.6f} {sy:.6f} 0\n")
    return swc, syn, (bx, by)


