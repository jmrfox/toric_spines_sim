"""Show the original (pre-fix) SWC-node -> Arbor-location mapping bug.

Uses the Aug-2026 BFS branch numbering kept in arbor_model/swc_buggy_original.py.
Run: uv run --group arbor-evidence python scripts/arbor_evidence/check_gj_placement.py data/swc/microns/TS*_wsink_r10um.swc
"""
import sys, numpy as np, arbor as A
from pathlib import Path
from arbor_model.swc_buggy_original import arbor_locations_for_swc_nodes, iter_swc_samples, segment_tree_branches
for f in sys.argv[1:]:
    st = A.load_swc_arbor(f).segment_tree
    m = A.morphology(st)
    pw = A.place_pwlin(m)
    samples = {s[0]: s for s in iter_swc_samples(Path(f))}
    locs = arbor_locations_for_swc_nodes(Path(f), m, st, list(samples))
    errs = []
    for nid, loc in locs.items():
        p = pw.at(loc)
        s = samples[nid]
        errs.append(np.linalg.norm([p.x - s[2], p.y - s[3], p.z - s[4]]))
    errs = np.array(errs)
    # compare BFS branch ordering vs arbor's
    ours = segment_tree_branches(st)
    mism = 0
    for bid, segs in enumerate(ours):
        ms = m.branch_segments(bid)
        if len(ms) != len(segs) or any(abs(ms[k].dist.x - st.segments[segs[k]].dist.x) > 1e-9 for k in range(len(segs))):
            mism += 1
    print(f"{Path(f).name}: nodes={len(errs)} branches={m.num_branches} max_xyz_err={errs.max():.4g} n_bad(>1e-3)={(errs>1e-3).sum()} branch_order_mismatches={mism}")
