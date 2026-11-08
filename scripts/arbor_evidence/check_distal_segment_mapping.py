import sys, numpy as np, arbor as A
from pathlib import Path
from toric_spines_sim.geometry.swc import iter_swc_samples
for f in sys.argv[1:]:
    f = Path(f)
    st = A.load_swc_arbor(str(f)).segment_tree
    m = A.morphology(st); pw = A.place_pwlin(m)
    samples = iter_swc_samples(f)
    non_root = [s for s in samples if s[6] != -1]
    labels = {f"n{s[0]}": f"(distal (segment {k}))" for k, s in enumerate(non_root)}
    cell = A.cable_cell(m, A.decor(), A.label_dict(labels))
    errs = []; multi = 0
    for k, s in enumerate(non_root):
        locs = cell.locations(f'"n{s[0]}"')
        multi += len(locs) != 1
        p = pw.at(locs[0])
        errs.append(np.linalg.norm([p.x-s[2], p.y-s[3], p.z-s[4]]))
    print(f.name, "max err", max(errs), "non-unique", multi)
