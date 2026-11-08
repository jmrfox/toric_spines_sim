"""Continuous 100um cable vs two 50um halves joined by a gap junction.
Synapse-like current pulse at x=0; compare V at x=100 (far end) and at the junction."""
import numpy as np, arbor as A
from arbor import units as U
R=0.3; L=50.0; CM=1.0; RL=100.0; GPAS=1.7e-5; DX=0.1
def tree(x0, x1):
    t=A.segment_tree(); t.append(A.mnpos, A.mpoint(x0,0,0,R), A.mpoint(x1,0,0,R), tag=3); return t
class Rec(A.recipe):
    def __init__(s, split, w, dt):
        super().__init__(); s.split=split; s.w=w
        s.p=A.neuron_cable_properties(); s.p.set_property(Vm=-65*U.mV, cm=CM*U.uF/U.cm2, rL=RL*U.Ohm*U.cm)
    def num_cells(s): return 2 if s.split else 1
    def cell_kind(s,g): return A.cell_kind.cable
    def global_properties(s,k): return s.p
    def cell_description(s,g):
        d=A.decor().paint("(all)", A.density("pas/e=-65", {"g":GPAS}))
        if g==0: d.place("(location 0 0)", A.i_clamp(1*U.ms, 2*U.ms, 0.05*U.nA))
        if s.split: d.place("(location 0 1)" if g==0 else "(location 0 0)", A.junction("gj"), "gj")
        t=tree(0,2*L) if not s.split else (tree(0,L) if g==0 else tree(L,2*L))
        return A.cable_cell(A.morphology(t), d, A.label_dict(), A.cv_policy_max_extent(DX*U.um))
    def gap_junctions_on(s,g):
        if not s.split: return []
        return [A.gap_junction_connection((1-g,"gj"),"gj",s.w)]
    def probes(s,g):
        return [A.cable_probe_membrane_voltage("(location 0 1)","far"), A.cable_probe_membrane_voltage("(location 0 0)","near")]
def run(split, w=1.0, dt=0.025):
    rec=Rec(split,w,dt); sim=A.simulation(rec)
    if split:
        hs={"far":sim.sample((1,"far"),A.regular_schedule(dt*U.ms)), "j0":sim.sample((0,"far"),A.regular_schedule(dt*U.ms)), "j1":sim.sample((1,"near"),A.regular_schedule(dt*U.ms))}
    else:
        hs={"far":sim.sample((0,"far"),A.regular_schedule(dt*U.ms))}
    sim.run(20*U.ms, dt*U.ms)
    return {k: sim.samples(h)[0][0][:,1] for k,h in hs.items()}
# axial conductance of one CV-length of this cable, in uS
g_ax = np.pi*(R*1e-4)**2/(RL*DX*1e-4)*1e6
C_cv = CM*1e-6*2*np.pi*R*1e-4*DX*1e-4  # F
print(f"one-CV axial conductance g_ax = {g_ax:.4g} uS ; CV capacitance = {C_cv*1e12:.4g} pF")
for dt in (0.025, 0.005):
    ref=run(False, dt=dt); pk=(ref['far']+65).max()
    print(f"\n dt={dt} ms  reference far-end peak depolarization = {pk:.4f} mV")
    for w in (1e-4,1e-3, 0.015, 0.1, 1.0, 10.0, 100.0):
        r=run(True,w,dt); far=(r['far']+65)
        jump=np.abs(r['j0']-r['j1']).max()
        osc=np.abs(np.diff(np.sign(np.diff(r['j0']-r['j1'])))).sum()/2
        print(f"  w={w:<7g} g={w:<7g}uS  g/g_ax={w/g_ax:9.3g}  g*dt/C={w*1e-6*dt*1e-3/C_cv:9.3g}  far peak={far.max():.4f} ({100*(far.max()/pk-1):+.2f}%)  max|dV| across gj={jump:.4f} mV  sign flips={osc:.0f}")
