"""Simulate a :class:`LoopCableModel` in jaxley, with loops closed implicitly.

Export (technique from cabaret ``jaxley_io``, plan 007):

- Each graph node becomes a one-compartment jaxley branch on a BFS spanning
  tree. A branch's ``resistive_load_in`` carries the resistance of its edge to
  its parent (jaxley resistance = load x r_L, so load = R / (r_L * 1e4));
  ``resistive_load_out`` is a negligible branch-point load.
- Membrane is set per compartment: ``area`` is chosen so that
  ``capacitance`` = 1 µF/cm² reproduces the node's capacitance; leak density
  follows from the node's conductance. Channel densities are rescaled to that
  area, so totals match the geometric membrane they belong to.
- Each graph edge not in the spanning tree is a loop a-b with resistance R:
  a dummy compartment without membrane hangs under a with load_in = R/2, and
  an extra row in jaxley's private ``_comp_edges`` joins it to b using
  load_in(dummy) + load_out(b). This needs jaxley 0.14 and the ``jax.sparse``
  voltage solver; ``jaxley.dhs`` assumes a tree and is refused for loops.

Numerics (each was found the hard way, see the migration plan):

- float64 is required: the ~1 Ω branch-point loads make the system ~1e9
  stiffer than the membrane terms, and float32 invents a conductance to ground.
- The cell runs in deviations from rest (v0 = 0): jaxley's internal
  branch-point nodes are not initialized by ``cell.set("v", ...)``. Reversals
  are shifted, voltage-dependent rates see ``v + v_rest``, and rest is added
  back to the output.
- jaxley applies a clamped state after computing currents, so clamp series
  are shifted one step earlier.

Synapses: conductance time courses come from
:mod:`toric_spines_sim.model.synaptic` and are clamped onto per-compartment
channel states; the Mg block is evaluated from the voltage in the channel.

:class:`JaxleySimulation` builds and jit-compiles once per model structure;
:meth:`~JaxleySimulation.run` and :meth:`~JaxleySimulation.run_batch` then take
only event times. :func:`get_simulation` caches instances.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from toric_spines_sim.model import synaptic
from toric_spines_sim.model.model import HHSpec

from .reference import LoopCableModel

R_L_OHM_CM = 100.0  # arbitrary: loads carry the resistances
BRANCHPOINT_LOAD = 1e-6  # 1/µm, ~1 Ω at R_L = 100 (cabaret: do not go far lower)
DUMMY_AREA_UM2 = 1e-4
JAXLEY_SUPPORTED = ("0.14",)


def _enable_x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def _check_jaxley_version() -> None:
    import jaxley

    if not jaxley.__version__.startswith(JAXLEY_SUPPORTED):
        raise RuntimeError(
            f"jaxley {jaxley.__version__} is not supported: the loop export edits the "
            f"private `_comp_edges` table of jaxley {JAXLEY_SUPPORTED}. Pin jaxley<0.15."
        )


def _channel_classes():
    """Channel classes, built lazily so importing this module needs no jaxley."""
    import jax.numpy as jnp
    from jaxley.channels import HH, Channel

    class ClampedSyn(Channel):
        """Conductance density ``g`` (S/cm²) supplied by ``data_clamp``; optional Mg block."""

        def __init__(self, name: str, blocked: bool):
            self.current_is_in_mA_per_cm2 = True
            super().__init__(name)
            self.blocked = blocked
            n = self._name
            self.channel_params = {f"{n}_e": 0.0, f"{n}_mg": 1.0, f"{n}_vrest": -65.0}
            self.channel_states = {f"{n}_g": 0.0}
            self.current_name = f"i_{n}"

        def update_states(self, states, dt, v, params):
            return {}

        def compute_current(self, states, v, params):
            n = self._name
            g = states[f"{n}_g"]
            if self.blocked:
                v_abs = v + params[f"{n}_vrest"]
                g = g / (1.0 + jnp.exp(-0.062 * v_abs) * params[f"{n}_mg"] / 3.57)
            return g * (v - params[f"{n}_e"])

        def init_state(self, states, v, params, delta_t):
            return {}

    class ShiftedHH(HH):
        """jaxley HH (same rates as ``hhnotemp``) for a cell run in deviations from rest."""

        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params[f"{self._name}_vrest"] = -65.0

        def update_states(self, states, dt, v, params):
            return super().update_states(states, dt, v + params[f"{self._name}_vrest"], params)

        def init_state(self, states, v, params, delta_t):
            return super().init_state(states, v + params[f"{self._name}_vrest"], params, delta_t)

    return ClampedSyn, ShiftedHH


def _spanning_tree(model: LoopCableModel, root: int):
    n = model.n_nodes
    lap = model.G_axial.tocoo()
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j, v in zip(lap.row, lap.col, lap.data):
        if i < j and v < 0:
            g.add_edge(int(i), int(j), resistance=1.0 / (-v))
    tree = nx.bfs_tree(g, root)
    loops = [(a, b, d["resistance"]) for a, b, d in g.edges(data=True)
             if not tree.has_edge(a, b) and not tree.has_edge(b, a)]
    for k, (a, _b, r) in enumerate(loops):
        g.add_edge(a, n + k, resistance=r / 2)
        tree.add_edge(a, n + k)
    order = list(nx.bfs_tree(tree, root))
    pred = {v: next(iter(tree.predecessors(v))) for v in order if v != root}
    return g, order, pred, loops


class JaxleySimulation:
    """A compiled jaxley simulation of one model structure.

    Parameters
    ----------
    model : LoopCableModel
        Graph and synapses (``model.synapses``); gmax and kinetics are read at
        each :meth:`run`, so only the synapse *placement* and (e, Mg) layout
        are fixed here.
    T_ms, dt_ms, v_rest_mV
        Duration, time step and resting potential (initial voltage).
    record_nodes : sequence of int
        Graph nodes to record.
    hh : HHSpec, optional
    """

    def __init__(
        self,
        model: LoopCableModel,
        *,
        T_ms: float,
        dt_ms: float,
        v_rest_mV: float,
        record_nodes: Sequence[int],
        hh: Optional[HHSpec] = None,
        root: int = 0,
    ):
        _enable_x64()
        _check_jaxley_version()
        import jax
        import jax.numpy as jnp
        import jaxley as jx
        import pandas as pd
        from jaxley.channels import Leak

        ClampedSyn, ShiftedHH = _channel_classes()
        self.model = model
        self.T_ms, self.dt_ms, self.v_rest = float(T_ms), float(dt_ms), float(v_rest_mV)
        self.n_steps = int(round(T_ms / dt_ms))
        self.record_nodes = [int(n) for n in record_nodes]
        n = model.n_nodes

        g, order, pred, loops = _spanning_tree(model, root)
        comp_of = {v: k for k, v in enumerate(order)}
        self.comp_of_node = np.array([comp_of[v] for v in range(n)])
        self.n_loops = len(loops)
        cell = jx.Cell([jx.Branch(jx.Compartment(), ncomp=1)] * len(order),
                       parents=[-1 if v == root else comp_of[pred[v]] for v in order])
        cell.insert(Leak())
        cell.set("radius", 1.0)
        cell.set("length", 1.0)
        cell.set("axial_resistivity", R_L_OHM_CM)
        cell.set("capacitance", 1.0)

        self.area_cm2 = model.C_uF.copy()  # capacitance 1 µF/cm² -> area in cm²
        area_um2 = np.full(len(order), DUMMY_AREA_UM2)
        g_leak = np.zeros(len(order))
        e_leak = np.zeros(len(order))
        load_in = np.empty(len(order))
        for v in order:
            c = comp_of[v]
            if v < n:
                area_um2[c] = self.area_cm2[v] * 1e8
                g_leak[c] = model.G_leak_S[v] / self.area_cm2[v]
                e_leak[c] = model.E_leak_mV[v] - self.v_rest
            r = 0.0 if v == root else g.edges[v, pred[v]]["resistance"]
            load_in[c] = max(r, 1.0) * 1e-4 / R_L_OHM_CM
        nodes = cell.base.nodes
        nodes["area"] = area_um2
        nodes["Leak_gLeak"] = g_leak
        nodes["Leak_eLeak"] = e_leak
        nodes["resistive_load_in"] = load_in
        nodes["resistive_load_out"] = BRANCHPOINT_LOAD
        for k, (_a, b, _r) in enumerate(loops):
            rows = pd.DataFrame({"source": [comp_of[b], comp_of[n + k]],
                                 "sink": [comp_of[n + k], comp_of[b]],
                                 "ordered": [1, 0], "type": [0, 0]})
            cell._comp_edges = pd.concat([cell._comp_edges, rows], ignore_index=True)
        if loops:
            cell._init_solvers()
        if not {"source", "sink", "ordered", "type"} <= set(cell._comp_edges.columns):
            raise RuntimeError("jaxley `_comp_edges` layout changed; the loop export needs updating")

        # Synaptic channels: one channel type per (e, mg) group, clamped per node.
        group_types: Dict[tuple, str] = {}
        self.slots: List[Tuple[int, tuple]] = []  # (node, (e, mg)) per clamp row
        for node, comps in model.synapses:
            for c in comps:
                key = synaptic.group_key(c)
                if key not in group_types:
                    group_types[key] = f"Syn{len(group_types)}"
                slot = (int(node), key)
                if slot not in self.slots:
                    self.slots.append(slot)
        inserted = set()
        for node, key in self.slots:
            name = group_types[key]
            comp = int(self.comp_of_node[node])
            if (comp, name) in inserted:
                continue
            cell.branch(comp).insert(ClampedSyn(name, blocked=key[1] is not None))
            br = cell.branch(comp)  # views are snapshots: fetch again after insert
            br.set(f"{name}_e", key[0] - self.v_rest)
            br.set(f"{name}_vrest", self.v_rest)
            if key[1] is not None:
                br.set(f"{name}_mg", key[1])
            inserted.add((comp, name))
        self._slot_names = [group_types[key] for _, key in self.slots]

        # HH on the area of the given tags, rescaled to each compartment's jaxley area.
        self.hh = hh
        if hh is not None:
            a_hh = sum(model.area_by_tag.get(t, np.zeros(n)) for t in hh.tags)
            hh_nodes = np.nonzero(a_hh > 0)[0]
            if len(hh_nodes) == 0:
                raise ValueError(f"no membrane with tags {hh.tags} for HH")
            for nd in hh_nodes:
                c = int(self.comp_of_node[nd])
                scale = (a_hh[nd] * 1e-8) / self.area_cm2[nd]  # geometric HH area / jaxley area
                cell.branch(c).insert(ShiftedHH())
                br = cell.branch(c)
                br.set("ShiftedHH_gNa", hh.gna * scale)
                br.set("ShiftedHH_gK", hh.gk * scale)
                br.set("ShiftedHH_gLeak", hh.gl * scale)
                br.set("ShiftedHH_eNa", hh.ena - self.v_rest)
                br.set("ShiftedHH_eK", hh.ek - self.v_rest)
                br.set("ShiftedHH_eLeak", hh.el - self.v_rest)
                br.set("ShiftedHH_vrest", self.v_rest)

        cell.set("v", 0.0)
        cell.init_states()
        # jaxley records a compartment once even if asked twice: record unique
        # nodes and expand back to the requested columns.
        unique = sorted(set(self.record_nodes))
        self._col_index = np.array([unique.index(nd) for nd in self.record_nodes], dtype=int)
        for nd in unique:
            cell.branch(int(self.comp_of_node[nd])).comp(0).record("v", verbose=False)
        self.cell = cell

        solver = "jax.sparse" if loops else "jaxley.dhs"
        slot_comps = [int(self.comp_of_node[nd]) for nd, _ in self.slots]
        slot_states = [f"{nm}_g" for nm in self._slot_names]
        dt, T = self.dt_ms, self.T_ms

        def simulate(clamps):
            data_clamps = None
            for i, (comp, state) in enumerate(zip(slot_comps, slot_states)):
                data_clamps = cell.branch(comp).comp(0).data_clamp(state, clamps[i], data_clamps)
            kwargs = dict(delta_t=dt, t_max=T, voltage_solver=solver)
            if data_clamps is not None:
                kwargs["data_clamps"] = data_clamps
            return jx.integrate(cell, **kwargs)

        self._simulate = jax.jit(simulate)
        # jax's sparse solver (needed for loops) has no vmap batching rule, so
        # looped cells run batches sequentially inside one compiled lax.map;
        # tree cells (jaxley.dhs) vectorize with vmap.
        if loops:
            self._simulate_batch = jax.jit(lambda c: jax.lax.map(simulate, c))
        else:
            self._simulate_batch = jax.jit(jax.vmap(simulate))
        self._jnp = jnp

    def _clamps(self, events: Mapping[int, Sequence[float]]) -> np.ndarray:
        groups = synaptic.node_conductances(self.model.synapses, events, self.n_steps, self.dt_ms)
        out = np.zeros((max(len(self.slots), 1), self.n_steps))
        for i, (node, key) in enumerate(self.slots):
            s = groups.get((node, *key))
            if s is None:
                continue
            density = s * 1e-6 / self.area_cm2[node]  # µS -> S/cm² of the jaxley area
            out[i] = np.concatenate([density[1:], density[-1:]])  # applied one step late
        return out

    def _finish(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        if v.shape[-1] < self.n_steps + 1:
            raise RuntimeError(f"jaxley returned {v.shape[-1]} samples, expected {self.n_steps + 1}")
        v = np.swapaxes(v[..., : self.n_steps + 1], -1, -2) + self.v_rest
        return v[..., self._col_index]

    def run(self, events: Mapping[int, Sequence[float]]) -> np.ndarray:
        """``V[time, k]`` (mV) for ``record_nodes``, times ``0, dt, ..., T``."""
        return self._finish(self._simulate(self._jnp.asarray(self._clamps(events))))

    def run_batch(self, events_list: Sequence[Mapping[int, Sequence[float]]]) -> np.ndarray:
        """``V[batch, time, k]`` for several event sets in one compiled call.

        Vectorized (``vmap``) for tree cells; sequential (``lax.map``) for
        looped cells, whose sparse solve cannot be batched in jax.
        """
        clamps = np.stack([self._clamps(ev) for ev in events_list])
        return self._finish(self._simulate_batch(self._jnp.asarray(clamps)))


def _structure_key(model: LoopCableModel, **kwargs) -> str:
    h = hashlib.sha1()
    for arr in (model.C_uF, model.G_leak_S, model.E_leak_mV, model.G_axial.data,
                model.G_axial.indices, model.G_axial.indptr):
        h.update(np.ascontiguousarray(arr).tobytes())
    for t in sorted(model.area_by_tag):
        h.update(str(t).encode())
        h.update(np.ascontiguousarray(model.area_by_tag[t]).tobytes())
    layout = sorted({(int(nd), synaptic.group_key(c)) for nd, comps in model.synapses for c in comps},
                    key=repr)
    h.update(repr(layout).encode())
    h.update(repr(sorted(kwargs.items())).encode())
    return h.hexdigest()


_CACHE: Dict[str, JaxleySimulation] = {}
_CACHE_MAX = 8


def get_simulation(model: LoopCableModel, *, T_ms: float, dt_ms: float, v_rest_mV: float,
                   record_nodes: Sequence[int], hh: Optional[HHSpec] = None) -> JaxleySimulation:
    """Cached :class:`JaxleySimulation` for this structure (rebuilt on any change)."""
    key = _structure_key(model, T_ms=float(T_ms), dt_ms=float(dt_ms), v_rest=float(v_rest_mV),
                         record=tuple(int(n) for n in record_nodes), hh=hh)
    sim = _CACHE.get(key)
    if sim is None:
        sim = JaxleySimulation(model, T_ms=T_ms, dt_ms=dt_ms, v_rest_mV=v_rest_mV,
                               record_nodes=record_nodes, hh=hh)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = sim
    else:
        sim.model = model  # same structure; gmax / kinetics may differ
    return sim


def run(model: LoopCableModel, events_ms: Mapping[int, Sequence[float]], *, T_ms: float,
        dt_ms: float, v0_mV: float, record_nodes: Sequence[int], hh: Optional[HHSpec] = None,
        root: int = 0) -> np.ndarray:
    """One-shot helper: ``V[time, k]`` for ``record_nodes`` (incl. t = 0)."""
    sim = get_simulation(model, T_ms=T_ms, dt_ms=dt_ms, v_rest_mV=v0_mV,
                         record_nodes=record_nodes, hh=hh)
    return sim.run(events_ms)
