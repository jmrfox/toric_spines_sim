"""Loop-capable cable graph and reference solver.

SWC is a tree, so MASCAF cuts each mesh cycle and records the cut as
``# CYCLE_BREAK reconnect i j`` (extra necks: ``# MULTI_NECK``). This module
builds the cable *graph*: colocated reconnect pairs are merged into one node
(separated pairs are joined by a frustum), so loops are real loops. The same
graph is exported to jaxley (``simulation.jaxley_backend``); integrating it
here gives an independent check that needs only scipy.

Discretization: each SWC sample defines a frustum from its parent (child's
tag, radii from a :class:`~toric_spines_sim.geometry.morphology.SegmentTable`).
The frustum is cut into pieces no longer than ``dx_um``; each piece gives half
its lateral area to each end node and an axial conductance
``pi * r1 * r2 / (rL * L)``. Time stepping is implicit Euler.

Scope of :meth:`LoopCableModel.run`: passive membrane (per-tag ``cm`` /
``g_pas`` overrides) plus conductance synapses from
:mod:`toric_spines_sim.model.synaptic`; the Mg block uses the voltage at the
start of the step. Active channels run in jaxley.

Units: µm, ms, mV, µF/cm², S/cm², Ω·cm, µS.

Examples
--------
>>> model = LoopCableModel.from_swc("TS1_wsink_r10um.swc", dx_um=0.1,
...     cm_uF_per_cm2=0.9, g_pas_S_per_cm2=1.5e-5, e_pas_mV=-66, rL_ohm_cm=100)  # doctest: +SKIP
>>> s = model.add_synapse((34.8, 6.7, 16.6), "ampa", gmax_uS=0.001, tau_ms=2.0)  # doctest: +SKIP
>>> res = model.run({s: [10.0]}, T_ms=60, dt_ms=0.025, v0_mV=-66)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from toric_spines_sim.geometry.swc import iter_swc_samples, parse_reconnect_pairs
from toric_spines_sim.model import synaptic
from toric_spines_sim.model.synaptic import Component, exp2_peak_factor  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

XYZ = Tuple[float, float, float]


@dataclass
class ReferenceResult:
    """Voltages for every node: ``V[time_index, node]`` in mV."""

    t_ms: np.ndarray
    V: np.ndarray
    node_xyz: np.ndarray
    node_tag: np.ndarray

    def at(self, xyz: XYZ) -> np.ndarray:
        """Trace at the node nearest ``xyz``."""
        d = np.linalg.norm(self.node_xyz - np.asarray(xyz, float), axis=1)
        return self.V[:, int(np.argmin(d))]


@dataclass
class LoopCableModel:
    """Compartmental model on a graph (loops allowed). Build with ``from_swc``.

    ``synapses`` holds ``(node, [Component, ...])`` pairs; ``area_by_tag``
    maps each tag to the per-node membrane area (µm²) contributed by
    frusta of that tag (used to paint channels by tag).
    """

    node_xyz: np.ndarray
    node_tag: np.ndarray
    C_uF: np.ndarray
    G_leak_S: np.ndarray
    E_leak_mV: np.ndarray
    G_axial: sp.csr_matrix  # graph Laplacian of axial conductances, S
    area_by_tag: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _pieces: List[Tuple[int, int]] = field(default_factory=list, repr=False)
    synapses: List[Tuple[int, List[Component]]] = field(default_factory=list)
    n_loops: int = 0

    @property
    def n_nodes(self) -> int:
        return len(self.node_xyz)

    @property
    def area_um2(self) -> np.ndarray:
        return sum(self.area_by_tag.values()) if self.area_by_tag else np.zeros(self.n_nodes)

    @classmethod
    def from_swc(
        cls,
        swc_path: Union[str, Path],
        *,
        dx_um: float,
        cm_uF_per_cm2: float,
        g_pas_S_per_cm2: float,
        e_pas_mV: float,
        rL_ohm_cm: float,
        restore_loops: bool = True,
        tag_overrides: Optional[Mapping[int, Mapping[str, float]]] = None,
        segments=None,
    ) -> "LoopCableModel":
        """Build from an SWC with ``# CYCLE_BREAK`` / ``# MULTI_NECK`` headers.

        Parameters
        ----------
        restore_loops
            Merge reconnect pairs (colocated) or join them with a frustum
            (separated). ``False`` gives the tree stored in the SWC.
        tag_overrides
            ``{tag: {"cm": µF/cm², "g_pas": S/cm², "e_pas": mV}}`` for pieces
            whose (child) tag matches, e.g. a lumped sink.
        segments
            Optional :class:`~toric_spines_sim.geometry.morphology.SegmentTable`
            for this SWC (e.g. with rescaled sink or neck radii); its prox/dist
            radii are used for every segment.
        """
        swc_path = Path(swc_path)
        samples = iter_swc_samples(swc_path)
        by_id = {s[0]: s for s in samples}
        overrides = {int(k): dict(v) for k, v in (tag_overrides or {}).items()}
        seg_radii: Dict[int, Tuple[float, float]] = {}
        if segments is not None:
            non_root = [x[0] for x in samples if x[6] != -1]
            if len(segments) != len(non_root):
                raise ValueError("segment table does not match SWC sample count")
            seg_radii = segments.radii_by_sample()

        parent_uf = {sid: sid for sid in by_id}

        def find(a: int) -> int:
            while parent_uf[a] != a:
                parent_uf[a] = parent_uf[parent_uf[a]]
                a = parent_uf[a]
            return a

        extra_edges: List[Tuple[int, int]] = []
        if restore_loops:
            for i, j in parse_reconnect_pairs(swc_path):
                pi_, pj = np.array(by_id[i][2:5]), np.array(by_id[j][2:5])
                if np.linalg.norm(pi_ - pj) < 1e-6:
                    parent_uf[find(j)] = find(i)
                else:
                    extra_edges.append((i, j))

        node_index: Dict[int, int] = {}
        xyz: List[np.ndarray] = []
        tags: List[int] = []

        def swc_node(sid: int) -> int:
            root = find(sid)
            if root not in node_index:
                node_index[root] = len(xyz)
                xyz.append(np.array(by_id[root][2:5], float))
                tags.append(by_id[root][1])
            return node_index[root]

        C: List[float] = []
        G: List[float] = []
        E: List[float] = []
        A_tag: Dict[int, List[float]] = {}
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        pieces: List[Tuple[int, int]] = []

        def grow(n: int) -> None:
            while len(C) < n:
                C.append(0.0)
                G.append(0.0)
                E.append(0.0)
            for lst in A_tag.values():
                while len(lst) < n:
                    lst.append(0.0)

        def add_frustum(a_id: int, b_id: int, tag: int, use_table: bool = True) -> None:
            pa = np.array(by_id[a_id][2:5], float)
            pb = np.array(by_id[b_id][2:5], float)
            if use_table and b_id in seg_radii:
                ra, rb = seg_radii[b_id]
            else:
                ra, rb = by_id[a_id][5], by_id[b_id][5]
            length = float(np.linalg.norm(pb - pa))
            na, nb = swc_node(a_id), swc_node(b_id)
            if length <= 0.0:
                raise ValueError(f"zero-length segment {a_id}->{b_id} in {swc_path}")
            n_pieces = max(1, int(math.ceil(length / dx_um - 1e-9)))
            mem = overrides.get(tag, {})
            cm = mem.get("cm", cm_uF_per_cm2)
            gp = mem.get("g_pas", g_pas_S_per_cm2)
            ep = mem.get("e_pas", e_pas_mV)
            chain = [na]
            for k in range(1, n_pieces):
                f = k / n_pieces
                chain.append(len(xyz))
                xyz.append(pa + f * (pb - pa))
                tags.append(tag)
            chain.append(nb)
            A_tag.setdefault(tag, [])
            grow(len(xyz))
            area_t = A_tag[tag]
            for k in range(n_pieces):
                f0, f1 = k / n_pieces, (k + 1) / n_pieces
                r0 = ra + f0 * (rb - ra)
                r1 = ra + f1 * (rb - ra)
                rm = 0.5 * (r0 + r1)
                l = length / n_pieces
                for node, (q0, q1) in ((chain[k], (r0, rm)), (chain[k + 1], (rm, r1))):
                    area_um2 = math.pi * (q0 + q1) * math.hypot(0.5 * l, q1 - q0)
                    area_cm2 = area_um2 * 1e-8
                    C[node] += cm * area_cm2
                    G[node] += gp * area_cm2
                    E[node] += gp * area_cm2 * ep  # weighted; normalized below
                    area_t[node] += area_um2
                g_ax = math.pi * (r0 * 1e-4) * (r1 * 1e-4) / (rL_ohm_cm * l * 1e-4)
                u, v = chain[k], chain[k + 1]
                rows.extend([u, v, u, v])
                cols.extend([u, v, v, u])
                vals.extend([g_ax, g_ax, -g_ax, -g_ax])
                pieces.append((u, v))

        for s in samples:
            sid, tag, *_rest, par = s
            swc_node(sid)
            if par != -1:
                add_frustum(par, sid, tag)
        for i, j in extra_edges:
            add_frustum(i, j, by_id[j][1], use_table=False)
        grow(len(xyz))

        n = len(xyz)
        C_arr = np.array(C[:n])
        G_arr = np.array(G[:n])
        E_arr = np.where(G_arr > 0, np.array(E[:n]) / np.where(G_arr > 0, G_arr, 1), e_pas_mV)
        lap = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        if np.any(C_arr <= 0):
            raise ValueError("Some nodes have no membrane (isolated or zero-length)")
        n_components = connected_components(lap, directed=False)[0]
        if n_components != 1:
            raise ValueError(f"Graph has {n_components} disconnected components")
        coo = lap.tocoo()
        n_edges = int(((coo.row < coo.col) & (coo.data < 0)).sum())
        logger.info(
            "LoopCableModel %s: %d nodes, %d loops (%s)",
            swc_path.name, n, n_edges - n + 1, "restored" if restore_loops else "cut",
        )
        return cls(
            node_xyz=np.array(xyz),
            node_tag=np.array(tags),
            C_uF=C_arr,
            G_leak_S=G_arr,
            E_leak_mV=E_arr,
            G_axial=lap,
            area_by_tag={t: np.array(a[:n]) for t, a in A_tag.items()},
            _pieces=pieces,
            n_loops=n_edges - n + 1,
        )

    # ------------------------------------------------------------------
    def closest_node(self, xyz: XYZ) -> int:
        """Node nearest the closest point on any cable piece."""
        p = np.asarray(xyz, float)
        if not hasattr(self, "_piece_a"):
            u = np.array([a for a, _ in self._pieces])
            v = np.array([b for _, b in self._pieces])
            object.__setattr__(self, "_piece_a", u)
            object.__setattr__(self, "_piece_b", v)
        u, v = self._piece_a, self._piece_b
        a, b = self.node_xyz[u], self.node_xyz[v]
        ab = b - a
        denom = np.maximum((ab * ab).sum(1), 1e-30)
        t = np.clip(((p - a) * ab).sum(1) / denom, 0.0, 1.0)
        d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
        k = int(np.argmin(d))
        return int(u[k] if t[k] < 0.5 else v[k])

    def add_components(self, xyz: XYZ, comps: Sequence[Component]) -> int:
        """Add a synapse made of ``comps`` at the cable point nearest ``xyz``."""
        self.synapses.append((self.closest_node(xyz), list(comps)))
        return len(self.synapses) - 1

    def add_synapse(self, xyz: XYZ, kind: str = "ampa", **params: float) -> int:
        """Add a registry synapse (``ampa``, ``nmda``, ``gabaa``, ``gabab``, ``effexc``).

        ``params`` use the ``synapse_params`` names (``gmax_uS``, ``tau_ms``,
        ``tau_r_ms``, ``tau_d_ms``, ``e_mV``, ``mg_mM``, ...).
        """
        return self.add_components(xyz, synaptic.components(kind, params))

    def run(
        self,
        events_ms: Mapping[int, Sequence[float]],
        *,
        T_ms: float,
        dt_ms: float,
        v0_mV: float,
        record_every: int = 1,
        iclamps: Sequence[Tuple[XYZ, float, float, float]] = (),
    ) -> ReferenceResult:
        """Implicit-Euler integration. ``events_ms`` maps synapse index → times.

        ``iclamps``: ``(xyz, start_ms, duration_ms, amplitude_nA)`` boxes
        (positive = depolarizing).

        The matrix is factored once; the time-varying synaptic conductances
        on the few synapse nodes enter through a Woodbury update each step.
        """
        n = self.n_nodes
        n_steps = int(round(T_ms / dt_ms))
        c_dt = self.C_uF / dt_ms * 1e-3  # S
        base = (sp.diags(c_dt + self.G_leak_S) + self.G_axial).tocsc()
        lu = spla.splu(base)
        leak_rhs = self.G_leak_S * self.E_leak_mV

        groups = synaptic.node_conductances(self.synapses, events_ms, n_steps, dt_ms)
        keys = list(groups)
        g_series = np.array([groups[k] for k in keys]) * 1e-6 if keys else np.zeros((0, n_steps))
        g_node = np.array([k[0] for k in keys], dtype=int)
        g_e = np.array([k[1] for k in keys], dtype=float)
        g_mg = np.array([np.nan if k[2] is None else k[2] for k in keys], dtype=float)
        blocked = ~np.isnan(g_mg)
        snodes = np.unique(g_node) if keys else np.zeros(0, dtype=int)
        col = {nd: i for i, nd in enumerate(snodes)}
        g_col = np.array([col[nd] for nd in g_node], dtype=int)
        if len(snodes):
            U = sp.csc_matrix((np.ones(len(snodes)), (snodes, np.arange(len(snodes)))),
                              shape=(n, len(snodes)))
            Z = lu.solve(U.toarray())  # B^-1 U
            M = Z[snodes, :]           # U^T B^-1 U
        clamps = [(self.closest_node(xyz), t0, t0 + dur, amp * 1e-6) for xyz, t0, dur, amp in iclamps]

        v = np.full(n, float(v0_mV))
        out_t = [0.0]
        out_v = [v.copy()]
        eye = np.eye(len(snodes))
        for k in range(n_steps):
            rhs = c_dt * v + leak_rhs
            t_mid = (k + 0.5) * dt_ms
            for node, t0, t1, amp_mA in clamps:  # S*mV = mA
                if t0 <= t_mid < t1:
                    rhs[node] += amp_mA
            gk = g_series[:, k] if keys else None
            if keys and np.any(gk > 0):
                if blocked.any():
                    gk = gk.copy()
                    gk[blocked] *= synaptic.mg_block(v[g_node[blocked]], g_mg[blocked])
                d = np.zeros(len(snodes))
                np.add.at(d, g_col, gk)
                np.add.at(rhs, g_node, gk * g_e)
                y = lu.solve(rhs)
                # (B + U D U^T)^-1 r = y - Z (I + D M)^-1 D y_S
                w = np.linalg.solve(eye + d[:, None] * M, d * y[snodes])
                v = y - Z @ w
            else:
                v = lu.solve(rhs)
            if (k + 1) % record_every == 0:
                out_t.append((k + 1) * dt_ms)
                out_v.append(v.copy())
        return ReferenceResult(np.array(out_t), np.array(out_v), self.node_xyz, self.node_tag)


def lumped_sink_overrides(
    swc_path: Union[str, Path],
    sink_tags: Sequence[int],
    cell_C_pF: float,
    cell_Rin_MOhm: float,
    segments=None,
) -> Dict[int, Dict[str, float]]:
    """``tag_overrides`` that give the sink the whole cell's C and 1/Rin.

    The sink stands in for the rest of the neuron; its geometric area is far
    below a real cell's, so its specific cm / g_pas are scaled so the totals
    equal ``cell_C_pF`` and 1 / ``cell_Rin_MOhm``. Areas are frustum lateral
    areas of segments with a sink tag (from ``segments`` if given).
    """
    from toric_spines_sim.geometry.morphology import SegmentTable

    table = segments if segments is not None else SegmentTable.from_swc(swc_path)
    area_um2 = table.lateral_area_by_tags(sink_tags)
    if area_um2 <= 0:
        raise ValueError(f"no sink membrane for tags {list(sink_tags)} in {swc_path}")
    area_cm2 = area_um2 * 1e-8
    cm = cell_C_pF * 1e-12 / area_cm2 * 1e6
    g = 1.0 / (cell_Rin_MOhm * 1e6) / area_cm2
    return {int(t): {"cm": cm, "g_pas": g} for t in sink_tags}
