"""Loop attenuation: peak depolarization vs distance, loops closed vs cut.

One synapse fires once at a distal site of a toric spine. The peak membrane
potential at every SWC segment centre is plotted against the geodesic
distance from the synapse along the **looped** cable graph, twice: with the
MASCAF ``# CYCLE_BREAK`` loops closed (the real morphology) and with them
cut (the tree stored in the SWC). This is MASCAF Fig 12 and the SfN 2026
poster loop panel, for any spine.

Typical use::

    from toric_spines_sim.attenuation import loop_attenuation, plot_loop_attenuation
    res = loop_attenuation("TS1")
    res.table          # one row per segment
    res.summary        # peaks at the synapse and the neck, closed vs cut
    plot_loop_attenuation(res)

All spines: ``uv run python -m simulations.loop_attenuation``.

Defaults match the MASCAF revision's Fig 12: ICx in-vitro bank (cm 0.9
µF/cm², lumped sink with the Sanculi whole-cell C and Rin), one AMPA
synapse of 5 nS (chosen for signal size), dx = 0.25 µm, dt = 0.025 ms.
The reference solver runs both arms; ``cross_check=True`` also runs the
closed arm in jaxley and reports the largest difference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from jscip import ParameterSet
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from toric_spines_sim.geometry.morphology import SegmentTable
from toric_spines_sim.geometry.sink import neck_point_from_swc_file
from toric_spines_sim.model import SynapsePopulation, TSModel
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_icx_parameter_bank
from toric_spines_sim.utils import load_xyz_points

logger = logging.getLogger(__name__)

XYZ = Tuple[float, float, float]
SPINE_TAG = 3
DEFAULTS = dict(gmax_uS=0.005, T_ms=40.0, dt_ms=0.025, dx_um=0.25, event_ms=5.0)


@dataclass
class AttenuationResult:
    """Output of :func:`loop_attenuation`.

    Attributes
    ----------
    name : str
        Spine name (or SWC stem).
    table : pandas.DataFrame
        One row per SWC segment: ``probe, tag, x, y, z, geodesic_um,
        peak_closed_mV, peak_cut_mV`` (+ ``peak_jaxley_mV`` with cross-check).
        Peaks are depolarization from rest.
    summary : dict
        Stimulus site, loop count, peaks at the synapse and at the neck for
        both arms, neck/synapse transfer ratios, and cross-check error.
    parameters : ParameterSet
        The sampled parameter set that was simulated.
    """

    name: str
    table: pd.DataFrame
    summary: Dict[str, object]
    parameters: ParameterSet = field(repr=False)

    @classmethod
    def load(cls, directory: Union[str, Path], name: str) -> "AttenuationResult":
        """Read a result written by :meth:`save` (``parameters`` is not stored, so None)."""
        directory = Path(directory)
        table = pd.read_csv(directory / f"{name}_attenuation.csv")
        summary = pd.read_csv(directory / f"{name}_attenuation_summary.csv").iloc[0].to_dict()
        summary["loops"] = int(summary["loops"])
        return cls(name=name, table=table, summary=summary, parameters=None)

    def save(self, directory: Union[str, Path]) -> Tuple[Path, Path]:
        """Write ``{name}_attenuation.csv`` and ``{name}_attenuation_summary.csv``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        table = directory / f"{self.name}_attenuation.csv"
        summary = directory / f"{self.name}_attenuation_summary.csv"
        self.table.to_csv(table, index=False, float_format="%.6g")
        pd.DataFrame([self.summary]).to_csv(summary, index=False, float_format="%.6g")
        return table, summary


def _default_parameters(dx_um: float, dt_ms: float, T_ms: float, gmax_uS: float) -> ParameterSet:
    bank = make_icx_parameter_bank("in_vitro")
    bank["discretization_um"].value = dx_um
    bank["dt_sim_ms"].value = dt_ms
    bank["dt_record_ms"].value = dt_ms
    bank["T_ms"].value = T_ms
    bank["ampa_gmax_uS"].value = gmax_uS
    p = bank.sample()
    p["hh_tags"] = []
    return p


def _geodesic(graph, source: int) -> np.ndarray:
    coo = graph.G_axial.tocoo()
    keep = coo.row != coo.col
    lens = np.linalg.norm(graph.node_xyz[coo.row[keep]] - graph.node_xyz[coo.col[keep]], axis=1)
    g = csr_matrix((lens, (coo.row[keep], coo.col[keep])), shape=graph.G_axial.shape)
    return dijkstra(g, indices=source)


def _resolve_inputs(spine: Union[str, Path], synpts: Optional[Union[str, Path]]):
    """``(name, swc, synpts or None)`` from a spine id (``"TS1"``) or an SWC path."""
    path = Path(spine)
    if path.suffix == ".swc":
        name = path.stem.split("_wsink")[0]
        return name, path, (Path(synpts) if synpts else None)
    name = str(spine).upper()
    swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
    if synpts is None:
        candidate = get_pointset_path(f"{name}_synpts.txt", units="microns")
        synpts = candidate if candidate.exists() else None
    return name, swc, (Path(synpts) if synpts else None)


def choose_distal_site(graph, segments: SegmentTable, neck_xyz: XYZ,
                       synapse_xyz: Optional[Sequence[XYZ]] = None) -> Tuple[XYZ, str]:
    """The site farthest from the neck along the looped cable.

    Among the synapse sites when given; otherwise among the spine segments'
    distal points (spines without a synapse file, e.g. TS21 / TS24).
    """
    dist = _geodesic(graph, graph.closest_node(neck_xyz))
    if synapse_xyz is not None and len(synapse_xyz):
        d = [dist[graph.closest_node(tuple(x))] for x in synapse_xyz]
        k = int(np.argmax(d))
        return tuple(map(float, synapse_xyz[k])), f"synapse {k}"
    pts = [(s.dist.x, s.dist.y, s.dist.z) for s in segments.segments if s.tag == SPINE_TAG]
    d = [dist[graph.closest_node(x)] for x in pts]
    k = int(np.argmax(d))
    return tuple(map(float, pts[k])), "farthest spine point"


def loop_attenuation(
    spine: Union[str, Path],
    *,
    synpts: Optional[Union[str, Path]] = None,
    parameters: Optional[ParameterSet] = None,
    stimulus_xyz: Optional[XYZ] = None,
    gmax_uS: float = DEFAULTS["gmax_uS"],
    T_ms: float = DEFAULTS["T_ms"],
    dt_ms: float = DEFAULTS["dt_ms"],
    dx_um: float = DEFAULTS["dx_um"],
    event_ms: float = DEFAULTS["event_ms"],
    cross_check: bool = False,
) -> AttenuationResult:
    """Peak depolarization vs geodesic distance, with loops closed and cut.

    Parameters
    ----------
    spine
        Spine id (``"TS1"``; uses ``TS1_wsink_r10um.swc`` and its synpts) or
        a path to a sink SWC.
    synpts
        Synapse points file; the stimulus is the synapse farthest from the
        neck. Without one, the farthest spine point is used.
    parameters
        Sampled parameters. Default: in-vitro ICx bank with ``dx_um``,
        ``dt_ms``, ``T_ms`` and ``gmax_uS`` applied. When given, those keyword
        values are ignored.
    stimulus_xyz
        Override the stimulus site.
    cross_check
        Also run the closed arm in jaxley (adds a one-off compile, ~10-20 s).
    """
    name, swc, synpts = _resolve_inputs(spine, synpts)
    p = parameters if parameters is not None else _default_parameters(dx_um, dt_ms, T_ms, gmax_uS)
    T, dt = float(p["T_ms"]), float(p["dt_sim_ms"])
    rest = float(p["Vrest_mV"])
    neck_xyz = tuple(neck_point_from_swc_file(swc))

    # Record at every segment centre (same labels as record_points="all").
    segments = SegmentTable.from_swc(swc)
    record = {f"probe_seg_{k}": s.center for k, s in enumerate(segments.segments)}

    closed = TSModel(swc, {}, record, p).build()
    syn_xyz = load_xyz_points(synpts) if synpts is not None else None
    if stimulus_xyz is None:
        stimulus_xyz, source = choose_distal_site(closed.graph, closed.segments, neck_xyz, syn_xyz)
    else:
        stimulus_xyz, source = tuple(map(float, stimulus_xyz)), "given"
    syn = SynapsePopulation("ampa", [stimulus_xyz], p, label_prefix="stim").synapses

    peaks = {}
    built = {}
    for arm, restore in (("closed", True), ("cut", False)):
        b = TSModel(swc, syn, record, p, restore_loops=restore).build()
        built[arm] = b
        res = b.graph.run({0: [event_ms]}, T_ms=T, dt_ms=dt, v0_mV=rest)
        nodes = [b.record_nodes[lab] for lab in record]
        peaks[arm] = res.V[:, nodes].max(axis=0) - rest

    g = built["closed"].graph
    dist = _geodesic(g, g.closest_node(stimulus_xyz))
    labels = list(record)
    rec_nodes = [built["closed"].record_nodes[lab] for lab in labels]
    table = pd.DataFrame({
        "probe": labels,
        "tag": [s.tag for s in segments.segments],
        "x": [record[lab][0] for lab in labels],
        "y": [record[lab][1] for lab in labels],
        "z": [record[lab][2] for lab in labels],
        "geodesic_um": dist[rec_nodes],
        "peak_closed_mV": peaks["closed"],
        "peak_cut_mV": peaks["cut"],
    })

    cross = None
    if cross_check:
        from toric_spines_sim.simulation import jaxley_backend

        v = jaxley_backend.run(g, {0: [event_ms]}, T_ms=T, dt_ms=dt, v0_mV=rest, record_nodes=rec_nodes)
        table["peak_jaxley_mV"] = v.max(axis=0) - rest
        cross = float(np.abs(table["peak_jaxley_mV"] - table["peak_closed_mV"]).max())

    def at(arm: str, xyz: XYZ) -> float:
        b = built[arm]
        res_node = b.graph.closest_node(xyz)
        # peak at the record probe nearest that node
        k = int(np.argmin(np.linalg.norm(table[["x", "y", "z"]].to_numpy() - b.graph.node_xyz[res_node], axis=1)))
        return float(table[f"peak_{arm}_mV"].iloc[k])

    spine_rows = table["tag"] == SPINE_TAG
    summary = {
        "spine": name,
        "loops": g.n_loops,
        "stimulus": source,
        "stimulus_x": stimulus_xyz[0], "stimulus_y": stimulus_xyz[1], "stimulus_z": stimulus_xyz[2],
        "stimulus_to_neck_um": float(dist[g.closest_node(neck_xyz)]),
        "gmax_nS": float(syn["stim_0"].synapse_params["gmax_uS"]) * 1e3,
        "peak_synapse_closed_mV": float(table.loc[spine_rows, "peak_closed_mV"].max()),
        "peak_synapse_cut_mV": float(table.loc[spine_rows, "peak_cut_mV"].max()),
        "peak_neck_closed_mV": at("closed", neck_xyz),
        "peak_neck_cut_mV": at("cut", neck_xyz),
        "jaxley_max_diff_mV": cross,
    }
    for arm in ("closed", "cut"):
        summary[f"neck_transfer_{arm}"] = summary[f"peak_neck_{arm}_mV"] / summary[f"peak_synapse_{arm}_mV"]
    summary["cut_overestimate_at_synapse"] = (
        summary["peak_synapse_cut_mV"] / summary["peak_synapse_closed_mV"] - 1.0
    )
    logger.info("%s: %d loops, synapse peak closed %.2f mV, cut %.2f mV", name, g.n_loops,
                summary["peak_synapse_closed_mV"], summary["peak_synapse_cut_mV"])
    return AttenuationResult(name=name, table=table, summary=summary, parameters=p)


STYLES = {
    "paper": dict(closed="#1f5fbf", cut="0.6", sink="#c4501a", font=None, size=11),
    # SfN 2026 poster palette and type
    "poster": dict(closed="#2E6FB0", cut="#5F6368", sink="#C8661A",
                   font=["Calibri", "Carlito", "DejaVu Sans"], size=16),
}


_FONT_DIRS = (Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts"), Path.home() / ".fonts")


def _ensure_fonts(names) -> None:
    """Register the first available of ``names`` with matplotlib (e.g. Calibri from Windows under WSL)."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in names:
        if name in installed:
            return
        for d in _FONT_DIRS:
            files = sorted(d.glob(f"{name.lower()}*.ttf")) if d.exists() else []
            if files:
                for f in files:
                    font_manager.fontManager.addfont(str(f))
                return


def style_rc(style: str) -> dict:
    """matplotlib rcParams for a STYLES entry (fonts registered as needed)."""
    st = STYLES[style]
    rc = {"font.size": st["size"], "axes.spines.top": False, "axes.spines.right": False,
          "pdf.fonttype": 42}
    if style == "poster":
        rc.update({"axes.edgecolor": "#5F6368", "axes.labelcolor": "#1A1A1A",
                   "xtick.color": "#5F6368", "ytick.color": "#5F6368",
                   "xtick.labelsize": st["size"] * 0.85, "ytick.labelsize": st["size"] * 0.85,
                   "axes.titlesize": st["size"] * 1.05})
    if st["font"]:
        _ensure_fonts(st["font"])
        rc["font.family"] = st["font"]
    return rc


def plot_loop_attenuation(result: AttenuationResult, ax=None, style: str = "paper",
                          sink: bool = True, title: Optional[str] = None, x_pad: float = 0.04,
                          legend: bool = True):
    """Scatter of peak depolarization (mV from rest) vs geodesic distance.

    Open grey circles: loops cut. Filled blue: loops closed. The sink, whose
    peak is nearly the same at every distance, is drawn as a horizontal band
    of its mean ± 1 standard deviation (loops closed) behind the points; the
    sink is nearly isopotential (σ ~ 0.1 µV), so the band is usually no wider
    than its centre line, and the legend states σ. The x axis spans the spine
    (tag 3) segments only. Returns the axes.
    """
    import matplotlib.pyplot as plt

    st = STYLES[style]
    with plt.rc_context(style_rc(style)):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 3.6))
        t = result.table
        spine = t["tag"] == SPINE_TAG
        if sink and (~spine).any():
            v = t.loc[~spine, "peak_closed_mV"]
            mu, sd = float(v.mean()), float(v.std(ddof=0))
            ax.axhspan(mu - sd, mu + sd, color=st["sink"], alpha=0.25, lw=0, zorder=1,
                       label=f"sink {mu:.2f} mV ± {sd * 1e3:.2g} µV (1σ)")
            ax.axhline(mu, color=st["sink"], lw=1.0, alpha=0.8, zorder=1)
        ax.scatter(t.loc[spine, "geodesic_um"], t.loc[spine, "peak_cut_mV"], s=26,
                   facecolors="white", edgecolors=st["cut"], linewidths=1.2, zorder=2,
                   label="loops cut (SWC tree)")
        ax.scatter(t.loc[spine, "geodesic_um"], t.loc[spine, "peak_closed_mV"], s=26,
                   color=st["closed"], edgecolors="white", linewidths=0.5, zorder=3, label="loops closed")
        x_max = float(t.loc[spine, "geodesic_um"].max())
        ax.set_xlim(-x_pad * x_max, (1 + x_pad) * x_max)
        ax.set_xlabel("geodesic distance from synapse (µm)")
        ax.set_ylabel("peak depolarization (mV)")
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color="#E3E3E3", zorder=0)
        if legend:
            ax.legend(frameon=False, fontsize=st["size"] * 0.8, loc="upper right")
        loops = result.summary["loops"]
        ax.set_title(title if title is not None else f"{result.name} ({loops} loop{'s' if loops != 1 else ''})")
    return ax
