"""Model description and build for toric spine simulations.

``TSModel`` holds what to simulate: an SWC (typically ``TS*_wsink_r*um.swc``),
synapses, record points and a sampled ``ParameterSet``. ``TSModel.build()``
turns it into a :class:`BuiltModel` that both solvers use:

- a :class:`~toric_spines_sim.geometry.morphology.SegmentTable` (sink / neck
  radius scaling applied) and its :class:`Morphology`;
- the cable graph (:class:`~toric_spines_sim.simulation.reference.LoopCableModel`)
  with ``# CYCLE_BREAK`` / ``# MULTI_NECK`` loops closed, passive membrane,
  the lumped sink when ``sink_lumped`` is set, and the synapses placed in
  label order;
- graph nodes for each record point, and an :class:`HHSpec` when
  ``hh_tags`` / ``hh_scale`` ask for Hodgkin-Huxley channels.

Nothing here depends on a simulator; see ``simulation.core.TSSimulator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from jscip import ParameterSet

from toric_spines_sim.geometry.morphology import Morphology, SegmentTable
from toric_spines_sim.geometry.sink import neck_point_from_swc_file
from toric_spines_sim.geometry.swc import parse_reconnect_pairs

from . import synaptic
from .synapse import SynapsePoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HHSpec:
    """Hodgkin-Huxley channels on frusta with the given tags (densities in S/cm²)."""

    tags: Tuple[int, ...]
    gna: float
    gk: float
    gl: float
    ena: float
    ek: float
    el: float


def lumped_sink_membrane(
    segments: SegmentTable,
    sink_tags: Sequence[int],
    cell_C_pF: float,
    cell_Rin_MOhm: float,
) -> dict:
    """Specific cm / g_pas that make the sink carry the whole cell's C and 1/Rin.

    The sink is a stand-in for the rest of the neuron. Its geometric area
    (~1,300 µm² for the 10 µm sink) is far below a real SSN membrane
    (~23,000 µm² from Sanculi whole-cell C / 0.9 µF/cm²), so its specific
    properties are scaled to match the measured totals.

    Returns ``{"cm": µF/cm², "g_pas": S/cm², "area_um2": µm²}``.
    """
    area_um2 = segments.lateral_area_by_tags(sink_tags)
    if area_um2 <= 0:
        raise ValueError(f"No sink membrane found for tags {list(sink_tags)}")
    area_cm2 = area_um2 * 1e-8
    cm = cell_C_pF * 1e-12 / area_cm2 * 1e6
    g = 1.0 / (cell_Rin_MOhm * 1e6) / area_cm2
    return {"cm": cm, "g_pas": g, "area_um2": area_um2}


def hh_spec_from_parameters(p: ParameterSet) -> Optional[HHSpec]:
    """HH spec from ``hh_tags`` / ``hh_scale`` (None when the cell is passive)."""
    tags = [int(t) for t in p["hh_tags"]]
    scale = float(p["hh_scale"]) if "hh_scale" in p.index else 0.0
    if not tags or scale <= 0:
        return None
    return HHSpec(
        tags=tuple(tags),
        gna=float(p["Na_gbar_S_per_cm2"]) * scale,
        gk=float(p["K_gbar_S_per_cm2"]) * scale,
        gl=float(p["hh_leak_g_S_per_cm2"]) * scale,
        ena=float(p["Na_revpot_mV"]),
        ek=float(p["K_revpot_mV"]),
        el=float(p["hh_leak_e_mV"]),
    )


@dataclass
class BuiltModel:
    """Everything a solver needs, built from a :class:`TSModel`."""

    segments: SegmentTable
    morphology: Morphology
    graph: object  # LoopCableModel
    synapse_labels: List[str]
    record_nodes: Dict[str, int]
    reconnects: List[Tuple[int, int]]
    membrane_overrides: Optional[Dict[int, Dict[str, float]]] = None
    hh: Optional[HHSpec] = None


@dataclass
class TSModel:
    """What to simulate.

    Attributes
    ----------
    swc_path : Path
        SWC morphology (``# CYCLE_BREAK`` / ``# MULTI_NECK`` headers are loops).
    synapses : dict[str, SynapsePoint]
        Synapses by label (``syn_0``, ...); event streams map to this order.
    record_points : dict[str, tuple]
        Record site labels to XYZ (µm, same frame as the SWC).
    parameters : ParameterSet
        Sampled parameters (``ParameterBank.sample()``).
    restore_loops : bool
        Close the SWC's loops (default). ``False`` simulates the cut tree.
    """

    swc_path: Path
    synapses: Dict[str, SynapsePoint]
    record_points: Dict[str, Tuple[float, float, float]]
    parameters: ParameterSet
    restore_loops: bool = True

    def build(self) -> BuiltModel:
        from toric_spines_sim.simulation.reference import LoopCableModel

        p = self.parameters
        swc = Path(self.swc_path)
        neck_xyz = None
        if float(p["neck_radius_scale"]) != 1.0:
            neck_xyz = neck_point_from_swc_file(swc)
        segments = SegmentTable.from_swc(
            swc,
            sink_radii_scale=float(p["sink_radii_scale"]),
            sink_tag=int(p["sink_tag"]),
            neck_radius_scale=float(p["neck_radius_scale"]),
            neck_xyz=neck_xyz,
        )
        overrides = None
        if "sink_lumped" in p.index and float(p["sink_lumped"]) > 0:
            sink_tags = [int(p["sink_tag"]), int(p["sink_tip_tag"])]
            mem = lumped_sink_membrane(segments, sink_tags, float(p["cell_C_pF"]),
                                       float(p["cell_Rin_MOhm"]))
            overrides = {t: {"cm": mem["cm"], "g_pas": mem["g_pas"]} for t in sink_tags}
            logger.info(
                "Lumped sink: area %.0f um2 -> cm=%.4g uF/cm2, g_pas=%.4g S/cm2 (C=%.0f pF, Rin=%.0f MOhm)",
                mem["area_um2"], mem["cm"], mem["g_pas"], p["cell_C_pF"], p["cell_Rin_MOhm"],
            )
        graph = LoopCableModel.from_swc(
            swc,
            dx_um=float(p["discretization_um"]),
            cm_uF_per_cm2=float(p["cm_uF_per_cm2"]),
            g_pas_S_per_cm2=float(p["pas_leak_g_S_per_cm2"]),
            e_pas_mV=float(p["pas_leak_e_mV"]),
            rL_ohm_cm=float(p["rL_ohm_cm"]),
            restore_loops=self.restore_loops,
            tag_overrides=overrides,
            segments=segments,
        )
        labels = list(self.synapses)
        for label in labels:
            syn = self.synapses[label]
            graph.add_components(syn.location, synaptic.components(syn.model, syn.synapse_params))
        record_nodes = {lab: graph.closest_node(xyz) for lab, xyz in self.record_points.items()}
        hh = hh_spec_from_parameters(p)
        if hh is not None and overrides is not None and set(hh.tags) & set(overrides):
            logger.warning(
                "HH on a lumped sink: channel densities apply to the geometric sink area, "
                "while capacitance and leak are lumped to the whole cell."
            )
        logger.info(
            "Built model %s: %d compartments, %d loops, %d synapses, %d record points",
            swc.name, graph.n_nodes, graph.n_loops, len(labels), len(record_nodes),
        )
        return BuiltModel(
            segments=segments,
            morphology=Morphology(segments),
            graph=graph,
            synapse_labels=labels,
            record_nodes=record_nodes,
            reconnects=parse_reconnect_pairs(swc) if self.restore_loops else [],
            membrane_overrides=overrides,
            hh=hh,
        )
