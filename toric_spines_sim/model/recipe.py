"""Arbor recipe for toric spine models."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import arbor as A
from arbor import units as U
from jscip import ParameterSet
import pynapple as nap

from toric_spines_sim.model.synapse import SynapsePoint
from toric_spines_sim.model.gj import GapJunctionPoint


class TSRecipe(A.recipe):
    """Arbor recipe for toric spines models with per-synapse event streams.

    Parameters
    ----------
    cell : A.cable_cell
        The cell constructed from `TSMorphology.build_cell(...)`.
    events : TsGroup | Dict[str, List[float]] | None
        Optional TsGroup or dict mapping synapse labels to event times (ms).
        TsGroup is preferred; dict is supported for backward compatibility.
    """

    def __init__(
        self,
        cell: A.cable_cell,
        synapses: Dict[str, SynapsePoint],
        gap_junctions: Dict[str, GapJunctionPoint],
        record_points: Dict[str, Tuple[float, float, float]],
        events: Optional[Union[nap.TsGroup, Dict[str, List[float]]]] = None,
        parameters: Optional[ParameterSet] = None,
        custom_catalogue: Optional[A.catalogue] = None,
    ):
        super().__init__()
        self._cell = cell
        self._synapses = synapses
        self._gap_junctions = gap_junctions
        self._record_points = record_points
        self._parameters = parameters
        self._custom_catalogue = custom_catalogue

        # Convert TsGroup to dict if needed
        if isinstance(events, nap.TsGroup):
            self._events_ms = self._tsgroup_to_dict(events, list(synapses.keys()))
        else:
            self._events_ms = events

        # process cell parameters
        if self._parameters is not None:
            Vrest = self._parameters["Vrest_mV"] * U.mV
            tempK = self._parameters["temp_K"] * U.Kelvin
            cm = self._parameters["cm_uF_per_cm2"] * U.uF / U.cm2
            rL = self._parameters["rL_ohm_cm"] * U.Ohm * U.cm
        else:  # default values
            Vrest = -65 * U.mV
            tempK = 280 * U.Kelvin
            cm = 1.0 * U.uF / U.cm2
            rL = 35.4 * U.Ohm * U.cm

        # Global properties (passive defaults; catalog)
        self._gprop = A.cable_global_properties()
        self._gprop.catalogue = A.default_catalogue()

        # Extend with custom catalogue if provided 
        if self._custom_catalogue is not None:
            logger.info("Extending catalogue with custom mechanisms")
            self._gprop.catalogue.extend(self._custom_catalogue, "")

        self._gprop.set_property(Vm=Vrest, cm=cm, rL=rL, tempK=tempK)
        self._gprop.set_ion(
            "ca",
            valence=2,
            int_con=self._parameters["Ca_intcon_mM"] * U.mM,
            ext_con=self._parameters["Ca_extcon_mM"] * U.mM,
            rev_pot=self._parameters["Ca_revpot_mV"] * U.mV,
        )
        self._gprop.set_ion(
            "na",
            valence=1,
            int_con=self._parameters["Na_intcon_mM"] * U.mM,
            ext_con=self._parameters["Na_extcon_mM"] * U.mM,
            rev_pot=self._parameters["Na_revpot_mV"] * U.mV,
        )
        self._gprop.set_ion(
            "k",
            valence=1,
            int_con=self._parameters["K_intcon_mM"] * U.mM,
            ext_con=self._parameters["K_extcon_mM"] * U.mM,
            rev_pot=self._parameters["K_revpot_mV"] * U.mV,
        )
        logger.info(
            "Initialized TSRecipe with %d synapses, %d gap junctions, %d record points",
            len(self._synapses) if self._synapses else 0,
            len(self._gap_junctions) if self._gap_junctions else 0,
            len(self._record_points) if self._record_points else 0,
        )

        # Precompute explicit schedules per-synapse if provided
        self._evgens: List[A.event_generator] = []
        if self._events_ms is not None and self._synapses:
            syn_labels = list(self._synapses.keys())
            if len(self._events_ms) != len(syn_labels):
                logger.error(
                    "events length %d must match number of synapses %d",
                    len(self._events_ms),
                    len(syn_labels),
                )
                raise ValueError(
                    f"events length {len(self._events_ms)} must match number of synapses {len(syn_labels)}"
                )
            for syn_label, time_list in self._events_ms.items():
                times_quantities = [float(t) * U.ms for t in (time_list or [])]
                schedule = A.explicit_schedule(times_quantities)
                self._evgens.append(A.event_generator(syn_label, 1.0, schedule))  # gmax is set in the synapse mechanism parameters
            logger.debug("Created %d event generators", len(self._evgens))

    def _tsgroup_to_dict(
        self, tsgroup: nap.TsGroup, synapse_labels: List[str]
    ) -> Dict[str, List[float]]:
        """Convert TsGroup to dict mapping synapse labels to event times.

        Args:
            tsgroup: TsGroup with event timestamps
            synapse_labels: Ordered list of synapse labels to map indices to

        Returns:
            Dict mapping synapse label to list of event times in milliseconds
        """
        result = {}
        for idx, ts in tsgroup.items():
            if idx < len(synapse_labels):
                label = synapse_labels[idx]
                # Pynapple stores times in seconds, convert to milliseconds
                result[label] = (ts.index.values * 1000).tolist()
            else:
                logger.warning(f"TsGroup index {idx} out of range for synapse labels")
        return result

    # --- Arbor recipe API ---
    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self._cell

    def global_properties(self, kind):
        return self._gprop if kind == A.cell_kind.cable else None

    def event_generators(self, gid):
        return list(self._evgens) if gid == 0 else []

    def connections_on(self, gid):
        return []

    def probes(self, gid):
        if gid == 0:
            if self._record_points:
                probes = [
                    A.cable_probe_membrane_voltage(f'"{label}"', f"v_{label}")
                    for label in self._record_points.keys()
                ]
            else:
                probes = [A.cable_probe_membrane_voltage('"root"', "v_root")]
            logger.debug("Providing %d probes for gid=%s", len(probes), gid)
            return probes
        else:
            return []

    def gap_junctions_on(self, gid):
        connections = []
        if not self._gap_junctions:
            return connections
        for gj_label, gj in self._gap_junctions.items():
            a_label = gj_label + "_a"
            b_label = gj_label + "_b"
            weight = float(gj.weight or 1.0)
            peer_b = A.cell_global_label((gid, b_label))
            loc_a = A.cell_local_label(a_label, A.selection_policy.round_robin)
            connections.append(A.gap_junction_connection(peer_b, loc_a, weight))
            peer_a = A.cell_global_label((gid, a_label))
            loc_b = A.cell_local_label(b_label, A.selection_policy.round_robin)
            connections.append(A.gap_junction_connection(peer_a, loc_b, weight))
        logger.debug(
            "Providing %d gap junction connections for gid=%s", len(connections), gid
        )
        return connections
