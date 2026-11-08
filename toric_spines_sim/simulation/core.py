"""TSSimulator: build a toric-spine model and run it with loops closed.

Solvers
-------
``"auto"`` (default)
    ``"reference"`` for passive models, ``"jaxley"`` when HH is painted.
``"jaxley"``
    jaxley 0.14 with the SWC's ``# CYCLE_BREAK`` / ``# MULTI_NECK`` loops
    closed inside its sparse implicit solve (``simulation.jaxley_backend``).
    Passive membrane, all registry synapses, and Hodgkin-Huxley channels on
    ``hh_tags``; differentiable. Compiled once per model structure and cached.
``"reference"``
    scipy implicit Euler on the same graph (``simulation.reference``), with
    one LU factorization and a low-rank synapse update per step. Passive only.

Both solvers share geometry, membrane, synapse placement and synaptic
kinetics (``model.synaptic``) and agree to ~1e-5 mV on passive models. On
CPU the reference solver is ~6-8x faster for passive models (TS3, 1 s
simulated: 2.9 s vs 20 s; ``scripts/diagnostics/jaxley_timing.py``), hence
the ``"auto"`` default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pynapple as nap
from jscip import ParameterSet

from toric_spines_sim.geometry.swc import get_center_coordinates_for_all_segments
from toric_spines_sim.model import BuiltModel, TSModel
from toric_spines_sim.model.synapse import SynapsePopulation

from .results import SimulationResults

logger = logging.getLogger(__name__)

Events = Union[nap.TsGroup, Mapping[str, Sequence[float]], None]


class TSSimulator:
    """Run a toric-spine SWC with synapses from a points file.

    Synapses are built from the points file (``syn_0``, ``syn_1``, ... in file
    order) with ``synapse_model`` (default AMPA). For mixed receptor types,
    build a :class:`~toric_spines_sim.model.TSModel` yourself.

    Event channels are mapped by **TsGroup index** to that synapse order (a
    dict maps synapse labels to times in ms). Generators emit axon-order
    channels (``A0S0``, ...); call ``remap_axon_channel_events_to_synapses``
    first if axon assignments are not already point-file order.

    Examples
    --------
    >>> from toric_spines_sim.simulation import TSSimulator, make_icx_parameter_bank
    >>> p = make_icx_parameter_bank("in_vitro").sample()  # doctest: +SKIP
    >>> sim = TSSimulator("data/swc/microns/TS1_wsink_r10um.swc",
    ...                   "data/pointsets/microns/TS1_synpts.txt",
    ...                   {"syn_0": [10.0]}, p)  # doctest: +SKIP
    >>> results = sim.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        swc_filepath: Union[str, Path],
        synpts_filepath: Union[str, Path],
        events: Events,
        parameters: ParameterSet,
        record_points: Union[Literal["all"], Dict[str, Tuple[float, float, float]]] = "all",
        solver: Literal["auto", "jaxley", "reference"] = "auto",
        synapse_model: str = "ampa",
        restore_loops: bool = True,
    ):
        if solver not in ("auto", "jaxley", "reference"):
            raise ValueError(f"solver must be 'auto', 'jaxley' or 'reference', got {solver!r}")
        self._swc_filepath = Path(swc_filepath)
        self._synpts_filepath = Path(synpts_filepath)
        self._events = events
        self._parameters = parameters
        self._record_points_spec = record_points
        self._solver = solver
        self._synapse_model = synapse_model
        self._restore_loops = restore_loops

        self._synapses: Optional[Dict] = None
        self._record_points_resolved: Optional[Dict[str, Tuple[float, float, float]]] = None
        self._built: Optional[BuiltModel] = None
        logger.info(
            "Initialized TSSimulator: swc=%s, synpts=%s, solver=%s, record_points=%s",
            self._swc_filepath, self._synpts_filepath, solver,
            record_points if record_points != "all" else "all",
        )

    # ------------------------------------------------------------------
    @property
    def swc_filepath(self) -> Path:
        return self._swc_filepath

    @property
    def synpts_filepath(self) -> Path:
        return self._synpts_filepath

    @property
    def parameters(self) -> ParameterSet:
        return self._parameters

    @property
    def solver(self) -> str:
        """Resolved solver: ``"jaxley"`` or ``"reference"``."""
        if self._solver != "auto":
            return self._solver
        from toric_spines_sim.model import hh_spec_from_parameters

        return "jaxley" if hh_spec_from_parameters(self._parameters) is not None else "reference"

    @property
    def record_points_spec(self):
        return self._record_points_spec

    @property
    def events(self) -> Events:
        return self._events

    # ------------------------------------------------------------------
    def build_synapses(self) -> Dict:
        if self._synapses is None:
            self._synapses = SynapsePopulation.from_file(
                self._synpts_filepath, model=self._synapse_model, global_parameters=self._parameters
            ).synapses
        return self._synapses

    def build_record_points(self) -> Dict[str, Tuple[float, float, float]]:
        if self._record_points_resolved is None:
            if self._record_points_spec == "all":
                self._record_points_resolved = get_center_coordinates_for_all_segments(self._swc_filepath)
            else:
                self._record_points_resolved = dict(self._record_points_spec)
        return self._record_points_resolved

    def build_model(self) -> BuiltModel:
        if self._built is None:
            self._built = TSModel(
                swc_path=self._swc_filepath,
                synapses=self.build_synapses(),
                record_points=self.build_record_points(),
                parameters=self._parameters,
                restore_loops=self._restore_loops,
            ).build()
        return self._built

    @property
    def synapses(self) -> Dict:
        return self.build_synapses()

    @property
    def record_points(self) -> Dict[str, Tuple[float, float, float]]:
        return self.build_record_points()

    @property
    def model(self) -> BuiltModel:
        return self.build_model()

    @property
    def graph(self):
        return self.build_model().graph

    @property
    def morphology(self):
        return self.build_model().morphology

    @property
    def segment_tree(self):
        """The :class:`~toric_spines_sim.geometry.morphology.SegmentTable` (scaled radii)."""
        return self.build_model().segments

    @property
    def reconnects(self) -> List[Tuple[int, int]]:
        return self.build_model().reconnects

    # ------------------------------------------------------------------
    def _events_by_index(self, events: Events) -> Dict[int, List[float]]:
        labels = self.build_model().synapse_labels
        if events is None:
            return {}
        if isinstance(events, nap.TsGroup):
            out = {}
            for idx, ts in events.items():
                if idx < len(labels):
                    out[int(idx)] = (ts.index.values * 1000.0).tolist()  # pynapple stores s
                else:
                    logger.warning("TsGroup index %s out of range for %d synapses", idx, len(labels))
            return out
        index = {lab: i for i, lab in enumerate(labels)}
        unknown = set(events) - set(index)
        if unknown:
            raise ValueError(f"events for unknown synapses: {sorted(unknown)[:5]}")
        return {index[lab]: [float(t) for t in times] for lab, times in events.items()}

    def events_ms(self, events: Events = None) -> Dict[str, List[float]]:
        """Event times (ms) each synapse label will receive (default: constructor events).

        A TsGroup maps stream index ``i`` to ``syn_i``; pynapple stores seconds,
        so times are converted with ``* 1000``. Missing streams are silent.
        """
        labels = self.build_model().synapse_labels
        by_index = self._events_by_index(self._events if events is None else events)
        return {labels[i]: by_index.get(i, []) for i in range(len(labels))}

    def _record_every(self) -> int:
        dt = float(self._parameters["dt_sim_ms"])
        return max(1, int(round(float(self._parameters["dt_record_ms"]) / dt)))

    def _record_columns(self) -> Tuple[List[str], List[int]]:
        built = self.build_model()
        labels = sorted(built.record_nodes)
        return labels, [built.record_nodes[lab] for lab in labels]

    def _jaxley_sim(self):
        from . import jaxley_backend

        built = self.build_model()
        _labels, nodes = self._record_columns()
        p = self._parameters
        return jaxley_backend.get_simulation(
            built.graph, T_ms=float(p["T_ms"]), dt_ms=float(p["dt_sim_ms"]),
            v_rest_mV=float(p["Vrest_mV"]), record_nodes=nodes, hh=built.hh,
        )

    def _simulate(self, events_list: Sequence[Dict[int, List[float]]]) -> List[np.ndarray]:
        """Full-resolution traces ``V[time, column]`` for each event set."""
        built = self.build_model()
        p = self._parameters
        _labels, nodes = self._record_columns()
        if self.solver == "reference":
            if built.hh is not None:
                raise ValueError("solver='reference' is passive-only; hh_tags/hh_scale paint HH")
            out = []
            for ev in events_list:
                res = built.graph.run(ev, T_ms=float(p["T_ms"]), dt_ms=float(p["dt_sim_ms"]),
                                      v0_mV=float(p["Vrest_mV"]))
                out.append(res.V[:, nodes])
            return out
        sim = self._jaxley_sim()
        if len(events_list) == 1:
            return [sim.run(events_list[0])]
        return list(sim.run_batch(list(events_list)))

    def _package(self, v_full: np.ndarray, events: Events) -> SimulationResults:
        built = self.build_model()
        labels, _nodes = self._record_columns()
        p = self._parameters
        every = self._record_every()
        dt = float(p["dt_sim_ms"])
        t_full = np.arange(v_full.shape[0]) * dt
        keep = np.arange(0, v_full.shape[0], every)
        keep = keep[t_full[keep] < float(p["T_ms"]) - 1e-9]
        voltage = nap.TsdFrame(t=t_full[keep], d=v_full[keep], time_units="ms", columns=labels)
        return SimulationResults({
            "voltage_traces": voltage,
            "input_events": events,
            "synapses": self.build_synapses(),
            "reconnects": built.reconnects,
            "record_points": self.build_record_points(),
            "segment_tree": built.segments,
            "morphology": built.morphology,
            "graph": built.graph,
            "solver": self.solver,
            "swc_filepath": self._swc_filepath,
            "synpts_filepath": self._synpts_filepath,
        })

    def run(self) -> SimulationResults:
        """Simulate with the events given at construction."""
        logger.info("Running %s: T=%s ms, dt=%s ms", self.solver, self._parameters["T_ms"],
                    self._parameters["dt_sim_ms"])
        v = self._simulate([self._events_by_index(self._events)])[0]
        return self._package(v, self._events)

    def run_batch(self, events_list: Sequence[Events]) -> List[SimulationResults]:
        """Simulate several event sets on the same model (one jaxley call)."""
        evs = [self._events_by_index(e) for e in events_list]
        return [self._package(v, e) for v, e in zip(self._simulate(evs), events_list)]
