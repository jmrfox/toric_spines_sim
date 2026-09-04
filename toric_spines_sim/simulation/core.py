"""Core simulation execution class."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Literal, Optional

logger = logging.getLogger(__name__)

import arbor as A
from jscip import ParameterSet
import pynapple as nap
import numpy as np

from toric_spines_sim.model import TSModel, TSRecipe
from toric_spines_sim.model.synapse import SynapsePopulation
from toric_spines_sim.model.gj import prepare_gap_junctions
from toric_spines_sim.geometry.swc import get_center_coordinates_for_all_segments

from .results import SimulationResults


class TSSimulator:
    """Toric Spines Simulator for Arbor simulations of toric spine morphologies.

    All configuration is provided at construction time. Intermediate pipeline
    objects (synapses, cell, recipe, etc.) are built lazily on first use and
    cached for the lifetime of the instance. ``run()`` may be called multiple
    times; each call reuses cached pipeline objects and creates a fresh Arbor
    simulation.

    When using stochastic event generators, pass ``parameters["seed"]`` to the
    generator so that repeated ``run()`` calls with the same instance are
    deterministic.

    Example:
        >>> from toric_spines_sim.simulation import TSSimulator
        >>> from toric_spines_sim.simulation import make_default_parameter_bank
        >>> from toric_spines_sim.events import StochasticEventGenerator, FlatRateCurve
        >>>
        >>> pb = make_default_parameter_bank()
        >>> parameters = pb.sample()
        >>> events = StochasticEventGenerator(
        ...     rate_curves=[FlatRateCurve(rate_hz=50.0)],
        ...     n_synapses_per_axon=[25],
        ...     T_ms=parameters["T_ms"],
        ...     seed=int(parameters["seed"]),
        ... ).generate()
        >>> sim = TSSimulator(
        ...     "data/swc/microns/TS1_wsink_r10um.swc",
        ...     "data/pointsets/microns/TS1_synpts.txt",
        ...     events,
        ...     parameters,
        ... )
        >>> results = sim.run()
    """

    def __init__(
        self,
        swc_filepath: Union[str, Path],
        synpts_filepath: Union[str, Path],
        events: nap.TsGroup,
        parameters: ParameterSet,
        record_points: Union[
            Literal["all"], Dict[str, Tuple[float, float, float]]
        ] = "all",
    ):
        self._swc_filepath = Path(swc_filepath)
        self._synpts_filepath = Path(synpts_filepath)
        self._events = events
        self._parameters = parameters
        self._record_points_spec = record_points

        # Lazy pipeline caches (built once per instance)
        self._synapses: Optional[Dict] = None
        self._gap_junctions: Optional[Dict] = None
        self._record_points_resolved: Optional[
            Dict[str, Tuple[float, float, float]]
        ] = None
        self._build_cell_results: Optional[Dict] = None
        self._recipe: Optional[TSRecipe] = None

        logger.info(
            "Initialized TSSimulator: swc=%s, synpts=%s, events=%d streams, record_points=%s",
            self._swc_filepath,
            self._synpts_filepath,
            len(events),
            record_points if record_points != "all" else "all",
        )

    # -------------------------------------------------------------------------
    # Read-only configuration properties
    # -------------------------------------------------------------------------

    @property
    def swc_filepath(self) -> Path:
        """Path to the SWC morphology file."""
        return self._swc_filepath

    @property
    def synpts_filepath(self) -> Path:
        """Path to the synapse points file."""
        return self._synpts_filepath

    @property
    def parameters(self) -> ParameterSet:
        """Sampled simulation parameters."""
        return self._parameters

    @property
    def record_points_spec(
        self,
    ) -> Union[Literal["all"], Dict[str, Tuple[float, float, float]]]:
        """Recording point specification passed at construction."""
        return self._record_points_spec

    # -------------------------------------------------------------------------
    # Building methods (build what's needed, reuse what's cached)
    # -------------------------------------------------------------------------

    def build_synapses(self) -> Dict:
        """Build synapses and gap junctions from morphology files.

        Returns:
            Dictionary of synapse specifications.
        """
        if self._synapses is not None:
            return self._synapses

        logger.info("Building synapses and gap junctions...")
        self._synapses = SynapsePopulation.from_file(
            self._synpts_filepath,
            model="ampa",
            global_parameters=self._parameters,
        ).synapses
        self._gap_junctions = prepare_gap_junctions(
            self._swc_filepath, parameters=self._parameters
        )
        logger.info(
            "Built %d synapses and %d gap junctions",
            len(self._synapses),
            len(self._gap_junctions),
        )

        return self._synapses

    def build_record_points(self) -> Dict[str, Tuple[float, float, float]]:
        """Resolve record points (either "all" or explicit dict).

        Returns:
            Dictionary mapping probe labels to (x, y, z) coordinates.
        """
        if self._record_points_resolved is not None:
            return self._record_points_resolved

        if self._record_points_spec == "all":
            logger.info("Resolving record points from all segment centers")
            self._record_points_resolved = get_center_coordinates_for_all_segments(
                self._swc_filepath
            )
            logger.info(
                "Resolved %d recording points",
                len(self._record_points_resolved),
            )
        else:
            self._record_points_resolved = self._record_points_spec
            logger.info(
                "Using %d specified recording points",
                len(self._record_points_resolved),
            )

        return self._record_points_resolved

    def build_events(self) -> nap.TsGroup:
        """Return input events.

        Returns:
            TsGroup mapping stream indices to Ts objects (timestamps in ms).
        """
        return self._events

    def build_cell(self) -> A.cable_cell:
        """Build the Arbor cable cell.

        Returns:
            The built Arbor cable_cell.
        """
        if self._build_cell_results is not None:
            return self._build_cell_results["cell"]

        synapses = self.build_synapses()
        gap_junctions = self._gap_junctions
        record_points = self.build_record_points()

        logger.info("Building cell morphology...")
        tsm = TSModel(
            swc_path=self._swc_filepath,
            synapses=synapses,
            gap_junctions=gap_junctions,
            record_points=record_points,
            parameters=self._parameters,
        )
        self._build_cell_results = tsm.build_cell()
        logger.info("Cell built successfully")

        return self._build_cell_results["cell"]

    def build_recipe(self) -> TSRecipe:
        """Build the Arbor recipe.

        Returns:
            The built TSRecipe.
        """
        if self._recipe is not None:
            return self._recipe

        cell = self.build_cell()
        synapses = self.build_synapses()
        gap_junctions = self._gap_junctions
        record_points = self.build_record_points()
        events = self.build_events()

        logger.info("Creating Arbor recipe...")
        self._recipe = TSRecipe(
            cell,
            synapses=synapses,
            gap_junctions=gap_junctions,
            record_points=record_points,
            events=events,
            parameters=self._parameters,
            custom_catalogue=self._build_cell_results["custom_catalogue"],
        )

        return self._recipe

    # -------------------------------------------------------------------------
    # Properties for accessing intermediate objects
    # -------------------------------------------------------------------------

    @property
    def synapses(self) -> Dict:
        """Synapse specifications (builds if needed)."""
        if self._synapses is None:
            self.build_synapses()
        return self._synapses

    @property
    def gap_junctions(self) -> Dict:
        """Gap junction specifications (builds synapses if needed)."""
        if self._gap_junctions is None:
            self.build_synapses()
        return self._gap_junctions

    @property
    def record_points(self) -> Dict[str, Tuple[float, float, float]]:
        """Resolved record points (builds if needed)."""
        if self._record_points_resolved is None:
            self.build_record_points()
        return self._record_points_resolved

    @property
    def events(self) -> nap.TsGroup:
        """Input event timestamps."""
        return self._events

    @property
    def cell(self) -> A.cable_cell:
        """The built Arbor cable_cell (builds if needed)."""
        if self._build_cell_results is None:
            self.build_cell()
        return self._build_cell_results["cell"]

    @property
    def morphology(self) -> A.morphology:
        """The cell morphology (builds cell if needed)."""
        if self._build_cell_results is None:
            self.build_cell()
        return self._build_cell_results["morphology"]

    @property
    def segment_tree(self) -> A.segment_tree:
        """The segment tree (builds cell if needed)."""
        if self._build_cell_results is None:
            self.build_cell()
        return self._build_cell_results["segment_tree"]

    @property
    def decor(self):
        """The cell decor (builds cell if needed)."""
        if self._build_cell_results is None:
            self.build_cell()
        return self._build_cell_results["decor"]

    @property
    def labels(self):
        """The label dictionary (builds cell if needed)."""
        if self._build_cell_results is None:
            self.build_cell()
        return self._build_cell_results["labels"]

    @property
    def cvp(self):
        """The control volume policy (builds cell if needed)."""
        if self._build_cell_results is None:
            self.build_cell()
        return self._build_cell_results["cvp"]

    def write_cell(self, filename: str) -> None:
        """Write the cable cell to file using Arbor's write_component.

        Args:
            filename: Path to the output file (should have .acc extension)
        """
        A.write_component(self.cell, filename)
        logger.info("Wrote cable cell to %s", filename)

    @property
    def recipe(self) -> TSRecipe:
        """The Arbor recipe (builds if needed)."""
        if self._recipe is None:
            self.build_recipe()
        return self._recipe

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def run(self) -> SimulationResults:
        """Run the simulation and return results.

        May be called multiple times on the same instance. Pipeline objects
        are reused; a fresh Arbor simulation is created on each call.

        Returns:
            SimulationResults containing all simulation data.
        """
        recipe = self.build_recipe()
        record_points = self.build_record_points()
        events = self.build_events()

        logger.info("Setting up Arbor simulation...")
        ctx = A.context()
        dec = A.partition_load_balance(recipe, ctx)
        sim = A.simulation(recipe, ctx, dec)
        logger.info("Simulation context: %s", ctx)

        logger.info("Setting up %d voltage probes...", len(record_points))
        probe_handles = {}
        dt_record_ms = self._parameters["dt_record_ms"]
        for probe_label in record_points.keys():
            probe_id = f"v_{probe_label}"
            handle = sim.sample(
                0,
                probe_id,
                A.regular_schedule(dt_record_ms * A.units.ms),
            )
            probe_handles[probe_label] = handle
        logger.info("Probes recording at dt=%f ms", dt_record_ms)

        sim.record(A.spike_recording.all)
        logger.info(
            "Running simulation: T=%f ms, dt=%f ms...",
            self._parameters["T_ms"],
            self._parameters["dt_sim_ms"],
        )

        sim.run(
            self._parameters["T_ms"] * A.units.ms,
            self._parameters["dt_sim_ms"] * A.units.ms,
        )
        logger.info("Simulation completed")

        logger.info("Collecting voltage data from probes...")
        voltage_data_raw = {}
        for probe_label, handle in probe_handles.items():
            voltage_data_raw[probe_label] = sim.samples(handle)
        logger.info("Collected data from %d probes", len(voltage_data_raw))

        data_dict = {}
        probe_times = None
        for probe_label, samples in voltage_data_raw.items():
            if len(samples) > 0:
                data_array, location = samples[0]
                if probe_times is None:
                    probe_times = data_array[:, 0]
                data_dict[probe_label] = data_array[:, 1]

        if probe_times is not None and len(data_dict) > 0:
            voltage_tsdframe = nap.TsdFrame(
                t=probe_times,
                d=np.column_stack([data_dict[k] for k in sorted(data_dict.keys())]),
                time_units="ms",
                columns=sorted(data_dict.keys()),
            )
        else:
            voltage_tsdframe = nap.TsdFrame(
                t=[],
                d=np.array([]).reshape(0, len(record_points)),
                time_units="ms",
                columns=list(record_points.keys()),
            )

        logger.info("Created voltage TsdFrame with shape %s", voltage_tsdframe.shape)

        simulation_results_dict = {
            "voltage_traces": voltage_tsdframe,
            "input_events": events,
            "synapses": self._synapses,
            "gap_junctions": self._gap_junctions,
            "record_points": record_points,
            "cell": self._build_cell_results["cell"],
            "morphology": self._build_cell_results["morphology"],
            "segment_tree": self._build_cell_results["segment_tree"],
            "decor": self._build_cell_results["decor"],
            "labels": self._build_cell_results["labels"],
            "cvp": self._build_cell_results["cvp"],
            "swc_filepath": self._swc_filepath,
            "synpts_filepath": self._synpts_filepath,
        }

        logger.info("Simulation complete")
        return SimulationResults(simulation_results_dict)
