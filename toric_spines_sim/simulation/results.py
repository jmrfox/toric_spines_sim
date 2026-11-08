"""Simulation results container with analysis methods."""

import logging
import dill
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal, Optional, Set

logger = logging.getLogger(__name__)

import numpy as np
import pynapple as nap


class SimulationResults:
    """
    Container for simulation results with analysis methods.

    This class wraps the raw simulation results and provides
    convenient methods for analyzing voltage traces, including integration
    by segment tag with optional surface-area weighting.

    Attributes:
        voltage_traces: TsdFrame with voltage data (rows=time, columns=probes)
        input_events: TsGroup mapping stream indices to event timestamps
        synapses: Dict of synapse specifications (``SynapsePoint``)
        reconnects: SWC reconnect pairs closed as loops (``# CYCLE_BREAK`` / ``# MULTI_NECK``)
        record_points: Dict mapping probe labels to (x, y, z) coordinates
        segment_tree: ``geometry.morphology.SegmentTable`` (one frustum per SWC sample)
        morphology: ``geometry.morphology.Morphology`` (branches, for plotting)
        graph: ``simulation.reference.LoopCableModel`` that was simulated
        solver: ``"jaxley"`` or ``"reference"``
        swc_filepath: Path to SWC morphology file
        synpts_filepath: Path to synapse points file
    """

    def __init__(self, results_dict: Dict):
        """Initialize from a simulation results dictionary."""
        self.voltage_traces = results_dict["voltage_traces"]
        self.input_events = results_dict["input_events"]
        self.synapses = results_dict.get("synapses", {})
        self.reconnects = results_dict.get("reconnects", [])
        self.record_points = results_dict["record_points"]
        self.segment_tree = results_dict.get("segment_tree")
        self.morphology = results_dict.get("morphology")
        self.graph = results_dict.get("graph")
        self.swc_filepath = results_dict.get("swc_filepath")
        self.synpts_filepath = results_dict.get("synpts_filepath")
        self.solver = results_dict.get("solver")
        if self.segment_tree is None and self.swc_filepath and Path(self.swc_filepath).exists():
            from toric_spines_sim.geometry.morphology import Morphology, SegmentTable

            self.segment_tree = SegmentTable.from_swc(self.swc_filepath)
            self.morphology = Morphology(self.segment_tree)

        # Cache for segment properties
        self._segment_tags_cache = None
        self._segment_areas_cache = None
        self._probe_to_segment_cache = None

    def _get_segment_tags(self) -> Dict[int, int]:
        """Get mapping from segment index to tag."""
        if self._segment_tags_cache is None:
            self._segment_tags_cache = {}
            for i, seg in enumerate(self.segment_tree.segments):
                self._segment_tags_cache[i] = seg.tag
        return self._segment_tags_cache

    def _get_segment_surface_areas(self) -> Dict[int, float]:
        """
        Calculate surface area for each segment (frustum).

        Surface area of a frustum: π * (r1 + r2) * sqrt((r1 - r2)^2 + h^2)
        where r1, r2 are the radii at the two ends and h is the length.
        """
        if self._segment_areas_cache is None:
            self._segment_areas_cache = {}
            for i, seg in enumerate(self.segment_tree.segments):
                p, d = seg.prox, seg.dist
                r1 = p.radius
                r2 = d.radius

                # Calculate length
                dx = d.x - p.x
                dy = d.y - p.y
                dz = d.z - p.z
                length = np.sqrt(dx**2 + dy**2 + dz**2)

                # Frustum surface area (lateral surface only, excluding caps)
                slant_height = np.sqrt((r1 - r2) ** 2 + length**2)
                area = np.pi * (r1 + r2) * slant_height

                self._segment_areas_cache[i] = area
        return self._segment_areas_cache

    def _map_probes_to_segments(self) -> Dict[str, int]:
        """
        Map probe labels to their nearest segment indices.

        This uses the probe coordinates to find the closest segment center.
        """
        if self._probe_to_segment_cache is None:
            self._probe_to_segment_cache = {}

            # Calculate segment centers
            segment_centers = []
            for seg in self.segment_tree.segments:
                p, d = seg.prox, seg.dist
                cx = (p.x + d.x) / 2
                cy = (p.y + d.y) / 2
                cz = (p.z + d.z) / 2
                segment_centers.append((cx, cy, cz))

            # Map each probe to nearest segment
            for probe_label, (px, py, pz) in self.record_points.items():
                min_dist = float("inf")
                nearest_seg = 0

                for seg_idx, (cx, cy, cz) in enumerate(segment_centers):
                    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_seg = seg_idx

                self._probe_to_segment_cache[probe_label] = nearest_seg

        return self._probe_to_segment_cache

    def integrate_voltages_by_tag(
        self,
        tags: Union[int, List[int], Set[int]],
        method: Literal["average", "surface_weighted"] = "average",
    ) -> nap.Tsd:
        """
        Integrate voltage traces from all segments with specified tag(s).

        Args:
            tags: Single tag or collection of tags to integrate over.
            method: Integration method:
                - "average": Simple arithmetic mean of all voltage traces
                - "surface_weighted": Weighted average by segment surface area

        Returns:
            Pynapple Tsd with time in seconds and integrated voltage in mV.

        Example:
            >>> results = SimulationResults(raw_results)
            >>> # Get average voltage across all spine segments (tag 3)
            >>> v_spine = results.integrate_voltages_by_tag(3, method="average")
            >>>
            >>> # Get surface-weighted voltage across sink segments (tag 5)
            >>> v_sink = results.integrate_voltages_by_tag(5, method="surface_weighted")
            >>>
            >>> # Combine multiple tags
            >>> v_combined = results.integrate_voltages_by_tag([3, 5])
        """
        # Normalize tags to a set
        if isinstance(tags, int):
            tags_set = {tags}
        else:
            tags_set = set(tags)

        # Get mappings
        segment_tags = self._get_segment_tags()
        probe_to_segment = self._map_probes_to_segments()

        # Find probes that correspond to segments with the target tags
        matching_probes = []
        for probe_label, seg_idx in probe_to_segment.items():
            if segment_tags.get(seg_idx) in tags_set:
                matching_probes.append((probe_label, seg_idx))

        if not matching_probes:
            raise ValueError(f"No segments found with tag(s) {tags_set}")

        logger.info(
            f"Integrating {len(matching_probes)} voltage traces for tag(s) {tags_set}"
        )

        # Extract matching column names from TsdFrame
        probe_labels = [p[0] for p in matching_probes]

        # Check which probes exist in the voltage_traces
        available_probes = [p for p in probe_labels if p in self.voltage_traces.columns]
        if not available_probes:
            raise ValueError(f"No voltage data found for tag(s) {tags_set}")

        # Get subset of TsdFrame
        voltage_subset = self.voltage_traces[available_probes]

        # Get time array from TsdFrame index
        time = voltage_subset.index.values

        # Integrate based on method
        if method == "average":
            # Simple average using TsdFrame.mean(axis=1)
            integrated_voltage = voltage_subset.mean(axis=1).values
            logger.debug(f"Computed simple average of {len(available_probes)} traces")

        elif method == "surface_weighted":
            # Weighted average by surface area
            segment_areas = self._get_segment_surface_areas()

            # Map probe labels to their segment indices
            probe_to_seg = {p: s for p, s in matching_probes}

            weights = []
            for probe_label in available_probes:
                seg_idx = probe_to_seg[probe_label]
                weights.append(segment_areas[seg_idx])

            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

            # Weighted sum: multiply each column by its weight and sum
            weighted_data = voltage_subset.values * weights[np.newaxis, :]
            integrated_voltage = weighted_data.sum(axis=1)

            logger.debug(
                f"Computed surface-weighted average of {len(available_probes)} traces"
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'average' or 'surface_weighted'"
            )

        # Return as pynapple Tsd (time in seconds, voltage in mV)
        return nap.Tsd(t=time, d=integrated_voltage, time_units="s")

    def get_tags(self) -> Set[int]:
        """Get all unique tags present in the morphology."""
        return set(self._get_segment_tags().values())

    def get_segments_by_tag(self, tag: int) -> List[int]:
        """Get list of segment indices with the specified tag."""
        segment_tags = self._get_segment_tags()
        return [seg_idx for seg_idx, seg_tag in segment_tags.items() if seg_tag == tag]

    def to_dict(self) -> Dict:
        """Convert back to a dictionary (for saving)."""
        return {
            "voltage_traces": self.voltage_traces,
            "input_events": self.input_events,
            "synapses": self.synapses,
            "reconnects": self.reconnects,
            "record_points": self.record_points,
            "morphology": self.morphology,
            "segment_tree": self.segment_tree,
            "graph": self.graph,
            "solver": self.solver,
            "swc_filepath": self.swc_filepath,
            "synpts_filepath": self.synpts_filepath,
        }

    def to_serializable_dict(self) -> Dict:
        """Convert to a dictionary containing only serializable data.

        Geometry (segment table, morphology, graph) is not stored; ``load``
        rebuilds the segment table from ``swc_filepath``.
        """
        # Convert TsdFrame to serializable format
        # Pynapple stores times internally in seconds; retrieve as ms
        times_ms = self.voltage_traces.as_units("ms").index.values
        serializable_voltages = {
            "t": times_ms.tolist(),
            "d": self.voltage_traces.values.tolist(),
            "columns": list(self.voltage_traces.columns),
            "time_units": "ms",
        }

        # Convert TsGroup to serializable format
        serializable_events = {}
        for idx, ts in self.input_events.items():
            label = (
                self.input_events.get_info("label")[idx]
                if "label" in self.input_events.metadata
                else str(idx)
            )
            # Retrieve event times in ms
            times_ms = ts.as_units("ms").index.values
            serializable_events[idx] = {
                "t": times_ms.tolist(),
                "label": label,
                "time_units": "ms",
            }

        return {
            "voltage_traces": serializable_voltages,
            "input_events": serializable_events,
            "record_points": self.record_points,
            "synapses": self.synapses,
            "reconnects": list(self.reconnects),
            "solver": self.solver,
            "swc_filepath": str(self.swc_filepath) if self.swc_filepath else None,
            "synpts_filepath": (
                str(self.synpts_filepath) if self.synpts_filepath else None
            ),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save simulation results to a pickle file.

        Saves voltages, events, record points, synapses and metadata. Geometry
        is rebuilt from ``swc_filepath`` on :meth:`load`.

        Args:
            filepath: Path where to save the results (.pkl file).

        Example:
            >>> results.save("simulations/ts1/results.pkl")
        """
        filepath = Path(filepath)
        logger.info(f"Saving simulation results to {filepath}")

        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save traces, events, record points and synapses; geometry is rebuilt on load
            serializable_data = self.to_serializable_dict()

            with open(filepath, "wb") as f:
                dill.dump(serializable_data, f)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Results saved successfully ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SimulationResults":
        """
        Load simulation results from a pickle file.

        The segment table and morphology are rebuilt from ``swc_filepath``
        when that file exists, so ``integrate_voltages_by_tag`` works on
        loaded results. The simulated graph is not stored.

        Args:
            filepath: Path to the saved results file (.pkl).

        Returns:
            SimulationResults instance with limited functionality.

        Raises:
            FileNotFoundError: If the file does not exist.

        Example:
            >>> results = SimulationResults.load("simulations/ts1/results.pkl")
            >>> voltage_trace = results.voltage_traces["probe_seg_0"]
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Simulation results file not found: {filepath}")

        logger.info(f"Loading simulation results from {filepath}")

        try:
            with open(filepath, "rb") as f:
                results_dict = dill.load(f)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Results loaded successfully ({file_size_mb:.2f} MB)")

            # Reconstruct TsdFrame from serializable format
            voltage_data = results_dict.get("voltage_traces", {})
            if isinstance(voltage_data, dict) and "t" in voltage_data:
                # Reconstruct from serialized format
                voltage_traces = nap.TsdFrame(
                    t=voltage_data["t"],
                    d=voltage_data["d"],
                    time_units=voltage_data.get("time_units", "ms"),
                    columns=voltage_data["columns"],
                )
            else:
                # Legacy format - empty TsdFrame
                voltage_traces = nap.TsdFrame(t=[], d=[], time_units="ms")

            # Reconstruct TsGroup from serializable format
            events_data = results_dict.get("input_events", {})
            if isinstance(events_data, dict):
                ts_dict = {}
                labels = []
                for idx, event_info in events_data.items():
                    if isinstance(event_info, dict):
                        ts_dict[int(idx)] = nap.Ts(
                            t=event_info["t"],
                            time_units=event_info.get("time_units", "ms"),
                        )
                        labels.append(event_info.get("label", str(idx)))
                    else:
                        # Legacy format: just a list of times
                        ts_dict[int(idx)] = nap.Ts(t=event_info, time_units="ms")
                        labels.append(str(idx))
                input_events = nap.TsGroup(ts_dict, label=labels)
            else:
                # Legacy format - empty TsGroup
                input_events = nap.TsGroup({})

            full_dict = {
                "voltage_traces": voltage_traces,
                "input_events": input_events,
                "synapses": results_dict.get("synapses", {}),
                "reconnects": results_dict.get("reconnects", []),
                "record_points": results_dict.get("record_points", {}),
                "solver": results_dict.get("solver"),
                "swc_filepath": results_dict.get("swc_filepath"),
                "synpts_filepath": results_dict.get("synpts_filepath"),
            }

            # Log some basic info about the loaded results
            logger.info(f"Loaded voltage TsdFrame with shape {voltage_traces.shape}")
            logger.info(f"Loaded events TsGroup with {len(input_events)} streams")
            if "swc_filepath" in results_dict:
                logger.info(f"SWC file: {results_dict['swc_filepath']}")
            if "synpts_filepath" in results_dict:
                logger.info(f"Synapse points file: {results_dict['synpts_filepath']}")

            return cls(full_dict)

        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
