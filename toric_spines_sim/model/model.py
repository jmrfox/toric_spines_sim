"""Morphology specification and cell building for toric spine models."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from jscip import ParameterSet

import arbor as A
from arbor import units as U

from toric_spines_sim.geometry.rescale import (
    scale_radii_in_segment_tree_by_tag,
    scale_one_radius_in_segment_tree_by_coordinates,
)
from toric_spines_sim.geometry.sink import neck_point_from_swc_file
from toric_spines_sim.utils import join_tags_dsl
from .synapse import SynapsePoint
from .gj import GapJunctionPoint

CUSTOM_CATALOGUE_PATH = (
    Path(__file__).parent.parent / "mechanisms" / "custom-catalogue.so"
)


@dataclass
class TSModel:
    """Morphology specification for toric spine models.

    Attributes
    ----------
    swc_path : Path
        Path to the base SWC morphology for the toric spine.
    synapses : dict[str, SynapsePoint]
        Dictionary of synapse labels to instances of `SynapsePoint`.
    gap_junctions : dict[str, GapJunctionPoint]
        Dictionary of gap junction labels to instances of `GapJunctionPoint`.
    record_points : dict[str, Tuple[float, float, float]]
        Dictionary of record site labels to 3D points (x,y,z).
    parameters : ParameterSet
        Sampled simulation parameters (from ``ParameterBank.sample()``).
    """

    swc_path: Path
    synapses: Dict[str, SynapsePoint]
    gap_junctions: Dict[str, GapJunctionPoint]
    record_points: Dict[str, Tuple[float, float, float]]
    parameters: ParameterSet

    def build_cell(self) -> dict[str, object]:
        """Construct an Arbor cable_cell from the SWC and explicit placements.

        Returns
        -------
        cell : A.cable_cell
            The fully constructed cell with placements.
        """
        # Load pre-built custom mechanism catalogue (ampasyn, nmdasyn, hh, ...)
        logger.info("Loading custom catalogue from %s", CUSTOM_CATALOGUE_PATH)
        custom_catalogue = A.load_catalogue(str(CUSTOM_CATALOGUE_PATH))
        
        # Load swc morphology (fallback for older Arbor without raw=True)
        logger.info(
            "Building cable_cell from %s with discretization %.3f um",
            self.swc_path,
            self.parameters["discretization_um"],
        )
        loaded_morphology = A.load_swc_arbor(str(self.swc_path))
        original_segment_tree = getattr(
            loaded_morphology, "segment_tree", loaded_morphology
        )
        segment_tree = original_segment_tree
        if self.parameters["sink_radii_scale"] != 1.0:
            segment_tree = scale_radii_in_segment_tree_by_tag(
                segment_tree,
                self.parameters["sink_radii_scale"],
                self.parameters["sink_tag"],
            )
            logger.debug(
                "Scaled sink radii by %.4f",
                self.parameters["sink_radii_scale"],
            )
        if self.parameters["neck_radius_scale"] != 1.0:
            neck_xyz = neck_point_from_swc_file(self.swc_path)
            segment_tree = scale_one_radius_in_segment_tree_by_coordinates(
                segment_tree,
                self.parameters["neck_radius_scale"],
                neck_xyz,
            )
            logger.debug(
                "Scaled neck radius at %s by %.4f",
                neck_xyz,
                self.parameters["neck_radius_scale"],
            )

        morphology = A.morphology(segment_tree)

        hh_tags = [int(tag) for tag in self.parameters["hh_tags"]]
        hh_scale = (
            float(self.parameters["hh_scale"])
            if "hh_scale" in self.parameters.index
            else 0.0
        )
        label_map = {
            "all": "(all)",
            "root": "(root)",
            "sink": f"(tag {self.parameters['sink_tag']})",
            "spine": f"(tag {self.parameters['spine_tag']})",
        }
        if len(hh_tags) > 0 and hh_scale > 0:
            label_map["hh_area"] = join_tags_dsl(hh_tags)

        decor = A.decor()
        
        # DENSITY MECHANISMS ===============================
        # Passive leak everywhere (independent of HH leak parameters).
        pas_mechanism_name = f"pas/e={self.parameters['pas_leak_e_mV']}"
        decor.paint(
            label_map["all"],
            A.density(
                pas_mechanism_name,
                {
                    "g": self.parameters["pas_leak_g_S_per_cm2"],
                },
            ),
        )
        # Hodgkin-Huxley (only if hh_tags are specified and hh_scale > 0)
        use_hh_notemp = True
        if "hh_area" in label_map:
            hh_mechanism_name = "hhnotemp" if use_hh_notemp else "hh"
            decor.paint(
                label_map["hh_area"],
                A.density(
                    hh_mechanism_name,
                    {
                        "gnabar": self.parameters["Na_gbar_S_per_cm2"] * hh_scale,
                        "gkbar": self.parameters["K_gbar_S_per_cm2"] * hh_scale,
                        "gl": self.parameters["hh_leak_g_S_per_cm2"] * hh_scale,
                        "el": self.parameters["hh_leak_e_mV"],
                    },
                ),
            )
            logger.info(
                "Painted %s on tags %s with scale %.4g "
                "(gnabar=%.6g, gkbar=%.6g g/cm2)",
                hh_mechanism_name,
                hh_tags,
                hh_scale,
                self.parameters["Na_gbar_S_per_cm2"] * hh_scale,
                self.parameters["K_gbar_S_per_cm2"] * hh_scale,
            )

        # Map synapse 3D points to closest Arbor locations; place individually under unique labels.
        piecewise_placer = A.place_pwlin(morphology)
        if self.synapses:
            logger.debug("Placing %d synapses", len(self.synapses))
            for syn_label, synapse in self.synapses.items():
                x, y, z = synapse.location
                location, _ = piecewise_placer.closest(float(x), float(y), float(z))
                location_expr = f"(location {location.branch} {location.pos:.6f})"

                decor.place(
                    location_expr,
                    A.synapse(synapse.mechanism, synapse.mechanism_params),
                    syn_label,
                )
                label_map[syn_label] = location_expr
                logger.debug(
                    "Placed synapse %s at %s (mechanism=%s, params=%s)",
                    syn_label,
                    location_expr,
                    synapse.mechanism,
                    synapse.mechanism_params,
                )
        else:
            logger.info("No synapses provided.")

        # Make gap junction connections from input dict (connect index pairs)
        if self.gap_junctions:
            logger.debug("Placing %d gap junctions", len(self.gap_junctions))
            for gj_label, gj in self.gap_junctions.items():
                # i, j = gj.index_pair
                gj_label_a = f"{gj_label}_a"
                gj_label_b = f"{gj_label}_b"
                location_i, _ = piecewise_placer.closest(
                    float(gj.location[0]), float(gj.location[1]), float(gj.location[2])
                )
                location_j, _ = piecewise_placer.closest(
                    float(gj.location[0]), float(gj.location[1]), float(gj.location[2])
                )
                location_expr_i = f"(location {location_i.branch} {location_i.pos:.6f})"
                location_expr_j = f"(location {location_j.branch} {location_j.pos:.6f})"
                decor.place(location_expr_i, A.junction("gj"), gj_label_a)
                decor.place(location_expr_j, A.junction("gj"), gj_label_b)
                label_map[gj_label_a] = location_expr_i
                label_map[gj_label_b] = location_expr_j
                label_map[gj_label] = location_expr_i
                logger.debug(
                    "Placed gap junction %s at %s (a/b labels: %s, %s)",
                    gj_label,
                    location_expr_i,
                    gj_label_a,
                    gj_label_b,
                )
        else:
            logger.info("No gap junctions provided.")

        if self.record_points:
            logger.debug("Placing %d record points", len(self.record_points))
            for record_label, record_point in self.record_points.items():
                x, y, z = record_point
                location, _ = piecewise_placer.closest(float(x), float(y), float(z))
                location_expr = f"(location {location.branch} {location.pos:.6f})"
                label_map[record_label] = location_expr
                logger.debug(
                    "Placed record point %s at %s", record_label, location_expr
                )
        else:
            logger.info("No record points provided.")

        labels = A.label_dict(label_map)
        cvp = A.cv_policy_max_extent(
            self.parameters["discretization_um"] * U.um
        )
        cell = A.cable_cell(morphology, decor, labels, cvp)
        logger.info("Built cable_cell with %d labels", len(label_map))
        output = {
            "cell": cell,
            "morphology": morphology,
            "segment_tree": segment_tree,
            "decor": decor,
            "labels": labels,
            "cvp": cvp,
            "custom_catalogue": custom_catalogue,
        }
        return output
