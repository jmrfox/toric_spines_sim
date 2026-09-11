"""Morphology specification and cell building for toric spine models.

``TSModel`` loads an SWC (typically ``TS*_wsink_r*um.swc``), applies passive
leak (and optional HH), and places synapses, gap junctions, and voltage
probes. Gap junctions restore SWC cycle-breaks and extra necks: each pair of
SWC node IDs is mapped to distinct Arbor locations on the segment tree.
"""

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
from toric_spines_sim.geometry.swc import arbor_locations_for_swc_nodes
from toric_spines_sim.utils import join_tags_dsl
from .synapse import SynapsePoint
from .gj import GapJunctionPoint

CUSTOM_CATALOGUE_PATH = (
    Path(__file__).parent.parent / "mechanisms" / "custom-catalogue.so"
)
CATALOGUE_SOURCE_DIR = Path(__file__).parent.parent / "mechanisms" / "my_catalogue"
_CATALOGUE_BUILD_CMD = "uv run bash scripts/make_custom_catalogue.sh"


def required_catalogue_mechanisms() -> list[str]:
    """NMODL mechanism names compiled from ``mechanisms/my_catalogue/*.mod``."""
    return sorted(path.stem for path in CATALOGUE_SOURCE_DIR.glob("*.mod"))


def _catalogue_has_mechanism(catalogue: A.catalogue, name: str) -> bool:
    try:
        return name in catalogue
    except Exception:
        try:
            catalogue[name]
            return True
        except Exception:
            return False


def check_catalogue(path: Path | None = None) -> A.catalogue:
    """Load the custom NMODL catalogue, or raise if it is missing or unusable.

    Checks that ``custom-catalogue.so`` exists, that Arbor can load it, and
    that every mechanism in ``mechanisms/my_catalogue/*.mod`` is present.
    Rebuild after changing ``.mod`` files, upgrading Arbor, or changing
    OS/compiler; do not copy a ``.so`` between machines.

    Parameters
    ----------
    path : Path, optional
        Catalogue file. Default: ``CUSTOM_CATALOGUE_PATH``.

    Returns
    -------
    arbor.catalogue
    """
    catalogue_path = Path(path) if path is not None else CUSTOM_CATALOGUE_PATH
    if not catalogue_path.is_file():
        raise FileNotFoundError(
            f"Custom mechanism catalogue not found at {catalogue_path}. "
            "Build it from the repository root (see README Getting started):\n"
            f"  {_CATALOGUE_BUILD_CMD}"
        )
    logger.info("Loading custom catalogue from %s", catalogue_path)
    try:
        catalogue = A.load_catalogue(str(catalogue_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load NMODL catalogue at {catalogue_path}. "
            "Rebuild after changing .mod files, upgrading Arbor, or changing "
            "OS/compiler. Do not copy a .so between machines.\n"
            f"  {_CATALOGUE_BUILD_CMD}"
        ) from exc
    expected = required_catalogue_mechanisms()
    missing = [
        name for name in expected if not _catalogue_has_mechanism(catalogue, name)
    ]
    if missing:
        raise RuntimeError(
            f"Catalogue at {catalogue_path} is missing mechanisms {missing}. "
            f"Expected {expected}. Rebuild with:\n"
            f"  {_CATALOGUE_BUILD_CMD}"
        )
    return catalogue


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
        """Construct an Arbor cable cell from the SWC and explicit placements.

        Returns
        -------
        dict
            Intermediate objects keyed by:

            - ``cell`` : ``arbor.cable_cell``
            - ``morphology`` : ``arbor.morphology``
            - ``segment_tree`` : ``arbor.segment_tree``
            - ``decor`` : ``arbor.decor``
            - ``labels`` : ``arbor.label_dict``
            - ``cvp`` : CV policy
            - ``custom_catalogue`` : loaded NMODL catalogue

        Examples
        --------
        >>> build_result = model.build_cell()  # doctest: +SKIP
        >>> cell = build_result["cell"]
        """
        # Load pre-built custom mechanism catalogue (ampasyn, nmdasyn, hhnotemp, ...)
        custom_catalogue = check_catalogue()
        
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

        # Restore cycles / extra necks: place two junction labels on the
        # distinct SWC samples named by index_pair (not a shared closest XYZ).
        if self.gap_junctions:
            node_ids = []
            for gj in self.gap_junctions.values():
                node_ids.extend(gj.index_pair)
            node_locations = arbor_locations_for_swc_nodes(
                self.swc_path, morphology, segment_tree, node_ids
            )
            logger.debug("Placing %d gap junctions", len(self.gap_junctions))
            for gj_label, gj in self.gap_junctions.items():
                node_i, node_j = gj.index_pair
                gj_label_a = f"{gj_label}_a"
                gj_label_b = f"{gj_label}_b"
                location_i = node_locations[node_i]
                location_j = node_locations[node_j]
                location_expr_i = f"(location {location_i.branch} {location_i.pos:.6f})"
                location_expr_j = f"(location {location_j.branch} {location_j.pos:.6f})"
                if (
                    location_i.branch == location_j.branch
                    and abs(location_i.pos - location_j.pos) < 1e-9
                ):
                    logger.warning(
                        "Gap junction %s maps both SWC nodes %s and %s to %s; "
                        "the connection may be electrically inert",
                        gj_label,
                        node_i,
                        node_j,
                        location_expr_i,
                    )
                decor.place(location_expr_i, A.junction("gj"), gj_label_a)
                decor.place(location_expr_j, A.junction("gj"), gj_label_b)
                label_map[gj_label_a] = location_expr_i
                label_map[gj_label_b] = location_expr_j
                label_map[gj_label] = location_expr_i
                logger.debug(
                    "Placed gap junction %s at %s / %s (SWC nodes %s, %s)",
                    gj_label,
                    location_expr_i,
                    location_expr_j,
                    node_i,
                    node_j,
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
