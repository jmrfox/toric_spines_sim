"""
Simulations of barn-owl toric spines (Sanculi et al., 2020) with cable loops
closed exactly (jaxley, or the scipy reference solver).

Package map
-----------
geometry/     SWC I/O, cycle/multi-neck reconnects, sink append, dendrite helpers
model/        TSModel (description + build), synapses and their conductance kinetics
simulation/   parameter banks, TSSimulator, axon inputs, SimulationResults
events/       rate curves and event generators
viz/          Plotly / Dash / animation helpers
kmatrix.py    pairwise integration (k-matrix)
report.py     PDF reports
paths.py      repo-root data and simulation path helpers
"""

# Submodules
from . import simulation  # noqa: F401
from . import model  # noqa: F401
from . import geometry  # noqa: F401
from . import events  # noqa: F401
from . import viz  # noqa: F401
from . import paths  # noqa: F401
from . import utils  # noqa: F401
from . import report  # noqa: F401
from . import kmatrix  # noqa: F401

# Core classes
from .model.synapse import SynapsePoint, SynapsePopulation  # noqa: F401
from .model import BuiltModel, HHSpec, TSModel  # noqa: F401
from .simulation import (
    SimulationResults,
    make_default_parameter_bank,
    make_icx_parameter_bank,
    make_icx_parameter_bank_invitro,
    make_icx_parameter_bank_invivo,
    TSSimulator,
)  # noqa: F401

from .geometry import (
    Morphology,
    SegmentTable,
    compute_geodesic_distances,
    classify_compartments,
    map_probes_to_nodes,
    map_xyz_to_nearest_probes,
)  # noqa: F401

# Path utilities
from .paths import (
    PathConfig,
    get_data_path,
    get_mesh_path,
    get_skeleton_path,
    get_swc_path,
    get_pointset_path,
    get_simulation_path,
)  # noqa: F401

# General utilities
from .utils import (
    load_xyz_points,
    equal_vectors,
    join_tags_dsl,
    read_nff_s_points,
    read_nff_points_and_write_txt_file,
)  # noqa: F401

__all__ = [
    # Submodules
    "simulation",
    "model",
    "geometry",
    "events",
    "viz",
    "paths",
    "utils",
    "report",
    "kmatrix",
    # Core classes
    "SynapsePoint",
    "SynapsePopulation",
    "BuiltModel",
    "HHSpec",
    "TSModel",
    "SimulationResults",
    "make_default_parameter_bank",
    "make_icx_parameter_bank",
    "make_icx_parameter_bank_invitro",
    "make_icx_parameter_bank_invivo",
    "TSSimulator",
    # Geometry utilities
    "Morphology",
    "SegmentTable",
    "compute_geodesic_distances",
    "classify_compartments",
    "map_probes_to_nodes",
    "map_xyz_to_nearest_probes",
    # Path utilities
    "PathConfig",
    "get_data_path",
    "get_mesh_path",
    "get_skeleton_path",
    "get_swc_path",
    "get_pointset_path",
    "get_simulation_path",
    # General utilities
    "load_xyz_points",
    "equal_vectors",
    "join_tags_dsl",
    "read_nff_s_points",
    "read_nff_points_and_write_txt_file",
]
