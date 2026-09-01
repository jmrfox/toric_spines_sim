"""Visualization utilities for morphology and simulation results."""

from .arbor import (
    COLORS,
    VizConfig,
    draw_morphology,
    draw_morphology_frusta,
    plot_morphology_3d,
    plot_morphology_frusta_3d,
    plot_cable_cell_with_locations,
    terminals_from_morphology,
    locations_to_points,
    locations_from_location_exprs,
    parse_location_expr,
    scatter_locations,
    scatter_points,
    plot_morph_and_locations,
    plotly_morphology_traces,
    plotly_locations_trace,
    plotly_points_trace,
    plotly_morphology_frusta_trace,
)
from .plotting import (
    arbor_samples_to_arrays,
    TimeSeriesPlotter,
    HistogramGridPlotter,
    RasterPlotter,
)
from .hypergrid_dash import create_hypergrid_dash_app
from .simulation_dash import (
    create_simulation_dash_app,
    load_or_build_playback_cache,
    prepare_simulation_dashboard_data,
)
from .animation import Animation, AnimationFrameCache
from .mesh_compare import (
    figure_mesh_and_skeleton,
    figure_mesh_and_swc,
    read_polylines_txt,
)

__all__ = [
    "COLORS",
    "VizConfig",
    "draw_morphology",
    "draw_morphology_frusta",
    "plot_morphology_3d",
    "plot_morphology_frusta_3d",
    "plot_cable_cell_with_locations",
    "terminals_from_morphology",
    "locations_to_points",
    "locations_from_location_exprs",
    "parse_location_expr",
    "scatter_locations",
    "scatter_points",
    "plot_morph_and_locations",
    "plotly_morphology_traces",
    "plotly_locations_trace",
    "plotly_points_trace",
    "plotly_morphology_frusta_trace",
    "arbor_samples_to_arrays",
    "TimeSeriesPlotter",
    "HistogramGridPlotter",
    "RasterPlotter",
    "create_hypergrid_dash_app",
    "create_simulation_dash_app",
    "load_or_build_playback_cache",
    "prepare_simulation_dashboard_data",
    "Animation",
    "AnimationFrameCache",
    "read_polylines_txt",
    "figure_mesh_and_skeleton",
    "figure_mesh_and_swc",
]
