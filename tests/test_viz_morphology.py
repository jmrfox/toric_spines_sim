"""Smoke tests for the morphology plotting functions (ported from Arbor)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from toric_spines_sim import viz  # noqa: E402
from toric_spines_sim.geometry import Morphology  # noqa: E402
from toric_spines_sim.paths import get_swc_path  # noqa: E402

SWC = get_swc_path("TS2_wsink_r10um.swc", units="microns")


def test_draw_from_path_and_morphology():
    ax = viz.draw_morphology(SWC)
    assert len(ax.lines) > 0
    morph = Morphology.from_swc(SWC)
    ax = viz.draw_morphology_frusta(morph, n_sides=6)
    assert ax is not None


def test_terminals_and_points_lie_on_cable():
    morph = Morphology.from_swc(SWC)
    terms = viz.terminals_from_morphology(morph)
    assert len(terms) > 0
    pts = viz.locations_to_points(morph, terms)
    for loc, p in zip(terms, pts):
        seg = morph.branch_segments(loc.branch)[-1]
        assert np.allclose([p.x, p.y, p.z], [seg.dist.x, seg.dist.y, seg.dist.z])


@pytest.mark.parametrize("backend", ["mpl", "plotly"])
def test_plot_morphology_3d_with_overlays(backend):
    overlays = {"syn": ["(location 0 0.5)"], "pts": [(0.0, 0.0, 0.0)]}
    fig = viz.plot_morphology_3d(SWC, overlays=overlays, backend=backend)
    assert fig is not None


def test_frusta_plotly_trace_and_results_object():
    pytest.importorskip("plotly")
    trace = viz.plotly_morphology_frusta_trace(Morphology.from_swc(SWC))
    assert trace is not None

    class R:  # anything with .morphology, like SimulationResults / BuiltModel
        morphology = Morphology.from_swc(SWC)

    ax = viz.plot_cable_cell_with_locations(R(), overlays={"syn": ["(location 1 0.25)"]})
    assert ax is not None


def test_deprecated_alias_module():
    from toric_spines_sim.viz import arbor as old

    assert old.plot_morphology_3d is viz.plot_morphology_3d
