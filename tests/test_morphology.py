"""SegmentTable / Morphology / PlacePwlin (plain-Python replacement for Arbor geometry)."""

import logging

import numpy as np
import pytest

from toric_spines_sim.geometry.morphology import Location, Morphology, PlacePwlin, SegmentTable
from toric_spines_sim.geometry.swc import iter_swc_samples
from toric_spines_sim.paths import get_swc_path

SPINE_SINK = """\
1 3 0 0 0 1.0 -1
2 3 1 0 0 1.0 1
3 3 2 0 0 1.0 2
4 5 3 0 0 0.5 3
5 5 4 0 0 0.5 4
"""
# segments: 0: 1->2 (tag 3), 1: 2->3 (tag 3), 2: 3->4 (tag 5, prox is the neck node), 3: 4->5 (tag 5)


@pytest.fixture
def spine_sink(tmp_path):
    p = tmp_path / "spine_sink.swc"
    p.write_text(SPINE_SINK)
    return p


class TestSinkScaling:
    def test_boundary_proximal_not_scaled(self, spine_sink):
        segs = SegmentTable.from_swc(spine_sink, sink_radii_scale=2.0, sink_tag=5).segments
        assert segs[2].prox.radius == pytest.approx(1.0)
        assert segs[2].dist.radius == pytest.approx(1.0)

    def test_interior_sink_fully_scaled(self, spine_sink):
        segs = SegmentTable.from_swc(spine_sink, sink_radii_scale=2.0, sink_tag=5).segments
        assert segs[3].prox.radius == pytest.approx(1.0)
        assert segs[3].dist.radius == pytest.approx(1.0)

    def test_spine_unchanged_and_tags_kept(self, spine_sink):
        segs = SegmentTable.from_swc(spine_sink, sink_radii_scale=2.0, sink_tag=5).segments
        for s in segs[:2]:
            assert (s.prox.radius, s.dist.radius) == (1.0, 1.0)
        assert [s.tag for s in segs] == [3, 3, 5, 5]

    def test_scale_one_is_noop(self, spine_sink):
        a = SegmentTable.from_swc(spine_sink).segments
        b = SegmentTable.from_swc(spine_sink, sink_radii_scale=1.0).segments
        assert a == b


class TestNeckScaling:
    def test_scales_matching_endpoints_only(self, spine_sink):
        segs = SegmentTable.from_swc(spine_sink, neck_radius_scale=0.5, neck_xyz=(2.0, 0, 0)).segments
        assert segs[1].dist.radius == pytest.approx(0.5)
        assert segs[2].prox.radius == pytest.approx(0.5)
        assert (segs[0].prox.radius, segs[0].dist.radius) == (1.0, 1.0)
        assert (segs[3].prox.radius, segs[3].dist.radius) == (0.5, 0.5)

    def test_tolerance(self, spine_sink):
        segs = SegmentTable.from_swc(spine_sink, neck_radius_scale=3.0,
                                     neck_xyz=(2.0 + 1e-8, 0, 0), tolerance=1e-6).segments
        assert segs[1].dist.radius == pytest.approx(3.0)

    def test_no_match_warns(self, spine_sink, caplog):
        with caplog.at_level(logging.WARNING):
            SegmentTable.from_swc(spine_sink, neck_radius_scale=2.0, neck_xyz=(99, 99, 99))
        assert "no endpoints matched" in caplog.text


def test_lateral_area_matches_frustum_formula(spine_sink):
    t = SegmentTable.from_swc(spine_sink)
    expected = np.pi * (1.0 + 0.5) * np.hypot(1.0, 0.5) + np.pi * 1.0 * 1.0
    assert t.lateral_area_by_tags([5]) == pytest.approx(expected)


def test_branches_split_at_forks(tmp_path):
    p = tmp_path / "y.swc"
    p.write_text("1 3 0 0 0 1 -1\n2 3 1 0 0 1 1\n3 3 2 1 0 1 2\n4 3 2 -1 0 1 2\n5 3 3 1 0 1 3\n")
    m = Morphology.from_swc(p)
    assert m.num_branches == 3
    assert sorted(len(m.branch_segments(b)) for b in range(3)) == [1, 1, 2]
    root = next(b for b in range(3) if m.branch_parent(b) == -1)
    assert sorted(m.branch_children(root)) == sorted(b for b in range(3) if b != root)


@pytest.mark.parametrize("name", ["TS1", "TS3", "TS76"])
def test_every_sample_is_placed_exactly(name):
    swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
    place = PlacePwlin(Morphology.from_swc(swc))
    for s in iter_swc_samples(swc):
        loc, d = place.closest(*s[2:5])
        assert d < 1e-9
        q = place.at(loc)
        assert np.linalg.norm([q.x - s[2], q.y - s[3], q.z - s[4]]) < 1e-6


def test_at_endpoints(spine_sink):
    m = Morphology.from_swc(spine_sink)
    place = PlacePwlin(m)
    assert m.num_branches == 1
    a, b = place.at(Location(0, 0.0)), place.at(Location(0, 1.0))
    assert (a.x, b.x) == (0.0, 4.0)
    assert place.at(Location(0, 0.5)).x == pytest.approx(2.0)
