"""Tests for arborgeometry.py segment tree scaling functions."""

import arbor as A
import pytest

from toric_spines_sim.geometry.rescale import (
    scale_one_radius_in_segment_tree_by_coordinates,
    scale_radii_in_segment_tree_by_tag,
)


def _make_spine_with_sink_tree():
    """Build a minimal segment tree simulating a spine (tag=3) with a sink (tag=5).

    Layout (all along x-axis for simplicity):
        seg 0: (0,0,0,r=1) -> (1,0,0,r=1)  tag=3  parent=-1  (spine)
        seg 1: (1,0,0,r=1) -> (2,0,0,r=1)  tag=3  parent=0   (spine)
        seg 2: (2,0,0,r=1) -> (3,0,0,r=0.5) tag=5  parent=1  (sink, prox is neck node)
        seg 3: (3,0,0,r=0.5) -> (4,0,0,r=0.5) tag=5  parent=2  (sink interior)

    The neck node is at (2,0,0) with radius 1.0.  It is the proximal end of
    seg 2 (tag=5) whose parent seg 1 has tag=3.
    """
    tree = A.segment_tree()
    tree.append(
        A.mnpos,
        A.mpoint(0, 0, 0, 1),
        A.mpoint(1, 0, 0, 1),
        3,
    )
    tree.append(
        0,
        A.mpoint(1, 0, 0, 1),
        A.mpoint(2, 0, 0, 1),
        3,
    )
    tree.append(
        1,
        A.mpoint(2, 0, 0, 1),
        A.mpoint(3, 0, 0, 0.5),
        5,
    )
    tree.append(
        2,
        A.mpoint(3, 0, 0, 0.5),
        A.mpoint(4, 0, 0, 0.5),
        5,
    )
    return tree


class TestScaleRadiiByTag:
    def test_boundary_proximal_not_scaled(self):
        """The neck node (prox of first tag=5 seg) must NOT be scaled."""
        tree = _make_spine_with_sink_tree()
        scaled = scale_radii_in_segment_tree_by_tag(tree, 2.0, scale_tag=5)
        segs = scaled.segments

        # seg 2 (first sink): prox is the neck node — should be unchanged
        assert segs[2].prox.radius == pytest.approx(1.0)
        # seg 2 dist should be scaled
        assert segs[2].dist.radius == pytest.approx(1.0)  # 0.5 * 2.0

    def test_interior_sink_fully_scaled(self):
        """Interior sink segments (parent also tag=5) should be fully scaled."""
        tree = _make_spine_with_sink_tree()
        scaled = scale_radii_in_segment_tree_by_tag(tree, 2.0, scale_tag=5)
        segs = scaled.segments

        # seg 3: both prox and dist should be scaled
        assert segs[3].prox.radius == pytest.approx(1.0)  # 0.5 * 2.0
        assert segs[3].dist.radius == pytest.approx(1.0)  # 0.5 * 2.0

    def test_spine_segments_unchanged(self):
        """Spine segments (tag=3) should not be affected by scaling tag=5."""
        tree = _make_spine_with_sink_tree()
        scaled = scale_radii_in_segment_tree_by_tag(tree, 2.0, scale_tag=5)
        segs = scaled.segments

        for i in (0, 1):
            assert segs[i].prox.radius == pytest.approx(1.0)
            assert segs[i].dist.radius == pytest.approx(1.0)

    def test_tags_preserved(self):
        tree = _make_spine_with_sink_tree()
        scaled = scale_radii_in_segment_tree_by_tag(tree, 2.0, scale_tag=5)
        segs = scaled.segments
        assert [s.tag for s in segs] == [3, 3, 5, 5]

    def test_scale_factor_one_is_noop(self):
        tree = _make_spine_with_sink_tree()
        scaled = scale_radii_in_segment_tree_by_tag(tree, 1.0, scale_tag=5)
        for orig, new in zip(tree.segments, scaled.segments):
            assert new.prox.radius == pytest.approx(orig.prox.radius)
            assert new.dist.radius == pytest.approx(orig.dist.radius)


class TestScaleRadiusAtNode:
    def test_scales_matching_endpoints(self):
        """Scaling at (2,0,0) should hit the dist of seg 1 and prox of seg 2."""
        tree = _make_spine_with_sink_tree()
        scaled = scale_one_radius_in_segment_tree_by_coordinates(
            tree, 0.5, target_xyz=(2.0, 0.0, 0.0)
        )
        segs = scaled.segments

        # seg 1 dist is at (2,0,0) — should be scaled
        assert segs[1].dist.radius == pytest.approx(0.5)  # 1.0 * 0.5
        # seg 2 prox is at (2,0,0) — should be scaled
        assert segs[2].prox.radius == pytest.approx(0.5)  # 1.0 * 0.5

    def test_non_matching_endpoints_unchanged(self):
        tree = _make_spine_with_sink_tree()
        scaled = scale_one_radius_in_segment_tree_by_coordinates(
            tree, 0.5, target_xyz=(2.0, 0.0, 0.0)
        )
        segs = scaled.segments

        # seg 0 should be completely untouched
        assert segs[0].prox.radius == pytest.approx(1.0)
        assert segs[0].dist.radius == pytest.approx(1.0)
        # seg 3 should be completely untouched
        assert segs[3].prox.radius == pytest.approx(0.5)
        assert segs[3].dist.radius == pytest.approx(0.5)

    def test_no_match_warns(self, caplog):
        tree = _make_spine_with_sink_tree()
        import logging

        with caplog.at_level(logging.WARNING):
            scale_one_radius_in_segment_tree_by_coordinates(
                tree, 2.0, target_xyz=(99.0, 99.0, 99.0)
            )
        assert "no endpoints matched" in caplog.text

    def test_tolerance(self):
        """A point slightly off should still match within tolerance."""
        tree = _make_spine_with_sink_tree()
        scaled = scale_one_radius_in_segment_tree_by_coordinates(
            tree, 3.0, target_xyz=(2.0 + 1e-8, 0.0, 0.0), tolerance=1e-6
        )
        segs = scaled.segments
        # Should still match (2,0,0)
        assert segs[1].dist.radius == pytest.approx(3.0)
        assert segs[2].prox.radius == pytest.approx(3.0)

    def test_tags_preserved(self):
        tree = _make_spine_with_sink_tree()
        scaled = scale_one_radius_in_segment_tree_by_coordinates(
            tree, 2.0, target_xyz=(2.0, 0.0, 0.0)
        )
        assert [s.tag for s in scaled.segments] == [3, 3, 5, 5]
