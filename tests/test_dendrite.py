"""Tests for toric_spines_sim.geometry.dendrite."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from toric_spines_sim.geometry.dendrite import (
    SpinyDendriteParams,
    ToricSpineMatchParams,
    build_from_toric_spine,
    build_spiny_dendrite,
    distribute_spine_counts,
    swc_subsystem_surface_area,
    write_subsystem,
)
from toric_spines_sim.geometry.swc import read_swc_points
from toric_spines_sim.utils import load_xyz_points


def _default_params(**overrides) -> SpinyDendriteParams:
    base = dict(
        length=20.0,
        trunk_neck_radius=0.5,
        trunk_tip_radius=0.4,
        n_spines=6,
        spine_length=1.0,
        spine_neck_radius=0.1,
        spine_head_radius=0.25,
        spine_neck_length_fraction=0.5,
        max_spines_per_node=2,
        distribution="even",
        seed=0,
        azimuth0=0.0,
        axis="z",
    )
    base.update(overrides)
    return SpinyDendriteParams(**base)


class TestDistributeSpineCounts:
    def test_even_balanced(self):
        assert distribute_spine_counts(10, 4, 3, "even") == [3, 3, 2, 2]

    def test_even_exact(self):
        assert distribute_spine_counts(6, 3, 2, "even") == [2, 2, 2]

    def test_random_respects_cap_and_total(self):
        counts = distribute_spine_counts(10, 4, 3, "random", seed=42)
        assert sum(counts) == 10
        assert all(c <= 3 for c in counts)
        assert len(counts) == 4

    def test_capacity_error(self):
        with pytest.raises(ValueError, match="Cannot place"):
            distribute_spine_counts(10, 2, 3, "even")


class TestBuildSpinyDendrite:
    def test_node_counts_and_neck_has_no_spines(self):
        morph = build_spiny_dendrite(_default_params(n_spines=6, max_spines_per_node=2))
        # n_attach = ceil(6/2)=3 → trunk nodes = 4; spines = 6*(2 nodes each)=12; total=16
        assert len(morph.trunk_node_ids) == 4
        assert morph.n_spines == 6
        assert len(morph.nodes) == 4 + 12
        assert morph.spines_per_attach_node == [2, 2, 2]
        # Neck is first trunk node; no spine parents to it
        neck_id = morph.trunk_node_ids[0]
        spine_parents = {
            n.parent for n in morph.nodes if n.node_id not in morph.trunk_node_ids
        }
        # spine necks parent to trunk attach nodes only
        assert neck_id not in spine_parents
        assert morph.neck_point == (0.0, 0.0, 0.0)

    def test_uniform_angles_for_multiple_spines(self):
        morph = build_spiny_dendrite(
            _default_params(n_spines=3, max_spines_per_node=3, length=10.0)
        )
        # All 3 spines on the single attach/tip node
        assert morph.spines_per_attach_node == [3]
        tip_id = morph.trunk_node_ids[-1]
        tip = next(n for n in morph.nodes if n.node_id == tip_id)
        tip_pos = np.array([tip.x, tip.y, tip.z])
        # Head directions in xy (axis=z)
        angles = []
        for head_id in morph.spine_head_node_ids:
            head = next(n for n in morph.nodes if n.node_id == head_id)
            vec = np.array([head.x, head.y, head.z]) - tip_pos
            # Should be perpendicular to z
            assert abs(vec[2]) < 1e-9
            angles.append(math.atan2(vec[1], vec[0]))
        angles = sorted(angles)
        # Spaced by 2π/3
        deltas = [
            (angles[(i + 1) % 3] - angles[i]) % (2 * math.pi) for i in range(3)
        ]
        assert deltas == pytest.approx([2 * math.pi / 3] * 3, rel=1e-6)

    def test_az_at_head_nodes(self):
        morph = build_spiny_dendrite(_default_params())
        assert len(morph.az_points) == morph.n_spines
        for az, head_id in zip(morph.az_points, morph.spine_head_node_ids):
            head = next(n for n in morph.nodes if n.node_id == head_id)
            assert az == pytest.approx((head.x, head.y, head.z))

    def test_radius_taper_linear(self):
        morph = build_spiny_dendrite(
            _default_params(
                n_spines=0,
                length=10.0,
                trunk_neck_radius=1.0,
                trunk_tip_radius=0.0,
            )
        )
        # n_spines=0 → n_attach=1 → 2 trunk nodes
        radii = [
            next(n for n in morph.nodes if n.node_id == tid).radius
            for tid in morph.trunk_node_ids
        ]
        assert radii[0] == pytest.approx(1.0)
        assert radii[-1] == pytest.approx(0.0)

        morph4 = build_spiny_dendrite(
            _default_params(
                n_spines=3,
                max_spines_per_node=1,
                length=30.0,
                trunk_neck_radius=1.0,
                trunk_tip_radius=0.4,
            )
        )
        # 1 neck + 3 attach = 4 nodes; t = 0, 1/3, 2/3, 1
        radii = [
            next(n for n in morph4.nodes if n.node_id == tid).radius
            for tid in morph4.trunk_node_ids
        ]
        expected = [1.0 - t * 0.6 for t in (0.0, 1 / 3, 2 / 3, 1.0)]
        assert radii == pytest.approx(expected)

    def test_default_tags_all_three(self):
        morph = build_spiny_dendrite(_default_params())
        assert all(n.tag == 3 for n in morph.nodes)

    def test_optional_distinct_tags(self):
        morph = build_spiny_dendrite(
            _default_params(trunk_tag=3, spine_neck_tag=4, spine_head_tag=7)
        )
        by_id = {n.node_id: n for n in morph.nodes}
        for tid in morph.trunk_node_ids:
            assert by_id[tid].tag == 3
        for head_id in morph.spine_head_node_ids:
            head = by_id[head_id]
            assert head.tag == 7
            neck = by_id[head.parent]
            assert neck.tag == 4


class TestWriteSubsystem:
    def test_round_trip(self, temp_dir: Path):
        morph = build_spiny_dendrite(_default_params(n_spines=4, max_spines_per_node=2))
        swc_path = temp_dir / "dend.swc"
        az_path = temp_dir / "dend_AZ.txt"
        neck_path = temp_dir / "dend_neckpoint.txt"
        write_subsystem(morph, swc_path, az_path, neck_path)

        pts = read_swc_points(swc_path)
        assert len(pts) == len(morph.nodes)
        az = load_xyz_points(az_path)
        assert len(az) == 4
        np.testing.assert_allclose(az, morph.az_points, atol=1e-6)
        neck = load_xyz_points(neck_path)
        assert len(neck) == 1
        assert neck[0] == pytest.approx((0.0, 0.0, 0.0))


class TestBuildFromToricSpine:
    def _write_simple_ts(self, temp_dir: Path, n_az: int = 4):
        """Cylinder-like TS body (tag 3) plus sink (tag 5) and AZ file."""
        swc = temp_dir / "ts.swc"
        # Simple cylinder along z, radius 1, length 10 — SA = 2*pi*1*10
        lines = [
            "# test toric-like cylinder\n",
            "1 3 0.0 0.0 0.0 1.0 -1\n",
            "2 3 0.0 0.0 5.0 1.0 1\n",
            "3 3 0.0 0.0 10.0 1.0 2\n",
            "4 5 0.0 0.0 -1.0 2.0 1\n",
            "5 5 0.0 0.0 -5.0 2.0 4\n",
        ]
        swc.write_text("".join(lines))
        az = temp_dir / "ts_AZ.txt"
        az.write_text("\n".join(f"{i}.0 0.0 1.0" for i in range(n_az)) + "\n")
        neck = temp_dir / "ts_neckpoint.txt"
        neck.write_text("0.000000 0.000000 0.000000\n")
        return swc, az, neck

    def test_relative_matches_n_and_sa(self, temp_dir: Path):
        swc, az, neck = self._write_simple_ts(temp_dir, n_az=4)
        target_sa = swc_subsystem_surface_area(swc)
        # Body only: two segments of length 5, r=1 → 2 * 2*pi*1*5 = 20*pi
        assert target_sa == pytest.approx(20.0 * math.pi)

        morph = build_from_toric_spine(
            swc,
            az,
            match=ToricSpineMatchParams(
                scale_strategy="relative",
                max_spines_per_node=2,
                distribution="even",
            ),
            neck_path=neck,
        )
        assert morph.n_spines == 4
        assert morph.surface_area() == pytest.approx(target_sa, rel=1e-6)
        assert morph.diagnostics["surface_area_rel_error"] < 1e-6

    def test_absolute_spines_matches_sa(self, temp_dir: Path):
        swc, az, _neck = self._write_simple_ts(temp_dir, n_az=3)
        target_sa = swc_subsystem_surface_area(swc)
        morph = build_from_toric_spine(
            swc,
            az,
            match=ToricSpineMatchParams(
                scale_strategy="absolute_spines",
                max_spines_per_node=1,
            ),
        )
        assert morph.n_spines == 3
        # Spine absolute sizes unchanged from template defaults
        assert morph.params.spine_length == pytest.approx(1.0)
        assert morph.params.spine_neck_radius == pytest.approx(0.1)
        assert morph.surface_area() == pytest.approx(target_sa, rel=1e-5)

    def test_excludes_sink_from_target_sa(self, temp_dir: Path):
        swc, az, _ = self._write_simple_ts(temp_dir, n_az=2)
        body_sa = swc_subsystem_surface_area(swc)
        sa_with_sink = swc_subsystem_surface_area(swc, sink_tags=[])
        assert sa_with_sink > body_sa
        morph = build_from_toric_spine(swc, az)
        assert morph.diagnostics["target_surface_area"] == pytest.approx(body_sa)
