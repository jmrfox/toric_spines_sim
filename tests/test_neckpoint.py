"""Tests for mesh-based neckpoint detection."""

from __future__ import annotations

import numpy as np
import pytest

from toric_spines_sim.geometry.neckpoint import (
    NeckCandidate,
    NeckpointParams,
    _apply_max_necks,
    compute_neck_points,
    compute_neck_points_for_spine,
    default_neckpoint_path,
)
from toric_spines_sim.paths import MESH_DIR, get_cell_mesh_path


trimesh = pytest.importorskip("trimesh")


def _box(bounds_min, bounds_max):
    extents = np.asarray(bounds_max, dtype=float) - np.asarray(bounds_min, dtype=float)
    center = 0.5 * (np.asarray(bounds_min, dtype=float) + np.asarray(bounds_max, dtype=float))
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center - mesh.bounds.mean(axis=0))
    return mesh


class TestApplyMaxNecks:
    def test_keeps_largest_n(self):
        cands = [
            NeckCandidate(
                centroid=np.array([0.0, 0.0, 0.0]),
                area=10.0,
                n_faces=1,
                planarity=0.0,
                mean_signed=-10.0,
                mean_align=-1.0,
                ray_frac_inside=1.0,
                dendrite_frac=0.5,
            ),
            NeckCandidate(
                centroid=np.array([1.0, 0.0, 0.0]),
                area=50.0,
                n_faces=1,
                planarity=0.0,
                mean_signed=-10.0,
                mean_align=-1.0,
                ray_frac_inside=1.0,
                dendrite_frac=0.5,
            ),
            NeckCandidate(
                centroid=np.array([2.0, 0.0, 0.0]),
                area=20.0,
                n_faces=1,
                planarity=0.0,
                mean_signed=-10.0,
                mean_align=-1.0,
                ray_frac_inside=1.0,
                dendrite_frac=0.5,
            ),
        ]
        kept = _apply_max_necks(cands, 2)
        assert [c.area for c in kept] == [50.0, 20.0]

    def test_none_keeps_all_sorted(self):
        cands = [
            NeckCandidate(
                centroid=np.array([0.0, 0.0, 0.0]),
                area=3.0,
                n_faces=1,
                planarity=0.0,
                mean_signed=-1.0,
                mean_align=-1.0,
                ray_frac_inside=1.0,
                dendrite_frac=0.5,
            ),
            NeckCandidate(
                centroid=np.array([1.0, 0.0, 0.0]),
                area=9.0,
                n_faces=1,
                planarity=0.0,
                mean_signed=-1.0,
                mean_align=-1.0,
                ray_frac_inside=1.0,
                dendrite_frac=0.5,
            ),
        ]
        kept = _apply_max_necks(cands, None)
        assert [c.area for c in kept] == [9.0, 3.0]


class TestSyntheticCap:
    def test_detects_closing_cap(self):
        # Large cell cube; small spine protruding through +z with closing face inside.
        cell = _box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        spine = _box((4.0, 4.0, 8.0), (6.0, 6.0, 13.0))
        # Scale-friendly thresholds for this toy geometry (units ~1, not EM voxels).
        params = NeckpointParams(
            crop_margin=2.0,
            align_thr=0.3,
            align_signed_max=-0.5,
            signed_thr=-1.5,
            min_faces=1,
            planarity_max=0.25,
            mean_signed_max=-0.5,
            ray_frac_min=0.4,
            ray_offsets=(0.5, 1.0, 1.5, 2.0),
            dendrite_radius=4.0,
            dendrite_far_thr=1.5,
            dendrite_frac_floor=0.0,
            dendrite_frac_of_max=0.0,
            max_necks=1,
        )
        points = compute_neck_points(spine, cell, params)
        assert len(points) >= 1
        # Cap should be near the interior face around z≈8, xy≈5.
        xyz = points[0]
        assert abs(xyz[0] - 5.0) < 1.5
        assert abs(xyz[1] - 5.0) < 1.5
        assert abs(xyz[2] - 8.0) < 2.0

    def test_max_necks_truncates(self):
        cell = _box((0.0, 0.0, 0.0), (20.0, 10.0, 10.0))
        # Two protrusions with interior closing faces.
        spine_a = _box((2.0, 4.0, 8.0), (4.0, 6.0, 13.0))
        spine_b = _box((14.0, 4.0, 8.0), (18.0, 6.0, 13.0))  # larger footprint
        spine = trimesh.util.concatenate([spine_a, spine_b])
        params = NeckpointParams(
            crop_margin=2.0,
            align_thr=0.3,
            align_signed_max=-0.5,
            signed_thr=-1.5,
            min_faces=1,
            planarity_max=0.25,
            mean_signed_max=-0.5,
            ray_frac_min=0.4,
            ray_offsets=(0.5, 1.0, 1.5, 2.0),
            dendrite_radius=5.0,
            dendrite_far_thr=1.5,
            dendrite_frac_floor=0.0,
            dendrite_frac_of_max=0.0,
            max_necks=None,
        )
        all_pts = compute_neck_points(spine, cell, params)
        if len(all_pts) < 2:
            pytest.skip("Toy dual-cap scene did not yield two necks")
        params.max_necks = 1
        one = compute_neck_points(spine, cell, params)
        assert len(one) == 1


class TestPaths:
    def test_default_neckpoint_path(self):
        path = default_neckpoint_path("TS1")
        assert path.name == "TS1_neckpoint.txt"
        assert path.parent.name == "pixels"

    def test_get_cell_mesh_path(self):
        assert get_cell_mesh_path().name == "cell_wrapped_simplified.obj"


@pytest.mark.integration
class TestRealMeshes:
    def test_ts1_near_curated(self, temp_dir):
        ts_path = MESH_DIR / "TS1.obj"
        cell_path = get_cell_mesh_path()
        if not ts_path.is_file() or not cell_path.is_file():
            pytest.skip("Real meshes not present")
        gt_path = MESH_DIR.parent / "pointsets" / "pixels" / "TS1_neckpoint.txt"
        if not gt_path.is_file():
            pytest.skip("Curated TS1 neckpoint missing")
        gt = np.loadtxt(gt_path).reshape(-1, 3)[0]
        output_path = temp_dir / "TS1_neckpoint.txt"
        points = compute_neck_points_for_spine(
            "TS1.obj",
            params=NeckpointParams(max_necks=1),
            write=True,
            overwrite=True,
            output_path=output_path,
        )
        assert len(points) == 1
        err = float(np.linalg.norm(np.asarray(points[0]) - gt))
        assert err < 100.0, f"TS1 neckpoint error {err:.1f} px exceeds 100"
