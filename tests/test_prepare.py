"""Tests for pixel→micron preparation (NFF AZ, synpts, sink append)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toric_spines_sim.geometry.prepare import (
    append_sink_write_microns,
    convert_nff_active_zone,
    list_ts_spine_swcs,
    nff_spine_stem,
    resolve_swc_path,
    resolve_swc_targets,
    scale_swc_file,
    write_synpts_microns,
)
from toric_spines_sim.geometry.swc import parse_cycle_breaks, read_swc_points
from toric_spines_sim.paths import SWC_PIXELS_DIR, UM_PER_PX
from toric_spines_sim.utils import load_xyz_points, read_nff_s_points


class TestUmPerPx:
    def test_project_conversion_factor(self):
        assert UM_PER_PX == pytest.approx(5.0 / 1000.0)


class TestSwcTargets:
    def test_list_ts_spine_swcs_excludes_wsink(self):
        stems = {p.stem for p in list_ts_spine_swcs()}
        assert "TS1" in stems
        assert all("_wsink_" not in s for s in stems)
        assert all(p.parent == SWC_PIXELS_DIR for p in list_ts_spine_swcs())

    def test_resolve_bare_stem(self):
        path = resolve_swc_path("TS1")
        assert path == (SWC_PIXELS_DIR / "TS1.swc").resolve()

    def test_resolve_all_xor_explicit(self):
        with pytest.raises(ValueError, match="either --all"):
            resolve_swc_targets(["TS1"], all_swcs=True)

    def test_nff_spine_stem(self):
        assert nff_spine_stem(Path("TS1_AZ.nff")) == "TS1"
        assert nff_spine_stem(Path("TS21_AZ.nff")) == "TS21"


class TestConvertNff:
    def test_writes_s_points(self, sample_nff_file, temp_dir):
        out = temp_dir / "TS1_AZ.txt"
        written = convert_nff_active_zone(sample_nff_file, out)
        assert written == out
        points = np.loadtxt(out)
        expected = np.asarray(read_nff_s_points(sample_nff_file, return_numpy=True))
        np.testing.assert_array_almost_equal(points, expected, decimal=3)

    def test_empty_nff_does_not_write_file(self, temp_dir):
        nff = temp_dir / "empty.nff"
        nff.write_text("# object with 1 contours.\nf 1 0 0 0 0 0 0 0\n")
        out = temp_dir / "empty.txt"
        convert_nff_active_zone(nff, out)
        assert not out.exists()


class TestScaleSwc:
    def test_scales_coords_and_preserves_cycle_breaks(self, sample_swc_file, temp_dir):
        out = temp_dir / "scaled.swc"
        scale_swc_file(sample_swc_file, out, UM_PER_PX)
        pts_in = read_swc_points(sample_swc_file)
        pts_out = read_swc_points(out)
        assert set(pts_in) == set(pts_out)
        nid = next(iter(pts_in))
        for a, b in zip(pts_in[nid][:4], pts_out[nid][:4]):
            assert b == pytest.approx(a * UM_PER_PX)
        assert parse_cycle_breaks(out) == parse_cycle_breaks(sample_swc_file)


class TestSynptsAndSink:
    def test_write_synpts_scales_projected_points(
        self, sample_swc_file, sample_synpts_file, temp_dir
    ):
        out = temp_dir / "synpts.txt"
        write_synpts_microns(sample_swc_file, sample_synpts_file, out, um_per_px=UM_PER_PX)
        pts = np.loadtxt(out)
        assert pts.ndim == 2 and pts.shape[1] == 3
        # Input AZ are O(1) px; output must be O(UM_PER_PX) microns.
        assert pts[:, 0].max() < 4.0 * UM_PER_PX + 1e-3
        assert np.all(np.abs(pts) < 4.0 * UM_PER_PX + 1e-3)

    def test_append_sink_writes_micron_geometry(self, sample_swc_file, temp_dir):
        neck = temp_dir / "neck.txt"
        neck.write_text("0.0 0.0 0.0\n")
        swc_out = temp_dir / "with_sink.swc"
        swc_out_px = temp_dir / "with_sink_px.swc"
        neck_out = temp_dir / "neck_um.txt"
        written = append_sink_write_microns(
            sample_swc_file,
            neck_file=neck,
            radius_um=10.0,
            connector_length_um=5.0,
            n_cylinders=5,
            um_per_px=UM_PER_PX,
            swc_out=swc_out,
            swc_out_px=swc_out_px,
            neck_out=neck_out,
        )
        assert written == swc_out.resolve()
        assert swc_out_px.is_file()
        header = swc_out.read_text()
        assert "# SINK:" in header
        assert "radius=10" in header
        assert "last_segment_tag=6" in header
        pts = read_swc_points(swc_out)
        # Original node 1 was at (0,0,0) r=1; in microns r=0.005.
        assert pts[1][3] == pytest.approx(UM_PER_PX)
        # Sink nodes use radius 10 µm (much larger than scaled spine radii).
        sink_radii = [r for (_x, _y, _z, r) in pts.values() if r > 1.0]
        assert sink_radii
        assert all(r == pytest.approx(10.0) for r in sink_radii)
        # Pixel SWC sink radius is 10 / UM_PER_PX.
        pts_px = read_swc_points(swc_out_px)
        sink_radii_px = [r for (_x, _y, _z, r) in pts_px.values() if r > 100.0]
        assert sink_radii_px
        assert all(r == pytest.approx(10.0 / UM_PER_PX) for r in sink_radii_px)
        neck_um = load_xyz_points(neck_out)
        assert neck_um[0] == pytest.approx((0.0, 0.0, 0.0))
        assert parse_cycle_breaks(swc_out) == parse_cycle_breaks(sample_swc_file)

    def test_scale_swc_scales_sink_header(self, sample_swc_file, temp_dir):
        from toric_spines_sim.geometry.prepare import append_sink_write

        neck = temp_dir / "neck.txt"
        neck.write_text("0.0 0.0 0.0\n")
        px_out = temp_dir / "px.swc"
        um_out = temp_dir / "um.swc"
        append_sink_write(
            sample_swc_file,
            neck_file=neck,
            radius_um=10.0,
            swc_out_px=px_out,
            swc_out_um=um_out,
            neck_out_um=temp_dir / "neck_um.txt",
        )
        px_sink = next(
            line for line in px_out.read_text().splitlines() if line.startswith("# SINK:")
        )
        um_sink = next(
            line for line in um_out.read_text().splitlines() if line.startswith("# SINK:")
        )
        assert "radius=2000" in px_sink or "radius=2000.0" in px_sink
        assert "radius=10" in um_sink
        assert "neck_xyz=0.000000 0.000000 0.000000" in um_sink

    def test_append_sink_falls_back_to_swc_root(self, sample_swc_file, temp_dir):
        swc_out = temp_dir / "with_sink.swc"
        neck_out = temp_dir / "neck_um.txt"
        append_sink_write_microns(
            sample_swc_file,
            radius_um=2.0,
            swc_out=swc_out,
            neck_out=neck_out,
        )
        assert "# SINK:" in swc_out.read_text()
        assert load_xyz_points(neck_out)[0] == pytest.approx((0.0, 0.0, 0.0))
