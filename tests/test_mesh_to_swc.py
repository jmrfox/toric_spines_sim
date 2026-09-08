"""Tests for mesh → skeleton → SWC path helpers and orchestration."""

from __future__ import annotations

import builtins
import os
from unittest.mock import MagicMock

import pytest

from toric_spines_sim.paths import (
    MESH_DIR,
    SKELETONS_DIR,
    SWC_PIXELS_DIR,
    get_mesh_path,
    get_skeleton_path,
    get_swc_path,
)


class TestMeshPathHelpers:
    def test_get_mesh_path(self):
        assert get_mesh_path("TS1.obj") == MESH_DIR / "TS1.obj"

    def test_get_skeleton_path(self):
        assert get_skeleton_path("TS1.polylines.txt") == (
            SKELETONS_DIR / "TS1.polylines.txt"
        )

    def test_default_swc_pixels_for_mesh_pipeline(self):
        assert get_swc_path("TS1.swc", units="pixels") == SWC_PIXELS_DIR / "TS1.swc"

    def test_mesh_dir_contains_ts_meshes(self):
        assert (MESH_DIR / "TS1.obj").is_file()


class TestMeshToSwcDefaults:
    def test_resolve_and_defaults(self, temp_dir):
        import toric_spines_sim.geometry.mesh_pipeline as m

        mesh = temp_dir / "Toy.obj"
        mesh.write_text("# empty\n")
        resolved = m.resolve_mesh_path(mesh)
        assert resolved == mesh.resolve()
        assert m.default_polylines_path(resolved).name == "Toy.polylines.txt"
        assert m.default_swc_path(resolved).name == "Toy.swc"

    def test_resolve_bare_filename(self):
        from toric_spines_sim.geometry.mesh_pipeline import resolve_mesh_path

        path = resolve_mesh_path("TS1.obj")
        assert path == (MESH_DIR / "TS1.obj").resolve()

    def test_resolve_missing_raises(self):
        from toric_spines_sim.geometry.mesh_pipeline import resolve_mesh_path

        with pytest.raises(FileNotFoundError):
            resolve_mesh_path("definitely_missing_mesh_xyz.obj")


class TestMeshToSwcOrchestration:
    def test_skeletonize_mesh_calls_pymcfs(self, temp_dir, monkeypatch):
        import toric_spines_sim.geometry.mesh_pipeline as m

        mesh = temp_dir / "m.obj"
        mesh.write_text("v 0 0 0\n")
        output_path = temp_dir / "m.polylines.txt"

        fake_skel = MagicMock()
        load_and_repair = MagicMock(return_value="mesh-obj")
        skeletonize = MagicMock(return_value=fake_skel)
        monkeypatch.setattr(
            m, "_require_pymcfs", lambda: (load_and_repair, skeletonize)
        )

        result = m.skeletonize_mesh(mesh, output_path, profile="auto", branching="sparse")
        assert result == output_path.resolve()
        load_and_repair.assert_called_once_with(str(mesh.resolve()))
        skeletonize.assert_called_once()
        assert skeletonize.call_args.kwargs.get("profile") == "auto"
        assert skeletonize.call_args.kwargs.get("branching") == "sparse"
        assert skeletonize.call_args.kwargs.get("extend_tips") is True
        fake_skel.write_polylines.assert_called_once_with(str(output_path))

    def test_fit_swc_calls_mascaf(self, temp_dir, monkeypatch):
        import toric_spines_sim.geometry.mesh_pipeline as m

        mesh = temp_dir / "m.obj"
        mesh.write_text("v 0 0 0\n")
        polylines = temp_dir / "m.polylines.txt"
        polylines.write_text("2 0 0 0 1 0 0\n")
        swc = temp_dir / "m.swc"

        BasisOptimizerOptions = MagicMock()
        CableFitter = MagicMock()
        FitOptions = MagicMock()
        MeshManager = MagicMock()
        mesh_mgr = MagicMock()
        mesh_mgr.bounding_box_diagonal.return_value = 25.0
        MeshManager.return_value = mesh_mgr
        SkeletonGraph = MagicMock()
        morphology = MagicMock()
        CableFitter.return_value.fit.return_value = morphology

        monkeypatch.setattr(
            m,
            "_require_mascaf",
            lambda: (
                BasisOptimizerOptions,
                CableFitter,
                FitOptions,
                MeshManager,
                SkeletonGraph,
            ),
        )

        result = m.fit_swc(
            mesh,
            polylines,
            swc,
            max_edge_length_frac=0.08,
            scale_radii=True,
            basis_optimize=False,
        )
        assert result == swc.resolve()
        FitOptions.assert_called_once()
        assert FitOptions.call_args.kwargs["max_edge_length"] == pytest.approx(2.0)
        assert FitOptions.call_args.kwargs["basis_optimizer_options"] is None
        morphology.scale_radii_to_match_mesh.assert_called_once()
        morphology.to_swc_file.assert_called_once_with(str(swc))

    def test_mesh_to_swc_polylines_only(self, temp_dir, monkeypatch):
        import toric_spines_sim.geometry.mesh_pipeline as m

        mesh = temp_dir / "m.obj"
        mesh.write_text("v 0 0 0\n")
        polylines = temp_dir / "out.polylines.txt"
        swc = temp_dir / "out.swc"

        skeletonize = MagicMock(return_value=polylines.resolve())
        fit = MagicMock()
        monkeypatch.setattr(m, "skeletonize_mesh", skeletonize)
        monkeypatch.setattr(m, "fit_swc", fit)

        result = m.mesh_to_swc(
            mesh,
            polylines_path=polylines,
            swc_path=swc,
            polylines_only=True,
        )
        skeletonize.assert_called_once()
        fit.assert_not_called()
        assert result.swc_path is None
        assert result.polylines_path == polylines.resolve()

    def test_mesh_to_swc_skip_skeletonize(self, temp_dir, monkeypatch):
        import toric_spines_sim.geometry.mesh_pipeline as m

        mesh = temp_dir / "m.obj"
        mesh.write_text("v 0 0 0\n")
        polylines = temp_dir / "out.polylines.txt"
        polylines.write_text("2 0 0 0 1 0 0\n")
        swc = temp_dir / "out.swc"

        skeletonize = MagicMock()
        fit = MagicMock(return_value=swc.resolve())
        monkeypatch.setattr(m, "skeletonize_mesh", skeletonize)
        monkeypatch.setattr(m, "fit_swc", fit)

        result = m.mesh_to_swc(
            mesh,
            polylines_path=polylines,
            swc_path=swc,
            skip_skeletonize=True,
        )
        skeletonize.assert_not_called()
        fit.assert_called_once()
        assert result.swc_path == swc.resolve()

    def test_require_pymcfs_missing_extra(self, monkeypatch):
        import toric_spines_sim.geometry.mesh_pipeline as m

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pymcfs" or name.startswith("pymcfs."):
                raise ImportError("No module named 'pymcfs'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="uv sync"):
            m._require_pymcfs()


class TestMeshTargetResolution:
    def test_list_ts_meshes(self):
        from toric_spines_sim.geometry.mesh_pipeline import list_ts_meshes

        meshes = list_ts_meshes()
        assert meshes
        assert all(p.name.startswith("TS") and p.suffix == ".obj" for p in meshes)
        assert meshes == sorted(meshes)

    def test_resolve_all(self):
        from toric_spines_sim.geometry.mesh_pipeline import (
            list_ts_meshes,
            resolve_mesh_targets,
        )

        targets = resolve_mesh_targets(all_meshes=True)
        assert targets == [p.resolve() for p in list_ts_meshes()]

    def test_resolve_explicit(self):
        from toric_spines_sim.geometry.mesh_pipeline import resolve_mesh_targets

        targets = resolve_mesh_targets(["TS1.obj"])
        assert len(targets) == 1
        assert targets[0].name == "TS1.obj"

    def test_resolve_all_with_explicit_raises(self):
        from toric_spines_sim.geometry.mesh_pipeline import resolve_mesh_targets

        with pytest.raises(ValueError, match="either --all or explicit"):
            resolve_mesh_targets(["TS1.obj"], all_meshes=True)

    def test_resolve_empty_raises(self):
        from toric_spines_sim.geometry.mesh_pipeline import resolve_mesh_targets

        with pytest.raises(ValueError, match="Provide one or more"):
            resolve_mesh_targets([])


class TestMeshComparePolylines:
    def test_read_polylines_txt(self, temp_dir):
        from toric_spines_sim.viz.mesh_compare import read_polylines_txt

        path = temp_dir / "s.polylines.txt"
        path.write_text("2 0 0 0 1 0 0\n3 0 0 0 0 1 0 0 0 1\n")
        polylines = read_polylines_txt(path)
        assert len(polylines) == 2
        assert polylines[0].shape == (2, 3)
        assert polylines[1].shape == (3, 3)

    def test_read_polylines_invalid(self, temp_dir):
        from toric_spines_sim.viz.mesh_compare import read_polylines_txt

        path = temp_dir / "bad.polylines.txt"
        path.write_text("2 0 0 0\n")
        with pytest.raises(ValueError, match="expected 6 coordinates"):
            read_polylines_txt(path)


class TestSteppedScripts:
    def test_skeletonize_script_calls_api(self, monkeypatch):
        import scripts.skeletonize_meshes as skel_script

        calls = []

        def fake_skeletonize(mesh_path, polylines_path, *, profile="auto", **kwargs):
            calls.append((mesh_path, polylines_path, profile, kwargs))
            return polylines_path

        monkeypatch.setattr(skel_script, "skeletonize_mesh", fake_skeletonize)
        monkeypatch.setattr(
            skel_script,
            "resolve_mesh_targets",
            lambda meshes, all_meshes=False: [MESH_DIR / "TS1.obj"],
        )
        monkeypatch.setattr(
            skel_script,
            "default_polylines_path",
            lambda mesh_path: SKELETONS_DIR / f"{mesh_path.stem}.polylines.txt",
        )

        assert skel_script.main(["TS1.obj"]) == 0
        assert len(calls) == 1
        assert calls[0][0] == MESH_DIR / "TS1.obj"
        assert calls[0][2] == "auto"
        assert calls[0][3].get("branching") == "sparse"
        assert calls[0][3].get("extend_tips") is True
        assert calls[0][3].get("timeout_seconds") == 300.0

    def test_fit_script_calls_api(self, monkeypatch, temp_dir):
        import scripts.fit_swc as fit_script

        calls = []
        mesh = MESH_DIR / "TS1.obj"
        polylines = temp_dir / "TS1.polylines.txt"
        polylines.write_text("2 0 0 0 1 0 0\n")
        swc = temp_dir / "TS1.swc"

        def fake_fit(mesh_path, polylines_path, swc_path, **kwargs):
            calls.append((mesh_path, polylines_path, swc_path, kwargs))
            return swc_path

        monkeypatch.setattr(fit_script, "fit_swc", fake_fit)
        monkeypatch.setattr(
            fit_script,
            "resolve_mesh_targets",
            lambda meshes, all_meshes=False: [mesh],
        )
        monkeypatch.setattr(
            fit_script, "default_polylines_path", lambda mesh_path: polylines
        )
        monkeypatch.setattr(fit_script, "default_swc_path", lambda mesh_path: swc)

        assert fit_script.main(["TS1.obj", "--max-edge-length-frac", "0.05"]) == 0
        assert len(calls) == 1
        assert calls[0][0] == mesh
        assert calls[0][3]["max_edge_length_frac"] == 0.05


@pytest.mark.slow
def test_mesh_pipeline_imports_optional():
    """Opt-in check that mesh extras import when installed.

    Skips unless ``RUN_MESH_PIPELINE_TESTS=1`` and both packages are importable.
    Does not run full TS skeletonization.
    """
    if os.environ.get("RUN_MESH_PIPELINE_TESTS") != "1":
        pytest.skip("Set RUN_MESH_PIPELINE_TESTS=1 to enable")

    pymcfs = pytest.importorskip("pymcfs")
    mascaf = pytest.importorskip("mascaf")
    assert hasattr(pymcfs, "skeletonize")
    assert hasattr(mascaf, "CableFitter")
