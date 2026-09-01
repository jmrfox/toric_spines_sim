"""Mesh → skeleton → SWC orchestration using pymcfs and mascaf.

Requires the optional ``mesh`` extra::

    uv sync --extra mesh

On Linux/WSL, CHOLMOD also needs::

    sudo apt install libsuitesparse-dev
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from toric_spines_sim.paths import (
    MESH_DIR,
    SKELETONS_DIR,
    SWC_PIXELS_DIR,
    get_mesh_path,
    get_skeleton_path,
    get_swc_path,
)

logger = logging.getLogger(__name__)

_MESH_EXTRA_HINT = (
    "Mesh pipeline dependencies are missing. Install with: "
    "`uv sync --extra mesh` "
    "(on Linux/WSL also: `sudo apt install libsuitesparse-dev`)."
)

PathLike = Union[str, Path]

# Defaults matching pymcfs ``toric_spines/scripts/batch_ts_skeletonize.py``
# (sparse oracle / profile="auto", refine with tip extension on).
TORIC_SPINES_SKELETONIZE_DEFAULTS: dict[str, Any] = {
    "profile": "auto",
    "branching": "sparse",
    "max_iterations": 500,
    "timeout_seconds": 300.0,
    "max_vertex_growth": 4.0,
    "resample": False,
    "prune_exterior": True,
    "prune_short_leaves": True,
    "prune_thick_hubs": True,
    "keep_hub_branches": 2,
    "extend_tips": True,
    "tip_extend_scale": 1.0,
    "validate": False,  # load_and_repair already validated upstream
}


def _require_pymcfs():
    try:
        from pymcfs import load_and_repair, skeletonize
    except ImportError as exc:  # pragma: no cover - exercised via mock tests
        raise ImportError(_MESH_EXTRA_HINT) from exc
    return load_and_repair, skeletonize


def _require_mascaf():
    try:
        from mascaf import (
            BasisOptimizerOptions,
            CableFitter,
            FitOptions,
            MeshManager,
            SkeletonGraph,
        )
    except ImportError as exc:  # pragma: no cover - exercised via mock tests
        raise ImportError(
            f"{_MESH_EXTRA_HINT} Underlying import error: {exc}"
        ) from exc
    return (
        BasisOptimizerOptions,
        CableFitter,
        FitOptions,
        MeshManager,
        SkeletonGraph,
    )


def resolve_mesh_path(mesh: PathLike) -> Path:
    """Resolve a mesh path; bare filenames are looked up under ``data/mesh/``."""
    path = Path(mesh)
    if path.is_file():
        return path.resolve()
    candidate = get_mesh_path(path.name)
    if candidate.is_file():
        return candidate.resolve()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"Mesh not found: {mesh} (also tried {candidate})")


def list_ts_meshes() -> list[Path]:
    """Return sorted ``TS*.obj`` paths under ``data/mesh/``."""
    return sorted(MESH_DIR.glob("TS*.obj"))


def resolve_mesh_targets(
    meshes: Optional[Sequence[str]] = None,
    *,
    all_meshes: bool = False,
) -> list[Path]:
    """Resolve CLI mesh targets.

    Parameters
    ----------
    meshes
        Explicit mesh names or paths. Ignored when ``all_meshes`` is True.
    all_meshes
        If True, return every ``TS*.obj`` under ``data/mesh/``.

    Raises
    ------
    ValueError
        If neither ``--all`` nor any mesh arguments were provided, or if
        ``--all`` is combined with explicit mesh arguments.
    FileNotFoundError
        If a requested mesh cannot be resolved, or if ``--all`` finds none.
    """
    if all_meshes and meshes:
        raise ValueError("Pass either --all or explicit mesh arguments, not both")
    if all_meshes:
        targets = list_ts_meshes()
        if not targets:
            raise FileNotFoundError(f"No TS*.obj meshes found under {MESH_DIR}")
        return [p.resolve() for p in targets]
    if not meshes:
        raise ValueError("Provide one or more mesh arguments, or pass --all")
    return [resolve_mesh_path(m) for m in meshes]


def default_polylines_path(mesh_path: Path) -> Path:
    """Default ``data/skeletons/<stem>.polylines.txt`` for a mesh."""
    return get_skeleton_path(f"{mesh_path.stem}.polylines.txt")


def default_swc_path(mesh_path: Path) -> Path:
    """Default ``data/swc/pixels/<stem>.swc`` for a mesh."""
    return get_swc_path(f"{mesh_path.stem}.swc", units="pixels")


@dataclass(frozen=True)
class MeshToSwcResult:
    """Outputs from :func:`mesh_to_swc`."""

    mesh_path: Path
    polylines_path: Path
    swc_path: Optional[Path]


def skeletonize_mesh(
    mesh_path: PathLike,
    polylines_path: PathLike,
    *,
    profile: str = "auto",
    branching: str = "sparse",
    **skeletonize_kwargs: Any,
) -> Path:
    """Skeletonize a closed triangle mesh with pymcfs and write polylines.

    Defaults match pymcfs toric-spine batch settings
    (``profile=\"auto\"``, ``branching=\"sparse\"``, tip extension on).
    See :data:`TORIC_SPINES_SKELETONIZE_DEFAULTS`.

    Parameters
    ----------
    mesh_path
        Mesh file path, or a bare filename under ``data/mesh/``.
    polylines_path
        Destination ``.polylines.txt`` path.
    profile
        pymcfs skeletonization profile (default ``\"auto\"`` for TS meshes).
    branching
        Branching preference when ``profile=\"auto\"`` (default ``\"sparse\"``).
    **skeletonize_kwargs
        Forwarded to ``pymcfs.skeletonize`` (overrides toric-spine defaults).

    Returns
    -------
    Path
        Path to the written polylines file.
    """
    load_and_repair, skeletonize = _require_pymcfs()

    mesh_path = resolve_mesh_path(mesh_path)
    polylines_path = Path(polylines_path)
    polylines_path.parent.mkdir(parents=True, exist_ok=True)

    options = dict(TORIC_SPINES_SKELETONIZE_DEFAULTS)
    options["profile"] = profile
    options["branching"] = branching
    options.update(skeletonize_kwargs)

    logger.info(
        "Skeletonizing mesh %s (profile=%s branching=%s extend_tips=%s)",
        mesh_path,
        options.get("profile"),
        options.get("branching"),
        options.get("extend_tips"),
    )
    mesh = load_and_repair(str(mesh_path))
    skeleton = skeletonize(mesh, **options)
    skeleton.write_polylines(str(polylines_path))
    logger.info("Wrote polylines to %s", polylines_path)
    return polylines_path.resolve()


def fit_swc(
    mesh_path: PathLike,
    polylines_path: PathLike,
    swc_path: PathLike,
    *,
    max_edge_length_frac: float = 0.08,
    radius_strategy: str = "equivalent_area",
    scale_radii: bool = True,
    basis_optimize: bool = False,
    basis_optimizer_options: Optional[dict[str, Any]] = None,
    scale_metric: str = "surface_area",
) -> Path:
    """Fit a cable SWC to a mesh + skeleton with mascaf.

    Parameters
    ----------
    mesh_path
        Mesh file path, or a bare filename under ``data/mesh/``.
    polylines_path
        Skeleton polylines text file.
    swc_path
        Destination SWC path.
    max_edge_length_frac
        ``FitOptions.max_edge_length`` as a fraction of the mesh bounding-box
        diagonal (default ``0.08``).
    radius_strategy
        Radius estimation strategy (default ``\"equivalent_area\"``).
    scale_radii
        If True, call ``scale_radii_to_match_mesh`` before export.
    basis_optimize
        If True, enable mascaf ``BasisOptimizer`` via ``FitOptions``.
    basis_optimizer_options
        Optional kwargs for ``BasisOptimizerOptions`` when ``basis_optimize``
        is True.
    scale_metric
        Metric for radius scaling (default ``\"surface_area\"``).

    Returns
    -------
    Path
        Path to the written SWC file.
    """
    (
        BasisOptimizerOptions,
        CableFitter,
        FitOptions,
        MeshManager,
        SkeletonGraph,
    ) = _require_mascaf()

    mesh_path = resolve_mesh_path(mesh_path)
    polylines_path = Path(polylines_path)
    swc_path = Path(swc_path)
    if not polylines_path.is_file():
        raise FileNotFoundError(f"Polylines file not found: {polylines_path}")

    swc_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = None
    if basis_optimize:
        opts = dict(basis_optimizer_options or {})
        optimizer = BasisOptimizerOptions(**opts)

    mesh_mgr = MeshManager(mesh_path=str(mesh_path))
    bbox_diag = float(mesh_mgr.bounding_box_diagonal())
    max_edge_length = float(max_edge_length_frac) * bbox_diag

    fit_options = FitOptions(
        max_edge_length=max_edge_length,
        radius_strategy=radius_strategy,
        basis_optimizer_options=optimizer,
    )

    logger.info(
        "Fitting SWC from mesh %s and skeleton %s "
        "(max_edge_length_frac=%s → max_edge_length=%s, bbox_diag=%s, "
        "radius_strategy=%s, scale_radii=%s, basis_optimize=%s)",
        mesh_path,
        polylines_path,
        max_edge_length_frac,
        max_edge_length,
        bbox_diag,
        radius_strategy,
        scale_radii,
        basis_optimize,
    )
    skeleton = SkeletonGraph.from_txt(str(polylines_path))
    morphology = CableFitter(fit_options).fit(mesh_mgr, skeleton)

    if scale_radii:
        morphology.scale_radii_to_match_mesh(mesh_mgr, metric=scale_metric)

    morphology.to_swc_file(str(swc_path))
    logger.info("Wrote SWC to %s", swc_path)
    return swc_path.resolve()


def mesh_to_swc(
    mesh_path: PathLike,
    *,
    polylines_path: Optional[PathLike] = None,
    swc_path: Optional[PathLike] = None,
    skip_skeletonize: bool = False,
    polylines_only: bool = False,
    profile: str = "auto",
    max_edge_length_frac: float = 0.08,
    radius_strategy: str = "equivalent_area",
    scale_radii: bool = True,
    basis_optimize: bool = False,
    basis_optimizer_options: Optional[dict[str, Any]] = None,
    scale_metric: str = "surface_area",
    **skeletonize_kwargs: Any,
) -> MeshToSwcResult:
    """Run mesh → polylines → SWC (or a subset of those steps).

    Default outputs are ``data/skeletons/<stem>.polylines.txt`` and
    ``data/swc/pixels/<stem>.swc``.
    """
    mesh_path = resolve_mesh_path(mesh_path)
    out_polylines = (
        Path(polylines_path)
        if polylines_path is not None
        else default_polylines_path(mesh_path)
    )
    out_swc = Path(swc_path) if swc_path is not None else default_swc_path(mesh_path)

    SKELETONS_DIR.mkdir(parents=True, exist_ok=True)
    SWC_PIXELS_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_skeletonize:
        skeletonize_mesh(
            mesh_path,
            out_polylines,
            profile=profile,
            **skeletonize_kwargs,
        )
    elif not out_polylines.is_file():
        raise FileNotFoundError(
            f"skip_skeletonize=True but polylines not found: {out_polylines}"
        )

    if polylines_only:
        return MeshToSwcResult(
            mesh_path=mesh_path,
            polylines_path=out_polylines.resolve(),
            swc_path=None,
        )

    written_swc = fit_swc(
        mesh_path,
        out_polylines,
        out_swc,
        max_edge_length_frac=max_edge_length_frac,
        radius_strategy=radius_strategy,
        scale_radii=scale_radii,
        basis_optimize=basis_optimize,
        basis_optimizer_options=basis_optimizer_options,
        scale_metric=scale_metric,
    )
    return MeshToSwcResult(
        mesh_path=mesh_path,
        polylines_path=out_polylines.resolve(),
        swc_path=written_swc,
    )
