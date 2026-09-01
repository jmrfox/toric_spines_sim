"""Prepare toric-spine SWCs and pointsets for simulation (pixels → microns).

Pixel-space inputs live under ``data/swc/pixels`` and ``data/nff``. This module
writes:

- ``data/pointsets/pixels/<stem>_AZ.txt`` — raw active-zone XYZ from NFF
- ``data/pointsets/microns/<stem>_synpts.txt`` — AZ projected onto the SWC, in µm
- ``data/swc/microns/<stem>_wsink_r<R>um.swc`` — spine + cylindrical sink, in µm
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from swctools import FrustaSet, PointSet, SWCModel

from toric_spines_sim.geometry.dendrite import write_xyz_points
from toric_spines_sim.geometry.sink import (
    SinkGeometry,
    append_sink_to_swc,
    append_sink_to_swc_multi_neck_points,
    optimal_sink_direction,
)
from toric_spines_sim.paths import (
    NFF_DIR,
    POINTSETS_PIXELS_DIR,
    SWC_PIXELS_DIR,
    UM_PER_PX,
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.utils import (
    load_xyz_points,
    read_nff_points_and_write_txt_file,
    read_nff_s_points,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

_TS_STEM = re.compile(r"^TS\d+$")

DEFAULT_SINK_RADIUS_UM = 10.0
DEFAULT_SINK_CONNECTOR_LENGTH_UM = 5.0
DEFAULT_SINK_N_CYLINDERS = 5
DEFAULT_SINK_TAG = 5
DEFAULT_SINK_TIP_TAG = 6


def list_ts_spine_swcs() -> list[Path]:
    """Return sorted ``TS{n}.swc`` paths under ``data/swc/pixels`` (no ``_wsink_``)."""
    return sorted(
        p for p in SWC_PIXELS_DIR.glob("TS*.swc") if _TS_STEM.match(p.stem)
    )


def resolve_swc_path(swc: PathLike) -> Path:
    """Resolve a spine SWC path or bare stem (e.g. ``TS1`` / ``TS1.swc``)."""
    path = Path(swc)
    if path.exists():
        return path.resolve()
    name = path.name
    if not name.endswith(".swc"):
        name = f"{name}.swc"
    candidate = SWC_PIXELS_DIR / name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"SWC not found: {swc} (also tried {candidate})")


def resolve_swc_targets(
    swcs: Optional[Sequence[str]] = None,
    *,
    all_swcs: bool = False,
) -> list[Path]:
    """Resolve CLI SWC targets under ``data/swc/pixels``."""
    if all_swcs and swcs:
        raise ValueError("Pass either --all or explicit SWC arguments, not both")
    if all_swcs:
        targets = list_ts_spine_swcs()
        if not targets:
            raise FileNotFoundError(f"No TS{{n}}.swc files found under {SWC_PIXELS_DIR}")
        return [p.resolve() for p in targets]
    if not swcs:
        raise ValueError("Provide one or more SWC arguments, or pass --all")
    return [resolve_swc_path(s) for s in swcs]


def nff_spine_stem(nff_path: Path) -> str:
    """``TS1_AZ.nff`` → ``TS1``."""
    stem = nff_path.stem
    if stem.endswith("_AZ"):
        return stem[: -len("_AZ")]
    return stem


def convert_nff_active_zone(nff_path: PathLike, out_path: Optional[PathLike] = None) -> Path:
    """Write NFF ``s`` points to ``data/pointsets/pixels/<stem>.txt``.

    If the NFF has no ``s`` points, writes an empty file and logs a warning.
    """
    nff_path = Path(nff_path)
    if out_path is None:
        POINTSETS_PIXELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = POINTSETS_PIXELS_DIR / f"{nff_path.stem}.txt"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    points = read_nff_s_points(nff_path)
    if not points:
        logger.warning("No s-points in %s; skipping AZ file", nff_path.name)
        if Path(out_path).exists() and Path(out_path).stat().st_size == 0:
            Path(out_path).unlink()
        return Path(out_path)
    read_nff_points_and_write_txt_file(nff_path, out_path)
    return Path(out_path)


def convert_all_nff_active_zones() -> list[Path]:
    """Convert every ``data/nff/*.nff`` file that has ``s`` points to a pixel AZ file."""
    written: list[Path] = []
    paths = sorted(NFF_DIR.glob("*.nff"))
    if not paths:
        raise FileNotFoundError(f"No .nff files found under {NFF_DIR}")
    for nff_path in paths:
        out = convert_nff_active_zone(nff_path)
        if out.exists() and out.stat().st_size > 0:
            written.append(out)
            logger.info("Wrote AZ pointset %s from %s", out, nff_path.name)
    return written


def scale_swc_file(swc_in: PathLike, swc_out: PathLike, scale: float) -> Path:
    """Scale SWC coordinates and radii, preserving CYCLE_BREAK header comments."""
    swc_in = Path(swc_in)
    swc_out = Path(swc_out)
    swc_out.parent.mkdir(parents=True, exist_ok=True)
    model = SWCModel.from_swc_file(str(swc_in), validate_reconnections=False)
    scaled = model.scale(scale)
    scaled.to_swc_file(str(swc_out))
    return swc_out.resolve()


def write_synpts_microns(
    swc_px: PathLike,
    az_px: PathLike,
    synpts_um: Optional[PathLike] = None,
    *,
    um_per_px: float = UM_PER_PX,
) -> Path:
    """Project pixel AZ points onto the SWC, scale to microns, and write synpts."""
    swc_px = Path(swc_px)
    az_px = Path(az_px)
    if synpts_um is None:
        synpts_um = get_pointset_path(f"{swc_px.stem}_synpts.txt", units="microns")
    else:
        synpts_um = Path(synpts_um)
    synpts_um.parent.mkdir(parents=True, exist_ok=True)

    swc_model = SWCModel.from_swc_file(str(swc_px), validate_reconnections=False)
    frusta = FrustaSet.from_swc_model(swc_model, sides=20, end_caps=False)
    az_pointset = PointSet.from_txt_file(str(az_px))
    if az_pointset is None or not az_pointset.points:
        raise ValueError(f"No AZ points in {az_px}")
    projected = az_pointset.project_onto_frusta(frusta)
    synpts = projected.scale(um_per_px)
    synpts.to_txt_file(str(synpts_um))
    logger.info(
        "Wrote %d synpts (µm) to %s",
        len(synpts.points),
        synpts_um,
    )
    return synpts_um.resolve()


def _swc_root_xyz(swc_path: Path) -> Tuple[float, float, float]:
    model = SWCModel.from_swc_file(str(swc_path), validate_reconnections=False)
    roots = model.roots()
    if not roots:
        raise ValueError(f"SWC has no root node: {swc_path}")
    if len(roots) > 1:
        logger.warning("%s has %d roots; using node %s", swc_path, len(roots), roots[0])
    xyz = model.get_node_xyz(roots[0])
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def resolve_neck_points_px(
    swc_px: Path,
    neck_file: Optional[PathLike] = None,
) -> Tuple[list[Tuple[float, float, float]], str]:
    """Load pixel-space neck point(s), or fall back to the SWC root."""
    if neck_file is not None:
        path = Path(neck_file)
        if not path.exists():
            raise FileNotFoundError(f"Neck point file not found: {path}")
        return load_xyz_points(path), f"file:{path}"

    default = POINTSETS_PIXELS_DIR / f"{swc_px.stem}_neckpoint.txt"
    if default.exists():
        return load_xyz_points(default), f"file:{default}"

    root_xyz = _swc_root_xyz(swc_px)
    logger.warning(
        "No neckpoint file for %s; attaching sink at SWC root (%.3f %.3f %.3f)",
        swc_px.stem,
        *root_xyz,
    )
    return [root_xyz], "swc_root"


def append_sink_write_microns(
    swc_px: PathLike,
    *,
    neck_file: Optional[PathLike] = None,
    radius_um: float = DEFAULT_SINK_RADIUS_UM,
    connector_length_um: float = DEFAULT_SINK_CONNECTOR_LENGTH_UM,
    n_cylinders: int = DEFAULT_SINK_N_CYLINDERS,
    um_per_px: float = UM_PER_PX,
    swc_out: Optional[PathLike] = None,
    neck_out: Optional[PathLike] = None,
    tag: int = DEFAULT_SINK_TAG,
    last_segment_tag: int = DEFAULT_SINK_TIP_TAG,
) -> Path:
    """Scale a pixel SWC to microns, append a sink, and write the combined file.

    Sink dimensions are specified in microns. The sink axis is the optimal
    direction away from the morphology at the neck.
    """
    swc_px = Path(swc_px)
    if swc_out is None:
        swc_out = get_swc_path(f"{swc_px.stem}_wsink_r{radius_um:g}um.swc", units="microns")
    else:
        swc_out = Path(swc_out)
    swc_out.parent.mkdir(parents=True, exist_ok=True)

    neck_xyzs_px, neck_source = resolve_neck_points_px(swc_px, neck_file)
    neck_xyzs_um = [
        (x * um_per_px, y * um_per_px, z * um_per_px) for x, y, z in neck_xyzs_px
    ]

    if neck_out is None:
        neck_out = get_pointset_path(f"{swc_px.stem}_neckpoint.txt", units="microns")
    write_xyz_points(neck_out, neck_xyzs_um)

    geom = SinkGeometry(
        radius=float(radius_um),
        length=2.0 * float(radius_um),
        n_cylinders=int(n_cylinders),
        connector_length=float(connector_length_um),
        axis="x",  # replaced below after direction is known
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        scaled_swc = tmp_dir / f"{swc_px.stem}.swc"
        scale_swc_file(swc_px, scaled_swc, um_per_px)
        neck_um_tmp = tmp_dir / "neck.txt"
        write_xyz_points(neck_um_tmp, neck_xyzs_um)

        direction = optimal_sink_direction(neck_um_tmp, scaled_swc)
        geom.axis = direction
        logger.info(
            "%s sink axis=%s neck_source=%s radius=%.3g µm",
            swc_px.stem,
            direction,
            neck_source,
            radius_um,
        )

        if len(neck_xyzs_um) > 1:
            written = append_sink_to_swc_multi_neck_points(
                swc_in=scaled_swc,
                swc_out=swc_out,
                neck_points=neck_um_tmp,
                geom=geom,
                tag=tag,
                last_segment_tag=last_segment_tag,
            )
        else:
            written = append_sink_to_swc(
                swc_in=scaled_swc,
                swc_out=swc_out,
                neck_coords=neck_xyzs_um[0],
                geom=geom,
                tag=tag,
                last_segment_tag=last_segment_tag,
            )

    logger.info("Wrote spine+sink (µm) to %s", written)
    return Path(written).resolve()
