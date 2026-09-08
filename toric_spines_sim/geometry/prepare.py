"""Prepare toric-spine SWCs and pointsets for simulation (pixels → microns).

Pixel-space inputs live under ``data/swc/pixels`` and ``data/nff``. This module
writes:

- ``data/pointsets/pixels/<spine_id>_AZ.txt`` — raw active-zone XYZ from NFF
- ``data/pointsets/microns/<spine_id>_synpts.txt`` — AZ projected onto the SWC, in µm
- ``data/pointsets/microns/<spine_id>_neckpoint.txt`` — neckpoints scaled to µm
- ``data/swc/pixels/<spine_id>_wsink_r<R>um.swc`` — spine + sink, pixel units
- ``data/swc/microns/<spine_id>_wsink_r<R>um.swc`` — spine + sink, micron units
"""

from __future__ import annotations

import logging
import re
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

_SPINE_ID_PATTERN = re.compile(r"^TS\d+$")
_SINK_LENGTH_RE = re.compile(r"(length=)([-+0-9.eE]+)")
_SINK_RADIUS_RE = re.compile(r"(radius=)([-+0-9.eE]+)")
_SINK_NECK_XYZ_RE = re.compile(
    r"(neck_xyz=)([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)

DEFAULT_SINK_RADIUS_UM = 10.0
DEFAULT_SINK_CONNECTOR_LENGTH_UM = 5.0
DEFAULT_SINK_N_CYLINDERS = 5
DEFAULT_SINK_TAG = 5
DEFAULT_SINK_TIP_TAG = 6


def list_ts_spine_swcs() -> list[Path]:
    """Return sorted ``TS{n}.swc`` paths under ``data/swc/pixels`` (no ``_wsink_``)."""
    return sorted(
        p for p in SWC_PIXELS_DIR.glob("TS*.swc") if _SPINE_ID_PATTERN.match(p.stem)
    )


def resolve_swc_path(swc: PathLike) -> Path:
    """Resolve a spine SWC path or bare spine id (e.g. ``TS1`` / ``TS1.swc``)."""
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


def nff_spine_id(nff_path: Path) -> str:
    """``TS1_AZ.nff`` → ``TS1``."""
    spine_id = nff_path.stem
    if spine_id.endswith("_AZ"):
        return spine_id[: -len("_AZ")]
    return spine_id


def convert_nff_active_zone(nff_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    """Write NFF ``s`` points to ``data/pointsets/pixels/<spine_id>.txt``.

    If the NFF has no ``s`` points, writes an empty file and logs a warning.
    """
    nff_path = Path(nff_path)
    if output_path is None:
        POINTSETS_PIXELS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = POINTSETS_PIXELS_DIR / f"{nff_path.stem}.txt"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    points = read_nff_s_points(nff_path)
    if not points:
        logger.warning("No s-points in %s; skipping AZ file", nff_path.name)
        if Path(output_path).exists() and Path(output_path).stat().st_size == 0:
            Path(output_path).unlink()
        return Path(output_path)
    read_nff_points_and_write_txt_file(nff_path, output_path)
    return Path(output_path)


def convert_all_nff_active_zones() -> list[Path]:
    """Convert every ``data/nff/*.nff`` file that has ``s`` points to a pixel AZ file."""
    written: list[Path] = []
    paths = sorted(NFF_DIR.glob("*.nff"))
    if not paths:
        raise FileNotFoundError(f"No .nff files found under {NFF_DIR}")
    for nff_path in paths:
        output_path = convert_nff_active_zone(nff_path)
        if output_path.exists() and output_path.stat().st_size > 0:
            written.append(output_path)
            logger.info("Wrote AZ pointset %s from %s", output_path, nff_path.name)
    return written


def _scale_sink_header_line(line: str, scale: float) -> str:
    """Scale length/radius/neck_xyz fields in a ``# SINK:`` header line."""
    if not line.startswith("# SINK:"):
        return line

    def _scale_num(match: re.Match[str]) -> str:
        return f"{match.group(1)}{float(match.group(2)) * scale:g}"

    line = _SINK_LENGTH_RE.sub(_scale_num, line)
    line = _SINK_RADIUS_RE.sub(_scale_num, line)

    def _scale_neck(match: re.Match[str]) -> str:
        x = float(match.group(2)) * scale
        y = float(match.group(3)) * scale
        z = float(match.group(4)) * scale
        return f"{match.group(1)}{x:.6f} {y:.6f} {z:.6f}"

    return _SINK_NECK_XYZ_RE.sub(_scale_neck, line)


def scale_swc_file(swc_in: PathLike, swc_out: PathLike, scale: float) -> Path:
    """Scale SWC coordinates and radii, preserving CYCLE_BREAK / SINK headers.

    ``SWCModel.scale`` keeps header text verbatim, so ``# SINK:`` length, radius,
    and ``neck_xyz`` are rewritten here to match the scaled geometry.
    """
    swc_in = Path(swc_in)
    swc_out = Path(swc_out)
    swc_out.parent.mkdir(parents=True, exist_ok=True)
    model = SWCModel.from_swc_file(str(swc_in), validate_reconnections=False)
    scaled = model.scale(scale)
    scaled.to_swc_file(str(swc_out))

    lines = swc_out.read_text(encoding="utf-8").splitlines(keepends=True)
    rewritten = [
        _scale_sink_header_line(line, scale) if line.startswith("# SINK:") else line
        for line in lines
    ]
    swc_out.write_text("".join(rewritten), encoding="utf-8")
    return swc_out.resolve()


def write_synpts_microns(
    swc_path_pixels: PathLike,
    az_path_pixels: PathLike,
    synpts_path_microns: Optional[PathLike] = None,
    *,
    um_per_px: float,
) -> Path:
    """Project pixel AZ points onto the SWC, scale to microns, and write synpts.

    Parameters
    ----------
    swc_path_pixels : path-like
        Pixel-space SWC.
    az_path_pixels : path-like
        Pixel-space active-zone points.
    synpts_path_microns : path-like, optional
        Output path. Default ``data/pointsets/microns/<spine_id>_synpts.txt``.
    um_per_px : float
        Scale factor (typically 0.005).

    Returns
    -------
    pathlib.Path
        Written micron synpts file.

    Examples
    --------
    >>> write_synpts_microns("TS1.swc", "TS1_AZ.txt", um_per_px=0.005)  # doctest: +SKIP
    """
    swc_path_pixels = Path(swc_path_pixels)
    az_path_pixels = Path(az_path_pixels)
    if synpts_path_microns is None:
        synpts_path_microns = get_pointset_path(
            f"{swc_path_pixels.stem}_synpts.txt", units="microns"
        )
    else:
        synpts_path_microns = Path(synpts_path_microns)
    synpts_path_microns.parent.mkdir(parents=True, exist_ok=True)

    swc_model = SWCModel.from_swc_file(str(swc_path_pixels), validate_reconnections=False)
    frusta = FrustaSet.from_swc_model(swc_model, sides=20, end_caps=False)
    az_pointset = PointSet.from_txt_file(str(az_path_pixels))
    if az_pointset is None or not az_pointset.points:
        raise ValueError(f"No AZ points in {az_path_pixels}")
    projected = az_pointset.project_onto_frusta(frusta)
    synpts = projected.scale(um_per_px)
    synpts.to_txt_file(str(synpts_path_microns))
    logger.info(
        "Wrote %d synpts (µm) to %s",
        len(synpts.points),
        synpts_path_microns,
    )
    return synpts_path_microns.resolve()


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
    swc_path_pixels: Path,
    neckpoint_path_pixels: Optional[PathLike] = None,
) -> Tuple[list[Tuple[float, float, float]], str]:
    """Load pixel-space neck point(s), or fall back to the SWC root."""
    if neckpoint_path_pixels is not None:
        path = Path(neckpoint_path_pixels)
        if not path.exists():
            raise FileNotFoundError(f"Neck point file not found: {path}")
        return load_xyz_points(path), f"file:{path}"

    default = POINTSETS_PIXELS_DIR / f"{swc_path_pixels.stem}_neckpoint.txt"
    if default.exists():
        return load_xyz_points(default), f"file:{default}"

    root_xyz = _swc_root_xyz(swc_path_pixels)
    logger.warning(
        "No neckpoint file for %s; attaching sink at SWC root (%.3f %.3f %.3f)",
        swc_path_pixels.stem,
        *root_xyz,
    )
    return [root_xyz], "swc_root"


def append_sink_write(
    swc_path_pixels: PathLike,
    *,
    neckpoint_path_pixels: Optional[PathLike] = None,
    radius_um: float = DEFAULT_SINK_RADIUS_UM,
    connector_length_um: float = DEFAULT_SINK_CONNECTOR_LENGTH_UM,
    n_cylinders: int = DEFAULT_SINK_N_CYLINDERS,
    um_per_px: float,
    output_swc_path_pixels: Optional[PathLike] = None,
    output_swc_path_microns: Optional[PathLike] = None,
    output_neckpoint_path_microns: Optional[PathLike] = None,
    tag: int = DEFAULT_SINK_TAG,
    last_segment_tag: int = DEFAULT_SINK_TIP_TAG,
) -> Tuple[Path, Path]:
    """Append a sink in pixel space, then write both pixel and micron SWCs.

    Sink dimensions are specified in microns and converted to pixels for
    attachment. Outputs:

    - ``data/swc/pixels/<spine_id>_wsink_r<R>um.swc``
    - ``data/swc/microns/<spine_id>_wsink_r<R>um.swc`` (scaled copy with SINK header fixed)
    - ``data/pointsets/microns/<spine_id>_neckpoint.txt``

    Returns ``(output_swc_path_pixels, output_swc_path_microns)``.
    """
    swc_path_pixels = Path(swc_path_pixels)
    spine_id = swc_path_pixels.stem
    if output_swc_path_pixels is None:
        output_swc_path_pixels = get_swc_path(
            f"{spine_id}_wsink_r{radius_um:g}um.swc", units="pixels"
        )
    else:
        output_swc_path_pixels = Path(output_swc_path_pixels)
    if output_swc_path_microns is None:
        output_swc_path_microns = get_swc_path(
            f"{spine_id}_wsink_r{radius_um:g}um.swc", units="microns"
        )
    else:
        output_swc_path_microns = Path(output_swc_path_microns)
    output_swc_path_pixels.parent.mkdir(parents=True, exist_ok=True)
    output_swc_path_microns.parent.mkdir(parents=True, exist_ok=True)

    neck_points_pixels, neck_source = resolve_neck_points_px(
        swc_path_pixels, neckpoint_path_pixels
    )
    neck_points_microns = [
        (x * um_per_px, y * um_per_px, z * um_per_px) for x, y, z in neck_points_pixels
    ]

    if output_neckpoint_path_microns is None:
        output_neckpoint_path_microns = get_pointset_path(
            f"{spine_id}_neckpoint.txt", units="microns"
        )
    write_xyz_points(output_neckpoint_path_microns, neck_points_microns)

    # Ensure a pixel neckpoint file exists for multi-neck append / direction.
    resolved_neckpoint_path_pixels = (
        Path(neckpoint_path_pixels)
        if neckpoint_path_pixels is not None
        else (POINTSETS_PIXELS_DIR / f"{spine_id}_neckpoint.txt")
    )
    if not resolved_neckpoint_path_pixels.exists():
        write_xyz_points(resolved_neckpoint_path_pixels, neck_points_pixels)

    radius_px = float(radius_um) / float(um_per_px)
    connector_length_px = float(connector_length_um) / float(um_per_px)
    direction = optimal_sink_direction(resolved_neckpoint_path_pixels, swc_path_pixels)

    geom = SinkGeometry(
        radius=radius_px,
        length=2.0 * radius_px,
        n_cylinders=int(n_cylinders),
        connector_length=connector_length_px,
        axis=direction,
    )
    logger.info(
        "%s sink axis=%s neck_source=%s radius=%.3g µm (%.3g px) necks=%d",
        spine_id,
        direction,
        neck_source,
        radius_um,
        radius_px,
        len(neck_points_pixels),
    )

    if len(neck_points_pixels) > 1:
        written_px = append_sink_to_swc_multi_neck_points(
            swc_in=swc_path_pixels,
            swc_out=output_swc_path_pixels,
            neck_points=resolved_neckpoint_path_pixels,
            geom=geom,
            tag=tag,
            last_segment_tag=last_segment_tag,
        )
    else:
        written_px = append_sink_to_swc(
            swc_in=swc_path_pixels,
            swc_out=output_swc_path_pixels,
            neck_coords=neck_points_pixels[0],
            geom=geom,
            tag=tag,
            last_segment_tag=last_segment_tag,
        )
    written_px = Path(written_px).resolve()
    written_um = scale_swc_file(written_px, output_swc_path_microns, um_per_px)
    logger.info("Wrote spine+sink (px) to %s", written_px)
    logger.info("Wrote spine+sink (µm) to %s", written_um)
    return written_px, written_um


def append_sink_write_microns(
    swc_path_pixels: PathLike,
    *,
    neckpoint_path_pixels: Optional[PathLike] = None,
    radius_um: float = DEFAULT_SINK_RADIUS_UM,
    connector_length_um: float = DEFAULT_SINK_CONNECTOR_LENGTH_UM,
    n_cylinders: int = DEFAULT_SINK_N_CYLINDERS,
    um_per_px: float,
    output_swc_path_microns: Optional[PathLike] = None,
    output_neckpoint_path_microns: Optional[PathLike] = None,
    tag: int = DEFAULT_SINK_TAG,
    last_segment_tag: int = DEFAULT_SINK_TIP_TAG,
    output_swc_path_pixels: Optional[PathLike] = None,
) -> Path:
    """Append a sink and write micron (and pixel) SWCs; return the micron path.

    Prefer :func:`append_sink_write` when both output paths are needed.
    """
    _px, um = append_sink_write(
        swc_path_pixels,
        neckpoint_path_pixels=neckpoint_path_pixels,
        radius_um=radius_um,
        connector_length_um=connector_length_um,
        n_cylinders=n_cylinders,
        um_per_px=um_per_px,
        output_swc_path_pixels=output_swc_path_pixels,
        output_swc_path_microns=output_swc_path_microns,
        output_neckpoint_path_microns=output_neckpoint_path_microns,
        tag=tag,
        last_segment_tag=last_segment_tag,
    )
    return um
