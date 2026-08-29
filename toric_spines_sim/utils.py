"""General-purpose utility functions for toric_spines_sim."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


def load_xyz_points(value: Union[str, Path]) -> List[Tuple[float, float, float]]:
    """Load XYZ coordinates from a whitespace-delimited file.

    Each row must contain at least 3 values (x, y, z). Extra columns are ignored.

    Args:
        value: Path to the coordinate file.

    Returns:
        List of (x, y, z) tuples.

    Raises:
        ValueError: If the file is empty or rows have fewer than 3 values.
    """
    arr = np.loadtxt(str(value))
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        raise ValueError("neck point file is empty")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError("each neck point line must contain at least 3 values: x y z")
    pts: List[Tuple[float, float, float]] = []
    for row in arr:
        pts.append((float(row[0]), float(row[1]), float(row[2])))
    return pts


def equal_vectors(v1: Sequence[float], v2: Sequence[float], tol: float = 1e-6) -> bool:
    """Return True if two vectors are equal within a tolerance.

    Args:
        v1: First vector.
        v2: Second vector.
        tol: Tolerance for comparison (L2 norm of difference).
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return bool(np.linalg.norm(v1 - v2) < tol)


def join_tags_dsl(tags: Sequence[int]) -> str:
    """Build an Arbor region DSL expression for the union of multiple tags.

    The DSL ``(join ...)`` operator accepts exactly two arguments, so multiple
    tags are combined via nested binary joins.

    Parameters
    ----------
    tags : Sequence[int]
        One or more integer region tags.

    Returns
    -------
    str
        An s-expression string, e.g. ``"(join (tag 3) (tag 5))"``.

    Raises
    ------
    ValueError
        If *tags* is empty.
    """
    if len(tags) == 0:
        raise ValueError("tags must be non-empty")
    expr = f"(tag {tags[0]})"
    for tag in tags[1:]:
        expr = f"(join {expr} (tag {tag}))"
    return expr


def _iter_nff_s_points(lines: Iterable[str]) -> Iterable[Tuple[float, float, float]]:
    """Yield (x, y, z) for every line that begins with token 's' (case-insensitive).

    Lines are split on whitespace. The expected format is:
        s x y z [i]
    where the trailing index 'i' is optional and ignored.
    """
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        parts = s.split()
        if not parts:
            continue
        if parts[0].lower() != "s":
            continue
        if len(parts) < 4:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue
        yield (x, y, z)


def read_nff_s_points(
    path: Union[str, Path], *, return_numpy: bool = False
) -> Union[List[Tuple[float, float, float]], np.ndarray]:
    """Read all 's x y z i' points from an NFF file and return them in order.

    Parameters
    ----------
    path
        Path to the .nff file.
    return_numpy
        If True, return an (N, 3) NumPy array of dtype float. Otherwise, return a
        Python list of (x, y, z) tuples.

    Returns
    -------
    list[tuple[float, float, float]] | np.ndarray
        The sequence of (x, y, z) triples extracted from the file (possibly empty
        if no 's' lines are present).
    """
    p = Path(path)
    with p.open("r", errors="ignore") as f:
        points = list(_iter_nff_s_points(f))
    if return_numpy:
        return np.asarray(points, dtype=float)
    return points


def read_nff_points_and_write_txt_file(
    input_path: Union[str, Path], output_path: Union[str, Path]
):
    points = read_nff_s_points(input_path)
    np.savetxt(output_path, points, fmt="%.3f")
