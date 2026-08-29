"""Path utilities for toric_spines_sim."""

from pathlib import Path

# Project root is two levels up from this file (toric_spines_sim/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SWC_DIR = DATA_DIR / "swc"
SWC_PIXELS_DIR = SWC_DIR / "pixels"
SWC_MICRONS_DIR = SWC_DIR / "microns"
POINTSETS_DIR = DATA_DIR / "pointsets"
POINTSETS_PIXELS_DIR = POINTSETS_DIR / "pixels"
POINTSETS_MICRONS_DIR = POINTSETS_DIR / "microns"
SIMULATIONS_DIR = PROJECT_ROOT / "simulations"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


class PathConfig:
    """Configuration class for accessing project paths."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.swc_pixels_dir = SWC_PIXELS_DIR
        self.swc_microns_dir = SWC_MICRONS_DIR
        self.pointsets_pixels_dir = POINTSETS_PIXELS_DIR
        self.pointsets_microns_dir = POINTSETS_MICRONS_DIR
        self.simulations_dir = SIMULATIONS_DIR
        self.notebooks_dir = NOTEBOOKS_DIR


def get_data_path(*parts: str) -> Path:
    """Get a path relative to the data directory."""
    return DATA_DIR.joinpath(*parts)


def get_swc_path(filename: str, units: str = "microns") -> Path:
    """Get path to an SWC file.

    Args:
        filename: Name of the SWC file.
        units: Either 'pixels' or 'microns'.
    """
    if units == "pixels":
        return SWC_PIXELS_DIR / filename
    elif units == "microns":
        return SWC_MICRONS_DIR / filename
    else:
        raise ValueError(f"units must be 'pixels' or 'microns', got {units}")


def get_pointset_path(filename: str, units: str = "microns") -> Path:
    """Get path to a pointset file.

    Args:
        filename: Name of the pointset file.
        units: Either 'pixels' or 'microns'.
    """
    if units == "pixels":
        return POINTSETS_PIXELS_DIR / filename
    elif units == "microns":
        return POINTSETS_MICRONS_DIR / filename
    else:
        raise ValueError(f"units must be 'pixels' or 'microns', got {units}")


def get_simulation_path(simulation_name: str, *parts: str) -> Path:
    """Get a path within a specific simulation directory."""
    path = SIMULATIONS_DIR / simulation_name
    if parts:
        path = path.joinpath(*parts)
    return path
