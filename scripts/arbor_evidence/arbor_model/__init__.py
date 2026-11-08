"""Frozen copy of the Arbor model code (toric_spines_sim, Nov 2026).

Kept only so the evidence scripts in ``scripts/arbor_evidence`` and the golden
baselines (``scripts/make_golden.py``) stay reproducible after Arbor was
removed from the package. Install with ``uv sync --group arbor-evidence`` and
build the catalogue with ``bash scripts/arbor_evidence/make_custom_catalogue.sh``.
Do not use for new work: Arbor cannot close cable loops.
"""

from .gj import GapJunctionPoint, prepare_gap_junctions
from .model import CUSTOM_CATALOGUE_PATH, TSModel, check_catalogue, lumped_sink_membrane
from .recipe import TSRecipe
from .simulator import ArborSimulator

__all__ = ["ArborSimulator", "CUSTOM_CATALOGUE_PATH", "GapJunctionPoint", "TSModel", "TSRecipe",
           "check_catalogue", "lumped_sink_membrane", "prepare_gap_junctions"]
