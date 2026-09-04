"""Per-spine experiment configuration for shared axon studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jscip import ParameterBank

from toric_spines_sim.paths import (
    get_pointset_path,
    get_simulation_path,
    get_swc_path,
)

from simulations.common.utils import count_assignment_axons


def default_calculations() -> dict[str, bool]:
    return {
        "pairwise_heatmaps": True,
        "sequential_axons": True,
        "synchronicity_sweep": True,
        "random_jitter_sweep": True,
    }


@dataclass
class ModelConfig:
    """Paths, parameter bank, and analysis knobs for one spine model.

    Users typically construct this in a per-model ``inputs.py``. Report
    defaults match the original TS1 axon study; override fields there.
    """

    stem: str
    sim_key: str
    swc_name: str
    synpts_name: str
    axon_assignment_file: Path
    make_parameter_bank: Callable[[], ParameterBank]
    sequential_axon_order: list[int] | None = None
    calculations: dict[str, bool] = field(default_factory=default_calculations)
    short_t_ms: float = 300.0
    pulse_time_ms: float = 50.0
    sync_x_max_ms: float = 80.0
    sync_x_step_ms: float = 20.0
    sync_n_trials: int = 10
    sync_rng_seed: int = 0
    sync_extra_axon_sets: list[list[int]] = field(
        default_factory=lambda: [[0, 1, 2, 3, 4]]
    )
    sync_hist_normalize_per_column: bool = False
    pair_delay_min_ms: float = -100.0
    pair_delay_max_ms: float = 100.0
    pair_delay_step_ms: float = 10.0

    def n_axons(self) -> int:
        return count_assignment_axons(self.axon_assignment_file)

    def resolved_sequential_axon_order(self) -> list[int]:
        if self.sequential_axon_order is not None:
            return list(self.sequential_axon_order)
        return list(range(self.n_axons()))

    def resolved_sync_extra_axon_sets(self) -> list[list[int]]:
        n_axons = self.n_axons()
        valid: list[list[int]] = []
        for axon_set in self.sync_extra_axon_sets:
            if axon_set and all(0 <= axon_idx < n_axons for axon_idx in axon_set):
                valid.append(list(axon_set))
        return valid

    def swc_path(self) -> Path:
        return get_swc_path(self.swc_name, units="microns")

    def synpts_path(self) -> Path:
        return get_pointset_path(self.synpts_name, units="microns")

    def results_dir(self) -> Path:
        return get_simulation_path(self.sim_key, "axons", "results")
