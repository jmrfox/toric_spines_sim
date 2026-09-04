"""Tests for per-spine axon assignment files and topology loading."""

from pathlib import Path

import pytest

from toric_spines_sim.paths import DATA_DIR, get_data_path, get_pointset_path
from toric_spines_sim.utils import load_xyz_points

from simulations.common.utils import count_assignment_axons, load_axon_topology

# TS1 keeps its original 10-axon map; others round-robin into min(10, N).
AXON_MAP_SPECS = {
    "ts1": {"stem": "TS1", "n_synapses": 25, "n_axons": 10},
    "ts2": {"stem": "TS2", "n_synapses": 6, "n_axons": 6},
    "ts3": {"stem": "TS3", "n_synapses": 46, "n_axons": 10},
    "ts4": {"stem": "TS4", "n_synapses": 23, "n_axons": 10},
    "ts48": {"stem": "TS48", "n_synapses": 18, "n_axons": 10},
    "ts67": {"stem": "TS67", "n_synapses": 8, "n_axons": 8},
    "ts76": {"stem": "TS76", "n_synapses": 5, "n_axons": 5},
}


def _parse_assignment_file(path: Path) -> list[list[int]]:
    axons: list[list[int]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        axons.append([int(part.strip()) for part in stripped.split(",")])
    return axons


@pytest.mark.parametrize("sim_key", list(AXON_MAP_SPECS))
def test_axon_assignment_covers_synpts(sim_key: str) -> None:
    spec = AXON_MAP_SPECS[sim_key]
    stem = spec["stem"]
    assignment_path = get_data_path("ts_axons", f"{sim_key}_axons.txt")
    synpts_path = get_pointset_path(f"{stem}_synpts.txt", units="microns")

    assert assignment_path.is_file(), assignment_path
    n_synapses = len(load_xyz_points(synpts_path))
    assert n_synapses == spec["n_synapses"]

    axons = _parse_assignment_file(assignment_path)
    assert len(axons) == spec["n_axons"]
    if sim_key == "ts1":
        assert len(axons) == 10
    else:
        assert len(axons) == min(10, n_synapses)

    assigned: list[int] = []
    for axon in axons:
        assert axon, "assignment file must not contain empty axons"
        assigned.extend(axon)

    assert sorted(assigned) == list(range(1, n_synapses + 1))


def test_load_axon_topology_derives_n_axons() -> None:
    path = get_data_path("ts_axons", "ts76_axons.txt")
    n_synapses_per_axon, axon_synapses = load_axon_topology(path)
    assert len(axon_synapses) == count_assignment_axons(path) == 5
    assert n_synapses_per_axon == [1, 1, 1, 1, 1]
    assert sorted(idx for synapses in axon_synapses for idx in synapses) == list(
        range(5)
    )


def test_all_assignment_files_are_known() -> None:
    axon_dir = DATA_DIR / "ts_axons"
    found = {path.stem.replace("_axons", "") for path in axon_dir.glob("*_axons.txt")}
    assert found == set(AXON_MAP_SPECS)


@pytest.mark.parametrize("sim_key", list(AXON_MAP_SPECS))
def test_model_config_n_axons_matches_file(sim_key: str) -> None:
    module = __import__(f"simulations.{sim_key}.inputs", fromlist=["MODEL"])
    model = module.MODEL
    spec = AXON_MAP_SPECS[sim_key]
    assert model.stem == spec["stem"]
    assert model.n_axons() == spec["n_axons"]
    assert model.swc_path().is_file()
    assert model.synpts_path().is_file()
    assert model.axon_assignment_file.is_file()
    order = model.resolved_sequential_axon_order()
    assert sorted(order) == list(range(spec["n_axons"]))
    if sim_key == "ts1":
        assert order == [5, 6, 9, 8, 2, 4, 7, 1, 3, 0]
    else:
        assert order == list(range(spec["n_axons"]))
