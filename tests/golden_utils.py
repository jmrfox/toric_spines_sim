"""Rebuild golden-baseline cases (tests/data/golden) as TSModel inputs."""

import json
from pathlib import Path

import numpy as np

from toric_spines_sim.model import SynapsePopulation, TSModel
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_icx_parameter_bank

GOLDEN = Path(__file__).parent / "data" / "golden"


def load(name):
    d = np.load(GOLDEN / f"{name}.npz", allow_pickle=True)
    return d, json.loads(str(d["meta"]))


def parameters(meta):
    case = meta["case"]
    bank = make_icx_parameter_bank(case["condition"])
    for key, value in meta["parameters"].items():
        if key in ("tau_m_ms", "gj_weight"):  # derived / removed since the baselines
            continue
        bank[key].value = value
    p = bank.sample()
    p["hh_tags"] = list(meta.get("hh_tags", []))
    return p


def built_model(name, restore_loops=False):
    """TSModel for a golden case; Arbor saw the loops cut, so they stay cut."""
    d, meta = load(name)
    case = meta["case"]
    p = parameters(meta)
    swc = get_swc_path(f"{case['spine']}_wsink_r10um.swc", units="microns")
    synpts = get_pointset_path(f"{case['spine']}_synpts.txt", units="microns")
    syn = SynapsePopulation.from_file(synpts, model=case["receptor"], global_parameters=p).synapses
    record = {f"p{k:02d}": tuple(x) for k, x in enumerate(d["probe_xyz"])}
    built = TSModel(swc, syn, record, p, restore_loops=restore_loops).build()
    events = {i: list(e) for i, e in enumerate(d["event_ms"]) if len(e)}
    nodes = [built.record_nodes[f"p{k:02d}"] for k in range(len(d["probe_xyz"]))]
    return d, meta, p, built, events, nodes
