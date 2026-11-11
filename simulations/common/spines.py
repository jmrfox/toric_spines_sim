"""Registry of toric spines and their axon-study configuration.

One place for what used to be seven copies of ``simulations/ts{id}/params.py``
and ``inputs.py``. Per-spine choices live in ``_OVERRIDES``; everything else
is shared. Results still go to ``simulations/ts{id}/axons/results/``.

    from simulations.common.spines import model_config, AXON_SPINES
    config = model_config("TS1")
"""

from __future__ import annotations

from typing import Callable, Dict

from jscip import ParameterBank

from toric_spines_sim.paths import get_data_path, get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_icx_parameter_bank

from simulations.common.model import ModelConfig

#: Every reconstructed toric spine with a sink SWC (``data/swc/microns``).
ALL_SPINES = ("TS1", "TS2", "TS3", "TS4", "TS21", "TS24", "TS48", "TS67", "TS76")
#: Spines with a synapse file and an axon map (axon studies).
AXON_SPINES = ("TS1", "TS2", "TS3", "TS4", "TS48", "TS67", "TS76")


def default_parameter_bank() -> ParameterBank:
    """Bank for the axon studies: living owl ICx (313 K, no background shunt)."""
    return make_icx_parameter_bank("in_vivo")


#: Per-spine differences from the shared defaults (ModelConfig fields).
_OVERRIDES: Dict[str, dict] = {
    "TS1": {"sequential_axon_order": [5, 6, 9, 8, 2, 4, 7, 1, 3, 0]},  # firing order for TS1
}


def normalize(spine: str) -> str:
    s = spine.upper()
    if not s.startswith("TS"):
        s = "TS" + s
    if s not in ALL_SPINES:
        raise ValueError(f"unknown spine {spine!r}; known: {', '.join(ALL_SPINES)}")
    return s


def model_config(spine: str, make_parameter_bank: Callable[[], ParameterBank] = default_parameter_bank) -> ModelConfig:
    """Axon-study configuration for ``spine`` (``"TS1"``, ``"ts1"`` or ``"1"``)."""
    s = normalize(spine)
    if s not in AXON_SPINES:
        raise ValueError(f"{s} has no synapse file / axon map; axon studies need one of {AXON_SPINES}")
    cfg = ModelConfig(
        spine_id=s,
        sim_key=s.lower(),
        swc_name=f"{s}_wsink_r10um.swc",
        synpts_name=f"{s}_synpts.txt",
        axon_assignment_file=get_data_path("ts_axons", f"{s.lower()}_axons.txt"),
        make_parameter_bank=make_parameter_bank,
    )
    for key, value in _OVERRIDES.get(s, {}).items():
        setattr(cfg, key, value)
    # fail early with a clear message if inputs are missing
    for path in (get_swc_path(cfg.swc_name, units="microns"),
                 get_pointset_path(cfg.synpts_name, units="microns"), cfg.axon_assignment_file):
        if not path.exists():
            raise FileNotFoundError(f"{s}: missing input {path}")
    return cfg
