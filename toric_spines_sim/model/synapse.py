"""
Synapse definitions and preparation utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from jscip import IndependentScalarParameter, ParameterBank, ParameterSet

logger = logging.getLogger(__name__)

Location = Tuple[float, float, float]

# model -> mechanism, global ParameterSet keys, and synapse/mechanism param names
MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "ampa": {
        "mechanism": "ampasyn",
        "global_keys": ["ampa_gmax_uS", "ampa_tau_ms", "ampa_e_mV"],
        "synapse_params": {
            "gmax_uS": "ampa_gmax_uS",
            "tau_ms": "ampa_tau_ms",
            "e_mV": "ampa_e_mV",
        },
        "mechanism_params": {
            "gmax": "gmax_uS",
            "tau": "tau_ms",
            "e": "e_mV",
        },
    },
    "nmda": {
        "mechanism": "nmdasyn",
        "global_keys": [
            "nmda_gmax_uS",
            "nmda_e_mV",
            "nmda_tau_r_ms",
            "nmda_tau_d_ms",
        ],
        "synapse_params": {
            "gmax_uS": "nmda_gmax_uS",
            "e_mV": "nmda_e_mV",
            "tau_r_ms": "nmda_tau_r_ms",
            "tau_d_ms": "nmda_tau_d_ms",
        },
        "mechanism_params": {
            "gmax": "gmax_uS",
            "e": "e_mV",
            "tau_r": "tau_r_ms",
            "tau_d": "tau_d_ms",
        },
    },
    "gabaa": {
        "mechanism": "gabaasyn",
        "global_keys": ["gabaa_gmax_uS", "gabaa_tau_ms", "gabaa_e_mV"],
        "synapse_params": {
            "gmax_uS": "gabaa_gmax_uS",
            "tau_ms": "gabaa_tau_ms",
            "e_mV": "gabaa_e_mV",
        },
        "mechanism_params": {
            "gmax": "gmax_uS",
            "tau": "tau_ms",
            "e": "e_mV",
        },
    },
    "gabab": {
        "mechanism": "gababsyn",
        "global_keys": [
            "gabab_gmax_uS",
            "gabab_e_mV",
            "gabab_tau_r_ms",
            "gabab_tau_d_ms",
        ],
        "synapse_params": {
            "gmax_uS": "gabab_gmax_uS",
            "e_mV": "gabab_e_mV",
            "tau_r_ms": "gabab_tau_r_ms",
            "tau_d_ms": "gabab_tau_d_ms",
        },
        "mechanism_params": {
            "gmax": "gmax_uS",
            "e": "e_mV",
            "tau_r": "tau_r_ms",
            "tau_d": "tau_d_ms",
        },
    },
    "effexc": {
        "mechanism": "effexcsyn",
        "global_keys": [
            "effexc_gmax_uS",
            "effexc_nmda_ratio",
            "effexc_tau_ampa_ms",
            "effexc_tau_nmda_rise_ms",
            "effexc_tau_nmda_decay_ms",
            "effexc_e_mV",
            "effexc_mg_mM",
        ],
        "synapse_params": {
            "gmax_uS": "effexc_gmax_uS",
            "nmda_ratio": "effexc_nmda_ratio",
            "tau_ampa_ms": "effexc_tau_ampa_ms",
            "tau_nmda_rise_ms": "effexc_tau_nmda_rise_ms",
            "tau_nmda_decay_ms": "effexc_tau_nmda_decay_ms",
            "e_mV": "effexc_e_mV",
            "mg_mM": "effexc_mg_mM",
        },
        "mechanism_params": {
            "gmax": "gmax_uS",
            "nmda_ratio": "nmda_ratio",
            "tau_ampa": "tau_ampa_ms",
            "tau_nmda_rise": "tau_nmda_rise_ms",
            "tau_nmda_decay": "tau_nmda_decay_ms",
            "e": "e_mV",
            "mg": "mg_mM",
        },
    },
}


@dataclass
class SynapsePoint:
    """Synapse point specification."""

    location: Location
    model: str
    mechanism: str
    synapse_params: Dict[str, float]
    mechanism_params: Dict[str, float]


def _validate_model(model: str) -> Dict[str, object]:
    if model not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Invalid model type: {model}. Must be one of: {valid}.")
    return MODEL_REGISTRY[model]


def _build_parameter_bank(
    model: str,
    global_parameters: ParameterSet,
    parameter_override: Optional[ParameterBank] = None,
) -> ParameterBank:
    """Build a per-population ParameterBank from global ParameterSet values."""
    spec = _validate_model(model)
    global_keys: Sequence[str] = spec["global_keys"]  # type: ignore[assignment]

    bank_params: Dict[str, IndependentScalarParameter] = {}
    if "seed" in global_parameters.index:
        bank_params["seed"] = IndependentScalarParameter(
            float(global_parameters["seed"]),
            is_sampled=False,
        )
    for key in global_keys:
        bank_params[key] = IndependentScalarParameter(
            float(global_parameters[key]),
            is_sampled=False,
        )

    bank = ParameterBank(bank_params)
    if parameter_override is not None:
        bank.merge(parameter_override, on_collision="overwrite")
    return bank


def _extract_synapse_params(
    sampled: ParameterSet,
    model: str,
) -> Dict[str, float]:
    spec = MODEL_REGISTRY[model]
    synapse_param_map: Mapping[str, str] = spec["synapse_params"]  # type: ignore[assignment]
    return {
        short_name: float(sampled[global_key])
        for short_name, global_key in synapse_param_map.items()
    }


def _build_mechanism_params(
    synapse_params: Dict[str, float],
    model: str,
) -> Dict[str, float]:
    spec = MODEL_REGISTRY[model]
    mechanism_param_map: Mapping[str, str] = spec["mechanism_params"]  # type: ignore[assignment]
    return {
        mech_name: synapse_params[synapse_name]
        for mech_name, synapse_name in mechanism_param_map.items()
    }


def _load_locations(points_file: Union[str, Path]) -> list[Location]:
    try:
        points = np.atleast_2d(np.loadtxt(points_file))
    except Exception:
        logger.error(
            "Failed to load synapse points from %s", points_file, exc_info=True
        )
        raise
    return [(float(x), float(y), float(z)) for x, y, z in points]


class SynapsePopulation:
    """A homogeneous population of synapses sharing one model type.

    Each synapse is assigned parameters by sampling an internal ParameterBank
    built from ``global_parameters``. By default all parameters have
    ``is_sampled=False``, so each synapse receives the same values.
    """

    def __init__(
        self,
        model: str,
        locations: Sequence[Location],
        global_parameters: ParameterSet,
        parameter_override: Optional[ParameterBank] = None,
        label_prefix: str = "syn",
    ):
        spec = _validate_model(model)
        self.model = model
        self.mechanism: str = spec["mechanism"]  # type: ignore[assignment]
        self._parameter_bank = _build_parameter_bank(
            model, global_parameters, parameter_override
        )
        self.synapses: Dict[str, SynapsePoint] = {}

        for index, location in enumerate(locations):
            sampled = self._parameter_bank.sample()
            synapse_params = _extract_synapse_params(sampled, model)
            mechanism_params = _build_mechanism_params(synapse_params, model)
            label = f"{label_prefix}_{index}"
            self.synapses[label] = SynapsePoint(
                location=location,
                model=model,
                mechanism=self.mechanism,
                synapse_params=synapse_params,
                mechanism_params=mechanism_params,
            )

        logger.debug(
            "Built %d %s synapses (mechanism=%s)",
            len(self.synapses),
            model,
            self.mechanism,
        )

    @classmethod
    def from_file(
        cls,
        points_file: Union[str, Path],
        model: str,
        global_parameters: ParameterSet,
        parameter_override: Optional[ParameterBank] = None,
        label_prefix: str = "syn",
    ) -> SynapsePopulation:
        """Load XYZ locations from a file and build a synapse population."""
        locations = _load_locations(points_file)
        population = cls(
            model=model,
            locations=locations,
            global_parameters=global_parameters,
            parameter_override=parameter_override,
            label_prefix=label_prefix,
        )
        logger.debug(
            "Prepared %d synapses from %s (model=%s)",
            len(population.synapses),
            points_file,
            model,
        )
        return population

    @staticmethod
    def merge(*populations: SynapsePopulation) -> Dict[str, SynapsePoint]:
        """Combine multiple populations into one label -> SynapsePoint dict."""
        merged: Dict[str, SynapsePoint] = {}
        for population in populations:
            for label, synapse in population.synapses.items():
                if label in merged:
                    raise ValueError(f"Duplicate synapse label: {label}")
                merged[label] = synapse
        return merged


def prepare_ampa_synapses(
    points_file: Union[str, Path],
    parameters: ParameterSet,
    parameter_override: Optional[ParameterBank] = None,
    label_prefix: str = "syn",
) -> Dict[str, SynapsePoint]:
    """Prepare AMPA synapses from a file of 3D points.

    Deprecated: use ``SynapsePopulation.from_file(..., model='ampa')`` instead.
    """
    return SynapsePopulation.from_file(
        points_file,
        model="ampa",
        global_parameters=parameters,
        parameter_override=parameter_override,
        label_prefix=label_prefix,
    ).synapses


def prepare_nmda_synapses(
    points_file: Union[str, Path],
    parameters: ParameterSet,
    parameter_override: Optional[ParameterBank] = None,
    label_prefix: str = "syn",
) -> Dict[str, SynapsePoint]:
    """Prepare NMDA synapses from a file of 3D points.

    Deprecated: use ``SynapsePopulation.from_file(..., model='nmda')`` instead.
    """
    return SynapsePopulation.from_file(
        points_file,
        model="nmda",
        global_parameters=parameters,
        parameter_override=parameter_override,
        label_prefix=label_prefix,
    ).synapses
