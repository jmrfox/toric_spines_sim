"""Pairwise synapse interaction (k-matrix) helpers.

For two synapses *i* and *j* driven at rates *(λ_i, λ_j)*, run three
simulations (i only, j only, both) and fit

    V_{ij} - V_i - V_j = k * V_i * V_j

on the baseline-subtracted sink (or other probe) voltage. Uses
``TSSimulator`` so custom NMODL synapses and gap junctions stay consistent
with the rest of the package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from jscip import ParameterBank, ParameterSet

from toric_spines_sim.events import (
    DeterministicEventGenerator,
    StochasticEventGenerator,
    FlatRateCurve,
)
from toric_spines_sim.simulation import TSSimulator, SimulationResults

logger = logging.getLogger(__name__)


def _voltage_and_time(results: SimulationResults, probe_label: str):
    """Return ``(time_ms, voltage_mV)`` for one probe column.

    Pynapple stores the TsdFrame index in seconds.
    """
    traces = results.voltage_traces
    time_ms = np.asarray(traces.t, dtype=float) * 1000.0
    voltage = np.asarray(traces[probe_label].values, dtype=float).reshape(-1)
    return time_ms, voltage


def _events_for_rates(
    input_rates_hz: List[float],
    parameters: ParameterSet,
    event_type: str,
    synapse_labels: List[str],
):
    curves = [FlatRateCurve(rate_hz) for rate_hz in input_rates_hz]
    n_syn = [1] * len(input_rates_hz)
    if event_type == "periodic":
        generator = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=n_syn,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
            labels=synapse_labels,
        )
    elif event_type == "poisson":
        generator = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=n_syn,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
            seed=int(parameters["seed"]),
            labels=synapse_labels,
        )
    else:
        raise ValueError("event_type must be 'periodic' or 'poisson'")
    return generator.generate()


def simulation(
    swc_filepath: Union[str, Path],
    synpts_filepath: Union[str, Path],
    input_rates_hz: List[float],
    parameters: ParameterSet,
    record_point: Tuple[float, float, float],
    event_type: str = "poisson",
    probe_label: str = "probe_0",
) -> SimulationResults:
    """Run one ``TSSimulator`` job with per-synapse rates and a single probe."""
    n_syn = len(input_rates_hz)
    synapse_labels = [f"syn_{i}" for i in range(n_syn)]
    events = _events_for_rates(
        input_rates_hz, parameters, event_type, synapse_labels
    )
    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        events,
        parameters,
        record_points={probe_label: record_point},
    )
    logger.info(
        "k-matrix simulation: swc=%s synapses=%d T=%s ms seed=%s",
        swc_filepath,
        n_syn,
        parameters["T_ms"],
        parameters["seed"],
    )
    return sim.run()


def simulation_probe_dict(
    swc_filepath: Union[str, Path],
    synpts_filepath: Union[str, Path],
    input_rates_hz: List[float],
    parameters: ParameterSet,
    record_point: Tuple[float, float, float],
    event_type: str = "poisson",
    probe_label: str = "probe_0",
) -> Dict:
    """Like ``simulation`` but returns the older ``{"probe": arbor-samples, ...}`` dict.

    Notebooks that call ``add_time_series_from_arbor(results["probe"])`` can
    import this as ``simulation``.
    """
    results = simulation(
        swc_filepath,
        synpts_filepath,
        input_rates_hz,
        parameters,
        record_point,
        event_type=event_type,
        probe_label=probe_label,
    )
    time_ms, voltage = _voltage_and_time(results, probe_label)
    probe = [(np.column_stack([time_ms, voltage]), None)]
    return {
        "probe": probe,
        "events": results.input_events,
        "synapses": results.synapses,
        "gap_junctions": results.gap_junctions,
        "record_points": results.record_points,
        "cell": results.cell,
        "morphology": results.morphology,
        "segment_tree": results.segment_tree,
        "decor": results.decor,
        "labels": results.labels,
        "cvp": results.cvp,
        "results": results,
    }


def compute_pairwise_voltages(
    total_synapses: int,
    active_synapse_pair: Tuple[int, int],
    rate_pair: Tuple[float, float],
    swc_filepath: str,
    synpts_filepath: str,
    parameter_bank: ParameterBank,
    record_point: Tuple[float, float, float],
    seeds: list[int],
    probe_label: str = "probe_0",
    event_type: str = "poisson",
):
    """Baseline-subtracted voltages V(i), V(j), and V(i+j) over seeds."""
    total_time = parameter_bank["T_ms"].value
    parameter_bank["T_ms"].value = 100.0
    input_rates_hz_0 = [0.0] * total_synapses
    results_0 = simulation(
        swc_filepath=swc_filepath,
        synpts_filepath=synpts_filepath,
        input_rates_hz=input_rates_hz_0,
        parameters=parameter_bank.sample(),
        record_point=record_point,
        event_type=event_type,
        probe_label=probe_label,
    )
    _t0, v0 = _voltage_and_time(results_0, probe_label)
    baseline = float(v0[-1])
    logger.debug("Baseline voltage: %f", baseline)

    v_1 = []
    v_2 = []
    v_12 = []
    parameter_bank["T_ms"].value = total_time
    time = None
    for seed in seeds:
        seed = int(seed)
        logger.debug("Seed: %s", seed)
        parameter_bank["seed"].value = seed
        parameters = parameter_bank.sample()

        input_rates_hz_1 = [0.0] * total_synapses
        input_rates_hz_1[active_synapse_pair[0]] = rate_pair[0]
        results_1 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_1,
            parameters=parameters,
            record_point=record_point,
            event_type=event_type,
            probe_label=probe_label,
        )
        t, v = _voltage_and_time(results_1, probe_label)
        v_1.append(v - baseline)

        input_rates_hz_2 = [0.0] * total_synapses
        input_rates_hz_2[active_synapse_pair[1]] = rate_pair[1]
        results_2 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_2,
            parameters=parameters,
            record_point=record_point,
            event_type=event_type,
            probe_label=probe_label,
        )
        _, v = _voltage_and_time(results_2, probe_label)
        v_2.append(v - baseline)

        input_rates_hz_12 = [0.0] * total_synapses
        input_rates_hz_12[active_synapse_pair[0]] = rate_pair[0]
        input_rates_hz_12[active_synapse_pair[1]] = rate_pair[1]
        results_12 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_12,
            parameters=parameters,
            record_point=record_point,
            event_type=event_type,
            probe_label=probe_label,
        )
        time, v = _voltage_and_time(results_12, probe_label)
        v_12.append(v - baseline)

    synapse_locations = np.loadtxt(synpts_filepath)
    synapse_1_location = synapse_locations[active_synapse_pair[0]]
    synapse_2_location = synapse_locations[active_synapse_pair[1]]
    distance = np.linalg.norm(synapse_1_location - synapse_2_location)
    return {
        "time": time,
        "v_1": np.array(v_1),
        "v_2": np.array(v_2),
        "v_12": np.array(v_12),
        "intersynapse_distance": distance,
    }


def solve_k_linreg(v_1, v_2, v_12):
    """Fit ``V_ij - V_i - V_j = k * V_i * V_j`` with a linear regression.

    Flattened arrays from one or more seeds are pooled. The slope is ``k``;
    an intercept is also returned (usually near zero).

    Parameters
    ----------
    v_1, v_2, v_12 : array_like
        Baseline-subtracted voltages for synapse i, j, and both.

    Returns
    -------
    dict
        ``coeffs`` (slope, intercept), ``uncertainty``, and the flattened
        ``a = v_1*v_2``, ``b = v_12-v_1-v_2`` used in the fit.

    Examples
    --------
    >>> import numpy as np
    >>> v_1 = np.array([1.0, 2.0])
    >>> v_2 = np.array([1.0, 2.0])
    >>> v_12 = v_1 + v_2 + 0.5 * v_1 * v_2
    >>> solve_k_linreg(v_1, v_2, v_12)["coeffs"][0]
    np.float64(0.5)
    """
    a = (v_1 * v_2).flatten()
    b = (v_12 - v_1 - v_2).flatten()
    coeffs, cov_matrix = np.polyfit(a, b, 1, cov="unscaled")
    return {
        "coeffs": coeffs,
        "uncertainty": np.sqrt(np.diag(cov_matrix)),
        "a": a,
        "b": b,
    }


def compute_k_matrix(
    total_synapses: int,
    active_synapse_pair: tuple[int, int],
    rate_list: list[float],
    swc_filepath: str,
    synpts_filepath: str,
    parameter_bank: ParameterBank,
    record_location: tuple[float, float, float],
    seeds: list[int],
    probe_label: str = "probe_0",
    event_type: str = "poisson",
) -> Dict:
    """Fill a symmetric k-matrix over an outer product of rates for one pair.

    For each unordered rate pair ``(rate_list[i], rate_list[j])`` with
    ``j <= i``, run ``compute_pairwise_voltages`` and store the fitted ``k``.

    Parameters
    ----------
    total_synapses : int
        Length of the synapse points file (inactive synapses get rate 0).
    active_synapse_pair : tuple of int
        0-based indices ``(i, j)`` of the two synapses that fire.
    rate_list : list of float
        Rates (Hz) whose outer product fills the matrix.
    swc_filepath, synpts_filepath : str
        Morphology and synapse-point files.
    parameter_bank : ParameterBank
        Mutated in place for ``T_ms`` / ``seed`` during the sweep.
    record_location : tuple of float
        Probe XYZ (often the sink tip).
    seeds : list of int
        Poisson seeds averaged in the linear fit.
    probe_label : str
        Column name in ``SimulationResults.voltage_traces``.
    event_type : {'poisson', 'periodic'}

    Returns
    -------
    dict
        ``k_matrix``, ``k_unc_matrix``, per-pair ``sim_results``,
        ``rate_pair_list``, and ``intersynapse_distance``.

    Examples
    --------
    >>> out = compute_k_matrix(  # doctest: +SKIP
    ...     total_synapses=25,
    ...     active_synapse_pair=(0, 1),
    ...     rate_list=[10.0, 50.0],
    ...     swc_filepath="data/swc/microns/TS1_wsink_r10um.swc",
    ...     synpts_filepath="data/pointsets/microns/TS1_synpts.txt",
    ...     parameter_bank=bank,
    ...     record_location=(0.0, 0.0, 0.0),
    ...     seeds=[0, 1],
    ... )
    >>> out["k_matrix"].shape
    (2, 2)
    """
    n_rates = len(rate_list)
    k_matrix = np.zeros((n_rates, n_rates))
    k_unc_matrix = np.zeros((n_rates, n_rates))
    sim_results = {}
    rate_pair_list = []
    intersynapse_distance = None
    for i in range(n_rates):
        for j in range(i + 1):
            rate_pair = (rate_list[i], rate_list[j])
            rate_pair_list.append(rate_pair)
            results = compute_pairwise_voltages(
                total_synapses,
                active_synapse_pair,
                rate_pair,
                swc_filepath,
                synpts_filepath,
                parameter_bank,
                record_location,
                seeds,
                probe_label=probe_label,
                event_type=event_type,
            )
            intersynapse_distance = results["intersynapse_distance"]
            sim_results[rate_pair] = {
                "v_1": results["v_1"],
                "v_2": results["v_2"],
                "v_12": results["v_12"],
                "time": results["time"],
            }
            linreg_results = solve_k_linreg(
                results["v_1"], results["v_2"], results["v_12"]
            )
            k = linreg_results["coeffs"][0]
            k_matrix[i, j] = k
            k_matrix[j, i] = k
            k_unc_matrix[i, j] = linreg_results["uncertainty"][0]
            k_unc_matrix[j, i] = linreg_results["uncertainty"][0]

    return {
        "k_matrix": k_matrix,
        "k_unc_matrix": k_unc_matrix,
        "sim_results": sim_results,
        "rate_pair_list": rate_pair_list,
        "intersynapse_distance": intersynapse_distance,
    }
