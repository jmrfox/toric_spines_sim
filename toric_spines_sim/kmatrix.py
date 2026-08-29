"""K-matrix computation utilities for pairwise synapse interactions."""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Tuple

import arbor as A
import numpy as np
from jscip import ParameterBank, ParameterSet

from toric_spines_sim.model.synapse import SynapsePopulation
from toric_spines_sim.model.gj import prepare_gap_junctions
from toric_spines_sim.model import TSModel, TSRecipe
from toric_spines_sim.events import (
    DeterministicEventGenerator,
    StochasticEventGenerator,
)


# ---


def simulation(
    swc_filepath,
    synpts_filepath,
    input_rates_hz,
    parameters: ParameterSet,
    record_point,
    event_type="poisson",
):
    synapses = SynapsePopulation.from_file(
        synpts_filepath, model="ampa", global_parameters=parameters
    ).synapses
    record_points = {"probe_0": record_point}  # single record point
    gap_junctions = prepare_gap_junctions(swc_filepath, parameters=parameters)
    synapse_labels = list(synapses.keys())
    if event_type == "periodic":
        event_generator = DeterministicEventGenerator(
            rates_hz=input_rates_hz,
            labels=synapse_labels,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
        )
    elif event_type == "poisson":
        event_generator = StochasticEventGenerator(
            rates_hz=input_rates_hz,
            labels=synapse_labels,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
            seed=parameters["seed"],
        )
    else:
        raise ValueError("event_type must be 'periodic' or 'poisson'")
    events = event_generator.generate()
    tsm = TSModel(
        swc_path=swc_filepath,
        synapses=synapses,
        gap_junctions=gap_junctions,
        record_points=record_points,
        parameters=parameters,
    )
    build_cell_results = tsm.build_cell()
    cell = build_cell_results["cell"]
    recipe = TSRecipe(
        cell,
        synapses=synapses,
        gap_junctions=gap_junctions,
        record_points=record_points,
        events=events,
        parameters=parameters,
    )
    ctx = A.context()
    dec = A.partition_load_balance(recipe, ctx)
    sim = A.simulation(recipe, ctx, dec)
    dt_record_ms = parameters["dt_record_ms"]
    handle = sim.sample(0, "v_probe_0", A.regular_schedule(dt_record_ms * A.units.ms))
    sim.record(A.spike_recording.all)
    T_ms = parameters["T_ms"]
    dt_sim_ms = parameters["dt_sim_ms"]
    sim.run(T_ms * A.units.ms, dt_sim_ms * A.units.ms)
    simulation_results = {
        "probe": sim.samples(handle),
        "events": events,
        "synapses": synapses,
        "gap_junctions": gap_junctions,
        "record_points": record_points,
        "cell": cell,
        "morphology": build_cell_results["morphology"],
        "segment_tree": build_cell_results["segment_tree"],
        "decor": build_cell_results["decor"],
        "labels": build_cell_results["labels"],
        "cvp": build_cell_results["cvp"],
    }
    return simulation_results


# ---


def compute_pairwise_voltages(
    total_synapses: int,
    active_synapse_pair: Tuple[int, int],
    rate_pair: Tuple[float, float],
    swc_filepath: str,
    synpts_filepath: str,
    parameter_bank: ParameterBank,
    record_point: Tuple[float, float, float],
    seeds: list[int],
):
    """For a given pair of synapses (1,2), compute the baseline-subtracted voltages V(1), V(2), and V(1+2).
    Do this for a given range of seeds: each seed produces a different input stream."""
    logger = logging.getLogger(__name__)
    total_time = parameter_bank["T_ms"].value
    # run simulation for short time and read out baseline voltage
    input_rates_hz_0 = [0.0] * total_synapses
    parameter_bank["T_ms"].value = 100.0
    results_0 = simulation(
        swc_filepath=swc_filepath,
        synpts_filepath=synpts_filepath,
        input_rates_hz=input_rates_hz_0,
        parameters=parameter_bank.sample(),
        record_point=record_point,
    )
    baseline = results_0["probe"][0][0][-1, 1]
    logger.debug("Baseline voltage: %f", baseline)

    # loop over samples and fill m lists
    v_1 = []
    v_2 = []
    v_12 = []
    parameter_bank["T_ms"].value = total_time
    logger.debug("Total time reset in parameter_bank: %f", parameter_bank["T_ms"].value)
    for seed in seeds:
        seed = int(seed)
        logger.debug(f"Seed: {seed}")
        parameter_bank["seed"].value = seed
        parameters = parameter_bank.sample()
        # run 1
        input_rates_hz_1 = [0.0] * total_synapses
        input_rates_hz_1[active_synapse_pair[0]] = rate_pair[0]
        results_1 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_1,
            parameters=parameters,
            record_point=record_point,
        )
        v_1.append(results_1["probe"][0][0][:, 1] - baseline)
        # run 2
        input_rates_hz_2 = [0.0] * total_synapses
        input_rates_hz_2[active_synapse_pair[1]] = rate_pair[1]
        results_2 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_2,
            parameters=parameters,
            record_point=record_point,
        )
        v_2.append(results_2["probe"][0][0][:, 1] - baseline)
        # run 1+2
        input_rates_hz_12 = [0.0] * total_synapses
        input_rates_hz_12[active_synapse_pair[0]] = rate_pair[0]
        input_rates_hz_12[active_synapse_pair[1]] = rate_pair[1]
        results_12 = simulation(
            swc_filepath=swc_filepath,
            synpts_filepath=synpts_filepath,
            input_rates_hz=input_rates_hz_12,
            parameters=parameters,
            record_point=record_point,
        )
        v_12.append(results_12["probe"][0][0][:, 1] - baseline)
    time = results_12["probe"][0][0][:, 0]
    v_1 = np.array(v_1)
    v_2 = np.array(v_2)
    v_12 = np.array(v_12)
    # compute distance between synapse locations
    synapse_locations = np.loadtxt(synpts_filepath)
    synapse_1_location = synapse_locations[active_synapse_pair[0]]
    synapse_2_location = synapse_locations[active_synapse_pair[1]]
    distance = np.linalg.norm(synapse_1_location - synapse_2_location)
    # done
    results = {
        "time": time,
        "v_1": v_1,
        "v_2": v_2,
        "v_12": v_12,
        "intersynapse_distance": distance,
    }
    return results


def solve_k_linreg(v_1, v_2, v_12):
    """Solve for k using linear regression, combining trials.
    a = v_1 * v_2
    b = v_12 - v_1 - v_2
    solve ak=b for k using linear fit."""
    a = v_1 * v_2
    b = v_12 - v_1 - v_2
    a = a.flatten()
    b = b.flatten()
    coeffs, cov_matrix = np.polyfit(a, b, 1, cov="unscaled")
    results = {
        "coeffs": coeffs,
        "uncertainty": np.sqrt(np.diag(cov_matrix)),
        "a": a,
        "b": b,
    }
    return results


# ---


def compute_k_matrix(
    total_synapses: int,
    active_synapse_pair: tuple[int, int],
    rate_list: list[float],
    swc_filepath: str,
    synpts_filepath: str,
    parameter_bank: dict[str, float],
    record_location: tuple[float, float, float],
    seeds: list[int],
):

    n_rates = len(rate_list)
    k_matrix = np.zeros((n_rates, n_rates))
    k_unc_matrix = np.zeros((n_rates, n_rates))
    sim_results = {}
    rate_pair_list = []
    for i in range(n_rates):
        for j in range(i + 1):
            rate_i = rate_list[i]
            rate_j = rate_list[j]
            rate_pair = (rate_i, rate_j)
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
            )
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
            intercept = linreg_results["coeffs"][1]
            k_matrix[i, j] = k
            k_matrix[j, i] = k
            k_unc_matrix[i, j] = linreg_results["uncertainty"][0]
            k_unc_matrix[j, i] = linreg_results["uncertainty"][0]

    results = {
        "k_matrix": k_matrix,
        "sim_results": sim_results,
        "rate_pair_list": rate_pair_list,
        "intersynapse_distance": results["intersynapse_distance"],
    }
    return results
