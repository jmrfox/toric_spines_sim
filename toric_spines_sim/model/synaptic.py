"""Synaptic conductance components shared by every solver.

Each registry model (``ampa``, ``nmda``, ``gabaa``, ``gabab``, ``effexc``) is
one or two *components*: a time course that does not depend on voltage
(single exponential, or a peak-normalized difference of exponentials), a
reversal potential, and an optional Jahr-Stevens Mg block evaluated from the
membrane voltage during the solve. The kinetics match the NMODL mechanisms
the package used with Arbor (now in ``scripts/arbor_evidence/arbor_model/mechanisms``):

- ``ampasyn`` / ``gabaasyn``: g = gmax * sum_k exp(-(t - t_k)/tau)
- ``nmdasyn``: peak-normalized exp2 with Mg block
- ``gababsyn``: peak-normalized exp2, no block
- ``effexcsyn``: gmax * [(1 - r) * exp(tau_ampa) + r * exp2(rise, decay) * block]

Events are delivered at the start of the step that contains them, and the
time course is sampled on the step grid, exactly as an event-driven
mechanism integrated with exponential decay per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Component:
    """One conductance time course of a synapse.

    ``kind`` is ``"exp"`` (uses ``tau_ms``) or ``"exp2"`` (uses ``tau_r_ms``
    and ``tau_d_ms``, normalized to peak 1). ``scale_uS`` is the peak
    conductance per event. ``mg_mM`` is None for no Mg block.
    """

    kind: str
    scale_uS: float
    e_mV: float
    tau_ms: float = 2.0
    tau_r_ms: float = 5.0
    tau_d_ms: float = 50.0
    mg_mM: Optional[float] = None


def exp2_peak_factor(tau_r_ms: float, tau_d_ms: float) -> float:
    """Factor that makes ``exp(-t/tau_d) - exp(-t/tau_r)`` peak at 1."""
    if tau_d_ms <= tau_r_ms:
        raise ValueError("tau_d must exceed tau_r")
    t_peak = tau_r_ms * tau_d_ms / (tau_d_ms - tau_r_ms) * math.log(tau_d_ms / tau_r_ms)
    return 1.0 / (math.exp(-t_peak / tau_d_ms) - math.exp(-t_peak / tau_r_ms))


def mg_block(v_mV, mg_mM):
    """Jahr & Stevens (1990) unblocked fraction."""
    return 1.0 / (1.0 + np.exp(-0.062 * v_mV) * mg_mM / 3.57)


def components(model: str, params: Mapping[str, float]) -> List[Component]:
    """Components for a registry model from its ``synapse_params`` dict."""
    p = dict(params)
    if model in ("ampa", "gabaa"):
        default_e = 0.0 if model == "ampa" else -75.0
        return [Component("exp", p["gmax_uS"], p.get("e_mV", default_e), tau_ms=p["tau_ms"])]
    if model == "nmda":
        return [Component("exp2", p["gmax_uS"], p.get("e_mV", 0.0), tau_r_ms=p["tau_r_ms"],
                          tau_d_ms=p["tau_d_ms"], mg_mM=p.get("mg_mM", 1.0))]
    if model == "gabab":
        return [Component("exp2", p["gmax_uS"], p.get("e_mV", -95.0), tau_r_ms=p["tau_r_ms"],
                          tau_d_ms=p["tau_d_ms"])]
    if model == "effexc":
        r = p["nmda_ratio"]
        e = p.get("e_mV", 0.0)
        return [
            Component("exp", p["gmax_uS"] * (1.0 - r), e, tau_ms=p["tau_ampa_ms"]),
            Component("exp2", p["gmax_uS"] * r, e, tau_r_ms=p["tau_nmda_rise_ms"],
                      tau_d_ms=p["tau_nmda_decay_ms"], mg_mM=p.get("mg_mM", 1.0)),
        ]
    raise ValueError(f"unknown synapse model {model!r}")


def series(comp: Component, times_ms: Sequence[float], n_steps: int, dt_ms: float) -> np.ndarray:
    """Conductance (µS) of ``comp`` on steps ``0..n_steps-1`` for the given events.

    Step ``k`` covers ``[k dt, (k+1) dt)``; an event at ``t`` is delivered at
    step ``floor(t / dt)``.
    """
    out = np.zeros(n_steps)
    if comp.scale_uS == 0 or len(times_ms) == 0:
        return out
    k = np.arange(n_steps)
    if comp.kind == "exp":
        d1 = math.exp(-dt_ms / comp.tau_ms)
    else:
        f = exp2_peak_factor(comp.tau_r_ms, comp.tau_d_ms)
        dr, dd = math.exp(-dt_ms / comp.tau_r_ms), math.exp(-dt_ms / comp.tau_d_ms)
    for t in times_ms:
        ke = int(math.floor(float(t) / dt_ms + 1e-9))
        if ke < 0 or ke >= n_steps:
            continue
        m = k[ke:] - ke
        if comp.kind == "exp":
            out[ke:] += d1 ** m
        else:
            out[ke:] += f * (dd ** m - dr ** m)
    return comp.scale_uS * out


def group_key(comp: Component) -> tuple:
    """Components with the same key can be summed on one compartment."""
    return (round(comp.e_mV, 9), None if comp.mg_mM is None else round(comp.mg_mM, 9))


def node_conductances(
    synapses: Sequence[tuple],
    events: Mapping[int, Sequence[float]],
    n_steps: int,
    dt_ms: float,
) -> Dict[tuple, np.ndarray]:
    """Sum conductances per (node, e, mg).

    ``synapses`` is a sequence of ``(node, [Component, ...])``; ``events``
    maps synapse index to event times (ms). Returns
    ``{(node, e_mV, mg_mM or None): series in µS}``.
    """
    out: Dict[tuple, np.ndarray] = {}
    for si, (node, comps) in enumerate(synapses):
        times = events.get(si, ())
        if len(times) == 0:
            continue
        for c in comps:
            key = (int(node), *group_key(c))
            s = series(c, times, n_steps, dt_ms)
            out[key] = out[key] + s if key in out else s
    return out
