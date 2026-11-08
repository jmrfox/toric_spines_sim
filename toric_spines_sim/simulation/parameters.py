"""Simulation parameter bank utilities.

Factories
---------
``make_default_parameter_bank()``
    Generic values (Arbor's historical defaults) used by the tutorial series.
``make_icx_parameter_bank(condition, temp_K=None, background=...)``
    Barn-owl ICx space-specific neuron, built in three layers:

    1. **Intrinsic (slice) values** at the Sanculi recording temperature
       (room temperature, ``temp_ref_K`` = 296 K). Passive values come from
       ``data/toric_spines_properties_imputed.csv`` (n = 20 cells, medians):
       C = 205 pF, Rin = 290 MΩ, τ ≈ 60 ms. Specific capacitance is the
       near-universal 0.9 µF/cm² (Gentet et al. 2000), so
       g_pas = cm / (Rin·C) ≈ 1.5e-5 S/cm².
    2. **Temperature layer.** Values are scaled from ``temp_ref_K`` to
       ``temp_K`` with Q10s (stored in the bank for provenance):
       leak conductance Q10 1.98 and axial resistivity Q10 0.80
       (Trevelyan & Jack 2002); Cm unchanged (Q10 0.96); AMPA kinetics
       Q10 1.4 (Postlethwaite et al. 2007); NMDA kinetics Q10 2.0 (Korinek
       et al. 2010); GABA kinetics Q10 2.0 (unverified placeholder).
    3. **Background layer** (in vivo only). Ongoing synaptic input lowers the
       rest-of-cell input resistance. There is no owl measurement, so it is a
       swept assumption: ``bg_rin_reduction`` is the fractional Rin drop
       applied to the lumped sink (tonic shunt at rest). Presets: ``none`` 0,
       ``moderate`` 0.5, ``high`` 0.8.

    ``condition="in_vitro"`` uses 296 K, no background, ACSF Mg 1.3 mM.
    ``condition="in_vivo"`` uses 313 K (barn owl ~40 °C), Vrest −67.6 mV
    (Peña & Konishi 2002), Mg 1.0 mM, and the requested background.

    Change temperature or background by calling the factory again, not by
    editing ``.value`` on derived quantities (``pas_leak_g_S_per_cm2``,
    ``rL_ohm_cm``, ``cell_Rin_MOhm``, synaptic τ), which are computed here.

``make_icx_parameter_bank_invitro()`` / ``make_icx_parameter_bank_invivo()``
    Backward-compatible shorthands for the two conditions.

Conventions
-----------
- Time in ms; temperature in Kelvin.
- ``cm_uF_per_cm2`` membrane capacitance; ``rL_ohm_cm`` axial resistivity;
  ``pas_leak_g_S_per_cm2`` leak.
- Synaptic ``*_gmax_uS`` is the **peak** conductance for one event (all
  NMODL synapses are peak-normalized).
- SWC tags: spine=3, sink cylinder=5, sink tip=6.
- ``sink_lumped = 1`` paints the sink with cm / g_pas scaled so its totals
  equal ``cell_C_pF`` and 1/``cell_Rin_MOhm`` (the sink stands in for the
  rest of the neuron). ``0`` uses the spine membrane on the sink geometry.
- ``hh_tags`` is a list of tags to apply Hodgkin–Huxley. The bank default is
  an empty vector, so assign tags on the **sampled** ``ParameterSet``:
  ``parameters["hh_tags"] = [5]``. Empty + ``hh_scale=0`` keeps the cell
  passive. ``hhnotemp`` is squid kinetics without temperature scaling.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
    ParameterSet,
)

T_REF_SANCULI_K = 296.0  # room temperature slice recordings
T_BARN_OWL_K = 313.0  # ~40 °C body temperature
BACKGROUND_PRESETS = {"none": 0.0, "moderate": 0.5, "high": 0.8}

# Sanculi et al. 2020 whole-cell medians (data/toric_spines_properties_imputed.csv)
SANCULI_CELL_C_PF = 205.0
SANCULI_CELL_RIN_MOHM = 290.0
CM_UF_PER_CM2 = 0.9
SANCULI_G_PAS = CM_UF_PER_CM2 * 1e-6 / (SANCULI_CELL_RIN_MOHM * 1e6 * SANCULI_CELL_C_PF * 1e-12)

Q10 = {
    "q10_leak": 1.98,
    "q10_rL": 0.80,
    "q10_ampa": 1.4,
    "q10_nmda": 2.0,
    "q10_gaba": 2.0,
}


def get_tau_m(pset: ParameterSet) -> float:
    """Membrane time constant (ms) of the spine membrane: cm / g_pas."""
    return pset["cm_uF_per_cm2"] / pset["pas_leak_g_S_per_cm2"] / 1000.0


def _fixed(value, lo=None, hi=None):
    if lo is None:
        return IndependentScalarParameter(value)
    return IndependentScalarParameter(value, is_sampled=False, range=(lo, hi))


def _base_values() -> Dict[str, object]:
    """Generic values shared by every bank (tutorial defaults, from Arbor's historical ones)."""
    return {
        "seed": 0,
        "T_ms": 1000.0,
        "delay_ms": 100.0,
        "discretization_um": 0.1,
        "dt_sim_ms": 0.02,
        "dt_record_ms": 0.1,
        "spine_tag": 3,
        "sink_tag": 5,
        "sink_tip_tag": 6,
        "sink_radii_scale": 1.0,
        "neck_radius_scale": 1.0,
        "temp_K": 280.0,
        "temp_ref_K": 280.0,
        **{k: 1.0 for k in Q10},
        "Vrest_mV": -65.0,
        "cm_uF_per_cm2": 1.0,
        "rL_ohm_cm": 35.0,
        "pas_leak_g_S_per_cm2": 0.001,
        "pas_leak_e_mV": -65.0,
        "sink_lumped": 0,
        "cell_C_pF": SANCULI_CELL_C_PF,
        "cell_Rin_MOhm": SANCULI_CELL_RIN_MOHM,
        "bg_rin_reduction": 0.0,
        "hh_leak_g_S_per_cm2": 0.0003,
        "hh_leak_e_mV": -54.3,
        "hh_scale": 1.0,
        "K_revpot_mV": -77.0,
        "K_intcon_mM": 54.4,
        "K_extcon_mM": 2.5,
        "K_gbar_S_per_cm2": 0.036,
        "Na_revpot_mV": 50.0,
        "Na_intcon_mM": 10.0,
        "Na_extcon_mM": 140.0,
        "Na_gbar_S_per_cm2": 0.12,
        "Ca_revpot_mV": 132.458,
        "Ca_intcon_mM": 0.00005,
        "Ca_extcon_mM": 2.0,
        "Ca_gbar_S_per_cm2": 0.0002,
        "ampa_gmax_uS": 0.002,
        "ampa_tau_ms": 2.0,
        "ampa_e_mV": 0.0,
        "nmda_gmax_uS": 0.002,
        "nmda_tau_r_ms": 5.0,
        "nmda_tau_d_ms": 50.0,
        "nmda_e_mV": 0.0,
        "nmda_mg_mM": 1.0,
        "gabaa_gmax_uS": 0.00008,
        "gabaa_tau_ms": 10.0,
        "gabaa_e_mV": -75.0,
        "gabab_gmax_uS": 0.00010,
        "gabab_e_mV": -95.0,
        "gabab_tau_r_ms": 30.0,
        "gabab_tau_d_ms": 200.0,
        "effexc_gmax_uS": 0.00010,
        "effexc_nmda_ratio": 0.50,
        "effexc_tau_ampa_ms": 2.0,
        "effexc_tau_nmda_rise_ms": 2.0,
        "effexc_tau_nmda_decay_ms": 50.0,
        "effexc_e_mV": 0.0,
        "effexc_mg_mM": 1.0,
    }


_RANGES = {
    "temp_K": (270.0, 320.0),
    "Vrest_mV": (-80.0, -50.0),
    "cm_uF_per_cm2": (0.1, 20.0),
    "rL_ohm_cm": (30.0, 200.0),
    "pas_leak_g_S_per_cm2": (1e-6, 0.1),
    "hh_leak_g_S_per_cm2": (0.00001, 0.01),
    "cell_C_pF": (50.0, 1000.0),
    "cell_Rin_MOhm": (1.0, 5000.0),
    "bg_rin_reduction": (0.0, 0.95),
    "ampa_gmax_uS": (0.0, 0.5),
    "ampa_tau_ms": (0.1, 10.0),
    "ampa_e_mV": (-70.0, 0.0),
    "nmda_gmax_uS": (0.0, 0.5),
    "nmda_tau_r_ms": (0.1, 20.0),
    "nmda_tau_d_ms": (1.0, 200.0),
    "nmda_e_mV": (-70.0, 0.0),
    "nmda_mg_mM": (0.0, 5.0),
    "gabaa_gmax_uS": (0.0, 0.5),
    "gabaa_tau_ms": (0.1, 50.0),
    "gabaa_e_mV": (-95.0, -60.0),
    "gabab_gmax_uS": (0.0, 0.5),
    "gabab_e_mV": (-100.0, -60.0),
    "gabab_tau_r_ms": (0.1, 100.0),
    "gabab_tau_d_ms": (1.0, 500.0),
    "effexc_gmax_uS": (0.0, 0.5),
    "effexc_nmda_ratio": (0.0, 1.0),
    "effexc_tau_ampa_ms": (0.1, 10.0),
    "effexc_tau_nmda_rise_ms": (0.1, 20.0),
    "effexc_tau_nmda_decay_ms": (1.0, 200.0),
    "effexc_e_mV": (-70.0, 0.0),
    "effexc_mg_mM": (0.0, 5.0),
}


def _bank(values: Dict[str, object]) -> ParameterBank:
    params: Dict[str, object] = {}
    for key, value in values.items():
        lo_hi = _RANGES.get(key)
        params[key] = _fixed(value, *lo_hi) if lo_hi else _fixed(value)
    params["tau_m_ms"] = DerivedScalarParameter(get_tau_m)
    for key in ("hh_tags", "iclamp_locations", "iclamp_amplitudes", "iclamp_durations"):
        params[key] = IndependentVectorParameter([])
    return ParameterBank(params)


def make_default_parameter_bank() -> ParameterBank:
    """Generic bank for tutorials. Call ``.sample()`` for a ParameterSet.

    Ion defaults (Arbor's): Na 50 mV (10 / 140 mM), K −77 mV (54.4 / 2.5 mM),
    Ca 132.458 mV (5e-5 / 2 mM).
    """
    return _bank(_base_values())


def icx_values(
    condition: Literal["in_vitro", "in_vivo"] = "in_vitro",
    temp_K: Optional[float] = None,
    background: str | float = "none",
) -> Dict[str, object]:
    """Plain dict of ICx values for a condition (see module docstring)."""
    if condition not in ("in_vitro", "in_vivo"):
        raise ValueError(f"condition must be 'in_vitro' or 'in_vivo', got {condition!r}")
    if isinstance(background, str):
        if background not in BACKGROUND_PRESETS:
            raise ValueError(f"background must be one of {list(BACKGROUND_PRESETS)} or a fraction")
        bg = BACKGROUND_PRESETS[background]
    else:
        bg = float(background)
    if not 0.0 <= bg < 1.0:
        raise ValueError("background Rin reduction must be in [0, 1)")
    if condition == "in_vitro" and bg > 0:
        raise ValueError("background synaptic activity applies to in_vivo only")

    v = _base_values()
    # Layer 1: intrinsic slice values at the Sanculi recording temperature.
    v.update(
        {
            "temp_ref_K": T_REF_SANCULI_K,
            **Q10,
            "Vrest_mV": -66.0,
            "pas_leak_e_mV": -66.0,
            "cm_uF_per_cm2": CM_UF_PER_CM2,
            "rL_ohm_cm": 100.0,
            "pas_leak_g_S_per_cm2": SANCULI_G_PAS,
            "sink_lumped": 1,
            "cell_C_pF": SANCULI_CELL_C_PF,
            "cell_Rin_MOhm": SANCULI_CELL_RIN_MOHM,
            # Sanculi ACSF / K-methanesulfonate pipette (only matters for HH).
            "K_extcon_mM": 1.3,
            "K_intcon_mM": 135.0,
            "Ca_extcon_mM": 2.5,
            "nmda_mg_mM": 1.3,
            "effexc_mg_mM": 1.3,
            # Synapses (peak conductances at the reference temperature)
            "ampa_gmax_uS": 0.001,
            "ampa_tau_ms": 2.0,
            "nmda_gmax_uS": 0.001,
            "nmda_tau_r_ms": 5.0,
            "nmda_tau_d_ms": 50.0,
            "gabaa_gmax_uS": 0.001,
            "gabaa_tau_ms": 2.0,
            "gabaa_e_mV": -70.0,
        }
    )
    if condition == "in_vivo":
        v.update(
            {
                "Vrest_mV": -67.6,
                "pas_leak_e_mV": -67.6,
                "K_extcon_mM": 3.0,
                "K_intcon_mM": 140.0,
                "Ca_extcon_mM": 2.0,
                "nmda_mg_mM": 1.0,
                "effexc_mg_mM": 1.0,
            }
        )
    T = float(temp_K) if temp_K is not None else (T_BARN_OWL_K if condition == "in_vivo" else T_REF_SANCULI_K)
    v["temp_K"] = T

    # Layer 2: temperature scaling from temp_ref_K to temp_K.
    f = (T - v["temp_ref_K"]) / 10.0
    leak = v["q10_leak"] ** f
    v["pas_leak_g_S_per_cm2"] *= leak
    v["cell_Rin_MOhm"] /= leak
    v["rL_ohm_cm"] *= v["q10_rL"] ** f
    for key, q in (
        ("ampa_tau_ms", "q10_ampa"),
        ("effexc_tau_ampa_ms", "q10_ampa"),
        ("nmda_tau_r_ms", "q10_nmda"),
        ("nmda_tau_d_ms", "q10_nmda"),
        ("effexc_tau_nmda_rise_ms", "q10_nmda"),
        ("effexc_tau_nmda_decay_ms", "q10_nmda"),
        ("gabaa_tau_ms", "q10_gaba"),
        ("gabab_tau_r_ms", "q10_gaba"),
        ("gabab_tau_d_ms", "q10_gaba"),
    ):
        v[key] /= v[q] ** f

    # Layer 3: background synaptic conductance on the rest of the cell.
    v["bg_rin_reduction"] = bg
    v["cell_Rin_MOhm"] *= 1.0 - bg
    return v


def make_icx_parameter_bank(
    condition: Literal["in_vitro", "in_vivo"] = "in_vitro",
    temp_K: Optional[float] = None,
    background: str | float = "none",
) -> ParameterBank:
    """ICx bank for ``condition`` (see module docstring for the three layers).

    Examples
    --------
    >>> bank = make_icx_parameter_bank("in_vivo", background="moderate")
    >>> p = bank.sample()
    >>> round(p["temp_K"])
    313
    """
    return _bank(icx_values(condition, temp_K, background))


def make_icx_parameter_bank_invitro() -> ParameterBank:
    """ICx bank matching Sanculi et al. 2020 slice patch clamp (296 K)."""
    return make_icx_parameter_bank("in_vitro")


def make_icx_parameter_bank_invivo(background: str | float = "none") -> ParameterBank:
    """ICx bank for the living owl (313 K; Peña & Konishi 2002 rest)."""
    return make_icx_parameter_bank("in_vivo", background=background)
