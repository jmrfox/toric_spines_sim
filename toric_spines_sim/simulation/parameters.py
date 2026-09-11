"""Simulation parameter bank utilities.

``make_default_parameter_bank()`` is the generic (Arbor-like) bank used by
the tutorial series. ``make_icx_parameter_bank_invitro()`` is Sanculi
patch-clamp ICx; ``make_icx_parameter_bank_invivo()`` is living-owl ICx
(Peña and Konishi, 2002, plus Sanculi leak scaled for in-vivo input
resistance). Axon studies return the invivo bank from
``simulations/ts{id}/params.py``; uncomment another factory or set
``.value`` for a per-spine change.

Conventions
-----------
- Time in ms; temperature in Kelvin (in vitro ~297 K; barn owl in vivo ~313 K).
- ``cm_uF_per_cm2`` membrane capacitance; ``rL_ohm_cm`` axial resistivity;
  ``pas_leak_g_S_per_cm2`` leak (Arbor default 0.001 S/cm²).
- SWC tags: spine=3, sink cylinder=5, sink tip=6.
- ``hh_tags`` is a list of tags to apply Hodgkin–Huxley. The bank default is an
  empty vector (shape ``(0,)``), so assign tags on the **sampled**
  ``ParameterSet``: ``parameters["hh_tags"] = [5]``. Empty + ``hh_scale=0``
  keeps the cell passive.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from jscip import ParameterBank, IndependentScalarParameter,\
    IndependentVectorParameter, DerivedScalarParameter, ParameterSet

def get_tau_m(pset: ParameterSet) -> float:
    """Compute the membrane time constant from membrane capacitance and passive conductance."""
    return pset["cm_uF_per_cm2"] / pset["pas_leak_g_S_per_cm2"] / 1000.0
    
def make_default_parameter_bank() -> ParameterBank:
    """
    Returns a ParameterBank with default values.
    Call ``bank.sample()`` to obtain a ParameterSet for simulation.

    Arbor ion channel defaults:
    Na: revpot = 50.0 mV, int_conc = 10.0 mM, ext_conc = 140.0 mM
    K: revpot = -77.0 mV, int_conc = 54.4 mM, ext_conc = 2.5 mM
    Ca: revpot = 132.458 mV, int_conc = 0.00005 mM, ext_conc = 2 mM


    """

    parameter_bank = ParameterBank(
        {
            "seed": IndependentScalarParameter(0),
            "T_ms": IndependentScalarParameter(1000.0),
            "delay_ms": IndependentScalarParameter(100.0),
            "discretization_um": IndependentScalarParameter(0.1),
            "dt_sim_ms": IndependentScalarParameter(0.02),
            "dt_record_ms": IndependentScalarParameter(0.1),
            "spine_tag": IndependentScalarParameter(3),
            "sink_tag": IndependentScalarParameter(5),
            "sink_tip_tag": IndependentScalarParameter(6),
            "sink_radii_scale": IndependentScalarParameter(1.0),
            "neck_radius_scale": IndependentScalarParameter(1.0),
            "temp_K": IndependentScalarParameter(
                280.0,
                is_sampled=False,
                range=(270.0, 320.0),
            ),
            "Vrest_mV": IndependentScalarParameter(
                -65.0,
                is_sampled=False,
                range=(-70.0, -60.0),
            ),
            "cm_uF_per_cm2": IndependentScalarParameter(
                1.0,
                is_sampled=False,
                range=(1.0, 20.0),
            ),
            "rL_ohm_cm": IndependentScalarParameter(
                35.0,
                is_sampled=False,
                range=(30.0, 200.0),
            ),
            "gj_weight": IndependentScalarParameter(1.0),
            "pas_leak_g_S_per_cm2": IndependentScalarParameter(
                0.001, is_sampled=False, range=(0.00001, 0.1),
            ),
            "pas_leak_e_mV": IndependentScalarParameter(-65.0),
            "tau_m_ms": DerivedScalarParameter(get_tau_m),
            "hh_leak_g_S_per_cm2": IndependentScalarParameter(
                0.0003, is_sampled=False, range=(0.00001, 0.01),
                ),
            "hh_leak_e_mV": IndependentScalarParameter(-54.3),
            "hh_tags": IndependentVectorParameter([]),
            "hh_scale": IndependentScalarParameter(1.0),
            "iclamp_locations": IndependentVectorParameter([]),
            "iclamp_amplitudes": IndependentVectorParameter([]),
            "iclamp_durations": IndependentVectorParameter([]),
            "K_revpot_mV": IndependentScalarParameter(-77.0),
            "K_intcon_mM": IndependentScalarParameter(54.4),
            "K_extcon_mM": IndependentScalarParameter(2.5),
            "K_gbar_S_per_cm2": IndependentScalarParameter(0.036),
            "Na_revpot_mV": IndependentScalarParameter(50.0),
            "Na_intcon_mM": IndependentScalarParameter(10.0),
            "Na_extcon_mM": IndependentScalarParameter(140.0),
            "Na_gbar_S_per_cm2": IndependentScalarParameter(0.12),
            "Ca_revpot_mV": IndependentScalarParameter(132.458),
            "Ca_intcon_mM": IndependentScalarParameter(0.00005),
            "Ca_extcon_mM": IndependentScalarParameter(2.0),
            "Ca_gbar_S_per_cm2": IndependentScalarParameter(0.0002),
            "ampa_gmax_uS": IndependentScalarParameter(
                0.002,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "ampa_tau_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 4.0),
            ),
            "ampa_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "nmda_gmax_uS": IndependentScalarParameter(
                0.002,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "nmda_tau_r_ms": IndependentScalarParameter(
                5.0,
                is_sampled=False,
                range=(0.1, 10.0),
            ),
            "nmda_tau_d_ms": IndependentScalarParameter(
                50.0,
                is_sampled=False,
                range=(0.1, 100.0),
            ),
            "nmda_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "gabaa_gmax_uS": IndependentScalarParameter(
                0.00008,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "gabaa_tau_ms": IndependentScalarParameter(
                10.0,
                is_sampled=False,
                range=(0.1, 50.0),
            ),
            "gabaa_e_mV": IndependentScalarParameter(
                -75.0,
                is_sampled=False,
                range=(-95.0, -60.0),
            ),
            "gabab_gmax_uS": IndependentScalarParameter(
                0.00010,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "gabab_e_mV": IndependentScalarParameter(
                -95.0,
                is_sampled=False,
                range=(-100.0, -60.0),
            ),
            "gabab_tau_r_ms": IndependentScalarParameter(
                30.0,
                is_sampled=False,
                range=(0.1, 100.0),
            ),
            "gabab_tau_d_ms": IndependentScalarParameter(
                200.0,
                is_sampled=False,
                range=(0.1, 500.0),
            ),
            "effexc_gmax_uS": IndependentScalarParameter(
                0.00010,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "effexc_nmda_ratio": IndependentScalarParameter(
                0.50,
                is_sampled=False,
                range=(0.0, 1.0),
            ),
            "effexc_tau_ampa_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 10.0),
            ),
            "effexc_tau_nmda_rise_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 20.0),
            ),
            "effexc_tau_nmda_decay_ms": IndependentScalarParameter(
                50.0,
                is_sampled=False,
                range=(0.1, 200.0),
            ),
            "effexc_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "effexc_mg_mM": IndependentScalarParameter(
                1.0,
                is_sampled=False,
                range=(0.0, 5.0),
            ),
        },
    )
    return parameter_bank


def make_icx_parameter_bank_invivo() -> ParameterBank:
    """
    ICx bank for comparison to living-owl intracellular recordings.
    Peña and Konishi (2002, Journal of Neuroscience): 
    mean rest potential = -67.6 ± 9.3 mV (n = 75 ICx neurons, sharp electrode). 
    Body temperature = ~313 K. 
    Input resistance is lower in vivo than in slice (~3×); 
    leak = Sanculi g_pas × 3 = 0.00042 S/cm². 
    Cm is still the Sanculi value (no in-vivo capacitance).

    - Longitudinal resistivity (ohm cm): ~100 ohm cm, typical values 30 - 200
    - Membrane capacitance (uF/cm^2): Cm = 8.559 +/- 50.475 (Sanculi; no in-vivo Cm)
    - AMPA: gmax = 0.2-2.0 nS, tau = 2 ms, revpot = 0 mV
    - NMDA: gmax = 0.2-2.0 nS, tau_r = 5 ms, tau_d = 50 ms, revpot = 0 mV
    - GABA_A: gmax = 0.5-2.0 nS, tau = 2 ms, revpot = -70 mV

    Arbor ion channel defaults:
    Na: revpot = 50.0 mV, int_conc = 10.0 mM, ext_conc = 140.0 mM
    K: revpot = -77.0 mV, int_conc = 54.4 mM, ext_conc = 2.5 mM
    Ca: revpot = 132.458 mV, int_conc = 0.00005 mM, ext_conc = 2 mM

    """

    parameter_bank = ParameterBank(
        {
            "seed": IndependentScalarParameter(0),
            "T_ms": IndependentScalarParameter(1000.0),
            "delay_ms": IndependentScalarParameter(100.0),
            "discretization_um": IndependentScalarParameter(0.1),
            "dt_sim_ms": IndependentScalarParameter(0.02),
            "dt_record_ms": IndependentScalarParameter(0.1),
            "spine_tag": IndependentScalarParameter(3),
            "sink_tag": IndependentScalarParameter(5),
            "sink_tip_tag": IndependentScalarParameter(6),
            "sink_radii_scale": IndependentScalarParameter(1.0),
            "neck_radius_scale": IndependentScalarParameter(1.0),
            "temp_K": IndependentScalarParameter(
                313.0,
                is_sampled=False,
                range=(270.0, 320.0),
            ),
            "Vrest_mV": IndependentScalarParameter(
                -67.6,
                is_sampled=False,
                range=(-80.0, -50.0),
            ),
            "cm_uF_per_cm2": IndependentScalarParameter(
                8.559,
                is_sampled=False,
                range=(1.0, 20.0),
            ),
            "rL_ohm_cm": IndependentScalarParameter(
                100.0,
                is_sampled=False,
                range=(30.0, 200.0),
            ),
            "gj_weight": IndependentScalarParameter(1.0),
            "pas_leak_g_S_per_cm2": IndependentScalarParameter(
                0.00042, is_sampled=False, range=(0.00001, 0.1),
            ),
            "pas_leak_e_mV": IndependentScalarParameter(-67.6),
            "tau_m_ms": DerivedScalarParameter(get_tau_m),
            "hh_leak_g_S_per_cm2": IndependentScalarParameter(
                0.0003, is_sampled=False, range=(0.00001, 0.01),
                ),
            "hh_leak_e_mV": IndependentScalarParameter(-54.3),
            "hh_tags": IndependentVectorParameter([]),
            "hh_scale": IndependentScalarParameter(1.0),
            "iclamp_locations": IndependentVectorParameter([]),
            "iclamp_amplitudes": IndependentVectorParameter([]),
            "iclamp_durations": IndependentVectorParameter([]),
            "K_revpot_mV": IndependentScalarParameter(-77.0),
            "K_intcon_mM": IndependentScalarParameter(54.4),
            "K_extcon_mM": IndependentScalarParameter(2.5),
            "K_gbar_S_per_cm2": IndependentScalarParameter(0.036),
            "Na_revpot_mV": IndependentScalarParameter(50.0),
            "Na_intcon_mM": IndependentScalarParameter(10.0),
            "Na_extcon_mM": IndependentScalarParameter(140.0),
            "Na_gbar_S_per_cm2": IndependentScalarParameter(0.12),
            "Ca_revpot_mV": IndependentScalarParameter(132.458),
            "Ca_intcon_mM": IndependentScalarParameter(0.00005),
            "Ca_extcon_mM": IndependentScalarParameter(2.0),
            "Ca_gbar_S_per_cm2": IndependentScalarParameter(0.0002),
            "ampa_gmax_uS": IndependentScalarParameter(
                0.001,
                is_sampled=False,
                range=(0.0002, 0.002),
            ),
            "ampa_tau_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 4.0),
            ),
            "ampa_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "nmda_gmax_uS": IndependentScalarParameter(
                0.001,
                is_sampled=False,
                range=(0.0002, 0.002),
            ),
            "nmda_tau_r_ms": IndependentScalarParameter(
                5.0,
                is_sampled=False,
                range=(0.1, 10.0),
            ),
            "nmda_tau_d_ms": IndependentScalarParameter(
                50.0,
                is_sampled=False,
                range=(0.1, 100.0),
            ),
            "nmda_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "gabaa_gmax_uS": IndependentScalarParameter(
                0.001,
                is_sampled=False,
                range=(0.0005, 0.002),
            ),
            "gabaa_tau_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 50.0),
            ),
            "gabaa_e_mV": IndependentScalarParameter(
                -70.0,
                is_sampled=False,
                range=(-95.0, -60.0),
            ),
            "gabab_gmax_uS": IndependentScalarParameter(
                0.00010,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "gabab_e_mV": IndependentScalarParameter(
                -95.0,
                is_sampled=False,
                range=(-100.0, -60.0),
            ),
            "gabab_tau_r_ms": IndependentScalarParameter(
                30.0,
                is_sampled=False,
                range=(0.1, 100.0),
            ),
            "gabab_tau_d_ms": IndependentScalarParameter(
                200.0,
                is_sampled=False,
                range=(0.1, 500.0),
            ),
            "effexc_gmax_uS": IndependentScalarParameter(
                0.00010,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "effexc_nmda_ratio": IndependentScalarParameter(
                0.50,
                is_sampled=False,
                range=(0.0, 1.0),
            ),
            "effexc_tau_ampa_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 10.0),
            ),
            "effexc_tau_nmda_rise_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 20.0),
            ),
            "effexc_tau_nmda_decay_ms": IndependentScalarParameter(
                50.0,
                is_sampled=False,
                range=(0.1, 200.0),
            ),
            "effexc_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "effexc_mg_mM": IndependentScalarParameter(
                1.0,
                is_sampled=False,
                range=(0.0, 5.0),
            ),
        },
    )
    return parameter_bank


def make_icx_parameter_bank_invitro() -> ParameterBank:
    """ICx bank for comparison to Sanculi et al. patch-clamp (in vitro).

    - Longitudinal resistivity (ohm cm): ~100 ohm cm, typical values 30 - 200
    - Membrane capacitance (uF/cm^2): Cm = 8.559 +/- 50.475 (Sanculi; no in-vivo Cm)
    - AMPA: gmax = 0.2-2.0 nS, tau = 2 ms, revpot = 0 mV
    - NMDA: gmax = 0.2-2.0 nS, tau_r = 5 ms, tau_d = 50 ms, revpot = 0 mV
    - GABA_A: gmax = 0.5-2.0 nS, tau = 2 ms, revpot = -70 mV

    Arbor ion channel defaults:
    Na: revpot = 50.0 mV, int_conc = 10.0 mM, ext_conc = 140.0 mM
    K: revpot = -77.0 mV, int_conc = 54.4 mM, ext_conc = 2.5 mM
    Ca: revpot = 132.458 mV, int_conc = 0.00005 mM, ext_conc = 2 mM
    """

    parameter_bank = ParameterBank(
        {
            "seed": IndependentScalarParameter(0),
            "T_ms": IndependentScalarParameter(1000.0),
            "delay_ms": IndependentScalarParameter(100.0),
            "discretization_um": IndependentScalarParameter(0.1),
            "dt_sim_ms": IndependentScalarParameter(0.02),
            "dt_record_ms": IndependentScalarParameter(0.1),
            "spine_tag": IndependentScalarParameter(3),
            "sink_tag": IndependentScalarParameter(5),
            "sink_tip_tag": IndependentScalarParameter(6),
            "sink_radii_scale": IndependentScalarParameter(1.0),
            "neck_radius_scale": IndependentScalarParameter(1.0),
            "temp_K": IndependentScalarParameter(
                297.0,
                is_sampled=False,
                range=(270.0, 320.0),
            ),
            "Vrest_mV": IndependentScalarParameter(
                -67.6,
                is_sampled=False,
                range=(-80.0, -50.0),
            ),
            "cm_uF_per_cm2": IndependentScalarParameter(
                8.559,
                is_sampled=False,
                range=(1.0, 20.0),
            ),
            "rL_ohm_cm": IndependentScalarParameter(
                100.0,
                is_sampled=False,
                range=(30.0, 200.0),
            ),
            "gj_weight": IndependentScalarParameter(1.0),
            "pas_leak_g_S_per_cm2": IndependentScalarParameter(
                0.000144, is_sampled=False, range=(0.00001, 0.1),
            ),
            "pas_leak_e_mV": IndependentScalarParameter(-67.6),
            "tau_m_ms": DerivedScalarParameter(get_tau_m),
            "hh_leak_g_S_per_cm2": IndependentScalarParameter(
                0.0003, is_sampled=False, range=(0.00001, 0.01),
                ),
            "hh_leak_e_mV": IndependentScalarParameter(-54.3),
            "hh_tags": IndependentVectorParameter([]),
            "hh_scale": IndependentScalarParameter(1.0),
            "iclamp_locations": IndependentVectorParameter([]),
            "iclamp_amplitudes": IndependentVectorParameter([]),
            "iclamp_durations": IndependentVectorParameter([]),
            "K_revpot_mV": IndependentScalarParameter(-77.0),
            "K_intcon_mM": IndependentScalarParameter(54.4),
            "K_extcon_mM": IndependentScalarParameter(2.5),
            "K_gbar_S_per_cm2": IndependentScalarParameter(0.036),
            "Na_revpot_mV": IndependentScalarParameter(50.0),
            "Na_intcon_mM": IndependentScalarParameter(10.0),
            "Na_extcon_mM": IndependentScalarParameter(140.0),
            "Na_gbar_S_per_cm2": IndependentScalarParameter(0.12),
            "Ca_revpot_mV": IndependentScalarParameter(132.458),
            "Ca_intcon_mM": IndependentScalarParameter(0.00005),
            "Ca_extcon_mM": IndependentScalarParameter(2.0),
            "Ca_gbar_S_per_cm2": IndependentScalarParameter(0.0002),
            "ampa_gmax_uS": IndependentScalarParameter(
                0.001,
                is_sampled=False,
                range=(0.0002, 0.002),
            ),
            "ampa_tau_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 4.0),
            ),
            "ampa_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "nmda_gmax_uS": IndependentScalarParameter(
                0.001,
                is_sampled=False,
                range=(0.0002, 0.002),
            ),
            "nmda_tau_r_ms": IndependentScalarParameter(
                5.0,
                is_sampled=False,
                range=(0.1, 10.0),
            ),
            "nmda_tau_d_ms": IndependentScalarParameter(
                50.0,
                is_sampled=False,
                range=(0.1, 100.0),
            ),
            "nmda_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "gabaa_gmax_uS": IndependentScalarParameter(
                0.001,
                is_sampled=False,
                range=(0.0005, 0.002),
            ),
            "gabaa_tau_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 50.0),
            ),
            "gabaa_e_mV": IndependentScalarParameter(
                -70.0,
                is_sampled=False,
                range=(-95.0, -60.0),
            ),
            "gabab_gmax_uS": IndependentScalarParameter(
                0.00010,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "gabab_e_mV": IndependentScalarParameter(
                -95.0,
                is_sampled=False,
                range=(-100.0, -60.0),
            ),
            "gabab_tau_r_ms": IndependentScalarParameter(
                30.0,
                is_sampled=False,
                range=(0.1, 100.0),
            ),
            "gabab_tau_d_ms": IndependentScalarParameter(
                200.0,
                is_sampled=False,
                range=(0.1, 500.0),
            ),
            "effexc_gmax_uS": IndependentScalarParameter(
                0.00010,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "effexc_nmda_ratio": IndependentScalarParameter(
                0.50,
                is_sampled=False,
                range=(0.0, 1.0),
            ),
            "effexc_tau_ampa_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 10.0),
            ),
            "effexc_tau_nmda_rise_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(0.1, 20.0),
            ),
            "effexc_tau_nmda_decay_ms": IndependentScalarParameter(
                50.0,
                is_sampled=False,
                range=(0.1, 200.0),
            ),
            "effexc_e_mV": IndependentScalarParameter(
                0.0,
                is_sampled=False,
                range=(-70.0, 0.0),
            ),
            "effexc_mg_mM": IndependentScalarParameter(
                1.0,
                is_sampled=False,
                range=(0.0, 5.0),
            ),
        },
    )
    return parameter_bank
    