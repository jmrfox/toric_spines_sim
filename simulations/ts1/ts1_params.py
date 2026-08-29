from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
)

from toric_spines_sim.simulation.parameters import get_tau_m


def make_ts1_parameter_bank():
    pb = ParameterBank(
        {
            "seed": IndependentScalarParameter(
                0, is_sampled=True, range=(0, 1), grid_points=2
            ),
            "discretization_um": IndependentScalarParameter(0.1),
            "dt_sim_ms": IndependentScalarParameter(0.02),
            "dt_record_ms": IndependentScalarParameter(0.1),
            "spine_tag": IndependentScalarParameter(3),
            "sink_tag": IndependentScalarParameter(5),
            "sink_tip_tag": IndependentScalarParameter(6),
            "sink_radii_scale": IndependentScalarParameter(1.0),
            "neck_radius_scale": IndependentScalarParameter(
                1.0, is_sampled=False, range=(0.4, 1.0),
            ),
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
                range=(0.1, 10.0),
                grid_points=[1.0, 10.0, 20.0],
            ),
            "rL_ohm_cm": IndependentScalarParameter(
                35.0,
                is_sampled=False,
                range=(30.0, 200.0),
            ),
            "gj_weight": IndependentScalarParameter(1.0),
            "pas_leak_g_S_per_cm2": IndependentScalarParameter(
                0.0005,
                is_sampled=False,
                range=(0.00001, 0.1),
            ),
            "pas_leak_e_mV": IndependentScalarParameter(-65.0),
            "tau_m_ms": DerivedScalarParameter(get_tau_m),
            "hh_leak_g_S_per_cm2": IndependentScalarParameter(0.0003),
            "hh_leak_e_mV": IndependentScalarParameter(-54.3),
            "hh_tags": IndependentVectorParameter([]),
            "hh_scale": IndependentScalarParameter(2.0),
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
                0.008,
                is_sampled=False,
                range=(0.0, 0.5),
            ),
            "ampa_tau_ms": IndependentScalarParameter(
                2.0,
                is_sampled=False,
                range=(1.0, 4.0),
            ),
            "ampa_e_mV": IndependentScalarParameter(0.0),
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
            "nmda_e_mV": IndependentScalarParameter(0.0),
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
    return pb
