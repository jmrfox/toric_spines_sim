"""TS3-specific ParameterBank overrides on top of the package defaults.

Tag convention used throughout: spine=3, sink=5, sink tip=6.
Temperature is in Kelvin (in vitro ~280 K; barn-owl in vivo ~313 K).
Capacitance is µF/cm²; leak is S/cm²; axial resistivity is Ω·cm.
``hh_tags`` is a list of SWC tags to apply Hodgkin–Huxley; empty means passive.

Uncomment the ``.value`` assignments below to override package defaults.
"""

from toric_spines_sim.simulation.parameters import make_default_parameter_bank


def make_parameter_bank():
    parameter_bank = make_default_parameter_bank()
    # parameter_bank["pas_leak_g_S_per_cm2"].value = 0.0005
    # parameter_bank["hh_scale"].value = 2.0
    # parameter_bank["ampa_gmax_uS"].value = 0.008
    return parameter_bank
