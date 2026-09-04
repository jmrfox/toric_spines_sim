"""TS67-specific ParameterBank overrides on top of the package defaults.

Tag convention used throughout: spine=3, sink=5, sink tip=6.
Temperature is in Kelvin (in vitro ~280 K; barn-owl in vivo ~313 K).
Capacitance is µF/cm²; leak is S/cm²; axial resistivity is Ω·cm.
``hh_tags`` is a list of SWC tags to paint Hodgkin–Huxley; empty means passive.

Uncomment the ``.value`` assignments below to override package defaults.
"""

from toric_spines_sim.simulation.parameters import make_default_parameter_bank


def make_parameter_bank():
    pb = make_default_parameter_bank()
    # pb["pas_leak_g_S_per_cm2"].value = 0.0005
    # pb["hh_scale"].value = 2.0
    # pb["ampa_gmax_uS"].value = 0.008
    return pb
