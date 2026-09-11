"""TS76-specific ParameterBank.
"""

from toric_spines_sim.simulation.parameters import (
    make_default_parameter_bank,
    make_icx_parameter_bank_invitro,
    make_icx_parameter_bank_invivo,
)


def make_parameter_bank():
    # parameter_bank = make_default_parameter_bank()
    # parameter_bank = make_icx_parameter_bank_invitro()
    parameter_bank = make_icx_parameter_bank_invivo()
    return parameter_bank
