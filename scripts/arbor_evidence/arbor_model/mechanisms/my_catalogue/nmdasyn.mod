: NMDA synapse: dual exponential, peak-normalized so gmax is the peak conductance (uS)
: for one unit-weight event (before Mg block).

NEURON {
    POINT_PROCESS nmdasyn
    RANGE gmax, e, tau_r, tau_d, mg
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (nA) = (nanoamp)
    (ms) = (millisecond)
}

PARAMETER {
    gmax = 0.001 (uS)
    e = 0 (mV)
    tau_r = 0.5 (ms)
    tau_d = 50 (ms)
    mg = 1.0 (mM)
}

ASSIGNED {
    mg_block
    g (uS)
    factor
}

STATE {
    r
    d
}

INITIAL {
    r = 0
    d = 0
    : peak normalization so one unit-weight event peaks at gmax (as Exp2Syn)
    LOCAL tp
    tp = (tau_r*tau_d)/(tau_d - tau_r)*log(tau_d/tau_r)
    factor = 1/(exp(-tp/tau_d) - exp(-tp/tau_r))
}

BREAKPOINT {
    SOLVE state_deriv METHOD cnexp
    : Jahr & Stevens (1990), same form as effexcsyn
    mg_block = 1 / (1 + exp(-0.062 * v) * mg / 3.57)
    g = gmax * factor * (d - r) * mg_block
    i = g * (v - e)
}

DERIVATIVE state_deriv {
    r' = -r / tau_r
    d' = -d / tau_d
}

NET_RECEIVE(w) {
    r = r + w * 1
    d = d + w * 1
}
