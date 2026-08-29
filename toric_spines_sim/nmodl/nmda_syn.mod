NEURON {
    POINT_PROCESS nmda_syn
    RANGE gmax, e, tau_r, tau_d
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
}

ASSIGNED {
    mg_block
    g (uS)
}

STATE {
    r
    d
}

INITIAL {
    r = 0
    d = 0
}

BREAKPOINT {
    SOLVE state_deriv METHOD cnexp
    mg_block = 1 / (1 + 0.33 * exp(-0.06 * v))
    g = gmax * (d - r) / (tau_d - tau_r) * mg_block
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
