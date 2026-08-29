: GABAb synapse mechanism with dual-exponential decay

NEURON {
    POINT_PROCESS gababsyn
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
    gmax = 0.00010 (uS)
    e = -95 (mV)
    tau_r = 30.0 (ms)
    tau_d = 200.0 (ms)
}

ASSIGNED {
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
    g = gmax * (d - r) / (tau_d - tau_r)
    i = g * (v - e)
}

DERIVATIVE state_deriv {
    r' = -r / tau_r
    d' = -d / tau_d
}

NET_RECEIVE(weight) {
    r = r + weight
    d = d + weight
}