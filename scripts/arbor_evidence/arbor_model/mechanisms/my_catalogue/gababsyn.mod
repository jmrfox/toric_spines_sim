: GABAb synapse: dual exponential, peak-normalized so gmax is the peak conductance (uS).

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
    g = gmax * factor * (d - r)
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