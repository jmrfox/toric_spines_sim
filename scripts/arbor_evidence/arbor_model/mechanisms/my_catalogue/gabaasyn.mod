: GABAa synapse mechanism with exponential decay

NEURON {
    POINT_PROCESS gabaasyn
    RANGE tau, e, gmax
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (nA) = (nanoamp)
    (ms) = (millisecond)
}

PARAMETER {
    tau = 10.0 (ms)
    e = -75 (mV)
    gmax = 0.00008 (uS)
}

ASSIGNED {}

STATE {
    g (uS)
}

INITIAL {
    g = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = gmax * g * (v - e)
}

DERIVATIVE state {
    g' = -g/tau
}

NET_RECEIVE(weight) {
    g = g + weight
}