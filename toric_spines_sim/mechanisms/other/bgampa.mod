NEURON {
    SUFFIX bgampa
    NONSPECIFIC_CURRENT i

    RANGE tau_mean, tau_noise
    RANGE g_rest, g_stim
    RANGE sigma
    RANGE e_rev
}

PARAMETER {
    tau_mean  = 50 (ms)
    tau_noise = 5  (ms)

    g_rest = 0.5 (nS)
    g_stim = 4.0 (nS)

    sigma = 0.2 (nS/ms)

    e_rev = 0 (mV)
}

ASSIGNED {
    :v (mV)
    :i (nA)
    target (nS)
}

STATE {
    g (nS)
    gbar (nS)
}

INITIAL {
    target = g_rest
    gbar = g_rest
    g = g_rest
}

WHITE_NOISE {
    W
}

BREAKPOINT {
    SOLVE state METHOD stochastic
    i = g*(v - e_rev)
}

DERIVATIVE state {
    gbar' = - (gbar - target)/tau_mean
    g'    = - (g - gbar)/tau_noise + sigma*W
}

NET_RECEIVE(weight) {
    if (weight > 0) {
        target = g_stim
    } else {
        target = g_rest
    }
}