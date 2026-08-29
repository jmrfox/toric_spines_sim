: Effective excitatory (AMPA + NMDA) synapse mechanism
: Designed for mixed-receptor and background excitatory synapses

NEURON {
    POINT_PROCESS effexcsyn
    RANGE gmax
    RANGE nmda_ratio
    RANGE tau_ampa
    RANGE tau_nmda_rise
    RANGE tau_nmda_decay
    RANGE e
    RANGE mg
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (nA) = (nanoamp)
    (ms) = (millisecond)
    (mM) = (milli/liter)
}

PARAMETER {
    gmax            = 0.00010 (uS)

    nmda_ratio      = 0.50

    tau_ampa        = 2.0  (ms)

    tau_nmda_rise   = 2.0  (ms)
    tau_nmda_decay  = 50.0 (ms)

    e               = 0.0  (mV)
    mg              = 1.0  (mM)
}

ASSIGNED {
    g (uS)
    B
    gNMDA
}

STATE {
    gA
    rN
    dN
}

INITIAL {
    gA = 0
    rN = 0
    dN = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    
    : Mg block
    B = 1/(1 + exp(-0.062*v)*(mg/3.57))
    
    : NMDA conductance, normalized double exponential
    gNMDA = (dN - rN)/(tau_nmda_decay - tau_nmda_rise)
    
    : Effective combination of AMPA and NMDA
    g = gmax * (
        (1.0 - nmda_ratio)*gA +
        nmda_ratio*B*gNMDA
    )

    i = g*(v - e)
}

DERIVATIVE state {
    gA' = -gA/tau_ampa

    rN' = -rN/tau_nmda_rise
    dN' = -dN/tau_nmda_decay
}

NET_RECEIVE(weight) {
    gA = gA + weight

    rN = rN + weight
    dN = dN + weight
}