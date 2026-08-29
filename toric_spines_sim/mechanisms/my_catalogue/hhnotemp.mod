NEURON {
    SUFFIX hhnotemp
    USEION na READ ena WRITE ina
    USEION k  READ ek  WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar, gkbar, gl, el
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar =   0.12   (S/cm2)
    gkbar  =   0.036  (S/cm2)
    gl     =   0.0003 (S/cm2)
    el     = -54.3    (mV)
}

STATE { m h n }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gk, gna, n2

    n2 = n*n
    gk = gkbar*n2*n2
    gna = gnabar*m*m*m*h
    ina = gna*(v - ena)
    ik  = gk*(v - ek)
    il  = gl*(v - el)
}

INITIAL {
    LOCAL alpha, beta

    : sodium activation system
    alpha = m_alpha(v)
    beta  = m_beta(v)
    m     = alpha/(alpha + beta)

    : sodium inactivation system
    alpha = h_alpha(v)
    beta  = h_beta(v)
    h     = alpha/(alpha + beta)

    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)
    n     = alpha/(alpha + beta)
}

DERIVATIVE states {
    LOCAL alpha, beta

    : sodium activation system
    alpha = m_alpha(v)
    beta  = m_beta(v)         
    m'    = (alpha - m*(alpha + beta))
                     
    : sodium inactivation system
    alpha = h_alpha(v)
    beta  = h_beta(v)
    h'    = (alpha - h*(alpha + beta))
                      
    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)                 
    n'    = (alpha - n*(alpha + beta))
}

FUNCTION m_alpha(v) { m_alpha = exprelr(-0.1*v - 4.0) }
FUNCTION h_alpha(v) { h_alpha = 0.07*exp(-0.05*v - 3.25) }
FUNCTION n_alpha(v) { n_alpha = 0.1*exprelr(-0.1*v - 5.5) }

FUNCTION m_beta(v)  { m_beta  = 4.0*exp(-(v + 65.0)/18.0) }
FUNCTION h_beta(v)  { h_beta  = 1.0/(exp(-0.1*v - 3.5) + 1.0) }
FUNCTION n_beta(v)  { n_beta  = 0.125*exp(-0.0125*v - 0.8125) }