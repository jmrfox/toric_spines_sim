NEURON {
  SUFFIX simple_vg
  RANGE gbar, erev
  USEION na READ ena WRITE ina  // optional, or define erev directly
}

PARAMETER {
  gbar = 0.001 (S/cm2)
  erev = 50 (mV)
}

STATE {
  m
}

ASSIGNED {
  v (mV)
  m_inf
  tau_m (ms)
  i (mA/cm2)
}

INITIAL {
  m = 1.0/(1.0 + exp((v - Vhalf)/k))
}

BREAKPOINT {
  SOLVE states METHOD cnexp
  i = gbar * m * (v - erev)
}

DERIVATIVE states {
  m_inf = 1.0/(1.0 + exp((Vhalf - v)/k))
  tau_m = tau0 + tau1/(1.0 + exp((v - Vtau)/ktau))
  m' = (m_inf - m)/tau_m
}
