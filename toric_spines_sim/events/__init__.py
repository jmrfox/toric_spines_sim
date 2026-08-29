"""Event generation and configuration for simulations."""

from .base import EventGenerator
from .generators import DeterministicEventGenerator, StochasticEventGenerator
from .rate_curves import (
    RateCurve,
    FlatRateCurve,
    LinearRateCurve,
    StepRateCurve,
    SineRateCurve,
)

__all__ = [
    "EventGenerator",
    "DeterministicEventGenerator",
    "StochasticEventGenerator",
    "RateCurve",
    "FlatRateCurve",
    "LinearRateCurve",
    "StepRateCurve",
    "SineRateCurve",
]
