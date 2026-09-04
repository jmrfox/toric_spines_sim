"""
Animation example script for swctools using Plotly.

Creates an interactive HTML animation with 3D controls (rotate, zoom, pan)
and animation playback controls (play/pause, time slider).
"""

import logging
from pathlib import Path
import numpy as np
from swctools import FrustaSet, animate_frusta_timeseries

from toric_spines_sim.paths import get_swc_path

logging.basicConfig(level=logging.INFO)

swc_filepath = get_swc_path("TS2_wsink_r10um.swc", units="microns")
frusta = FrustaSet.from_swc_file(swc_filepath)

animation_file = Path("animations/example.html")

n_frusta = frusta.n_frusta
T = 10
dt = 0.1
time_domain = np.linspace(0, T, int(T / dt) + 1)
freqs = np.random.uniform(3, 30, n_frusta) / T
V = np.zeros((len(time_domain), n_frusta))
for i, t in enumerate(time_domain):
    V[i, :] = np.sin(freqs * t)

fig = animate_frusta_timeseries(
    frusta,
    time_domain=time_domain,
    amplitudes=V,
    colorscale="Viridis",
    fps=30,
    stride=1,
    output_path=animation_file,
    auto_open=False,
)

print("Animation saved to", animation_file)
