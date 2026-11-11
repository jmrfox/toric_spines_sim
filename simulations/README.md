# Simulations

Full guide: [docs/simulations.md](../docs/simulations.md) ([online](https://jmrfox.github.io/toric_spines_sim/simulations/)).

Run from the repository root; every spine uses the same entry points:

```bash
uv run python -m simulations.axons TS1            # or: all
uv run python -m simulations.axons_dash TS1 --scenario sequential
uv run python -m simulations.axons_animation TS1
uv run python -m simulations.loop_attenuation     # all nine spines
uv run python -m simulations.ts2.ts2_integration --verbose
```

Per-spine settings: [common/spines.py](common/spines.py). Input scenarios:
[common/scenarios.py](common/scenarios.py). Passive studies run on the scipy
reference solver and HH studies on jaxley (`TSSimulator(solver="auto")`); both
close the spine's loops.

```
simulations/
  README.md
  axons.py                axon PDF study:        python -m simulations.axons TS1 [TS3 ...|all]
  axons_dash.py           3D Dash dashboard:     python -m simulations.axons_dash TS1 --scenario sequential
  axons_animation.py      HTML voltage movie:    python -m simulations.axons_animation TS1
  loop_attenuation.py     loops closed vs cut:   python -m simulations.loop_attenuation [TS1 ...]
  common/
    spines.py             spine registry: ids, inputs, parameter bank, per-spine overrides
    scenarios.py          named input scenarios (simultaneous, sequential, periodic, ...)
    axons_report.py       pairwise / sequential / synchrony PDF study
    axons_dash.py, axons_animation.py, inputs.py, model.py, utils.py
  ts1/neurosignature/     TS1 descriptor pipeline
  ts2/                    TS2 Poisson integration + frusta animation
  ts{id}/axons/results/   axon-study outputs (gitignored)
  attenuation/results/    loop-attenuation outputs
```
