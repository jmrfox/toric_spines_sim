"""Loop attenuation for every toric spine (peak depolarization vs distance).

For each spine: one 5 nS AMPA synapse at the site farthest from the neck
(its synapse file when there is one; TS21 / TS24 use the farthest spine
point), loops closed vs cut. See ``toric_spines_sim.attenuation``.

Run from the repository root::

    uv run python -m simulations.loop_attenuation                 # all spines
    uv run python -m simulations.loop_attenuation TS1 TS3         # some
    uv run python -m simulations.loop_attenuation --cross-check   # + jaxley check
    uv run python -m simulations.loop_attenuation --style poster
    uv run python -m simulations.loop_attenuation --replot        # redraw saved results
    uv run python -m simulations.loop_attenuation --replot --style poster \
        --panel TS1 TS3 TS21 TS24 --cols 2 --name poster_loops_2x2

Writes to ``simulations/attenuation/results/``: per spine
``{TS}_attenuation.csv`` (one row per segment), ``{TS}_attenuation_summary.csv``
and ``{TS}_attenuation.{png,pdf}``; for the whole run ``summary.csv`` and
``all_spines.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from toric_spines_sim.attenuation import (  # noqa: E402
    AttenuationResult,
    loop_attenuation,
    plot_loop_attenuation,
)
from toric_spines_sim.paths import get_simulation_path  # noqa: E402

from simulations.common.spines import ALL_SPINES  # noqa: E402

logger = logging.getLogger(__name__)
OUT = get_simulation_path("attenuation", "results")


def draw_panel(results, cols: int, style: str, name: str):
    """Grid of attenuation plots with one shared legend; returns the PNG path."""
    from toric_spines_sim.attenuation import STYLES, style_rc

    st = STYLES[style]
    n = len(results)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    with plt.rc_context(style_rc(style)):
        fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.6 * rows), squeeze=False)
        for ax, res in zip(axes.flat, results):
            plot_loop_attenuation(res, ax=ax, style=style, legend=False)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        # Axis labels on the outer panels only.
        for i, ax in enumerate(axes.flat[:n]):
            r, c = divmod(i, cols)
            if c > 0:
                ax.set_ylabel("")
            if r < rows - 1 and i + cols < n:
                ax.set_xlabel("")
        # One shared legend above the panels (the sink label omits per-spine values).
        handles, labels = axes.flat[0].get_legend_handles_labels()
        sigma_uv = max(r.table.query("tag != 3").peak_closed_mV.std(ddof=0) for r in results) * 1e3
        labels = [f"sink, mean ± 1σ (σ < {max(sigma_uv, 0.05):.1g} µV)" if lab.startswith("sink") else lab
                  for lab in labels]
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   fontsize=st["size"] * 0.85, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (3.6 * rows)))
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=200)
        plt.close(fig)
    return OUT / f"{name}.png"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spines", nargs="*", default=list(ALL_SPINES), help="spine ids (default: all)")
    ap.add_argument("--cross-check", action="store_true", help="also run the closed arm in jaxley")
    ap.add_argument("--style", choices=["paper", "poster"], default="paper")
    ap.add_argument("--gmax-nS", type=float, default=5.0, help="synapse peak conductance (nS)")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved CSVs (no simulation)")
    ap.add_argument("--panel", nargs="+", help="draw only a panel figure of these spines, in this order")
    ap.add_argument("--cols", type=int, default=3, help="panel columns")
    ap.add_argument("--name", default="all_spines", help="panel file name (without extension)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.panel:
        spines = [s.upper() for s in args.panel]
        results = [AttenuationResult.load(OUT, s) if args.replot else loop_attenuation(
            s, gmax_uS=args.gmax_nS * 1e-3, cross_check=args.cross_check) for s in spines]
        path = draw_panel(results, args.cols, args.style, args.name)
        print(f"Wrote {path}")
        return
    results = []
    for spine in [s.upper() for s in args.spines]:
        t0 = time.time()
        if args.replot:
            res = AttenuationResult.load(OUT, spine)
        else:
            res = loop_attenuation(spine, gmax_uS=args.gmax_nS * 1e-3, cross_check=args.cross_check)
            res.save(OUT)
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        plot_loop_attenuation(res, ax=ax, style=args.style)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{spine}_attenuation.{ext}", dpi=200)
        plt.close(fig)
        s = res.summary
        logger.info(
            "%s: %d loops, %s, synapse %.2f mV closed / %.2f mV cut, neck %.2f / %.2f mV (%.1fs)",
            spine, s["loops"], s["stimulus"], s["peak_synapse_closed_mV"], s["peak_synapse_cut_mV"],
            s["peak_neck_closed_mV"], s["peak_neck_cut_mV"], time.time() - t0,
        )
        results.append(res)

    summary = pd.DataFrame([r.summary for r in results])
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")

    draw_panel(results, 3, args.style, args.name)

    cols_show = ["spine", "loops", "stimulus", "peak_synapse_closed_mV", "peak_synapse_cut_mV",
                 "cut_overestimate_at_synapse", "peak_neck_closed_mV", "neck_transfer_closed"]
    if args.cross_check or summary["jaxley_max_diff_mV"].notna().any():
        cols_show.append("jaxley_max_diff_mV")
    print(summary[cols_show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
