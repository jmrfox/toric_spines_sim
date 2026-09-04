#!/usr/bin/env python3
"""TS2 single-cell simulation with synapses driven by spike-source cells.

This script:
- Loads morphology from data/swc/TS2_s50.swc.
- Builds a passive cable-cell with 'pas' density mechanism.
- Places N synapses at randomly chosen locations on the region "spine".
- Drives those synapses with a Poisson schedule via a population of
  spike_source_cell inputs (one per synapse) connected with a configurable
  delay.
- Samples membrane voltage at the root and plots the trace.

Usage (from repo root):
    python scripts/ts2_sim.py --n-syn 1 --freq-hz 5 --tfinal-ms 300

Notes:
- Default synapse mechanism is 'expsyn'. You can switch with --synapse exp2syn.
- Weight is in µS as interpreted by Arbor point synapses.
"""

from __future__ import annotations

import argparse
import random
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

import arbor as A
from arbor import units as U


def parse_cycle_breaks(swc_path: Path):
    """Parse '# CYCLE_BREAK reconnect i j' annotations from an SWC file.

    Returns a list of integer pairs [(i, j), ...].
    """
    pairs = []
    try:
        text = swc_path.read_text()
    except Exception:
        return pairs
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        # Example: # CYCLE_BREAK reconnect 71 35
        parts = line[1:].strip().split()
        if (
            len(parts) == 4
            and parts[0].upper() == "CYCLE_BREAK"
            and parts[1].lower() == "reconnect"
        ):
            try:
                a = int(parts[2])
                b = int(parts[3])
                pairs.append((a, b))
            except ValueError:
                continue
    return pairs


def read_swc_points(swc_path: Path):
    """Return dict id -> (x, y, z, r) from SWC content (ignores non-data lines)."""
    pts = {}
    try:
        for line in swc_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            if len(cols) < 7:
                continue
            try:
                nid = int(cols[0])
                x = float(cols[2])
                y = float(cols[3])
                z = float(cols[4])
                r = float(cols[5])
                pts[nid] = (x, y, z, r)
            except Exception:
                continue
    except Exception:
        pass
    return pts


def join_locset_terms(terms: List[str]) -> str:
    """Join single-location locset terms into a single locset expression string."""
    if not terms:
        raise ValueError("No location terms provided.")
    if len(terms) == 1:
        return terms[0]
    return "(join " + " ".join(terms) + ")"


class TS2Recipe(A.recipe):
    def __init__(
        self,
        cell: A.cable_cell,
        *,
        n_syn: int,
        weight_uS: float,
        freq_hz: float,
        tstart_ms: float,
        tstop_ms: float,
        seed: int,
        delay_ms: float,
        source_mode: str = "poisson",
        spike_csv: str | None = None,
        gj_pairs: list[tuple[str, str]] | None = None,
        gj_weight: float = 1.0,
    ):
        super().__init__()
        self._cell = cell
        self._gprop = A.cable_global_properties()
        # Ensure built-in mechanisms are available
        self._gprop.catalogue = A.default_catalogue()
        # Set global default properties to satisfy versions that require
        # globals for initial membrane potential and cable params.
        self._gprop.set_property(
            Vm=-70 * U.mV,
            cm=0.02 * U.F / U.m2,
            rL=30 * U.Ohm * U.cm,
            tempK=300 * U.Kelvin,
        )
        # Provide ion defaults (no active ion usage; set to zero effect)
        self._gprop.set_ion(
            "ca",
            valence=2,
            int_con=0 * U.mM,
            ext_con=0 * U.mM,
            rev_pot=0 * U.mV,
        )
        self._gprop.set_ion(
            "na",
            valence=1,
            int_con=0 * U.mM,
            ext_con=0 * U.mM,
            rev_pot=0 * U.mV,
        )
        self._gprop.set_ion(
            "k",
            valence=1,
            int_con=0 * U.mM,
            ext_con=0 * U.mM,
            rev_pot=0 * U.mV,
        )
        # input weight and schedules
        self._n_syn = n_syn
        self._weight = weight_uS
        # Store Arbor units quantities for core parameters.
        self._freq = freq_hz * U.Hz
        self._tstart = tstart_ms * U.ms
        self._tstop = tstop_ms * U.ms
        self._seed = seed
        # Delay is required as an Arbor units quantity for connections
        self._delay = delay_ms * U.ms
        self._gj_pairs = gj_pairs or []
        self._gj_weight = gj_weight
        # Precompute explicit schedules (times as units quantities) for each source gid
        self._src_sched: dict[int, A.explicit_schedule] = {}
        if self._n_syn > 0:
            if source_mode == "csv" and spike_csv:
                # Read CSV rows as source_id,time_ms and group by source_id
                times_by_src: dict[int, list[float]] = {
                    i: [] for i in range(1, 1 + self._n_syn)
                }
                try:
                    with open(spike_csv, "r", newline="") as f:
                        rdr = csv.reader(f)
                        for row in rdr:
                            if not row or len(row) < 2:
                                continue
                            try:
                                sid = int(row[0])
                                t = float(row[1])
                            except Exception:
                                # probably a header or malformed row; skip
                                continue
                            if 1 <= sid <= self._n_syn:
                                # Filter to [tstart, tstop) using units
                                t0 = self._tstart.value_as(U.ms)
                                t1 = self._tstop.value_as(U.ms)
                                if t0 <= t < t1:
                                    times_by_src[sid].append(t)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Spike CSV not found: {spike_csv}")
                # Sort and build explicit schedules
                for i in range(1, 1 + self._n_syn):
                    times = sorted(times_by_src.get(i, []))
                    times_q = [tm * U.ms for tm in times]
                    self._src_sched[i] = A.explicit_schedule(times_q)
            else:
                # Poisson mode: generate independent Poisson processes per source
                if self._freq.value_as(U.Hz) > 0:
                    for i in range(1, 1 + self._n_syn):
                        rng = random.Random(self._seed + i)
                        times_ms: list[float] = []
                        t = self._tstart.value_as(U.ms)
                        tstop = self._tstop.value_as(U.ms)
                        rate_hz = self._freq.value_as(U.Hz)
                        if rate_hz > 0:
                            while True:
                                # Exponential inter-arrival with mean 1/rate (in seconds)
                                dt_s = rng.expovariate(rate_hz)
                                t += dt_s * 1000.0  # convert to ms
                                if t >= tstop:
                                    break
                                times_ms.append(t)
                        times_q = [tm * U.ms for tm in times_ms]
                        self._src_sched[i] = A.explicit_schedule(times_q)
                else:
                    for i in range(1, 1 + self._n_syn):
                        self._src_sched[i] = A.explicit_schedule([])

    def num_cells(self):
        # 0 -> cable cell, 1..n_syn -> spike sources
        return 1 + self._n_syn

    def cell_kind(self, gid):
        return A.cell_kind.cable if gid == 0 else A.cell_kind.spike_source

    def cell_description(self, gid):
        if gid == 0:
            return self._cell
        # spike source cells emit spikes on label 'src' with explicit schedule
        sched = self._src_sched.get(gid, A.explicit_schedule([]))
        return A.spike_source_cell("src", sched)

    def global_properties(self, kind):
        return self._gprop if kind == A.cell_kind.cable else None

    def event_generators(self, gid):
        # No internal event generators: inputs come from spike_source_cell(s)
        return []

    def connections_on(self, gid):
        # Define incoming connections to gid.
        # Only the cable cell (gid 0) receives connections from spike sources.
        if gid != 0:
            return []
        dest = A.cell_local_label("syn", A.selection_policy.round_robin)
        conns = []
        for src_gid in range(1, 1 + self._n_syn):
            src = A.cell_global_label((src_gid, "src"))
            conns.append(A.connection(src, dest, self._weight, self._delay))
        return conns

    def probes(self, gid):
        # Record membrane voltage at the root location (only for cable cell gid 0)
        if gid == 0:
            return [A.cable_probe_membrane_voltage('"root"', "v_root")]
        return []

    def gap_junctions_on(self, gid):
        # For each pair of labels (a_label, b_label) placed on this cell, return
        # connections that tie them together. For a single-cell model, we connect
        # both directions with gid 0.
        conns = []
        for a_label, b_label in self._gj_pairs:
            peer_b = A.cell_global_label((gid, b_label))
            loc_a = A.cell_local_label(a_label, A.selection_policy.round_robin)
            conns.append(A.gap_junction_connection(peer_b, loc_a, self._gj_weight))
            peer_a = A.cell_global_label((gid, a_label))
            loc_b = A.cell_local_label(b_label, A.selection_policy.round_robin)
            conns.append(A.gap_junction_connection(peer_a, loc_b, self._gj_weight))
        return conns


def build_cell(
    swc_path: Path, synapse: str, n_syn: int, syn_tau_ms: float, seed: int
) -> tuple[A.cable_cell, list[tuple[str, str]], list[dict]]:
    # Load morphology (Arbor interpretation) and ensure we have an arbor.morphology
    try:
        # Prefer segment_tree -> morphology for broad compatibility
        st = A.load_swc_arbor(str(swc_path), raw=True)
        morph = A.morphology(st)
    except TypeError:
        # Fallback for versions without 'raw' kwarg: may return a loaded_morphology
        lm = A.load_swc_arbor(str(swc_path))
        morph = getattr(lm, "morphology", lm)

    # Labels: define basic regions and 'root' for probing
    labels = A.label_dict(
        {
            "all": "(all)",
            "spine": "(tag 3)",
            "root": "(root)",
        }
    )

    # Decor: passive properties + passive density mechanism
    decor = A.decor()
    # decor.set_property(
    #     Vm=-70 * U.mV,
    #     cm=0.02 * U.F / U.m2,
    #     rL=30 * U.Ohm * U.cm,
    #     tempK=300 * U.Kelvin,
    # )

    # Paint passive channel everywhere
    decor.paint("(all)", A.density("pas"))

    # Uniformly place synapses across the morphology
    loc_expr = f'(uniform (region "spine") 0 {n_syn-1} {seed})'

    # Place synapses under label 'syn'
    decor.place(loc_expr, A.synapse(synapse, {"tau": syn_tau_ms}), "syn")

    # Gap junction placements inferred from CYCLE_BREAK annotations
    gj_pairs = []  # list of (label_a, label_b)
    gj_info: list[dict] = []  # for optional printing
    cb_pairs = parse_cycle_breaks(swc_path)
    if cb_pairs:
        pw = A.place_pwlin(morph)
        pts = read_swc_points(swc_path)
        for idx, (a_id, b_id) in enumerate(cb_pairs):
            if a_id not in pts or b_id not in pts:
                continue
            ax, ay, az, ar = pts[a_id]
            bx, by, bz, br = pts[b_id]
            la, _ = pw.closest(ax, ay, az)
            lb, _ = pw.closest(bx, by, bz)
            # Build single-location locset strings from locations
            try:
                la_expr = f"(location {la.branch} {la.pos:.6f})"
                lb_expr = f"(location {lb.branch} {lb.pos:.6f})"
            except Exception:
                # Fallback: best-effort string (may fail if API differs)
                la_expr = f"(location {getattr(la,'branch',0)} {getattr(la,'pos',0.5)})"
                lb_expr = f"(location {getattr(lb,'branch',0)} {getattr(lb,'pos',0.5)})"

            a_label = f"gj_{idx}_a"
            b_label = f"gj_{idx}_b"
            # Ensure a junction mechanism is placed at each location
            decor.place(la_expr, A.junction("gj"), a_label)
            decor.place(lb_expr, A.junction("gj"), b_label)
            gj_pairs.append((a_label, b_label))
            gj_info.append(
                {
                    "index": idx,
                    "a_id": a_id,
                    "b_id": b_id,
                    "a_label": a_label,
                    "b_label": b_label,
                    "a_loc": la_expr,
                    "b_loc": lb_expr,
                }
            )

    cell = A.cable_cell(morph, decor, labels)
    # Discretization policy (reasonable default): expects float length in µm
    cell.discretization(A.cv_policy_max_extent(20.0))
    return cell, gj_pairs, gj_info


def run(args):
    repo_root = Path(__file__).resolve().parents[1]
    swc_path = Path(args.swc)
    if not swc_path.is_absolute():
        swc_path = (repo_root / args.swc).resolve()
    if not swc_path.exists():
        raise FileNotFoundError(f"SWC file not found: {swc_path}")

    cell, gj_pairs, gj_info = build_cell(
        swc_path, args.synapse, args.n_syn, args.syn_tau_ms, args.seed
    )

    # Build recipe and simulation
    # Resolve spike CSV path if provided
    spike_csv = None
    if getattr(args, "spike_csv", None):
        scp = Path(args.spike_csv)
        if not scp.is_absolute():
            scp = (repo_root / scp).resolve()
        spike_csv = str(scp)
        if args.source_mode == "csv" and not scp.exists():
            raise FileNotFoundError(f"Spike CSV file not found: {scp}")

    recipe = TS2Recipe(
        cell,
        n_syn=args.n_syn,
        weight_uS=args.weight_uS,
        freq_hz=args.freq_hz,
        tstart_ms=args.tstart_ms,
        tstop_ms=args.tfinal_ms,
        seed=args.seed,
        delay_ms=args.delay_ms,
        source_mode=args.source_mode,
        spike_csv=spike_csv,
        gj_pairs=gj_pairs,
        gj_weight=args.gj_weight,
    )
    if args.print_gjs:
        if not gj_info:
            print(
                "No CYCLE_BREAK reconnect pairs found in SWC; no gap junctions placed."
            )
        else:
            print(f"Placed {len(gj_info)} gap junction pairs from CYCLE_BREAK:")
            for item in gj_info:
                print(
                    f"  [{item['index']}] SWC {item['a_id']} <-> {item['b_id']}\n"
                    f"      labels: {item['a_label']} <-> {item['b_label']}\n"
                    f"      locs:   {item['a_loc']} <-> {item['b_loc']}"
                )
        # Also show how they resolve into connections on gid 0
        conns = recipe.gap_junctions_on(0)
        print(
            f"Recipe.gap_junctions_on(0): {len(conns)} connections (weight={args.gj_weight})"
        )
        for c in conns:
            print(f"  {c}")
    ctx = A.context()
    dec = A.partition_load_balance(recipe, ctx)
    sim = A.simulation(recipe, ctx, dec)

    # Sample voltage at the probe tagged 'v_root' on cell 0 at regular intervals
    handle = sim.sample(0, "v_root", A.regular_schedule(args.sample_dt_ms * U.ms))

    # Run
    sim.run(args.tfinal_ms * U.ms, args.dt_ms * U.ms)

    # Retrieve samples
    samples = sim.samples(handle)
    if not samples:
        raise RuntimeError("No samples returned; check probe configuration.")
    data, meta = samples[0]
    # data: Nx2 array [time(ms), value(mV)]
    t = [row[0] for row in data]
    v = [row[1] for row in data]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(t, v, label="V(root)")
    plt.xlabel("t (ms)")
    plt.ylabel("V (mV)")
    plt.title(
        f"TS2: {args.n_syn} {args.synapse} synapse(s), freq={args.freq_hz} Hz, weight={args.weight_uS} µS"
    )
    plt.legend()

    # Determine y-axis limits with padding; fallback to default range if signal is tiny
    try:
        v_min = min(v)
        v_max = max(v)
        v_range = v_max - v_min
    except ValueError:
        # Empty trace; keep defaults
        v_min = v_max = -70.0
        v_range = 0.0
    default_ylim = (-90.0, -40.0)
    small_range_threshold = 0.1  # mV; if range is smaller, use default ylim
    if v_range <= 0 or v_range < small_range_threshold:
        y0, y1 = default_ylim
    else:
        pad = max(0.3 * v_range, 1.0)  # padding
        y0 = v_min - pad
        y1 = v_max + pad
        if y1 <= y0:
            y0, y1 = default_ylim
    plt.ylim(y0, y1)

    plt.tight_layout()

    # Save figure (default) to avoid non-interactive backend issues
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    # Optionally show if requested
    if getattr(args, "show", False):
        plt.show()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--swc",
        type=str,
        default="data/swc/TS2_s50.swc",
        help="Path to SWC morphology (relative to repo root by default)",
    )
    p.add_argument(
        "--n-syn",
        dest="n_syn",
        type=int,
        default=1,
        help="Number of synapses to place randomly",
    )
    p.add_argument(
        "--synapse",
        type=str,
        default="expsyn",
        help="Synapse mechanism name (e.g., expsyn, exp2syn)",
    )
    p.add_argument(
        "--syn-tau-ms",
        dest="syn_tau_ms",
        type=float,
        default=1.0,
        help="Synapse time constant [ms]",
    )
    p.add_argument(
        "--weight-uS",
        dest="weight_uS",
        type=float,
        default=0.5,
        help="Synaptic weight [µS]",
    )
    p.add_argument(
        "--freq-hz", dest="freq_hz", type=float, default=5.0, help="Poisson rate [Hz]"
    )  #
    p.add_argument(
        "--tstart-ms",
        dest="tstart_ms",
        type=float,
        default=5.0,
        help="Event start time [ms]",
    )
    p.add_argument(
        "--tfinal-ms",
        dest="tfinal_ms",
        type=float,
        default=500.0,
        help="Simulation final time [ms]",
    )
    p.add_argument(
        "--dt-ms",
        dest="dt_ms",
        type=float,
        default=0.025,
        help="Simulation timestep [ms]",
    )
    p.add_argument(
        "--sample-dt-ms",
        dest="sample_dt_ms",
        type=float,
        default=0.1,
        help="Sampling interval [ms]",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Poisson schedule (placement uses label-language uniform)",
    )
    p.add_argument(
        "--save",
        type=str,
        default="out/ts2_trace.png",
        help="Path to save the plot image (default: out/ts2_trace.png). Use '' to disable saving.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (may fail under non-interactive backends).",
    )
    p.add_argument(
        "--gj-weight",
        dest="gj_weight",
        type=float,
        default=1.0,
        help="Gap junction connection weight (dimensionless)",
    )
    p.add_argument(
        "--print-gjs",
        action="store_true",
        help="Print inferred gap junction placements and resulting connections.",
    )
    p.add_argument(
        "--delay-ms",
        dest="delay_ms",
        type=float,
        default=0.1,
        help="Synaptic transmission delay [ms] for spike_source connections",
    )
    p.add_argument(
        "--source-mode",
        dest="source_mode",
        choices=["poisson", "csv"],
        default="poisson",
        help="Spike source mode: 'poisson' for constant-rate Poisson or 'csv' to read times from file",
    )
    p.add_argument(
        "--spike-csv",
        dest="spike_csv",
        type=str,
        default=None,
        help="CSV file with rows: source_id,time_ms (header optional). Used when --source-mode=csv",
    )
    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
