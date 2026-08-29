from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.ticker import FormatStrFormatter


def arbor_samples_to_arrays(samples: Sequence[Tuple[Iterable[Sequence[float]], Any]]):
    """Convert Arbor samples to plain arrays.

    Parameters
    ----------
    samples : sequence
        Typically the return value from `sim.samples(handle)` which is a list
        of (array Nx2, metadata) pairs, where each array row is [time, value].

    Returns
    -------
    list of (t, y, meta)
        t and y are lists of floats.
    """
    out = []
    for data, meta in samples:
        # data is typically a numpy-like Nx2 array; convert robustly
        t = [row[0] for row in data]
        y = [row[1] for row in data]
        out.append((t, y, meta))
    return out


class TimeSeriesPlotter:
    """Convenience Matplotlib plotter for time series."""

    def __init__(
        self,
        *,
        title: str | None = None,
        xlabel: str = "t (ms)",
        ylabel: str = "V (mV)",
        figsize: Tuple[int, int] = (8, 4),
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        nrows: int = 1,
        sharex: bool = True,
        ylabels: Optional[Sequence[str]] = None,
        dpi: int = 100,
    ):
        self._fig, axes = plt.subplots(
            nrows=nrows, ncols=1, sharex=sharex, figsize=figsize, dpi=dpi
        )
        if isinstance(axes, (list, tuple)):
            self._axes_list = list(axes)
        elif hasattr(axes, "ravel"):
            self._axes_list = list(axes.ravel())
        else:
            self._axes_list = [axes]
        self._ax = self._axes_list[0]
        if title:
            if len(self._axes_list) > 1:
                self._fig.suptitle(title)
            else:
                self._ax.set_title(title)
        if len(self._axes_list) > 1:
            self._axes_list[-1].set_xlabel(xlabel)
            if ylabels is not None:
                for a, yl in zip(self._axes_list, ylabels):
                    a.set_ylabel(yl)
            else:
                for a in self._axes_list:
                    a.set_ylabel(ylabel)
        else:
            self._ax.set_xlabel(xlabel)
            self._ax.set_ylabel(ylabel)
        # self._ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        # self._ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # self._ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self._lines: list = []
        if xlim:
            for a in self._axes_list:
                a.set_xlim(xlim)
        if ylim:
            for a in self._axes_list:
                a.set_ylim(ylim)

    @property
    def figure(self):
        return self._fig

    @property
    def axes(self):
        return self._ax

    @property
    def axes_list(self):
        return self._axes_list

    def add_time_series(
        self,
        t: Union[Sequence[float], nap.Tsd],
        y: Optional[Sequence[float]] = None,
        *,
        label: str | None = None,
        color: str | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        row: int = 0,
        show_legend: bool = False,
        summary: str | Sequence[str] = None,
        summary_style: str = "--",
        summary_alpha: float = 0.7,
        summary_color: str | None = None,
    ):
        # Handle pynapple Tsd objects (stored in seconds, convert to ms)
        if isinstance(t, nap.Tsd):
            t_array = t.as_units("ms").index.values
            y_array = t.values
        elif y is not None:
            t_array = t
            y_array = y
        else:
            raise ValueError("When t is not a Tsd, y must be provided")

        ax = self._axes_list[row]
        (line,) = ax.plot(
            t_array,
            y_array,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        self._lines.append(line)
        y_arr = np.asarray(y_array)
        if summary and y_arr.size:
            if isinstance(summary, str):
                stats = (summary,)
            elif isinstance(summary, bool):
                stats = ("min", "max", "mean", "median")
            else:
                stats = tuple(summary)
            base_color = color if color is not None else line.get_color()
            for s in stats:
                if s == "min":
                    v = float(np.min(y_arr))
                elif s == "max":
                    v = float(np.max(y_arr))
                elif s == "mean":
                    v = float(np.mean(y_arr))
                elif s == "median":
                    v = float(np.median(y_arr))
                else:
                    continue
                lab = None
                if label is not None:
                    lab = f"{label} {s}"
                hline = ax.axhline(
                    y=v,
                    color=summary_color if summary_color is not None else base_color,
                    linestyle=summary_style,
                    linewidth=linewidth,
                    alpha=summary_alpha,
                    label=lab,
                )
                self._lines.append(hline)
        if show_legend and label:
            ax.legend()

        # # if current span of ylim is smaller than (1 + ypadding) * y_span, update it
        # y_span = max(y) - min(y)
        # if ax.get_ylim()[1] - ax.get_ylim()[0] < (1 + ypadding) * y_span:
        #     ax.set_ylim(min(y) - ypadding * y_span, max(y) + ypadding * y_span)
        return line

    def add_time_series_from_arbor(
        self,
        samples: Sequence[Tuple[Iterable[Sequence[float]], Any]],
        *,
        label: str | None = None,
        color: str | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        row: int = 0,
        show_legend: bool = False,
        summary: str | Sequence[str] = None,
        summary_style: str = "--",
        summary_alpha: float = 0.7,
        summary_color: str | None = None,
    ):
        arrs = arbor_samples_to_arrays(samples)
        # If multiple arrays are provided, append an index to labels unless provided
        for i, (t, y, meta) in enumerate(arrs):
            lab = (
                label
                if label is not None
                else (
                    getattr(meta, "tag", f"trace_{i}") if hasattr(meta, "tag") else None
                )
            )
            self.add_time_series(
                t,
                y,
                label=lab,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                row=row,
                show_legend=show_legend,
                summary=summary,
                summary_style=summary_style,
                summary_alpha=summary_alpha,
                summary_color=summary_color,
            )

    def save(self, path: str, *, dpi: int = 150):
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=dpi)

    def show(self):
        self._fig.tight_layout()
        plt.show()


class HistogramGridPlotter:
    def __init__(
        self,
        mats_by_ij: Dict[Tuple[int, int], Any],
        *,
        title: str | None = None,
        bins: int | Sequence[float] = 10,
        figsize_per_cell: Tuple[float, float] = (3.0, 2.2),
        dpi: int = 100,
        share_limits: bool = True,
        wspace: float = 0.2,
        hspace: float = 0.2,
        major_pad: float = 0.03,
        hide_subplot_x: bool = False,
        hide_subplot_y: bool = True,
        x_ticks_bottom_per_column: bool = True,
        subplot_xlabel: Optional[str] = None,
        subplot_ylabel: Optional[str] = None,
        outer_tick_labelsize: int = 12,
        outer_labelsize: Optional[int] = None,
        outer_xlabel: Optional[str] = "j",
        outer_ylabel: Optional[str] = "i",
    ):
        if not mats_by_ij:
            raise ValueError("mats_by_ij is empty")

        iset = sorted({ij[0] for ij in mats_by_ij.keys()})
        jset = sorted({ij[1] for ij in mats_by_ij.keys()})

        self._i_min, self._i_max = min(iset), max(iset)
        self._j_min, self._j_max = min(jset), max(jset)
        self._i_vals = list(range(self._i_min, self._i_max + 1))
        self._j_vals = list(range(self._j_min, self._j_max + 1))

        nrows = len(self._i_vals)
        ncols = len(self._j_vals)
        fig_w = max(1, ncols) * figsize_per_cell[0]
        fig_h = max(1, nrows) * figsize_per_cell[1]
        self._fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False
        )
        self._axes = axes
        self._fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if title:
            self._fig.suptitle(title)

        x_min, x_max = None, None
        y_max = None
        if share_limits:
            vals = []
            for (i, j), M in mats_by_ij.items():
                if M is None:
                    continue
                v = np.asarray(M).ravel()
                if v.size:
                    v = v[np.isfinite(v)]
                    if v.size:
                        vals.append(v)
            if vals:
                all_vals = np.concatenate(vals)
                if all_vals.size:
                    x_min, x_max = np.nanmin(all_vals), np.nanmax(all_vals)
                    if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
                        pad = 0.02 * (x_max - x_min)
                        x_min, x_max = x_min - pad, x_max + pad
                    else:
                        x_min, x_max = None, None

        for ri, i in enumerate(self._i_vals):
            for cj, j in enumerate(self._j_vals):
                ax = self._axes[ri, cj]
                M = mats_by_ij.get((i, j))
                if M is None:
                    ax.axis("off")
                    continue
                v = np.asarray(M).ravel()
                v = v[np.isfinite(v)] if v.size else v
                if v.size == 0:
                    ax.axis("off")
                    continue
                n, _, _ = ax.hist(v, bins=bins)
                if share_limits:
                    ymax_local = float(np.nanmax(n)) if n.size else 0.0
                    if np.isfinite(ymax_local):
                        y_max = max(y_max or 0.0, ymax_local)
                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
        if share_limits and y_max is not None and np.isfinite(y_max):
            ylim_top = y_max * 1.05 if y_max > 0 else 1.0
            for ax in self._axes.ravel():
                if ax.has_data():
                    ax.set_ylim(0.0, ylim_top)

        for ax in self._axes.ravel():
            if hide_subplot_x:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )
                if subplot_xlabel is not None:
                    ax.set_xlabel("")
            if hide_subplot_y:
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )
                if subplot_ylabel is not None:
                    ax.set_ylabel("")

        if x_ticks_bottom_per_column:
            nrows, ncols = self._axes.shape
            for cj, j in enumerate(self._j_vals):
                target_ri = None
                for ri in range(nrows - 1, -1, -1):
                    i = self._i_vals[ri]
                    M = mats_by_ij.get((i, j))
                    if M is None:
                        continue
                    v = np.asarray(M).ravel()
                    v = v[np.isfinite(v)] if v.size else v
                    if v.size == 0:
                        continue
                    target_ri = ri
                    break
                for ri in range(nrows):
                    ax = self._axes[ri, cj]
                    show_bottom = target_ri is not None and ri == target_ri
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=show_bottom,
                        labelbottom=show_bottom,
                        top=False,
                        labeltop=False,
                    )
                    if show_bottom and subplot_xlabel is not None:
                        ax.set_xlabel(subplot_xlabel)

        self._outer_tick_labelsize = outer_tick_labelsize
        self._outer_labelsize = outer_labelsize
        self._outer_xlabel = outer_xlabel
        self._outer_ylabel = outer_ylabel
        self._major_pad = float(major_pad)
        self._outer_ax = None
        self._update_outer_axis()

    def _grid_bbox(self) -> Tuple[float, float, float, float]:
        positions = [ax.get_position() for ax in self._axes.ravel()]
        left = min(p.x0 for p in positions)
        right = max(p.x1 for p in positions)
        bottom = min(p.y0 for p in positions)
        top = max(p.y1 for p in positions)
        pad = self._major_pad
        left = max(0.0, left - pad)
        bottom = max(0.0, bottom - pad)
        right = min(1.0, right + pad)
        top = min(1.0, top + pad)
        return left, bottom, right - left, top - bottom

    def _update_outer_axis(self):
        left, bottom, width, height = self._grid_bbox()
        if getattr(self, "_outer_ax", None) is None:
            self._outer_ax = self._fig.add_axes(
                [left, bottom, width, height], frameon=True
            )
        else:
            self._outer_ax.set_position([left, bottom, width, height])

        self._outer_ax.set_facecolor("none")
        self._outer_ax.set_xlim(self._j_min - 0.5, self._j_max + 0.5)
        self._outer_ax.set_ylim(self._i_max + 0.5, self._i_min - 0.5)
        self._outer_ax.set_xticks(list(range(self._j_min, self._j_max + 1)))
        self._outer_ax.set_yticks(list(range(self._i_min, self._i_max + 1)))
        labelsize = getattr(self, "_outer_tick_labelsize", 12)
        self._outer_ax.tick_params(
            axis="x",
            which="both",
            labelsize=labelsize,
            bottom=True,
            labelbottom=True,
            top=False,
            labeltop=False,
        )
        self._outer_ax.tick_params(
            axis="y",
            which="both",
            labelsize=labelsize,
            left=True,
            labelleft=True,
            right=False,
            labelright=False,
        )
        for spine in self._outer_ax.spines.values():
            spine.set_visible(True)
        label_fs = getattr(self, "_outer_labelsize", None) or labelsize
        self._outer_ax.set_xlabel(self._outer_xlabel, fontsize=label_fs)
        self._outer_ax.set_ylabel(self._outer_ylabel, fontsize=label_fs)
        self._outer_ax.grid(False)

    @property
    def figure(self):
        return self._fig

    @property
    def axes(self):
        return self._axes

    def save(self, path: str, *, dpi: int = 150):
        self._fig.tight_layout()
        self._update_outer_axis()
        self._fig.savefig(path, dpi=dpi)

    def show(self):
        self._fig.tight_layout()
        self._update_outer_axis()
        plt.show()


class RasterPlotter:
    """Raster plotter for multiple event streams (rows over time)."""

    def __init__(
        self,
        *,
        ax=None,
        title: str | None = None,
        xlabel: str = "t (ms)",
        ylabel: str = "stream",
        figsize: Tuple[int, int] = (8, 4),
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        linewidth: float = 1.0,
    ):
        self._owns_figure = ax is None
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self._fig = ax.figure
            self._ax = ax
        self._linewidth = linewidth
        if self._owns_figure:
            if title:
                self._ax.set_title(title)
            self._ax.set_xlabel(xlabel)
            self._ax.set_ylabel(ylabel)
        self._ax.invert_yaxis()
        self._lines: list = []
        self._labels: List[str] = []
        self._label_to_row: Dict[str, int] = {}
        if xlim:
            self._ax.set_xlim(xlim)
        if ylim:
            self._ax.set_ylim(ylim)

    @property
    def figure(self):
        return self._fig

    @property
    def axes(self):
        return self._ax

    def _update_yticks(self):
        n = len(self._labels)
        self._ax.set_yticks(list(range(n)))
        self._ax.set_yticklabels(self._labels)
        if n > 0:
            self._ax.set_ylim(n - 0.5, -0.5)
        else:
            self._ax.set_ylim(0.5, -0.5)

    def add_stream(
        self,
        events: Union[Sequence[float], nap.Ts],
        *,
        label: str | None = None,
        color: str | None = None,
        linelength: float = 0.8,
        linewidth: float | None = None,
    ):
        lw = linewidth if linewidth is not None else self._linewidth
        # Handle pynapple Ts objects (stored in seconds, convert to ms)
        if isinstance(events, nap.Ts):
            times = events.as_units("ms").index.values
            # Use label from Ts metadata if available and not overridden
            if label is None and hasattr(events, "label"):
                label = events.label
        else:
            times = events

        if label is None:
            label = f"stream_{len(self._labels)}"
        if label in self._label_to_row:
            row = self._label_to_row[label]
        else:
            row = len(self._labels)
            self._labels.append(label)
            self._label_to_row[label] = row
        # Draw vertical ticks for this stream's events at the assigned row
        coll = self._ax.eventplot(
            [list(times)],
            lineoffsets=[row],
            linelengths=[linelength],
            colors=[color] if color else None,
            linewidths=[lw],
        )
        # eventplot returns a list of LineCollections
        self._lines.extend(coll)
        self._update_yticks()
        return coll

    def add_streams(
        self,
        streams: Union[Dict[str, Sequence[float]], nap.TsGroup],
        *,
        linelength: float = 0.8,
        linewidth: float | None = None,
    ):
        # Handle pynapple TsGroup objects
        if isinstance(streams, nap.TsGroup):
            # Get labels from metadata if available
            labels = None
            if "label" in streams.metadata:
                labels = streams.get_info("label")
            for idx in sorted(streams.keys()):
                ts = streams[idx]
                if labels is not None and idx in labels.index:
                    lab = labels[idx]
                elif isinstance(idx, int):
                    lab = f"syn_{idx}"
                else:
                    lab = str(idx)
                self.add_stream(
                    ts, label=lab, linelength=linelength, linewidth=linewidth
                )
        else:
            for lab, times in streams.items():
                self.add_stream(
                    times, label=lab, linelength=linelength, linewidth=linewidth
                )

    def save(self, path: str, *, dpi: int = 150):
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=dpi)

    def show(self):
        self._fig.tight_layout()
        plt.show()
