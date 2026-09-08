"""PDF report builder wrapping ReportLab (matplotlib and Plotly figures).

Use ``PdfReport`` to accumulate titles, tables, and figures, then ``build()``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import io
import math

try:
    import plotly.graph_objects as _go  # type: ignore

    _PLOTLY_FIGURE_CLS = _go.Figure  # type: ignore
except Exception:
    _PLOTLY_FIGURE_CLS = None  # type: ignore

try:
    from matplotlib.figure import Figure
except Exception:
    Figure = Any  # type: ignore

_DEFAULT_MAX_CONTENT_HEIGHT = 9.0 * inch


def figure_size_for_page(
    fig_width: float,
    fig_height: float,
    *,
    width_in: float,
    max_height_in: float,
) -> Tuple[float, float]:
    """Return (width, height) in inches that fit the page content area."""
    if fig_width <= 0 or fig_height <= 0:
        raise ValueError("fig_width and fig_height must be positive")
    aspect = fig_height / fig_width
    w = width_in
    h = w * aspect
    if h > max_height_in:
        h = max_height_in
        w = h / aspect
    return (w, h)


def _scale_image_to_page(
    nat_w: int,
    nat_h: int,
    *,
    page_width: float,
    max_height: float,
) -> Tuple[float, float]:
    """Scale a raster image to page width, capping height."""
    if nat_w <= 0 or nat_h <= 0:
        return page_width, None  # type: ignore[return-value]
    aspect = nat_h / nat_w
    tw = page_width
    th = tw * aspect
    if th > max_height:
        th = max_height
        tw = max_height / aspect
    return tw, th


class PdfReport:
    """Accumulate a letter-size PDF of text, tables, and figures.

    Parameters
    ----------
    output_path : str
        Destination ``.pdf`` path.
    pagesize, left_margin, right_margin, top_margin, bottom_margin
        ReportLab page setup (defaults: letter, 0.75 in margins).

    Examples
    --------
    >>> report = PdfReport("out.pdf")
    >>> report.add_title("TS1")
    >>> report.add_figure(fig)  # matplotlib or plotly
    >>> report.build()
    """
    def __init__(
        self,
        output_path: str,
        *,
        pagesize: Tuple[float, float] = letter,
        left_margin: float = 0.75 * inch,
        right_margin: float = 0.75 * inch,
        top_margin: float = 0.75 * inch,
        bottom_margin: float = 0.75 * inch,
    ):
        self._doc = SimpleDocTemplate(
            output_path,
            pagesize=pagesize,
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )
        self._story: List[Any] = []
        self._styles = getSampleStyleSheet()
        self._resources: List[io.BytesIO] = []
        self._left_margin = left_margin
        self._right_margin = right_margin
        self._top_margin = top_margin
        self._bottom_margin = bottom_margin

    @property
    def content_width(self) -> float:
        """Usable page width in points (ReportLab units)."""
        return getattr(self._doc, "width", 6.0 * inch)

    @property
    def content_width_in(self) -> float:
        """Usable page width in inches."""
        return self.content_width / inch

    @property
    def content_max_height(self) -> float:
        """Maximum embeddable figure height in points."""
        return _DEFAULT_MAX_CONTENT_HEIGHT

    @property
    def content_max_height_in(self) -> float:
        """Maximum embeddable figure height in inches."""
        return self.content_max_height / inch

    def figure_size(
        self,
        fig_width: float,
        fig_height: float,
    ) -> Tuple[float, float]:
        """Matplotlib figsize (inches) scaled to this report's content area."""
        return figure_size_for_page(
            fig_width,
            fig_height,
            width_in=self.content_width_in,
            max_height_in=self.content_max_height_in,
        )

    def add_title(self, text: str):
        """Append a title paragraph."""
        self._story.append(Paragraph(text, self._styles["Title"]))
        self._story.append(Spacer(1, 0.2 * inch))

    def add_heading(self, text: str, *, level: int = 1):
        """Append a heading (level 1–3)."""
        style_name = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(
            level, "Heading3"
        )
        self._story.append(Paragraph(text, self._styles[style_name]))
        self._story.append(Spacer(1, 0.1 * inch))

    def add_paragraph(self, text: str):
        """Append a body paragraph."""
        self._story.append(Paragraph(text, self._styles["BodyText"]))
        self._story.append(Spacer(1, 0.1 * inch))

    def add_spacer(self, height: float = 0.1 * inch):
        """Append vertical space."""
        self._story.append(Spacer(1, height))

    def add_page_break(self):
        """Start a new page."""
        self._story.append(PageBreak())

    def add_table(
        self,
        data: Sequence[Sequence[Any]],
        *,
        col_widths: Optional[Sequence[float]] = None,
        row_heights: Optional[Sequence[float]] = None,
        grid: bool = True,
        header_row: bool = False,
    ):
        """Append a grid table. First row is styled as a header if ``header_row``."""
        table = Table(data, colWidths=col_widths, rowHeights=row_heights)
        style_cmds: List[Tuple[str, Tuple[int, int], Tuple[int, int], Any]] = []
        if grid:
            style_cmds.append(("GRID", (0, 0), (-1, -1), 0.5, colors.grey))
        style_cmds.extend(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
        if header_row and len(data) > 0:
            style_cmds.extend(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ]
            )
        table.setStyle(TableStyle(style_cmds))
        self._story.append(table)
        self._story.append(Spacer(1, 0.1 * inch))

    def add_dict_table(
        self,
        data: Mapping[str, Any],
        *,
        column_names: Sequence[str] = ("Key", "Value"),
        columns: Optional[int] = None,
        grid: bool = True,
        header: bool = True,
    ):
        """Append a key/value table, optionally split across even column pairs."""
        items = list(data.items())
        if not items:
            return
        if columns is None or columns < 2:
            columns = 2
        if columns % 2 != 0:
            raise ValueError(
                "columns must be an even integer (pairs of key/value columns)"
            )
        groups = max(1, columns // 2)
        chunk_size = int(math.ceil(len(items) / groups))
        chunks: List[List[Tuple[Any, Any]]] = [
            items[i * chunk_size : (i + 1) * chunk_size] for i in range(groups)
        ]
        ncols = columns
        rows: List[List[str]] = []
        if header:
            header_row: List[str] = []
            for _ in range(groups):
                header_row.extend([str(column_names[0]), str(column_names[1])])
            rows.append(header_row)
        max_rows = max((len(c) for c in chunks), default=0)
        for r in range(max_rows):
            row: List[str] = []
            for chunk in chunks:
                if r < len(chunk):
                    k, v = chunk[r]
                    row.extend([str(k), str(v)])
                else:
                    row.extend(["", ""])
            rows.append(row)
        cw = None
        if ncols > 0:
            fw = getattr(self._doc, "width", 6.0 * inch)
            cw = [fw / ncols] * ncols
        self.add_table(rows, col_widths=cw, grid=grid, header_row=header)

    def add_image(
        self,
        image_source: Any,
        *,
        width: Optional[float] = None,
        height: Optional[float] = None,
        figure_spacer: float = 0.1 * inch,
    ):
        """Embed a raster image, scaled to content width by default."""
        if width is None and height is None:
            width = self.content_width
            height = None
        img = Image(image_source, width=width, height=height)
        try:
            img.hAlign = "CENTER"
        except Exception:
            pass
        self._story.append(img)
        self._story.append(Spacer(1, figure_spacer))

    def add_figure(
        self,
        fig: Any,
        *,
        width: Optional[float] = None,
        height: Optional[float] = None,
        dpi: int = 300,
        figure_spacer: Optional[float] = None,
    ):
        """Embed a matplotlib or Plotly figure as PNG."""
        if figure_spacer is None:
            figure_spacer = 0.05 * inch
        buf = io.BytesIO()
        handled = False
        if hasattr(fig, "savefig"):
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            handled = True
        elif (
            (_PLOTLY_FIGURE_CLS is not None and isinstance(fig, _PLOTLY_FIGURE_CLS))
            or hasattr(fig, "to_image")
            or hasattr(fig, "write_image")
        ):
            try:
                fw_in = self.content_width_in
                export_px = int(fw_in * dpi)
                export_px = max(600, min(2400, export_px))
                lw = getattr(getattr(fig, "layout", None), "width", None)
                lh = getattr(getattr(fig, "layout", None), "height", None)
                if isinstance(lw, (int, float)) and isinstance(lh, (int, float)) and lw > 0 and lh > 0:
                    aspect = lh / lw
                    export_h = int(export_px * aspect)
                    png_bytes = fig.to_image(format="png", width=export_px, height=export_h, scale=1)  # type: ignore[attr-defined]
                else:
                    png_bytes = fig.to_image(format="png", width=export_px, scale=1)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError(
                    "Plotly figure export requires the 'kaleido' package (pip install -U kaleido)."
                ) from e
            buf.write(png_bytes)
            handled = True
        if not handled:
            raise TypeError("Unsupported figure type; expected Matplotlib or Plotly figure.")
        buf.seek(0)
        self._resources.append(buf)
        if width is None and height is None:
            try:
                nat_w, nat_h = ImageReader(buf).getSize()
                width, height = _scale_image_to_page(
                    nat_w,
                    nat_h,
                    page_width=self.content_width,
                    max_height=self.content_max_height,
                )
            except Exception:
                width = self.content_width
                height = None
        self.add_image(buf, width=width, height=height, figure_spacer=figure_spacer)

    def add_section_figure(
        self,
        fig: Any,
        *,
        heading: Optional[str] = None,
        level: int = 3,
        paragraph: Optional[str] = None,
        page_break_after: bool = False,
        **figure_kwargs: Any,
    ) -> None:
        """Add optional heading/paragraph followed by a figure."""
        if heading:
            self.add_heading(heading, level=level)
        if paragraph:
            self.add_paragraph(paragraph)
        self.add_figure(fig, **figure_kwargs)
        if page_break_after:
            self.add_page_break()

    def add_simulation(
        self,
        name: str,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        figures: Optional[Sequence[Any]] = None,
        page_break_after: bool = False,
    ):
        """Append a named simulation block: parameter table, metrics, figures."""
        self.add_heading(name, level=2)
        rows: List[List[str]] = []
        if parameters:
            rows.append(["Parameter", "Value"])
            for k, v in parameters.items():
                rows.append([str(k), str(v)])
            self.add_table(rows, header_row=True)
        if metrics:
            mrows: List[List[str]] = [["Metric", "Value"]]
            for k, v in metrics.items():
                mrows.append([str(k), str(v)])
            self.add_table(mrows, header_row=True)
        if figures:
            for f in figures:
                self.add_figure(f, width=None, height=None)
        if page_break_after:
            self.add_page_break()

    def build(self):
        """Write the PDF to ``output_path``."""
        self._doc.build(self._story)
