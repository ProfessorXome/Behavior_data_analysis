"""Interactive Plotly views for notebook exploration (zoom / pan / hover)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    FRAMES_PER_MINUTE,
    MetricsResult,
    _gap_threshold,
)
from .plotting import (
    _active_windows,
    _ensure_frame_column,
    _frame_values,
    _times_to_frames,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Interactive plots require plotly. Install with: pip install plotly"
    ) from exc


def _gapped_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gap: Optional[float] = None,
) -> tuple[list, list]:
    """Insert None breaks so Plotly does not connect across data gaps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if gap is None:
        gap = _gap_threshold(x)
    xs: list = []
    ys: list = []
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or (x[i] - x[i - 1]) > gap:
            xs.extend(x[start:i].tolist())
            ys.extend(y[start:i].tolist())
            if i < len(x):
                xs.append(None)
                ys.append(None)
            start = i
    return xs, ys


def _active_shapes(
    frames: Optional[pd.DataFrame],
    *,
    frames_per_minute: int = FRAMES_PER_MINUTE,
    yref: str = "y",
    xref: str = "x",
    opacity: float = 0.22,
    fillcolor: str = "#a3b18a",
) -> list[dict]:
    frames = _ensure_frame_column(frames, frames_per_minute=frames_per_minute)
    if frames is None or frames.empty or "active" not in frames.columns:
        return []
    # Gap from the full frame axis (active + inactive) so early / dense runs
    # are not split by an over-tight threshold from active-only samples.
    gap = _gap_threshold(frames["frame"].to_numpy(dtype=float))
    if not np.isfinite(gap) or gap <= 0:
        gap = 3.0
    shapes = []
    for x0, x1 in _active_windows(frames, gap=float(gap), x_col="frame"):
        if x1 <= x0:
            x0, x1 = x0 - 0.5, x0 + 0.5
        shapes.append(
            dict(
                type="rect",
                xref=xref,
                yref=yref,
                x0=float(x0),
                x1=float(x1),
                y0=0,
                y1=1,
                fillcolor=fillcolor,
                opacity=opacity,
                line=dict(width=0),
                layer="below",
            )
        )
    return shapes


def _threshold_hline_traces(
    y_values: Sequence[tuple[float, str, str, str]],
    *,
    x0: float,
    x1: float,
) -> list[go.Scatter]:
    """
    Horizontal threshold lines as data traces (always visible / autoranged).

    ``y_values`` entries are (y, name, color, dash).
    """
    traces: list[go.Scatter] = []
    if not np.isfinite(x0) or not np.isfinite(x1):
        return traces
    if x1 <= x0:
        x1 = x0 + 1.0
    for y, name, color, dash in y_values:
        if not np.isfinite(y):
            continue
        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y, y],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.3, dash=dash),
                hoverinfo="skip",
                showlegend=True,
            )
        )
    return traces


def _base_layout(
    title: str,
    *,
    height: int = 420,
    ylabel: str = "speed",
) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Arial", size=13)),
        font=dict(family="Arial"),
        height=height,
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(
            title="frame",
            showgrid=True,
            minor=dict(showgrid=True),
            rangeslider=dict(visible=True, thickness=0.08),
        ),
        yaxis=dict(title=ylabel, showgrid=True, zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )


def _scatter_line(
    x,
    y,
    *,
    name: str,
    color: str,
    width: float = 1.2,
    opacity: float = 1.0,
    y_label: str = "y",
) -> go.Scattergl:
    return go.Scattergl(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        opacity=opacity,
        hovertemplate=f"frame=%{{x}}<br>{y_label}=%{{y:.3f}}<extra>{name}</extra>",
    )


def _marker_trace(
    x,
    y,
    *,
    name: str,
    color: str,
    symbol: str = "line-ns",
    size: float = 12,
    customdata=None,
    extra_hover: str = "",
) -> go.Scatter:
    hover = f"frame=%{{x}}<br>y=%{{y:.3f}}{extra_hover}<extra>{name}</extra>"
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name=name,
        marker=dict(color=color, size=size, symbol=symbol, line=dict(width=1, color="#333")),
        customdata=customdata,
        hovertemplate=hover,
    )


def _reversal_icon_traces(
    x,
    *,
    name: str,
    color: str,
    style: str,
    y: Optional[np.ndarray] = None,
    customdata=None,
    hovertemplate: Optional[str] = None,
    tick_half_height: float = 25.0,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> list[go.Scatter]:
    """
    Reversal markers that match the matplotlib icons.

    Styles:
      - ``tick``  → vertical | bars (kept reversals)
      - ``x``     → x markers (excluded)
      - ``diamond`` → diamond (unevaluated)
      - ``triangle`` → triangle-up (omega turns)
      - ``triangle-down`` → triangle-down (omega speed rejects)
      - ``star``  → star (legacy)
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        y = np.zeros(len(x), dtype=float)
    else:
        y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return []

    if hovertemplate is None:
        hovertemplate = f"frame=%{{x}}<br>y=%{{y:.3f}}<extra>{name}</extra>"

    traces: list[go.Scatter] = []
    if style == "tick":
        # Visible vertical tick segments (matplotlib ``|``).
        xs: list = []
        ys: list = []
        for xi, yi in zip(x, y):
            xs.extend([xi, xi, None])
            ys.extend([yi - tick_half_height, yi + tick_half_height, None])
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=name,
                line=dict(color=color, width=2.2),
                hoverinfo="skip",
                showlegend=True,
                legendgroup=name,
            )
        )
        # Invisible hover targets at the tick centers.
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(color=color, size=10, symbol="line-ns", opacity=0.01),
                customdata=customdata,
                hovertemplate=hovertemplate,
                showlegend=False,
                legendgroup=name,
            )
        )
    else:
        symbol = {
            "x": "x",
            "star": "star",
            "diamond": "diamond",
            "triangle": "triangle-up",
            "triangle-down": "triangle-down",
        }.get(style, "circle")
        size = {
            "x": 11,
            "star": 14,
            "diamond": 11,
            "triangle": 12,
            "triangle-down": 12,
        }.get(style, 10)
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=size,
                    symbol=symbol,
                    line=dict(
                        width=1.2 if style == "x" else 0.6,
                        color=color if style == "x" else "#333",
                    ),
                ),
                customdata=customdata,
                hovertemplate=hovertemplate,
                showlegend=True,
                legendgroup=name,
            )
        )
    return traces


def plot_metrics_board(
    result: MetricsResult,
    *,
    height: int = 720,
) -> go.Figure:
    """Interactive speed + activity-metric board for one worm."""
    m = result.metrics
    fpm = m.frames_per_minute
    frames = _ensure_frame_column(result.frames, frames_per_minute=fpm)
    valid = result.valid.copy()
    if "frame" not in valid.columns:
        valid["frame"] = _frame_values(valid, frames_per_minute=fpm)

    panel2_title = "centroid path-rate (BL/s)"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{m.worm_id}  |  active rev/min={m.active_reversals_per_min}  "
            f"active %fwd={m.active_pct_pos}  %active={m.pct_active}",
            panel2_title,
        ),
    )

    xs, ys = _gapped_xy(
        valid["frame"].to_numpy(dtype=float),
        valid["speed"].to_numpy(dtype=float),
    )
    fig.add_trace(_scatter_line(xs, ys, name="speed", color="#2c6e49", y_label="speed"), row=1, col=1)

    if frames is not None and "active_reversal" in frames.columns:
        revs = frames.loc[frames["active_reversal"] == 1]
        # Scale tick height to speed range so | icons stay visible.
        sp = valid["speed"].to_numpy(dtype=float)
        sp = sp[np.isfinite(sp)]
        tick_h = float(np.nanpercentile(np.abs(sp), 90) * 0.08) if len(sp) else 25.0
        tick_h = max(tick_h, 15.0)
        for tr in _reversal_icon_traces(
            revs["frame"],
            name="active reversals",
            color="#bc4749",
            style="tick",
            tick_half_height=tick_h,
        ):
            fig.add_trace(tr, row=1, col=1)
        if "is_approach_reversal_suppressed" in frames.columns:
            approach = frames.loc[frames["is_approach_reversal_suppressed"] == 1]
            for tr in _reversal_icon_traces(
                approach["frame"],
                name="approach reversal (not counted)",
                color="#1d4ed8",
                style="tick",
                tick_half_height=tick_h,
            ):
                fig.add_trace(tr, row=1, col=1)

    y2_title = "BL/s"
    if frames is not None and "centroid_bl_s" in frames.columns:
        thr = float(getattr(m, "centroid_threshold_bl_s", 0.1))
        xa, ya = _gapped_xy(
            frames["frame"].to_numpy(dtype=float),
            pd.to_numeric(frames["centroid_bl_s"], errors="coerce").to_numpy(dtype=float),
        )
        fig.add_trace(
            _scatter_line(xa, ya, name="centroid BL/s", color="#1d3557", y_label="BL/s"),
            row=2,
            col=1,
        )
        fig.add_hline(y=thr, line_dash="dot", line_color="#e76f51", row=2, col=1)

    annotation = (
        f"reversals {m.reversals} ({m.reversals_per_min}/min) | "
        f"active reversals {m.active_reversals} ({m.active_reversals_per_min}/min) | "
        f"active %fwd/rev {m.active_pct_pos}/{m.active_pct_neg}"
    )
    fig.update_layout(
        height=height,
        hovermode="closest",
        dragmode="zoom",
        template="plotly_white",
        font=dict(family="Arial"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=dict(text=annotation, font=dict(family="Arial", size=11)),
        margin=dict(l=60, r=20, t=80, b=40),
    )
    # Configure axes / rangeslider before shapes — rangeslider after shapes
    # can drop or mis-attach early active-window rects on the speed panel.
    fig.update_xaxes(title_text="frame", row=2, col=1, rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title_text="speed", row=1, col=1)
    fig.update_yaxes(title_text=y2_title, row=2, col=1)
    shapes = []
    for yref in ("y domain", "y2 domain"):
        shapes.extend(_active_shapes(frames, frames_per_minute=fpm, yref=yref))
    fig.update_layout(shapes=shapes)
    return fig





def plot_worm_summary(
    summary: pd.DataFrame,
    *,
    metrics: Sequence[str] = (
        "active_reversals_per_min",
        "active_pct_pos",
        "active_pct_neg",
        "pct_active",
    ),
    height: int = 420,
    figsize: Optional[tuple] = None,
) -> go.Figure:
    """Interactive bar comparison across worms."""
    if figsize is not None:
        # matplotlib-style (w, h) inches → approximate pixel height
        height = int(max(figsize[1], 3) * 90)
    metrics = [m for m in metrics if m in summary.columns]
    worm_ids = (
        summary["worm_id"].astype(str)
        if "worm_id" in summary.columns
        else summary.index.astype(str)
    )
    n = max(len(metrics), 1)
    fig = make_subplots(rows=1, cols=n, subplot_titles=tuple(metrics) if metrics else ("",))
    for i, metric in enumerate(metrics, start=1):
        fig.add_trace(
            go.Bar(
                x=worm_ids,
                y=summary[metric],
                name=metric,
                marker_color="#344e41",
                hovertemplate="worm=%{x}<br>" + metric + "=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Arial"),
        showlegend=False,
        title="Worm summary metrics",
        dragmode="zoom",
        hovermode="closest",
    )
    return fig



def show(fig) -> None:
    """Display a Plotly figure in the notebook (or browser)."""
    fig.show()
