"""Plotting helpers for worm speed traces and summary metrics."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from .metrics import (
    FRAMES_PER_MINUTE,
    MetricsResult,
    REVERSAL_DWELL_FRAMES,
    _gap_threshold,
)

# Default typeface for all matplotlib figures from this package.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

MAJOR_FRAME_TICK = 500
MINOR_FRAME_TICK = 250
MAJOR_SECONDS_TICK = 60
MINOR_SECONDS_TICK = 30
REVERSAL_MARKER_COLOR = "#c45c5c"
REVERSE_BOUT_COLOR = "#c45c5c"
SPEED_LINE_COLOR = "#2c6e49"
REVERSAL_CIRCLE_COLOR = "#ffdd33"
INACTIVE_LINE_COLOR = "#9aa0a6"


def _fps_from_frames_per_minute(frames_per_minute: int = FRAMES_PER_MINUTE) -> float:
    return float(frames_per_minute) / 60.0 if frames_per_minute else 6.0


def _resolve_plot_fps(
    *tables: Optional[pd.DataFrame],
    frames_per_minute: int = FRAMES_PER_MINUTE,
) -> float:
    """Prefer ``fps_true`` on any supplied table; else frames_per_minute / 60."""
    for tbl in tables:
        if tbl is None or tbl.empty or "fps_true" not in tbl.columns:
            continue
        try:
            v = float(tbl["fps_true"].iloc[0])
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            return v
    return _fps_from_frames_per_minute(frames_per_minute)


def _normalize_x_unit(x_unit: str) -> str:
    u = str(x_unit or "frame").strip().lower()
    if u in ("s", "sec", "secs", "second", "seconds", "time"):
        return "seconds"
    return "frame"


def _ensure_time_column(
    df: Optional[pd.DataFrame],
    *,
    fps: float,
) -> Optional[pd.DataFrame]:
    """Ensure a wall-clock ``time`` column (seconds) exists when possible."""
    if df is None or df.empty:
        return df
    if "time" in df.columns and pd.to_numeric(df["time"], errors="coerce").notna().any():
        return df
    if "frame" not in df.columns or not (np.isfinite(fps) and fps > 0):
        return df
    out = df.copy()
    out["time"] = pd.to_numeric(out["frame"], errors="coerce").to_numpy(dtype=float) / float(
        fps
    )
    return out


def _x_values(
    df: pd.DataFrame,
    *,
    x_unit: str = "frame",
    fps: float = 6.0,
) -> np.ndarray:
    """Return plot x coordinates in frames or seconds for rows of ``df``."""
    unit = _normalize_x_unit(x_unit)
    if unit == "seconds":
        if "time" in df.columns:
            t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(t).any():
                return t
        if "frame" in df.columns and np.isfinite(fps) and fps > 0:
            return (
                pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
                / float(fps)
            )
        raise ValueError("Need 'time' or 'frame' to plot x in seconds")
    if "frame" in df.columns:
        return pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    if "time" in df.columns and np.isfinite(fps) and fps > 0:
        return pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float) * float(
            fps
        )
    raise ValueError("Need 'frame' or 'time' to plot x in frames")


def _frames_to_x(
    frames_arr: np.ndarray,
    *,
    x_unit: str = "frame",
    fps: float = 6.0,
) -> np.ndarray:
    arr = np.asarray(frames_arr, dtype=float)
    if _normalize_x_unit(x_unit) == "seconds":
        if not (np.isfinite(fps) and fps > 0):
            raise ValueError("fps required to convert frames to seconds")
        return arr / float(fps)
    return arr


def _times_to_frames(
    times: np.ndarray,
    *,
    fps: Optional[float] = None,
    frames_per_minute: int = FRAMES_PER_MINUTE,
) -> np.ndarray:
    """
    Convert a timeline to frame index.

    If ``times`` already looks like frame indices (step ≈ 1), returns
    ``round(times)``. Otherwise treats ``times`` as true seconds and uses
    ``fps`` (or frames_per_minute/60).
    """
    from .video_fps import _values_look_like_frames

    arr = np.asarray(times, dtype=float)
    if _values_look_like_frames(arr):
        return np.round(arr)
    if fps is None:
        fps = _fps_from_frames_per_minute(frames_per_minute)
    return np.round(arr * float(fps))


def _frame_values(
    df: pd.DataFrame,
    *,
    fps: Optional[float] = None,
    frames_per_minute: int = FRAMES_PER_MINUTE,
) -> np.ndarray:
    """Prefer an existing ``frame`` column; otherwise derive from ``time``."""
    from .video_fps import ASSUMED_FPS, recover_frame_index

    if "frame" in df.columns and "time" in df.columns:
        return recover_frame_index(
            df["time"].to_numpy(dtype=float),
            assumed_fps=ASSUMED_FPS,
            existing_frame=pd.to_numeric(df["frame"], errors="coerce").to_numpy(
                dtype=float
            ),
        )
    if "frame" in df.columns:
        return pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    if "time" not in df.columns:
        raise ValueError("DataFrame needs a 'frame' or 'time' column for plotting")
    # Prefer fps_true on the frame table when present.
    use_fps = fps
    if use_fps is None and "fps_true" in df.columns and len(df):
        try:
            use_fps = float(df["fps_true"].iloc[0])
        except Exception:
            use_fps = None
    return recover_frame_index(
        df["time"].to_numpy(dtype=float),
        assumed_fps=ASSUMED_FPS if use_fps is None else float(use_fps),
        existing_frame=None,
    )


def _apply_tick_marks(ax: plt.Axes) -> None:
    """Visible major/minor tick marks on both axes."""
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=0.8,
        bottom=True,
        left=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3,
        width=0.6,
        bottom=True,
        left=True,
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def _apply_frame_xticks(ax: plt.Axes) -> None:
    """Label x as frames with major=500 and minor=250 tick spacing."""
    ax.set_xlabel("frame")
    ax.xaxis.set_major_locator(MultipleLocator(MAJOR_FRAME_TICK))
    ax.xaxis.set_minor_locator(MultipleLocator(MINOR_FRAME_TICK))
    _apply_tick_marks(ax)
    ax.grid(True, axis="x", which="major", color="#ddd", lw=0.6)
    ax.grid(True, axis="x", which="minor", color="#eee", lw=0.4)


def _apply_seconds_xticks(ax: plt.Axes) -> None:
    """Label x as seconds with major=60 and minor=30 tick spacing."""
    ax.set_xlabel("seconds")
    ax.xaxis.set_major_locator(MultipleLocator(MAJOR_SECONDS_TICK))
    ax.xaxis.set_minor_locator(MultipleLocator(MINOR_SECONDS_TICK))
    _apply_tick_marks(ax)
    ax.grid(True, axis="x", which="major", color="#ddd", lw=0.6)
    ax.grid(True, axis="x", which="minor", color="#eee", lw=0.4)


def _apply_x_xticks(ax: plt.Axes, x_unit: str = "frame") -> None:
    if _normalize_x_unit(x_unit) == "seconds":
        _apply_seconds_xticks(ax)
    else:
        _apply_frame_xticks(ax)


def _blend_on_white(color, alpha: float) -> tuple[float, float, float]:
    """Opaque RGB equivalent of ``color`` at ``alpha`` composited over white."""
    r, g, b = to_rgb(color)
    a = float(alpha)
    a = min(max(a, 0.0), 1.0)
    return (a * r + (1.0 - a), a * g + (1.0 - a), a * b + (1.0 - a))


def _zero_crossing(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    """Interpolate the point where the segment (x0,y0)→(x1,y1) crosses y=0."""
    if y0 == 0:
        return float(x0), 0.0
    if y1 == 0:
        return float(x1), 0.0
    if y1 == y0:
        return float(0.5 * (x0 + x1)), 0.0
    t = (0.0 - y0) / (y1 - y0)
    t = min(max(float(t), 0.0), 1.0)
    return float(x0 + t * (x1 - x0)), 0.0


def _plot_gapped_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    lw: float = 1.0,
    gap: Optional[float] = None,
) -> None:
    order = np.argsort(x)
    x, y = x[order], y[order]
    if gap is None:
        gap = _gap_threshold(x)
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or (x[i] - x[i - 1]) > gap:
            ax.plot(x[start:i], y[start:i], color=color, lw=lw)
            start = i


def _plot_gapped_colored_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, str],
    *,
    zero_cross_ids: Optional[set[int]] = None,
    lw: float = 1.0,
    gap: Optional[float] = None,
) -> None:
    """
    Plot a gapped speed line whose color follows integer ``labels``.

    Transitions between labels in ``zero_cross_ids`` (default 0 and 1,
    forward/reverse) split at the interpolated y=0 crossing. Other color
    changes (e.g. active ↔ inactive) split at the segment midpoint.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(x)
    x, y, labels = x[order], y[order], labels[order]
    if gap is None:
        gap = _gap_threshold(x)
    zero_ids = {0, 1} if zero_cross_ids is None else set(zero_cross_ids)

    pts: dict[int, tuple[list[float], list[float]]] = {
        int(k): ([], []) for k in colors
    }

    def _push(which: int, px: float, py: float, *, break_seg: bool = False) -> None:
        xs, ys = pts.setdefault(int(which), ([], []))
        if break_seg and xs and np.isfinite(xs[-1]):
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(float(px))
        ys.append(float(py))

    n = len(x)
    prev: Optional[int] = None
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            prev = None
            continue
        which = int(labels[i])
        if prev is None or (x[i] - x[prev]) > gap:
            _push(which, x[i], y[i], break_seg=True)
            prev = i
            continue
        prev_which = int(labels[prev])
        if prev_which != which:
            if prev_which in zero_ids and which in zero_ids:
                xc, yc = _zero_crossing(x[prev], y[prev], x[i], y[i])
            else:
                xc = 0.5 * (float(x[prev]) + float(x[i]))
                yc = 0.5 * (float(y[prev]) + float(y[i]))
            _push(prev_which, xc, yc)
            _push(which, xc, yc, break_seg=True)
        _push(which, x[i], y[i])
        prev = i

    for which, color in colors.items():
        xs, ys = pts.get(int(which), ([], []))
        if xs:
            ax.plot(xs, ys, color=color, lw=lw)


def _plot_gapped_line_by_mask(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    color_true: str,
    color_false: str,
    inactive_mask: Optional[np.ndarray] = None,
    inactive_color: Optional[str] = None,
    lw: float = 1.0,
    gap: Optional[float] = None,
) -> None:
    """
    Plot a gapped speed line whose color follows ``mask``.

    When consecutive samples differ in ``mask``, the segment is split at the
    interpolated y=0 crossing so the color flip sits on the zero line.
    Inactive samples (``inactive_mask``) use ``inactive_color`` and split at
    the segment midpoint instead of y=0.
    """
    labels = np.asarray(mask, dtype=bool).astype(int)
    colors = {0: color_false, 1: color_true}
    if inactive_mask is not None and inactive_color is not None:
        inactive_arr = np.asarray(inactive_mask, dtype=bool)
        if len(inactive_arr) == len(labels):
            labels = labels.copy()
            labels[inactive_arr] = 2
            colors[2] = inactive_color
    _plot_gapped_colored_line(
        ax,
        x,
        y,
        labels,
        colors,
        zero_cross_ids={0, 1},
        lw=lw,
        gap=gap,
    )


def _inactive_mask(
    frames: Optional[pd.DataFrame],
    n: int,
    plot_df: Optional[pd.DataFrame] = None,
) -> Optional[np.ndarray]:
    """True where the setup-cell activity filter marked the frame inactive."""
    if frames is None or frames.empty or "active" not in frames.columns:
        return None
    active = pd.to_numeric(frames["active"], errors="coerce").to_numpy(dtype=float)
    if len(active) == n:
        return np.asarray(active != 1, dtype=bool)
    if (
        plot_df is not None
        and "frame" in frames.columns
        and "frame" in plot_df.columns
    ):
        src = pd.DataFrame(
            {
                "frame": pd.to_numeric(frames["frame"], errors="coerce"),
                "active": active,
            }
        ).dropna(subset=["frame"])
        src = src.drop_duplicates(subset=["frame"], keep="last")
        mapped = pd.Series(src["active"].to_numpy(), index=src["frame"].to_numpy())
        keys = pd.to_numeric(plot_df["frame"], errors="coerce").to_numpy(dtype=float)
        aligned = mapped.reindex(keys).to_numpy(dtype=float)
        out = np.ones(n, dtype=bool)
        finite = np.isfinite(aligned)
        out[finite] = aligned[finite] != 1
        return out
    return None


def _dwell_reverse_mask(
    speed: np.ndarray,
    x: Optional[np.ndarray] = None,
    *,
    dwell: int = REVERSAL_DWELL_FRAMES,
) -> np.ndarray:
    """
    Color mask from the sign-dwell direction state (not counted reverse bouts).

    Counted bouts omit reverse runs that fail the activity filter, so a long
    negative stretch can sit inside a "forward" bout. This mask follows the
    same dwell machine as reversal detection:

      - red starts at the first negative frame of the confirming dwell
      - red runs through reverse state (including brief positive blips)
      - green starts at the first positive frame of the exit dwell
    """
    speed = np.asarray(speed, dtype=float)
    n = len(speed)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    need = max(1, int(dwell))
    x_arr = None if x is None else np.asarray(x, dtype=float)
    gap = (
        _gap_threshold(x_arr)
        if x_arr is not None and len(x_arr) == n
        else np.inf
    )

    state = 0  # +1 forward, -1 reverse, 0 unknown
    pending_sign = 0
    pending_count = 0
    pending_start = 0

    for i in range(n):
        if x_arr is not None and i > 0:
            dt = float(x_arr[i] - x_arr[i - 1])
            if not (dt > 0 and dt <= gap):
                state = 0
                pending_sign = 0
                pending_count = 0

        s = speed[i]
        if not np.isfinite(s) or s == 0:
            if state < 0:
                mask[i] = True
            continue

        sg = 1 if s > 0 else -1

        if state == 0:
            if pending_sign == sg:
                pending_count += 1
            else:
                pending_sign = sg
                pending_count = 1
                pending_start = i
            if pending_count >= need:
                state = int(sg)
                pending_sign = 0
                pending_count = 0
                if state < 0:
                    mask[pending_start : i + 1] = True
            continue

        if sg == state:
            pending_sign = 0
            pending_count = 0
            if state < 0:
                mask[i] = True
            continue

        if pending_sign != sg:
            pending_sign = sg
            pending_count = 1
            pending_start = i
        else:
            pending_count += 1

        if pending_count < need:
            if state < 0:
                mask[i] = True
            continue

        prev = state
        state = int(sg)
        if prev > 0 and state < 0:
            mask[pending_start : i + 1] = True
        elif prev < 0 and state > 0:
            mask[pending_start : i + 1] = False
        pending_sign = 0
        pending_count = 0

    return mask


def _contiguous_dt(x_arr: Optional[np.ndarray], a: int, b: int, gap: float) -> bool:
    if x_arr is None or a < 0 or b >= len(x_arr):
        return True
    dt = float(x_arr[b] - x_arr[a])
    return dt > 0 and dt <= gap


def _true_islands(
    mask: np.ndarray,
    x_arr: Optional[np.ndarray],
    gap: float,
) -> list[tuple[int, int]]:
    """Contiguous True runs; a tracking gap starts a new island."""
    islands: list[tuple[int, int]] = []
    start: Optional[int] = None
    prev: Optional[int] = None
    for i, v in enumerate(mask):
        if v:
            if start is None:
                start = i
            elif (
                prev is not None
                and x_arr is not None
                and not _contiguous_dt(x_arr, prev, i, gap)
            ):
                islands.append((start, prev))
                start = i
            prev = i
            continue
        if start is not None and prev is not None:
            islands.append((start, prev))
        start = None
        prev = None
    if start is not None and prev is not None:
        islands.append((start, prev))
    return islands


def _sign_gate_reverse_mask(mask: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """During reverse, color by speed sign: negative reverse, positive forward."""
    mask = np.asarray(mask, dtype=bool).copy()
    speed = np.asarray(speed, dtype=float)
    if len(speed) != len(mask):
        return mask
    return mask & np.isfinite(speed) & (speed < 0)


def _pipeline_reverse_mask(
    n: int,
    speed: np.ndarray,
    x: Optional[np.ndarray] = None,
    *,
    bouts: Optional[pd.DataFrame] = None,
    reversal_flags: Optional[np.ndarray] = None,
    dwell: int = REVERSAL_DWELL_FRAMES,
) -> np.ndarray:
    """
    Reverse coloring from ``compute_metrics`` (setup-cell dwell and activity).

    Sign-dwell sets which reverse runs are eligible. Within those runs, color
    follows speed sign: negative is reverse color, positive is forward color.
    """
    mask = np.zeros(n, dtype=bool)
    if n <= 0:
        return mask
    speed = np.asarray(speed, dtype=float)
    if len(speed) != n:
        return mask
    x_arr = None if x is None else np.asarray(x, dtype=float)
    gap = (
        _gap_threshold(x_arr)
        if x_arr is not None and len(x_arr) == n
        else np.inf
    )

    dwell_mask = _dwell_reverse_mask(speed, x_arr, dwell=int(dwell))
    justify = np.zeros(n, dtype=bool)

    if reversal_flags is not None:
        flags = np.asarray(reversal_flags).astype(bool)
        if len(flags) == n:
            justify |= flags

    if bouts is not None and len(bouts) and "direction" in bouts.columns:
        rev = bouts.loc[bouts["direction"] == "reverse"]
        if len(rev) and "frame_start" in rev.columns and "frame_end" in rev.columns:
            starts = pd.to_numeric(rev["frame_start"], errors="coerce").to_numpy()
            ends = pd.to_numeric(rev["frame_end"], errors="coerce").to_numpy()
            for i0, i1 in zip(starts, ends):
                if not (np.isfinite(i0) and np.isfinite(i1)):
                    continue
                a = int(max(0, min(n - 1, i0)))
                b = int(max(0, min(n - 1, i1)))
                if b < a:
                    a, b = b, a
                justify[a : b + 1] = True

    for a, b in _true_islands(dwell_mask, x_arr, gap):
        lo = max(0, a - 1)
        if justify[lo : b + 1].any():
            mask[a : b + 1] = True
    return _sign_gate_reverse_mask(mask, speed)


def _active_windows(
    frames: pd.DataFrame,
    *,
    gap: Optional[float] = None,
    x_col: str = "frame",
    value: int = 1,
) -> list[tuple[float, float]]:
    """
    Return (x_start, x_end) spans for contiguous ``active == value`` runs.

    A large gap between successive matching samples ends the window so
    missing-data stretches are not included in the span.
    """
    if frames is None or "active" not in frames.columns or frames.empty:
        return []
    work = frames.copy()
    if x_col not in work.columns:
        if x_col == "frame" and "time" in work.columns:
            work = work.copy()
            work["frame"] = _times_to_frames(work["time"].to_numpy(dtype=float))
        else:
            return []
    ordered = work.sort_values(x_col)
    xs = ordered[x_col].to_numpy(dtype=float)
    active = ordered["active"].to_numpy(dtype=int)
    if gap is None:
        gap = _gap_threshold(xs)

    windows: list[tuple[float, float]] = []
    start: Optional[float] = None
    prev_x: Optional[float] = None
    for i in range(len(active)):
        x = xs[i]
        if active[i] == value:
            if start is None:
                start = x
            elif prev_x is not None and (x - prev_x) > gap:
                windows.append((start, prev_x))
                start = x
            prev_x = x
        else:
            if start is not None and prev_x is not None:
                windows.append((start, prev_x))
            start = None
            prev_x = None
    if start is not None and prev_x is not None:
        windows.append((start, prev_x))
    return windows


def _ensure_frame_column(
    frames: Optional[pd.DataFrame],
    *,
    frames_per_minute: int = FRAMES_PER_MINUTE,
) -> Optional[pd.DataFrame]:
    if frames is None or frames.empty:
        return frames
    out = frames
    if "frame" not in out.columns:
        out = out.copy()
        out["frame"] = _times_to_frames(
            out["time"].to_numpy(dtype=float),
            frames_per_minute=frames_per_minute,
        )
    return out


def _shade_active_windows(
    ax: plt.Axes,
    frames: Optional[pd.DataFrame],
    *,
    frames_per_minute: int = FRAMES_PER_MINUTE,
    x_unit: str = "frame",
    fps: Optional[float] = None,
    color: str = "#a3b18a",
    alpha: float = 0.25,
    value: int = 1,
) -> None:
    frames = _ensure_frame_column(frames, frames_per_minute=frames_per_minute)
    unit = _normalize_x_unit(x_unit)
    x_col = "frame"
    if unit == "seconds":
        if fps is None:
            fps = _resolve_plot_fps(frames, frames_per_minute=frames_per_minute)
        frames = _ensure_time_column(frames, fps=float(fps))
        x_col = "time" if frames is not None and "time" in frames.columns else "frame"
    gap = None
    if frames is not None and x_col in frames.columns and len(frames):
        gap = _gap_threshold(frames[x_col].to_numpy(dtype=float))
    pad = 0.5 if x_col == "frame" else (0.5 / float(fps or 6.0))
    for x0, x1 in _active_windows(frames, gap=gap, x_col=x_col, value=value):
        if x1 <= x0:
            ax.axvspan(x0 - pad, x0 + pad, color=color, alpha=alpha, lw=0, zorder=0)
        else:
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _apply_activity_background(
    ax: plt.Axes,
    frames: Optional[pd.DataFrame],
    *,
    frames_per_minute: int = FRAMES_PER_MINUTE,
    x_unit: str = "frame",
    fps: Optional[float] = None,
    color: str = "#a3b18a",
    alpha: float = 0.25,
    shade_gaps: bool = False,
) -> None:
    """
    Shade activity on the speed axis.

    When ``shade_gaps`` is True, the axes background matches the active-window
    grey so data gaps stay grey; only inactive (active==0) stretches are white.
    """
    if shade_gaps and frames is not None and not frames.empty and "active" in frames.columns:
        ax.set_facecolor(_blend_on_white(color, alpha))
        _shade_active_windows(
            ax,
            frames,
            frames_per_minute=frames_per_minute,
            x_unit=x_unit,
            fps=fps,
            color="white",
            alpha=1.0,
            value=0,
        )
        return
    _shade_active_windows(
        ax,
        frames,
        frames_per_minute=frames_per_minute,
        x_unit=x_unit,
        fps=fps,
        color=color,
        alpha=alpha,
        value=1,
    )





def _mark_reversal_ticks(
    ax: plt.Axes,
    x: np.ndarray,
    *,
    color: str = REVERSAL_MARKER_COLOR,
    vlines: bool = False,
    vline_alpha: float = 0.5,
    label: Optional[str] = None,
    marker: str = "|",
    s: float = 28,
    zorder: int = 5,
    linewidths: float = 1.5,
    edgecolors: Optional[str] = None,
) -> None:
    """Scatter reversal markers; optionally a dashed full-height line at each."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    if vlines:
        for xi in x:
            ax.axvline(
                float(xi),
                color=color,
                ls="--",
                alpha=vline_alpha,
                lw=1.0,
                zorder=4,
            )
    style = str(marker or "|").strip().lower()
    if style in ("o", "circle", "dot"):
        ax.scatter(
            x,
            np.zeros(len(x)),
            c=color,
            s=s,
            zorder=max(zorder, 6),
            marker="o",
            linewidths=0.6 if edgecolors is not None else 0.7,
            edgecolors=edgecolors if edgecolors is not None else "#333333",
            label=label,
        )
        return
    scatter_kw = dict(
        c=color,
        s=s,
        zorder=zorder,
        marker=marker,
        linewidths=linewidths,
        label=label,
    )
    if edgecolors is not None:
        scatter_kw["edgecolors"] = edgecolors
    ax.scatter(x, np.zeros(len(x)), **scatter_kw)


def plot_speed_trace(
    df: pd.DataFrame,
    *,
    frames: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    show_activity: bool = True,
    show_legend: bool = True,
    ax: Optional[plt.Axes] = None,
    frames_per_minute: int = FRAMES_PER_MINUTE,
        x_unit: str = "frame",
    line_color: str = "#2c6e49",
    active_color: str = "#a3b18a",
    active_alpha: float = 0.25,
    shade_gaps: bool = False,
    reversal_color: str = "#bc4749",
    reversal_vlines: bool = False,
    bouts: Optional[pd.DataFrame] = None,
    color_reverse_bouts: bool = False,
    reverse_color: str = REVERSE_BOUT_COLOR,
    reversal_marker: str = "|",
    reversal_marker_size: float = 28,
    reversal_dwell_frames: Optional[int] = None,
    shade_activity: Optional[bool] = None,
    inactive_color: Optional[str] = None,
) -> plt.Axes:
    """Plot speed vs frame or seconds; mark reversals.

    When ``shade_activity`` is True (default when ``show_activity`` is True and
    ``inactive_color`` is None), active/inactive windows are shaded. When
    ``inactive_color`` is set, inactive speed is drawn in that color instead
    of shading; activity comes from ``frames['active']`` (setup-cell filter).

    When ``color_reverse_bouts`` is True, reverse vs forward color follows
    ``compute_metrics`` (setup-cell dwell and centroid activity).
    Red is the dwell reverse state after a kept reversal (or an overlapping
    reverse bout). Positive speeds at the tail of a reverse run that ends
    on a data gap (not a 3-frame positive dwell) are colored as forward.
    Color flips are interpolated at y=0. Inactive coloring overrides
    reverse/forward.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))

    unit = _normalize_x_unit(x_unit)
    frames = _ensure_frame_column(frames, frames_per_minute=frames_per_minute)
    plot_df = df.copy()
    if "frame" not in plot_df.columns:
        plot_df["frame"] = _frame_values(plot_df, frames_per_minute=frames_per_minute)
    fps = _resolve_plot_fps(plot_df, frames, frames_per_minute=frames_per_minute)
    if unit == "seconds":
        plot_df = _ensure_time_column(plot_df, fps=fps)
        frames = _ensure_time_column(frames, fps=fps)

    do_shade = show_activity if shade_activity is None else bool(shade_activity)
    if inactive_color is not None and shade_activity is None:
        do_shade = False
    if do_shade:
        _apply_activity_background(
            ax,
            frames,
            frames_per_minute=frames_per_minute,
            x_unit=unit,
            fps=fps,
            color=active_color,
            alpha=active_alpha,
            shade_gaps=shade_gaps,
        )

    x_speed = _x_values(plot_df, x_unit=unit, fps=fps)
    y_speed = plot_df["speed"].to_numpy(dtype=float)
    inactive = (
        _inactive_mask(frames, len(plot_df), plot_df)
        if inactive_color is not None
        else None
    )
    if color_reverse_bouts:
        rev_flags = None
        if frames is not None and len(frames) == len(plot_df):
            if "is_reversal" in frames.columns:
                rev_flags = frames["is_reversal"].to_numpy()
            elif "active_reversal" in frames.columns:
                rev_flags = frames["active_reversal"].to_numpy()
        have_pipeline = bouts is not None or rev_flags is not None
        dwell = (
            int(reversal_dwell_frames)
            if reversal_dwell_frames is not None
            else int(REVERSAL_DWELL_FRAMES)
        )
        if have_pipeline:
            rev_mask = _pipeline_reverse_mask(
                len(plot_df),
                y_speed,
                x_speed,
                bouts=bouts,
                reversal_flags=rev_flags,
                dwell=dwell,
            )
        else:
            rev_mask = _dwell_reverse_mask(y_speed, x_speed, dwell=dwell)
        rev_mask = _sign_gate_reverse_mask(rev_mask, y_speed)
        _plot_gapped_line_by_mask(
            ax,
            x_speed,
            y_speed,
            rev_mask,
            color_true=reverse_color,
            color_false=line_color,
            inactive_mask=inactive,
            inactive_color=inactive_color,
        )
    elif inactive is not None and inactive_color is not None:
        labels = np.zeros(len(plot_df), dtype=int)
        labels[inactive] = 2
        _plot_gapped_colored_line(
            ax,
            x_speed,
            y_speed,
            labels,
            {0: line_color, 2: inactive_color},
            zero_cross_ids=set(),
        )
    else:
        _plot_gapped_line(
            ax,
            x_speed,
            y_speed,
            color=line_color,
        )
    ax.axhline(0, color="#888", lw=0.8, ls="--")


    if show_activity and frames is not None and "active_reversal" in frames.columns:
        revs = frames.loc[frames["active_reversal"] == 1]
        if len(revs):
            _mark_reversal_ticks(
                ax,
                _x_values(revs, x_unit=unit, fps=fps),
                color=reversal_color,
                vlines=reversal_vlines,
                marker=reversal_marker,
                s=reversal_marker_size,
                label="active reversals" if show_legend else None,
            )

    ax.set_ylabel("speed")
    _apply_x_xticks(ax, unit)
    ax.set_title(title or "Speed trace")
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return ax



def plot_worm_summary(
    summary: pd.DataFrame,
    *,
    metrics: Sequence[str] = (
        "active_reversals_per_min",
        "active_pct_pos",
        "active_pct_neg",
        "pct_active",
    ),
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Bar comparison of selected metrics across worms."""
    metrics = [m for m in metrics if m in summary.columns]
    n = max(len(metrics), 1)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    worm_ids = summary["worm_id"] if "worm_id" in summary.columns else summary.index.astype(str)

    for ax, metric in zip(axes[0], metrics):
        ax.bar(worm_ids, summary[metric], color="#344e41")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_metrics_board(
    result: MetricsResult,
    *,
    figsize: tuple = (10, 7.5),
) -> plt.Figure:
    """Speed + centroid activity traces + key numbers for one worm."""
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 2.2, 1.0]}, sharex=True
    )
    m = result.metrics
    fpm = m.frames_per_minute
    plot_speed_trace(
        result.valid,
        frames=result.frames,
        bouts=result.bouts,
        color_reverse_bouts=True,
        title=(
            f"{m.worm_id}  |  active rev/min={m.active_reversals_per_min}  "
            f"active %fwd={m.active_pct_pos}  %active={m.pct_active}"
        ),
        ax=axes[0],
        frames_per_minute=fpm,
        reversal_dwell_frames=(
            1
            if not getattr(m, "use_reversal_sustain_filter", True)
            else m.reversal_dwell_frames
        ),
    )
    axes[0].set_xlabel("")

    if result.frames is not None and "centroid_bl_s" in result.frames.columns:
        thr = float(getattr(m, "centroid_threshold_bl_s", 0.1))
        win = float(getattr(m, "centroid_window_s", 15.0))
        frames = _ensure_frame_column(result.frames, frames_per_minute=fpm)
        _shade_active_windows(axes[1], frames, frames_per_minute=fpm)
        x = frames["frame"].to_numpy(dtype=float)
        y = pd.to_numeric(frames["centroid_bl_s"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y)
        _plot_gapped_line(axes[1], x[finite], y[finite], color="#1d3557")
        axes[1].axhline(0, color="#888", lw=0.8, ls="--")
        axes[1].axhline(
            thr, color="#e76f51", lw=0.9, ls=":", label=f">{thr:g} BL/s ({win:g}s)"
        )
        _apply_frame_xticks(axes[1])
        axes[1].set_ylabel("centroid BL/s")
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(loc="upper right", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "no centroid BL/s", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_ylabel("centroid BL/s")


    axes[2].axis("off")
    if m.use_reversal_sustain_filter:
        rev_rule = f"sign dwell ≥{m.reversal_dwell_frames} frames"
    else:
        rev_rule = "single +→−"
    thr = getattr(m, "centroid_threshold_bl_s", 0.1)
    win = getattr(m, "centroid_window_s", 15.0)
    activity_rule = (
        f"activity: centroid path-rate > {thr:g} BL/s "
        f"over {win:g}s window"
    )
    text = (
        f"duration {m.duration_min} min   "
        f"reversals {m.reversals} ({m.reversals_per_min}/min)   "
        f"active reversals {m.active_reversals} ({m.active_reversals_per_min}/min)   "
        f"[{rev_rule}]\n"
        f"active %fwd/rev {m.active_pct_pos}/{m.active_pct_neg}   "
        f"active time fwd/rev {m.active_time_pos}s/{m.active_time_neg}s   "
        f"({m.active_time_pos_per_min}/{m.active_time_neg_per_min} s per min)\n"
        f"avg active bout fwd {m.avg_active_fwd_bout_s}s "
        f"({m.avg_active_fwd_bout_frames} frames, n={m.active_fwd_bouts})   "
        f"rev {m.avg_active_rev_bout_s}s "
        f"({m.avg_active_rev_bout_frames} frames, n={m.active_rev_bouts})\n"
        f"{activity_rule}   "
        f"active frames {m.active_frames} ({m.pct_active}%)   "
        f"outliers removed={m.outliers_removed}"
    )
    axes[2].text(0.01, 0.5, text, va="center", fontsize=9, family="monospace")
    fig.tight_layout()
    return fig






