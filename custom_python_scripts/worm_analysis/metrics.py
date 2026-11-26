"""
Locomotion metrics with centroid path-rate activity filtering.

  - Outliers: speed > 5x mean of same-sign speeds (single pass) dropped
  - Reversals: sign-dwell state machine (when USE_REVERSAL_SUSTAIN_FILTER is
    True). Direction state is FORWARD or REVERSE and flips only after
    REVERSAL_DWELL_FRAMES consecutive opposite-sign speed samples (zeros /
    NaNs do not advance the counter). Tracking gaps reset state. With the
    filter off, dwell=1 (any sustained single-frame sign change).
  - Activity: centroid path-rate. Scanning window of
    CENTROID_ACTIVITY_WINDOW_S seconds on the full Tierpsy blob track; path
    length ÷ median body length ÷ duration; active when rate >
    CENTROID_ACTIVITY_THRESHOLD_BL_S; mapped onto the speed CSV timeline by
    frame (missing centroid → inactive). Missing speed values stay NaN
    (never filled with 0).
  - Active forward/reverse bouts: active same-sign runs. Forward bouts end
    only at kept reversals (after the dwell filter), so rejected sign-flip
    noise does not truncate bout lengths. Reverse→forward always ends a
    reverse bout.
  - Active-frame metrics exclude inactive frames from analysis
  - Active-reversal reorientation: directed head→tail body-axis turn
    (Tierpsy skeleton tip axes; see tierpsy_data.ht_axis). Pre vector is the
    most-extended skeleton before the reversal (REORIENTATION_WINDOW_S).
    Post vector is the most-extended skeleton within
    REORIENTATION_FWD_BEST_S of the next forward-bout start that begins
    within REORIENTATION_FWD_SEARCH_S of reverse-bout end. If no such
    bout (or no usable HT axis), the event is omitted from the
    reorientation count and from the percentage denominator. Count when
    angle > REORIENTATION_ANGLE_DEG (default 90°). If the worm or minute
    never reverses (``active_reversals == 0``), every reorientation
    metric is NaN and is omitted from across-strain means/plots.
  - Per-minute extremes: min/max active-reversal counts across full
    1-minute bins of the unit (worm recording or continuous-minute slice)
  - Per-minute rates and all seconds outputs use video ``frame`` index with
    ``fps_true`` / ``frames_per_minute`` (export timestamps are only used to
    recover frames). Fallback constant FRAMES_PER_MINUTE = 360 when unknown.
  - Continuous-minute mode (see compute_continuous_minute_metrics): extract
    non-overlapping FRAMES_PER_MINUTE speed windows from near-continuous
    stretches (gaps ≤ CONTINUOUS_SPEED_GAP_FRAMES), worm-agnostic across a
    dataset; requires a *_blob_perimeter.csv beside the speed file. Each minute carries
    recording_date / video_number from the source CSV name. When
    USE_CONTINUOUS_MINUTE_BACKFILL is on, a leftover of ≥ half a window at
    the end of a stretch (≥1 full window) becomes an overlapping end window
    backfilled into the prior minute instead of being discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

OUTLIER_FACTOR = 5.0
# Centroid path-rate activity
CENTROID_ACTIVITY_THRESHOLD_BL_S = 0.1  # active if window path-rate > this (BL/s)
CENTROID_ACTIVITY_WINDOW_S = 15.0  # centered scanning window (seconds)
CENTROID_PATH_MAX_GAP_FRAMES = 2  # ignore centroid steps with larger frame gaps
# Per-source caches so continuous-minute slices reuse featuresN / tip tracks /
# HT-invert sets instead of reloading once per minute window.
_CENTROID_TRACK_CACHE: Dict[tuple, pd.DataFrame] = {}
_CENTROID_ACTIVITY_CACHE: Dict[tuple, pd.DataFrame] = {}
_HT_INVERT_SET_CACHE: Dict[str, Optional[set]] = {}
FRAMES_PER_MINUTE = 360
# Worm inclusion: require at least this fraction of that video's max frame
# count (from video_fps.csv ``n_frames``, else max rows among worms in the
# same video). Set ``MIN_FRAMES`` to an int to use a fixed absolute floor
# instead; ``0`` keeps every file.
MIN_FRAMES_FRAC = 0.5
MIN_FRAMES = None
# Max spacing (in frames) between consecutive finite speed samples still
# treated as one continuous stretch for worm-agnostic minute extraction.
CONTINUOUS_SPEED_GAP_FRAMES = 36
# When a continuous stretch leftover is ≥ half a minute window, emit an
# overlapping end window that backfills into the previous minute instead of
# discarding the leftover samples.
USE_CONTINUOUS_MINUTE_BACKFILL = True
USE_REVERSAL_SUSTAIN_FILTER = True  # toggle sign-dwell reversal rule
REVERSAL_DWELL_FRAMES = 3  # consecutive opposite-sign frames to flip direction
# Back-compat aliases (both track the dwell length used in summary tables)
REVERSAL_MIN_NEG_FRAMES = REVERSAL_DWELL_FRAMES
REVERSAL_MIN_POS_FRAMES = REVERSAL_DWELL_FRAMES

# Active-reversal reorientation (head→tail axis; matches omega model)
REORIENTATION_ANGLE_DEG = 90.0  # count when pre→post HT angle exceeds this
REORIENTATION_WINDOW_S = 3.0  # pre-reversal most-extended skeleton search (s)
REORIENTATION_FWD_SEARCH_S = 10.0  # next forward bout must start within this of reverse-bout end
REORIENTATION_FWD_BEST_S = 1.0  # pick most-extended HT within this of that forward start


@dataclass
class WormMetrics:
    worm_id: str = ""
    recording_date: str = ""
    video_number: str = ""
    source_worm_id: str = ""
    source_file: str = ""
    minute_index: int = 0
    avg_speed: float = 0.0
    avg_pos_speed: float = 0.0
    avg_neg_speed: float = 0.0
    outliers_removed: int = 0
    # unfiltered (all valid speed frames)
    time_pos: float = 0.0
    time_neg: float = 0.0
    reversals: int = 0
    fwd_frames: int = 0
    rev_frames: int = 0
    total_frames: int = 0
    pct_pos: float = 0.0
    pct_neg: float = 0.0
    pct_captured: float = 0.0
    # activity-filtered
    active_frames: int = 0
    pct_active: float = 0.0
    active_reversals: int = 0
    # Extremes of active-reversal counts across 1-minute bins of this unit
    min_active_reversals_per_minute: int = 0
    max_active_reversals_per_minute: int = 0
    n_minute_bins: int = 0
    # Active reversals whose HT (or path) angle > REORIENTATION_ANGLE_DEG.
    # NaN when the unit never reverses (excluded from reorient analyses).
    active_reversals_reorient: float = float("nan")
    active_reversals_reorient_scored: float = float("nan")
    pct_active_reversals_reorient: float = float("nan")
    # (reorienting reversals per active min) / (active reversals per active min)
    reorient_per_reversal_rate: float = float("nan")
    reorientation_angle_deg: float = REORIENTATION_ANGLE_DEG
    reorientation_window_s: float = REORIENTATION_WINDOW_S
    reorientation_fwd_search_s: float = REORIENTATION_FWD_SEARCH_S
    reorientation_fwd_best_s: float = REORIENTATION_FWD_BEST_S
    active_time_pos: float = 0.0
    active_time_neg: float = 0.0
    active_pct_pos: float = 0.0
    active_pct_neg: float = 0.0
    # active forward / reverse bouts
    active_fwd_bouts: int = 0
    active_rev_bouts: int = 0
    avg_active_fwd_bout_frames: float = 0.0
    avg_active_rev_bout_frames: float = 0.0
    avg_active_fwd_bout_s: float = 0.0
    avg_active_rev_bout_s: float = 0.0
    # per-minute rates (360 frames = 1 min), activity-filtered where noted
    duration_min: float = 0.0
    reversals_per_min: float = 0.0
    active_reversals_per_min: float = 0.0
    active_time_pos_per_min: float = 0.0  # seconds of forward locomotion per minute of recording
    active_time_neg_per_min: float = 0.0
    duration_s: float = 0.0  # duration represented by captured speed frames
    active_duration_s: float = 0.0  # total active time used for active normalization
    active_duration_min: float = 0.0
    frames_per_minute: int = FRAMES_PER_MINUTE
    fps_true: float = 6.0
    centroid_threshold_bl_s: float = CENTROID_ACTIVITY_THRESHOLD_BL_S
    centroid_window_s: float = CENTROID_ACTIVITY_WINDOW_S
    use_reversal_sustain_filter: bool = USE_REVERSAL_SUSTAIN_FILTER
    reversal_dwell_frames: int = REVERSAL_DWELL_FRAMES
    reversal_min_neg_frames: int = REVERSAL_MIN_NEG_FRAMES  # alias of dwell
    reversal_min_pos_frames: int = REVERSAL_MIN_POS_FRAMES  # alias of dwell
    blob_perimeter_median: float = float("nan")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MetricsResult:
    metrics: WormMetrics
    frames: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_minute: pd.DataFrame = field(default_factory=pd.DataFrame)
    bouts: pd.DataFrame = field(default_factory=pd.DataFrame)
    perimeter: pd.DataFrame = field(default_factory=pd.DataFrame)


def _median_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return np.nan
    deltas = np.diff(times)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return np.nan
    return float(np.median(deltas))


def _gap_threshold(times: np.ndarray) -> float:
    median = _median_dt(times)
    if not np.isfinite(median) or median <= 0:
        return np.inf
    return median * 3


def _remove_outliers(speeds: np.ndarray) -> np.ndarray:
    keep = np.ones(len(speeds), dtype=bool)
    pos = speeds > 0
    neg = speeds < 0
    if pos.any():
        avg = float(speeds[pos].mean())
        if avg > 0:
            keep &= ~(pos & (speeds > avg * OUTLIER_FACTOR))
    if neg.any():
        avg = float(speeds[neg].mean())
        if avg < 0:
            keep &= ~(neg & (speeds < avg * OUTLIER_FACTOR))
    return keep


def _sample_interval(times: np.ndarray) -> float:
    """Typical inter-frame Δt from positive, non-gap intervals."""
    if len(times) < 2:
        return np.nan
    deltas = np.diff(times)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return np.nan
    median = float(np.median(deltas))
    regular = deltas[deltas <= median * 3]
    return float(np.median(regular)) if len(regular) else median




def _centroid_bl_s_activity(
    times: np.ndarray,
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    median_body_length_px: float,
    threshold_bl_s: Optional[float] = None,
    window_s: Optional[float] = None,
    max_gap_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scanning-window activity from centroid path rate (body lengths / second).

    For each sample ``i``, take a centered window of ``window_s`` seconds,
    sum consecutive centroid steps with frame gap in ``[1, max_gap_frames]``,
    and form::

        rate = path_px / median_body_length_px / duration_s

    where ``duration_s`` is the time span of finite positions in the window
    (at least one sample interval). Frame ``i`` is active when ``rate`` >
    ``threshold_bl_s``. Windows with fewer than two finite positions are
    inactive.

    Implemented with prefix sums + ``searchsorted`` (O(n log n)), not a
    per-frame full-array scan.
    """
    if threshold_bl_s is None:
        threshold_bl_s = CENTROID_ACTIVITY_THRESHOLD_BL_S
    if window_s is None:
        window_s = CENTROID_ACTIVITY_WINDOW_S
    if max_gap_frames is None:
        max_gap_frames = CENTROID_PATH_MAX_GAP_FRAMES

    n = len(times)
    active = np.zeros(n, dtype=int)
    rate = np.full(n, np.nan)
    if n == 0:
        return active, rate

    bl = float(median_body_length_px)
    if not np.isfinite(bl) or bl <= 0:
        return active, rate

    sample_dt = _sample_interval(times)
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        sample_dt = float(window_s) / max(n, 1)

    half = 0.5 * float(window_s)
    tol = 0.25 * sample_dt
    thr = float(threshold_bl_s)
    max_gap = max(1, int(max_gap_frames))
    times = np.asarray(times, dtype=float)
    frames = np.asarray(frames, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if n < 2:
        return active, rate

    # Consecutive step lengths; invalid / gapped steps contribute 0.
    dfr = frames[1:] - frames[:-1]
    finite = (
        np.isfinite(x[:-1])
        & np.isfinite(y[:-1])
        & np.isfinite(x[1:])
        & np.isfinite(y[1:])
    )
    ok = finite & (dfr >= 1.0) & (dfr <= float(max_gap))
    step = np.zeros(n - 1, dtype=float)
    step[ok] = np.hypot(x[1:][ok] - x[:-1][ok], y[1:][ok] - y[:-1][ok])
    path_prefix = np.empty(n, dtype=float)
    path_prefix[0] = 0.0
    np.cumsum(step, out=path_prefix[1:])

    # next/prev valid step index for duration endpoints
    next_valid = np.full(n - 1, -1, dtype=np.int64)
    prev_valid = np.full(n - 1, -1, dtype=np.int64)
    last = -1
    for i in range(n - 1):
        if ok[i]:
            last = i
        prev_valid[i] = last
    nxt = -1
    for i in range(n - 2, -1, -1):
        if ok[i]:
            nxt = i
        next_valid[i] = nxt

    left = np.searchsorted(times, times - half - tol, side="left")
    right = np.searchsorted(times, times + half + tol, side="right") - 1

    for i in range(n):
        L = int(left[i])
        R = int(right[i])
        if R - L < 1:
            active[i] = 0
            continue
        # Valid steps whose both endpoints lie in [L, R] use indices L .. R-1.
        first = int(next_valid[L]) if L < n - 1 else -1
        if first < 0 or first > R - 1:
            active[i] = 0
            continue
        last_step = int(prev_valid[R - 1])
        if last_step < L:
            active[i] = 0
            continue
        path = float(path_prefix[R] - path_prefix[L])
        t_first = float(times[first])
        t_last = float(times[last_step + 1])
        dur = max(t_last - t_first, sample_dt)
        r = (path / bl) / dur
        rate[i] = r
        active[i] = int(r > thr)

    return active, rate



_EMPTY_CENTROID_TRACK_COLS = (
    "frame",
    "x",
    "y",
    "length_px",
    "roi_size",
    "x_head",
    "y_head",
    "x_tail",
    "y_tail",
)


def clear_featuresn_track_caches() -> None:
    """Drop cached featuresN tracks, centroid activity, and HT-invert sets."""
    _CENTROID_TRACK_CACHE.clear()
    _CENTROID_ACTIVITY_CACHE.clear()
    _HT_INVERT_SET_CACHE.clear()


def clear_centroid_activity_cache() -> None:
    """Drop cached full-track centroid activity (+ related featuresN caches)."""
    clear_featuresn_track_caches()


def _source_key_from_df(df: Optional[pd.DataFrame]) -> str:
    if df is None or len(df) == 0 or "source" not in df.columns:
        return ""
    return str(df["source"].iloc[0] or "")


def load_centroid_track_for_speed_df(
    df: pd.DataFrame,
    *,
    min_track_frames: int = 1,
) -> pd.DataFrame:
    """
    Tierpsy blob centroid (x, y) + body length (px) for the worm behind ``df``.

    Uses ``featuresN`` trajectories (not video pixel scanning) for **every**
    tracked blob frame — independent of the FIXED speed CSV timeline.
    Returns ``frame``, ``x``, ``y``, ``length_px``, optional ``roi_size``, and
    Tierpsy tip columns ``x_head``/``y_head``/``x_tail``/``y_tail`` (pixels;
    NaN when not skeletonized) when timeseries tips are present.

    Cached per ``source`` CSV path so continuous-minute slices do not reload
    featuresN once per window.
    """
    empty = pd.DataFrame(columns=list(_EMPTY_CENTROID_TRACK_COLS))
    if df is None or len(df) == 0 or "source" not in df.columns:
        return empty

    source = _source_key_from_df(df)
    cache_key = (source, int(min_track_frames))
    cached = _CENTROID_TRACK_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    from .tierpsy_data import (
        get_video,
        load_video_signals,
        microns_per_pixel,
        parse_video_id_from_source,
        resolve_mask_worm,
        warn_once,
        worm_number_from_id,
        worm_number_from_source,
    )

    try:
        video = parse_video_id_from_source(source)
    except Exception:
        _CENTROID_TRACK_CACHE[cache_key] = empty
        return empty.copy()

    csv_worm = worm_number_from_source(source)
    if csv_worm is None and "worm_id" in df.columns:
        # Prefer source-path worm id: continuous-minute slices overwrite worm_id.
        csv_worm = worm_number_from_id(str(df["worm_id"].iloc[0]))
    if csv_worm is None:
        _CENTROID_TRACK_CACHE[cache_key] = empty
        return empty.copy()

    mask_worm = resolve_mask_worm(video, int(csv_worm))
    try:
        vp = get_video(video)
    except KeyError as exc:
        warn_once(
            f"no-features:{video}",
            f"{exc}. Without it, worms from {video} have no centroid track, so "
            "they are never 'active' and reorientation angles are not scored. "
            "See ../Sample_data/README.md (next to custom_python_scripts/) for the expected data layout.",
        )
        _CENTROID_TRACK_CACHE[cache_key] = empty
        return empty.copy()

    try:
        sig = load_video_signals(vp, min_track_frames=int(min_track_frames))
    except Exception:
        _CENTROID_TRACK_CACHE[cache_key] = empty
        return empty.copy()
    tip_src = ("coord_x_head", "coord_y_head", "coord_x_tail", "coord_y_tail")
    cols = ["frame", "x", "y", "length"]
    if "roi_size" in sig.columns:
        cols.append("roi_size")
    for c in tip_src:
        if c in sig.columns:
            cols.append(c)
    track = sig.loc[sig["worm"] == int(mask_worm), cols].copy()
    if track.empty:
        _CENTROID_TRACK_CACHE[cache_key] = empty
        return empty.copy()

    try:
        mpp = microns_per_pixel(vp)
    except Exception:
        mpp = float("nan")
    length = pd.to_numeric(track["length"], errors="coerce").to_numpy(dtype=float)
    length_px = length / float(mpp) if np.isfinite(mpp) and mpp > 0 else length
    out = pd.DataFrame(
        {
            "frame": pd.to_numeric(track["frame"], errors="coerce"),
            "x": pd.to_numeric(track["x"], errors="coerce"),
            "y": pd.to_numeric(track["y"], errors="coerce"),
            "length_px": length_px,
        }
    )
    if "roi_size" in track.columns:
        out["roi_size"] = pd.to_numeric(track["roi_size"], errors="coerce")
    else:
        out["roi_size"] = np.nan
    # Tips already converted to pixels in load_video_signals.
    tip_dst = ("x_head", "y_head", "x_tail", "y_tail")
    for src_col, dst in zip(tip_src, tip_dst):
        if src_col in track.columns:
            out[dst] = pd.to_numeric(track[src_col], errors="coerce")
        else:
            out[dst] = np.nan
    # Keep every tracked frame with a finite centroid (skeleton not required).
    out = out.dropna(subset=["frame", "x", "y"]).reset_index(drop=True)
    _CENTROID_TRACK_CACHE[cache_key] = out
    return out.copy()


def _ht_invert_frame_set_for_df(df: pd.DataFrame) -> Optional[set]:
    """Fixed-speed HT-invert frames for the worm behind ``df``, if available."""
    if df is None or len(df) == 0 or "source" not in df.columns:
        return None

    source = _source_key_from_df(df)
    if source in _HT_INVERT_SET_CACHE:
        return _HT_INVERT_SET_CACHE[source]

    from .tierpsy_data import (
        ht_invert_frame_set,
        parse_video_id_from_source,
        resolve_mask_worm,
        worm_number_from_id,
        worm_number_from_source,
    )

    try:
        video = parse_video_id_from_source(source)
    except Exception:
        _HT_INVERT_SET_CACHE[source] = None
        return None
    csv_worm = worm_number_from_source(source)
    if csv_worm is None and "worm_id" in df.columns:
        csv_worm = worm_number_from_id(str(df["worm_id"].iloc[0]))
    if csv_worm is None:
        _HT_INVERT_SET_CACHE[source] = None
        return None
    mask_worm = resolve_mask_worm(video, int(csv_worm))
    try:
        invert = ht_invert_frame_set(source, int(mask_worm))
    except Exception:
        invert = None
    _HT_INVERT_SET_CACHE[source] = invert
    return invert


def _median_body_length_px(track: pd.DataFrame) -> float:
    """Median skeleton length in px; fall back to median ROI size."""
    med_bl = float(np.nanmedian(track["length_px"].to_numpy(dtype=float)))
    if np.isfinite(med_bl) and med_bl > 0:
        return med_bl
    if "roi_size" in track.columns:
        med_roi = float(np.nanmedian(track["roi_size"].to_numpy(dtype=float)))
        if np.isfinite(med_roi) and med_roi > 0:
            return med_roi
    return float("nan")


def compute_centroid_bl_s_activity(
    track: pd.DataFrame,
    *,
    fps: float,
    threshold_bl_s: Optional[float] = None,
    window_s: Optional[float] = None,
    max_gap_frames: Optional[int] = None,
    median_body_length_px: Optional[float] = None,
) -> pd.DataFrame:
    """
    Centroid path-rate activity on the **full** blob track (no speed CSV join).

    ``track`` needs ``frame``, ``x``, ``y``; optional ``length_px`` / ``roi_size``
    for body-length scaling. Returns the track plus ``time``, ``centroid_bl_s``,
    ``active_centroid_bl_s``.
    """
    if threshold_bl_s is None:
        threshold_bl_s = CENTROID_ACTIVITY_THRESHOLD_BL_S
    if window_s is None:
        window_s = CENTROID_ACTIVITY_WINDOW_S
    if max_gap_frames is None:
        max_gap_frames = CENTROID_PATH_MAX_GAP_FRAMES

    out = track.copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out = out.dropna(subset=["frame", "x", "y"]).sort_values(
        "frame", kind="mergesort"
    ).reset_index(drop=True)
    if out.empty:
        out["time"] = []
        out["centroid_bl_s"] = []
        out["active_centroid_bl_s"] = []
        return out

    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    if median_body_length_px is None:
        median_body_length_px = _median_body_length_px(out)
    med_bl = float(median_body_length_px)
    if not np.isfinite(med_bl) or med_bl <= 0:
        raise ValueError("Could not estimate median body length from centroid track.")

    frames = out["frame"].to_numpy(dtype=float)
    times = frames / fps
    active, rate = _centroid_bl_s_activity(
        times,
        frames,
        out["x"].to_numpy(dtype=float),
        out["y"].to_numpy(dtype=float),
        median_body_length_px=med_bl,
        threshold_bl_s=float(threshold_bl_s),
        window_s=float(window_s),
        max_gap_frames=int(max_gap_frames),
    )
    out = out.assign(
        time=times,
        centroid_bl_s=rate,
        active_centroid_bl_s=active,
        median_body_length_px=med_bl,
    )
    return out


def _centroid_activity_cache_key(
    df: pd.DataFrame,
    *,
    fps: float,
    threshold_bl_s: float,
    window_s: float,
    max_gap_frames: int,
) -> tuple:
    source = ""
    if df is not None and len(df) and "source" in df.columns:
        source = str(df["source"].iloc[0] or "")
    return (
        source,
        round(float(fps), 6),
        round(float(threshold_bl_s), 6),
        round(float(window_s), 6),
        int(max_gap_frames),
    )


def _centroid_activity_on_speed_timeline(
    df: pd.DataFrame,
    frames_all: np.ndarray,
    *,
    fps: float,
    threshold_bl_s: Optional[float] = None,
    window_s: Optional[float] = None,
    max_gap_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-track centroid BL/s activity mapped onto the speed CSV frame axis.

    Missing centroid frames are inactive (``active=0``, rate NaN). Activity is
    computed on the Tierpsy blob track, then looked up by integer frame.
    Results are cached per source CSV + activity knobs.
    """
    n = len(frames_all)
    active = np.zeros(n, dtype=int)
    rate = np.full(n, np.nan)
    if threshold_bl_s is None:
        threshold_bl_s = CENTROID_ACTIVITY_THRESHOLD_BL_S
    if window_s is None:
        window_s = CENTROID_ACTIVITY_WINDOW_S
    if max_gap_frames is None:
        max_gap_frames = CENTROID_PATH_MAX_GAP_FRAMES

    key = _centroid_activity_cache_key(
        df,
        fps=float(fps),
        threshold_bl_s=float(threshold_bl_s),
        window_s=float(window_s),
        max_gap_frames=int(max_gap_frames),
    )
    cen = _CENTROID_ACTIVITY_CACHE.get(key)
    if cen is None:
        track = load_centroid_track_for_speed_df(df)
        if track is None or track.empty:
            _CENTROID_ACTIVITY_CACHE[key] = pd.DataFrame()
            return active, rate
        try:
            cen = compute_centroid_bl_s_activity(
                track,
                fps=float(fps),
                threshold_bl_s=threshold_bl_s,
                window_s=window_s,
                max_gap_frames=max_gap_frames,
            )
        except Exception:
            _CENTROID_ACTIVITY_CACHE[key] = pd.DataFrame()
            return active, rate
        _CENTROID_ACTIVITY_CACHE[key] = cen

    if cen is None or cen.empty or "active_centroid_bl_s" not in cen.columns:
        return active, rate

    fr = pd.to_numeric(cen["frame"], errors="coerce").to_numpy(dtype=float)
    act = pd.to_numeric(cen["active_centroid_bl_s"], errors="coerce").to_numpy(
        dtype=float
    )
    rates = pd.to_numeric(cen["centroid_bl_s"], errors="coerce").to_numpy(dtype=float)
    lookup: Dict[int, tuple[int, float]] = {}
    for f, a, r in zip(fr, act, rates):
        if not np.isfinite(f):
            continue
        lookup[int(round(float(f)))] = (
            int(a) if np.isfinite(a) else 0,
            float(r) if np.isfinite(r) else float("nan"),
        )
    for i, f in enumerate(np.asarray(frames_all, dtype=float)):
        if not np.isfinite(f):
            continue
        hit = lookup.get(int(round(float(f))))
        if hit is None:
            continue
        active[i], rate[i] = hit
    return active, rate



def _active_directed_bouts(
    times: np.ndarray,
    speeds: np.ndarray,
    active: np.ndarray,
    gap: float,
    *,
    kept_reversals: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Contiguous runs of active frames with the same speed sign.

    A bout breaks on inactive frames, zero speed, a large time gap, or a
    direction change.

    When ``kept_reversals`` is set (post-dwell / post-correlation flags), a
    forward→reverse transition ends the forward bout only if that transition
    is a kept reversal (flag on the last frame before the confirming reverse
    run). Correlation-rejected (or otherwise uncounted) sign blips are
    skipped: they do not end the forward bout early in a way that starts a
    reverse bout, and they do not create reverse bouts. Reverse→forward
    always ends a reverse bout.
    """
    n = len(speeds)
    deltas = np.diff(times)
    deltas = deltas[(deltas > 0) & (deltas <= gap)] if n > 1 else np.array([])
    typical_dt = float(np.median(deltas)) if len(deltas) else 0.0

    if kept_reversals is not None and len(kept_reversals) != n:
        raise ValueError("kept_reversals length must match speeds")

    def _sign(v: float) -> int:
        return 1 if v > 0 else -1

    def _gap_between(a: int, b: int) -> bool:
        dt = times[b] - times[a]
        return dt <= 0 or dt > gap

    rows = []
    i = 0
    while i < n:
        if active[i] != 1 or speeds[i] == 0 or not np.isfinite(speeds[i]):
            i += 1
            continue

        sign = _sign(speeds[i])
        start = i
        last_dir = i
        n_dir = 1
        j = i + 1
        # If we end because of an uncounted reverse island, skip past it so
        # those frames do not become their own reverse bout.
        skip_to: Optional[int] = None

        while j < n:
            if _gap_between(j - 1, j):
                break
            if active[j] != 1:
                break
            if not np.isfinite(speeds[j]) or speeds[j] == 0:
                break

            s = _sign(speeds[j])
            if s == sign:
                last_dir = j
                n_dir += 1
                j += 1
                continue

            # Opposite direction.
            if (
                kept_reversals is not None
                and sign > 0
                and s < 0
                and not bool(kept_reversals[j - 1])
            ):
                # Skip uncounted reverse island; resume if forward returns.
                k = j
                resumed = False
                while k < n:
                    if k > j and _gap_between(k - 1, k):
                        break
                    if active[k] != 1:
                        break
                    if not np.isfinite(speeds[k]) or speeds[k] == 0:
                        break
                    if speeds[k] > 0:
                        j = k
                        resumed = True
                        break
                    k += 1
                if resumed:
                    continue
                # No forward resume — end forward bout; skip the reverse island.
                skip_to = k
                break

            # Counted direction change (kept +→−, or any −→+, or unfiltered).
            break

        dur = (
            (times[last_dir] - times[start])
            if last_dir > start
            else typical_dt
        )
        rows.append(
            {
                "direction": "forward" if sign == 1 else "reverse",
                "frame_start": int(start),
                "frame_end": int(last_dir),
                "time_start": float(times[start]),
                "time_end": float(times[last_dir]),
                "n_frames": int(n_dir),
                "duration_s": float(dur),
            }
        )
        if skip_to is not None:
            i = max(skip_to, last_dir + 1)
        else:
            i = j if j > i else i + 1

    return pd.DataFrame(rows)

def _debounced_reversal_flags(
    speeds: np.ndarray,
    times: np.ndarray,
    gap: float,
    *,
    dwell: int,
) -> np.ndarray:
    """
    Sign-dwell reversal detector.

    Maintains FORWARD (+1) / REVERSE (-1) state. State flips only after
    ``dwell`` consecutive opposite-sign frames. Zeros and NaNs leave state
    unchanged and do not advance the opposite-sign counter. Tracking gaps
    (``dt > gap``) reset state to unknown.

    Flags True on the last frame *before* the confirming opposite-sign run
    when entering REVERSE.
    """
    n = len(speeds)
    flags = np.zeros(n, dtype=bool)
    need = max(1, int(dwell))
    state = 0  # +1 forward, -1 reverse, 0 unknown
    pending = 0
    pending_count = 0

    for i in range(n):
        if i > 0:
            dt = float(times[i] - times[i - 1])
            if not (dt > 0 and dt <= gap):
                state = 0
                pending = 0
                pending_count = 0

        s = speeds[i]
        if not np.isfinite(s) or s == 0:
            continue

        sg = 1 if s > 0 else -1

        if state == 0:
            if pending == sg:
                pending_count += 1
            else:
                pending = sg
                pending_count = 1
            if pending_count >= need:
                state = pending
                pending = 0
                pending_count = 0
            continue

        if sg == state:
            pending = 0
            pending_count = 0
            continue

        if pending == sg:
            pending_count += 1
        else:
            pending = sg
            pending_count = 1

        if pending_count < need:
            continue

        prev = state
        state = pending
        pending = 0
        pending_count = 0
        if prev == 1 and state == -1:
            mark = i - need
            if mark >= 0:
                flags[mark] = True

    return flags


def _reversal_flags(
    speeds: np.ndarray,
    times: np.ndarray,
    gap: float,
    *,
    use_sustain_filter: bool,
    dwell_frames: int,
    min_neg_frames: Optional[int] = None,
    min_pos_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Boolean mask of reversal events (last frame before confirming reverse).

    When ``use_sustain_filter`` is True, uses the sign-dwell state machine
    with ``dwell_frames`` (legacy ``min_neg_frames`` / ``min_pos_frames`` are
    accepted as aliases; the effective dwell is their max if both given).
    When False, dwell is 1 (immediate sign change after one opposite frame).
    """
    if min_neg_frames is not None or min_pos_frames is not None:
        vals = [int(dwell_frames)]
        if min_neg_frames is not None:
            vals.append(int(min_neg_frames))
        if min_pos_frames is not None:
            vals.append(int(min_pos_frames))
        dwell = max(vals)
    else:
        dwell = int(dwell_frames)
    if not use_sustain_filter:
        dwell = 1
    return _debounced_reversal_flags(speeds, times, gap, dwell=dwell)































def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _ht_helpers():
    """Lazy import of omega HT-axis pickers. Returns None if unavailable."""
    try:
        from .tierpsy_data import ht_axis, pick_extended_frame, skeleton_dropout_span

        return skeleton_dropout_span, pick_extended_frame, ht_axis
    except Exception:
        return None


def _ht_flip_at_frame(
    frame_val: float, invert_frames: Optional[set]
) -> int:
    if invert_frames is None or not np.isfinite(frame_val):
        return 0
    return 1 if int(round(float(frame_val))) in invert_frames else 0


def _pre_ht_axis_at_frame(
    frames: np.ndarray,
    head_xy: np.ndarray,
    tail_xy: np.ndarray,
    center_frame: float,
    *,
    window_s: float,
    fps: float,
    invert_frames: Optional[set] = None,
) -> Optional[np.ndarray]:
    """
    Most-extended head→tail axis before ``center_frame`` (same pre rule as
    the omega HT picker), searched in ``±window_s``.
    """
    helpers = _ht_helpers()
    if helpers is None:
        return None
    dropout_span, pick_extended, ht_axis = helpers
    if not np.isfinite(center_frame) or not np.isfinite(fps) or fps <= 0:
        return None
    frames = np.asarray(frames, dtype=float)
    head_xy = np.asarray(head_xy, dtype=float)
    tail_xy = np.asarray(tail_xy, dtype=float)
    half = max(1, int(round(float(window_s) * float(fps))))
    m = (
        (frames >= float(center_frame) - half)
        & (frames <= float(center_frame) + half)
        & np.isfinite(frames)
    )
    if int(m.sum()) < 1:
        return None
    fr = frames[m]
    h = head_xy[m]
    t = tail_xy[m]
    local_center = int(np.argmin(np.abs(fr - float(center_frame))))
    valid = np.isfinite(h).all(axis=1) & np.isfinite(t).all(axis=1)
    if int(valid.sum()) < 1:
        return None
    gap = dropout_span(valid, local_center)
    if gap is not None:
        lo, _hi = gap
        pre = np.flatnonzero(valid[:lo]) if lo > 0 else np.array([], dtype=int)
    else:
        pre = np.flatnonzero(valid[: max(local_center - 2, 1)])
    if len(pre) == 0:
        return None
    i_pre = pick_extended(pre, h, t)
    return ht_axis(h, t, int(i_pre), _ht_flip_at_frame(fr[i_pre], invert_frames))


def _best_ht_axis_near_time(
    frames: np.ndarray,
    head_xy: np.ndarray,
    tail_xy: np.ndarray,
    t_center: float,
    *,
    half_s: float,
    fps: float,
    invert_frames: Optional[set] = None,
) -> Optional[np.ndarray]:
    """Most-extended HT axis among skeletonized frames within ``±half_s`` of ``t_center``."""
    helpers = _ht_helpers()
    if helpers is None:
        return None
    _dropout_span, pick_extended, ht_axis = helpers
    if not np.isfinite(t_center) or not np.isfinite(fps) or fps <= 0:
        return None
    frames = np.asarray(frames, dtype=float)
    head_xy = np.asarray(head_xy, dtype=float)
    tail_xy = np.asarray(tail_xy, dtype=float)
    t = frames / float(fps)
    valid = (
        np.isfinite(t)
        & np.isfinite(head_xy).all(axis=1)
        & np.isfinite(tail_xy).all(axis=1)
    )
    m = valid & (np.abs(t - float(t_center)) <= float(half_s))
    idxs = np.flatnonzero(m)
    if len(idxs) == 0:
        return None
    i = pick_extended(idxs, head_xy, tail_xy)
    return ht_axis(
        head_xy, tail_xy, int(i), _ht_flip_at_frame(frames[i], invert_frames)
    )


def _reverse_bout_end_time(
    t_rev: float, rev_starts: np.ndarray, rev_ends: np.ndarray
) -> float:
    """End time of the reverse bout tied to a reversal at ``t_rev``."""
    if len(rev_starts) == 0:
        return float(t_rev)
    contain = (rev_starts <= t_rev) & (t_rev <= rev_ends)
    if contain.any():
        hits = np.flatnonzero(contain)
        i = hits[int(np.argmax(rev_starts[hits]))]
        return float(rev_ends[i])
    after = rev_starts >= t_rev
    if after.any():
        hits = np.flatnonzero(after)
        i = hits[int(np.argmin(rev_starts[hits]))]
        return float(rev_ends[i])
    return float(t_rev)


def _next_forward_start(
    t_end: float, fwd_starts: np.ndarray, search_s: float
) -> Optional[float]:
    if len(fwd_starts) == 0:
        return None
    ok = (fwd_starts >= t_end) & (fwd_starts <= t_end + float(search_s))
    if not ok.any():
        return None
    return float(np.min(fwd_starts[ok]))


def _active_reversal_reorientation(
    event_frames: np.ndarray,
    track: pd.DataFrame,
    *,
    fps: float,
    fwd_starts: np.ndarray,
    rev_starts: np.ndarray,
    rev_ends: np.ndarray,
    angle_deg: float = REORIENTATION_ANGLE_DEG,
    window_s: float = REORIENTATION_WINDOW_S,
    fwd_search_s: float = REORIENTATION_FWD_SEARCH_S,
    fwd_best_s: float = REORIENTATION_FWD_BEST_S,
    invert_frames: Optional[set] = None,
) -> tuple[np.ndarray, int, int]:
    """
    Reorientation at each active reversal.

    Pre: most-extended HT axis before the reversal (``window_s``).
    Post: most-extended HT within ``fwd_best_s`` of the next forward-bout
    start that begins within ``fwd_search_s`` of reverse-bout end. If no
    such bout or no usable HT axis, the event stays NaN and is omitted
    from both the >threshold count and the percentage denominator.
    Returns ``(angles, n_gt_threshold, n_scored)``.
    """
    event_frames = np.asarray(event_frames, dtype=float)
    fwd_starts = np.asarray(fwd_starts, dtype=float)
    rev_starts = np.asarray(rev_starts, dtype=float)
    rev_ends = np.asarray(rev_ends, dtype=float)
    angles = np.full(len(event_frames), np.nan)
    n_gt = 0
    n_scored = 0
    if track is None or track.empty or len(event_frames) == 0:
        return angles, 0, 0
    if not np.isfinite(fps) or fps <= 0:
        return angles, 0, 0
    fr = pd.to_numeric(track["frame"], errors="coerce").to_numpy(dtype=float)
    has_tips = all(c in track.columns for c in ("x_head", "y_head", "x_tail", "y_tail"))
    if has_tips:
        head_xy = np.column_stack(
            [
                pd.to_numeric(track["x_head"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(track["y_head"], errors="coerce").to_numpy(dtype=float),
            ]
        )
        tail_xy = np.column_stack(
            [
                pd.to_numeric(track["x_tail"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(track["y_tail"], errors="coerce").to_numpy(dtype=float),
            ]
        )
    else:
        head_xy = tail_xy = None
    thr = float(angle_deg)
    for i, ef in enumerate(event_frames):
        if not np.isfinite(ef):
            continue
        if head_xy is None or tail_xy is None:
            continue
        pre = _pre_ht_axis_at_frame(
            fr,
            head_xy,
            tail_xy,
            float(ef),
            window_s=window_s,
            fps=fps,
            invert_frames=invert_frames,
        )
        if pre is None:
            continue
        t_rev = float(ef) / float(fps)
        t_end = _reverse_bout_end_time(t_rev, rev_starts, rev_ends)
        t_fwd = _next_forward_start(t_end, fwd_starts, fwd_search_s)
        if t_fwd is None:
            continue
        post = _best_ht_axis_near_time(
            fr,
            head_xy,
            tail_xy,
            t_fwd,
            half_s=fwd_best_s,
            fps=fps,
            invert_frames=invert_frames,
        )
        if post is None:
            continue
        ang = _angle_between_deg(pre, post)
        if not np.isfinite(ang):
            continue
        angles[i] = ang
        n_scored += 1
        if ang > thr:
            n_gt += 1
    return angles, n_gt, n_scored


def _minute_active_reversal_extremes(
    active_reversal: np.ndarray,
    *,
    frames_per_minute: int,
    require_full_minute: bool = True,
) -> tuple[int, int, int]:
    """
    Min / max active-reversal counts across 1-minute bins.

    Returns ``(min_count, max_count, n_bins_used)``. With
    ``require_full_minute=True``, a trailing partial bin is ignored (unless it
    is the only bin).
    """
    flags = np.asarray(active_reversal, dtype=int)
    n = len(flags)
    fpm = max(1, int(frames_per_minute))
    if n == 0:
        return 0, 0, 0
    n_bins = int(np.ceil(n / fpm))
    counts: List[int] = []
    for b in range(n_bins):
        lo = b * fpm
        hi = min(n, (b + 1) * fpm)
        if require_full_minute and (hi - lo) < fpm and n_bins > 1:
            continue
        counts.append(int(flags[lo:hi].sum()))
    if not counts:
        return 0, 0, 0
    return int(min(counts)), int(max(counts)), int(len(counts))


def _per_minute_table(
    times: np.ndarray,
    speeds: np.ndarray,
    active: np.ndarray,
    *,
    frames_per_minute: int = FRAMES_PER_MINUTE,
    reversal_flags: Optional[np.ndarray] = None,
    active_reversal_flags: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """One row per frames_per_minute bin with reversals and % direction."""
    n = len(speeds)
    if reversal_flags is None:
        gap = _gap_threshold(times)
        reversal_flags = _reversal_flags(
            speeds,
            times,
            gap,
            use_sustain_filter=USE_REVERSAL_SUSTAIN_FILTER,
            dwell_frames=REVERSAL_DWELL_FRAMES,
        )
    if active_reversal_flags is None:
        active_reversal_flags = (
            np.asarray(reversal_flags, dtype=bool) & (np.asarray(active) == 1)
        )
    rows = []
    n_bins = int(np.ceil(n / frames_per_minute)) if n else 0
    for b in range(n_bins):
        lo = b * frames_per_minute
        hi = min(n, (b + 1) * frames_per_minute)
        sl = slice(lo, hi)
        s = speeds[sl]
        a = active[sl]
        t = times[sl]
        n_bin = hi - lo

        revs = int(np.asarray(reversal_flags[lo:hi]).sum())
        active_revs = int(np.asarray(active_reversal_flags[lo:hi]).sum())

        active_s = s[a == 1]
        n_act = len(active_s)
        rows.append(
            {
                "minute": b + 1,
                "frame_start": lo,
                "frame_end": hi - 1,
                "time_start": float(t[0]) if n_bin else np.nan,
                "time_end": float(t[-1]) if n_bin else np.nan,
                "n_frames": n_bin,
                "n_active": int(n_act),
                "pct_active": round(100 * n_act / n_bin, 1) if n_bin else 0.0,
                "reversals": revs,
                "active_reversals": active_revs,
                "pct_pos": round(100 * (s > 0).sum() / n_bin, 1) if n_bin else 0.0,
                "pct_neg": round(100 * (s < 0).sum() / n_bin, 1) if n_bin else 0.0,
                "active_pct_pos": round(100 * (active_s > 0).sum() / n_act, 1) if n_act else 0.0,
                "active_pct_neg": round(100 * (active_s < 0).sum() / n_act, 1) if n_act else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(
    df: pd.DataFrame,
    *,
    worm_id: Optional[str] = None,
    time_col: str = "time",
    speed_col: str = "speed",
    angular_col: str = "angular_velocity_midbody",
    centroid_threshold_bl_s: Optional[float] = None,
    centroid_window_s: Optional[float] = None,
    use_reversal_sustain_filter: Optional[bool] = None,
    reversal_dwell_frames: Optional[int] = None,
    reversal_min_neg_frames: Optional[int] = None,
    reversal_min_pos_frames: Optional[int] = None,
    frames_per_minute: Optional[int] = None,
) -> MetricsResult:
    """
    Compute locomotion metrics for one worm.

    Activity is centroid path-rate (BL/s) over a centered
    ``centroid_window_s`` window; a frame is active when rate >
    ``centroid_threshold_bl_s``. Mapped onto the speed timeline by frame.
    Angular velocity is optional (kept on the frame table when present).

    ``frames_per_minute`` defaults to ``df['frames_per_minute']`` when present
    (from ``video_fps.csv``), else the module ``FRAMES_PER_MINUTE`` constant.

    If activity / reversal filter kwargs are omitted, the current module-level
    constants are used (read at call time, so editing those constants takes
    effect after reload).
    """
    if frames_per_minute is None:
        if "frames_per_minute" in df.columns and len(df):
            try:
                frames_per_minute = int(df["frames_per_minute"].iloc[0])
            except Exception:
                frames_per_minute = FRAMES_PER_MINUTE
        else:
            frames_per_minute = FRAMES_PER_MINUTE
    frames_per_minute = max(1, int(frames_per_minute))
    if centroid_threshold_bl_s is None:
        centroid_threshold_bl_s = CENTROID_ACTIVITY_THRESHOLD_BL_S
    if centroid_window_s is None:
        centroid_window_s = CENTROID_ACTIVITY_WINDOW_S
    if use_reversal_sustain_filter is None:
        use_reversal_sustain_filter = USE_REVERSAL_SUSTAIN_FILTER
    if reversal_dwell_frames is None:
        # Prefer explicit dwell; fall back to legacy min_pos/min_neg aliases.
        if reversal_min_neg_frames is not None or reversal_min_pos_frames is not None:
            vals = []
            if reversal_min_neg_frames is not None:
                vals.append(int(reversal_min_neg_frames))
            if reversal_min_pos_frames is not None:
                vals.append(int(reversal_min_pos_frames))
            reversal_dwell_frames = max(vals) if vals else REVERSAL_DWELL_FRAMES
        else:
            reversal_dwell_frames = REVERSAL_DWELL_FRAMES
    reversal_dwell_frames = max(1, int(reversal_dwell_frames))
    # Keep legacy fields populated for summary tables / notebook displays.
    reversal_min_neg_frames = reversal_dwell_frames
    reversal_min_pos_frames = reversal_dwell_frames

    if worm_id is None:
        worm_id = str(df["worm_id"].iloc[0]) if "worm_id" in df.columns else ""

    recording_date = ""
    video_number = ""
    source_file = ""
    if "recording_date" in df.columns and len(df):
        recording_date = str(df["recording_date"].iloc[0] or "")
    if "video_number" in df.columns and len(df):
        video_number = str(df["video_number"].iloc[0] or "")
    if "source" in df.columns and len(df):
        source_file = str(df["source"].iloc[0] or "")
    if (not recording_date or not video_number) and source_file:
        try:
            from .io import parse_recording_metadata

            meta = parse_recording_metadata(source_file)
            recording_date = recording_date or meta.get("recording_date", "")
            video_number = video_number or meta.get("video_number", "")
        except Exception:
            pass

    has_angular = angular_col in df.columns
    cols = [time_col, speed_col]
    if has_angular:
        cols.append(angular_col)
    if "blob_perimeter" in df.columns:
        cols.append("blob_perimeter")
    if "frame" in df.columns:
        cols.append("frame")
    work = df[cols].copy()

    times_raw = pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float)
    existing_frame = None
    if "frame" in work.columns:
        existing_frame = pd.to_numeric(work["frame"], errors="coerce").to_numpy(
            dtype=float
        )
    from .video_fps import ASSUMED_FPS, recover_frame_index

    frames_all = recover_frame_index(
        times_raw,
        assumed_fps=ASSUMED_FPS,
        existing_frame=existing_frame,
    )
    work["frame"] = frames_all
    # Analysis axis is video frame index — never raw export timestamps.
    sorted_df = work.sort_values("frame", kind="mergesort").reset_index(drop=True)
    total_frames = len(sorted_df)

    frames_all = pd.to_numeric(sorted_df["frame"], errors="coerce").to_numpy(dtype=float)
    speeds_all = pd.to_numeric(sorted_df[speed_col], errors="coerce").to_numpy(dtype=float)
    if has_angular:
        angular_all = pd.to_numeric(
            sorted_df[angular_col], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        angular_all = np.full(len(sorted_df), np.nan)
    fps_true = float("nan")
    if "fps_true" in df.columns and len(df):
        try:
            fps_true = float(df["fps_true"].iloc[0])
        except Exception:
            fps_true = float("nan")
    if not np.isfinite(fps_true) or fps_true <= 0:
        fps_true = float(frames_per_minute) / 60.0
    # True wall-clock seconds for outputs / plots (frame / fps_true).
    times_all = frames_all / float(fps_true)
    peri_all = None
    if "blob_perimeter" in sorted_df.columns:
        peri_all = pd.to_numeric(
            sorted_df["blob_perimeter"], errors="coerce"
        ).to_numpy(dtype=float)

    # Outlier speeds become NaN for coverage / activity (never replaced with 0).
    speeds_for_activity = speeds_all.copy()
    finite_speed = np.isfinite(speeds_for_activity)
    outliers_removed = 0
    if finite_speed.any():
        keep_finite = _remove_outliers(speeds_for_activity[finite_speed])
        outlier_flags = np.zeros(len(speeds_for_activity), dtype=bool)
        outlier_flags[np.where(finite_speed)[0][~keep_finite]] = True
        speeds_for_activity[outlier_flags] = np.nan
        outliers_removed = int(outlier_flags.sum())

    # Activity on the full timeline so missing-data slots stay empty (NaN).
    # Windowing uses true seconds (frame / fps_true), not export timestamps.
    active_all, centroid_rate_all = _centroid_activity_on_speed_timeline(
        df,
        frames_all,
        fps=float(fps_true),
        threshold_bl_s=float(centroid_threshold_bl_s),
        window_s=float(centroid_window_s),
    )

    # Metrics use rows with usable (non-outlier) speed only.
    valid_mask = np.isfinite(speeds_for_activity)
    times = times_all[valid_mask]
    speeds = speeds_for_activity[valid_mask]
    angular = angular_all[valid_mask]
    frames_idx = frames_all[valid_mask]
    active = active_all[valid_mask]
    centroid_bl_s = centroid_rate_all[valid_mask]

    gap = _gap_threshold(times)
    n_valid = len(speeds)

    fwd_frames = int((speeds > 0).sum())
    rev_frames = int((speeds < 0).sum())
    pct_pos = (fwd_frames / n_valid * 100) if n_valid else 0.0
    pct_neg = (rev_frames / n_valid * 100) if n_valid else 0.0
    pct_captured = (n_valid / total_frames * 100) if total_frames else 0.0

    rev_candidates = _reversal_flags(
        speeds,
        times,
        gap,
        use_sustain_filter=bool(use_reversal_sustain_filter),
        dwell_frames=reversal_dwell_frames,
    )

    peri_median = float("nan")
    if peri_all is not None:
        finite_peri = peri_all[np.isfinite(peri_all)]
        if len(finite_peri):
            peri_median = float(np.nanmedian(finite_peri))

    rev_flags_for_count = rev_candidates
    reversals = int(rev_flags_for_count.sum())

    active_mask = active.astype(bool)
    n_active = int(active_mask.sum())
    pct_active = (n_active / n_valid * 100) if n_valid else 0.0
    active_reversals = int((rev_flags_for_count & (active == 1)).sum())

    rows = []
    for i in range(n_valid):
        is_cand = bool(rev_candidates[i])
        is_rev = bool(rev_flags_for_count[i])
        active_rev = int(is_rev and active[i] == 1)
        rows.append(
            {
                "time": times[i],
                "frame": float(frames_idx[i]) if np.isfinite(frames_idx[i]) else float("nan"),
                "speed": speeds[i],
                "angular_velocity_midbody": angular[i],
                "centroid_bl_s": centroid_bl_s[i],
                "active": active[i],
                "is_reversal_candidate": int(is_cand),
                "is_reversal": int(is_rev),
                "active_reversal": active_rev,
            }
        )
    frames = pd.DataFrame(rows)

    # Per-sample active-reversal flags.
    active_rev_flags = (
        frames["active_reversal"].to_numpy(dtype=int)
        if len(frames)
        else np.zeros(0, dtype=int)
    )
    min_ar_min, max_ar_min, n_min_bins = _minute_active_reversal_extremes(
        active_rev_flags,
        frames_per_minute=frames_per_minute,
        require_full_minute=True,
    )

    # Directional bouts (needed for reorientation post-vector and bout metrics).
    bouts = _active_directed_bouts(
        times,
        speeds,
        active,
        gap,
        kept_reversals=rev_flags_for_count,
    )
    if len(bouts):
        fwd_starts = bouts.loc[
            bouts["direction"] == "forward", "time_start"
        ].to_numpy(dtype=float)
        rev_starts = bouts.loc[
            bouts["direction"] == "reverse", "time_start"
        ].to_numpy(dtype=float)
        rev_ends = bouts.loc[
            bouts["direction"] == "reverse", "time_end"
        ].to_numpy(dtype=float)
    else:
        fwd_starts = rev_starts = rev_ends = np.array([], dtype=float)

    # HT-axis reorientation at each active reversal (> threshold degrees).
    frames["reorientation_deg"] = np.nan
    n_reorient_gt = 0
    n_reorient_scored = 0
    try:
        track_xy = load_centroid_track_for_speed_df(df)
    except Exception:
        track_xy = pd.DataFrame()
    invert_set = None
    try:
        invert_set = _ht_invert_frame_set_for_df(df)
    except Exception:
        invert_set = None
    if len(frames) and track_xy is not None and not track_xy.empty:
        ar_mask = active_rev_flags == 1
        ar_frames = frames.loc[ar_mask, "frame"].to_numpy(dtype=float)
        angles, n_reorient_gt, n_reorient_scored = _active_reversal_reorientation(
            ar_frames,
            track_xy,
            fps=float(fps_true),
            fwd_starts=fwd_starts,
            rev_starts=rev_starts,
            rev_ends=rev_ends,
            angle_deg=float(REORIENTATION_ANGLE_DEG),
            window_s=float(REORIENTATION_WINDOW_S),
            fwd_search_s=float(REORIENTATION_FWD_SEARCH_S),
            fwd_best_s=float(REORIENTATION_FWD_BEST_S),
            invert_frames=invert_set,
        )
        if ar_mask.any():
            frames.loc[ar_mask, "reorientation_deg"] = angles
    reorient = _reorient_metrics(
        active_reversals, n_reorient_gt, n_reorient_scored
    )

    if peri_all is not None:
        frames["blob_perimeter"] = peri_all[valid_mask]

    perimeter = pd.DataFrame()
    if peri_all is not None:
        perimeter = pd.DataFrame(
            {
                "time": times_all,
                "frame": frames_all,
                "blob_perimeter": peri_all,
            }
        )
    # Keep a compact valid dataframe for plotting speed (NaN speeds excluded).
    valid = pd.DataFrame(
        {
            "time": times,
            "frame": frames_idx,
            "speed": speeds,
            "angular_velocity_midbody": angular,
        }
    )

    active_speeds = speeds[active_mask]
    active_pct_pos = (
        100 * float((active_speeds > 0).sum()) / n_active if n_active else 0.0
    )
    active_pct_neg = (
        100 * float((active_speeds < 0).sum()) / n_active if n_active else 0.0
    )

    # Duration normalization is based only on frames with usable speed data.
    # Missing timestamps do not contribute time, regardless of how short the
    # gap is. At 360 frames/minute, each captured frame contributes 1/360 min.
    duration_min = n_valid / frames_per_minute if frames_per_minute else 0.0
    duration_s = duration_min * 60.0
    active_duration_min = (
        n_active / frames_per_minute if frames_per_minute else 0.0
    )
    active_duration_s = active_duration_min * 60.0

    seconds_per_frame = (
        60.0 / frames_per_minute if frames_per_minute else 0.0
    )
    # Directional time uses observed frames only; NaN speed is never converted
    # to zero and contributes no duration.
    time_pos = fwd_frames * seconds_per_frame
    time_neg = rev_frames * seconds_per_frame
    active_time_pos = int((active_speeds > 0).sum()) * seconds_per_frame
    active_time_neg = int((active_speeds < 0).sum()) * seconds_per_frame

    fwd_bouts = bouts.loc[bouts["direction"] == "forward"] if len(bouts) else bouts
    rev_bouts = bouts.loc[bouts["direction"] == "reverse"] if len(bouts) else bouts
    n_fwd_bouts = int(len(fwd_bouts))
    n_rev_bouts = int(len(rev_bouts))
    avg_fwd_frames = float(fwd_bouts["n_frames"].mean()) if n_fwd_bouts else 0.0
    avg_rev_frames = float(rev_bouts["n_frames"].mean()) if n_rev_bouts else 0.0
    avg_fwd_s = float(fwd_bouts["duration_s"].mean()) if n_fwd_bouts else 0.0
    avg_rev_s = float(rev_bouts["duration_s"].mean()) if n_rev_bouts else 0.0

    def per_min(count: float) -> float:
        return round(count / duration_min, 3) if duration_min > 0 else 0.0

    avg_speed = float(speeds.mean()) if n_valid else 0.0
    pos = speeds[speeds > 0]
    neg = speeds[speeds < 0]

    metrics = WormMetrics(
        worm_id=worm_id,
        recording_date=str(recording_date or ""),
        video_number=str(video_number or ""),
        source_worm_id=str(worm_id or ""),
        source_file=str(source_file or ""),
        avg_speed=round(avg_speed, 3),
        avg_pos_speed=round(float(pos.mean()) if len(pos) else 0.0, 3),
        avg_neg_speed=round(float(neg.mean()) if len(neg) else 0.0, 3),
        outliers_removed=outliers_removed,
        time_pos=round(time_pos, 2),
        time_neg=round(time_neg, 2),
        reversals=reversals,
        fwd_frames=fwd_frames,
        rev_frames=rev_frames,
        total_frames=total_frames,
        pct_pos=round(pct_pos, 1),
        pct_neg=round(pct_neg, 1),
        pct_captured=round(pct_captured, 1),
        active_frames=n_active,
        pct_active=round(pct_active, 1),
        active_reversals=active_reversals,
        min_active_reversals_per_minute=int(min_ar_min),
        max_active_reversals_per_minute=int(max_ar_min),
        n_minute_bins=int(n_min_bins),
        active_reversals_reorient=reorient["active_reversals_reorient"],
        active_reversals_reorient_scored=reorient["active_reversals_reorient_scored"],
        pct_active_reversals_reorient=reorient["pct_active_reversals_reorient"],
        reorient_per_reversal_rate=reorient["reorient_per_reversal_rate"],
        reorientation_angle_deg=float(REORIENTATION_ANGLE_DEG),
        reorientation_window_s=float(REORIENTATION_WINDOW_S),
        reorientation_fwd_search_s=float(REORIENTATION_FWD_SEARCH_S),
        reorientation_fwd_best_s=float(REORIENTATION_FWD_BEST_S),
        active_time_pos=round(active_time_pos, 2),
        active_time_neg=round(active_time_neg, 2),
        active_pct_pos=round(active_pct_pos, 1),
        active_pct_neg=round(active_pct_neg, 1),
        active_fwd_bouts=n_fwd_bouts,
        active_rev_bouts=n_rev_bouts,
        avg_active_fwd_bout_frames=round(avg_fwd_frames, 2),
        avg_active_rev_bout_frames=round(avg_rev_frames, 2),
        avg_active_fwd_bout_s=round(avg_fwd_s, 3),
        avg_active_rev_bout_s=round(avg_rev_s, 3),
        duration_min=round(duration_min, 3),
        reversals_per_min=per_min(reversals),
        active_reversals_per_min=per_min(active_reversals),
        active_time_pos_per_min=per_min(active_time_pos),
        active_time_neg_per_min=per_min(active_time_neg),
        duration_s=round(duration_s, 2),
        active_duration_s=round(active_duration_s, 2),
        active_duration_min=round(active_duration_min, 3),
        frames_per_minute=frames_per_minute,
        fps_true=round(float(fps_true), 6),
        centroid_threshold_bl_s=float(centroid_threshold_bl_s),
        centroid_window_s=float(centroid_window_s),
        use_reversal_sustain_filter=bool(use_reversal_sustain_filter),
        reversal_dwell_frames=int(reversal_dwell_frames),
        reversal_min_neg_frames=int(reversal_min_neg_frames),
        reversal_min_pos_frames=int(reversal_min_pos_frames),
        blob_perimeter_median=round(float(peri_median), 3)
        if np.isfinite(peri_median)
        else float("nan"),
    )

    per_minute = _per_minute_table(
        times,
        speeds,
        active,
        frames_per_minute=frames_per_minute,
        reversal_flags=rev_flags_for_count,
        active_reversal_flags=active_rev_flags,
    )
    return MetricsResult(
        metrics=metrics,
        frames=frames,
        valid=valid,
        per_minute=per_minute,
        bouts=bouts,
        perimeter=perimeter,
    )


def apply_activity_filter(frames: pd.DataFrame) -> pd.DataFrame:
    """Keep only frames where the activity flag is 1."""
    if "active" not in frames.columns:
        raise ValueError("Frame table has no 'active' column")
    return frames.loc[frames["active"] == 1].copy()


def _frame_spacing(times: np.ndarray) -> float:
    """Typical inter-sample Δt used to convert time gaps into frame counts."""
    dt = _sample_interval(times)
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return float(dt)


def _continuous_finite_speed_segments(
    times: np.ndarray,
    speeds: np.ndarray,
    *,
    max_gap_frames: int = CONTINUOUS_SPEED_GAP_FRAMES,
) -> List[np.ndarray]:
    """
    Split finite-speed sample indices into near-continuous stretches.

    Consecutive finite-speed samples remain in the same stretch when the
    time gap between them is ≤ ``max_gap_frames`` × typical sample interval.
    """
    times = np.asarray(times, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    idx = np.where(np.isfinite(times) & np.isfinite(speeds))[0]
    if len(idx) == 0:
        return []
    dt = _frame_spacing(times[idx])
    if not np.isfinite(dt) or dt <= 0:
        return [idx]
    max_dt = float(max_gap_frames) * dt
    segments: List[List[int]] = [[int(idx[0])]]
    for prev, cur in zip(idx[:-1], idx[1:]):
        gap = float(times[cur] - times[prev])
        if gap > max_dt + 1e-12:
            segments.append([int(cur)])
        else:
            segments[-1].append(int(cur))
    return [np.asarray(seg, dtype=int) for seg in segments if len(seg)]


def extract_continuous_minute_slices(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    speed_col: str = "speed",
    frames_per_minute: Optional[int] = None,
    max_gap_frames: int = CONTINUOUS_SPEED_GAP_FRAMES,
    require_siblings: bool = True,
    use_backfill: Optional[bool] = None,
) -> List[Dict]:
    """
    Carve continuous minutes from one worm DataFrame.

    By default windows are non-overlapping. When ``use_backfill`` is True
    (default: ``USE_CONTINUOUS_MINUTE_BACKFILL``), if a stretch has at least
    one full window and a leftover of ≥ half a window, an additional
    overlapping window is taken from the end of the stretch (backfilling
    into the previous minute) so that leftover data is not discarded.

    Window length defaults to ``df['frames_per_minute']`` (true FPS × 60)
    when present.

    Continuity and ``time_start`` / ``time_end`` use video ``frame`` indices
    converted with ``fps_true`` (never raw export timestamps).

    Only ``*_fixed.csv`` recordings that have every required sibling feature
    file on disk are eligible when ``require_siblings`` is True.

    Returns a list of dicts with keys:
      slice_df, recording_date, video_number, source_worm_id, source_file,
      minute_in_source, time_start, time_end, n_frames, backfilled
    """
    from .io import (
        has_continuous_minute_siblings,
        parse_recording_metadata,
    )
    from .video_fps import ASSUMED_FPS, recover_frame_index

    if use_backfill is None:
        use_backfill = USE_CONTINUOUS_MINUTE_BACKFILL
    if frames_per_minute is None:
        if "frames_per_minute" in df.columns and len(df):
            try:
                frames_per_minute = int(df["frames_per_minute"].iloc[0])
            except Exception:
                frames_per_minute = FRAMES_PER_MINUTE
        else:
            frames_per_minute = FRAMES_PER_MINUTE

    if time_col not in df.columns or speed_col not in df.columns:
        return []

    source_file = ""
    if "source" in df.columns and len(df):
        source_file = str(df["source"].iloc[0] or "")
    if require_siblings:
        if not source_file or not has_continuous_minute_siblings(source_file):
            return []

    meta = parse_recording_metadata(source_file) if source_file else {
        "recording_date": "",
        "video_number": "",
    }
    if "recording_date" in df.columns and len(df) and str(df["recording_date"].iloc[0]):
        meta["recording_date"] = str(df["recording_date"].iloc[0])
    if "video_number" in df.columns and len(df) and str(df["video_number"].iloc[0]):
        meta["video_number"] = str(df["video_number"].iloc[0])

    source_worm_id = (
        str(df["worm_id"].iloc[0]) if "worm_id" in df.columns and len(df) else ""
    )

    work = df.copy()
    t_raw = pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float)
    existing = None
    if "frame" in work.columns:
        existing = pd.to_numeric(work["frame"], errors="coerce").to_numpy(dtype=float)
    frames = recover_frame_index(
        t_raw, assumed_fps=ASSUMED_FPS, existing_frame=existing
    )
    work["frame"] = frames
    fps_true = float("nan")
    if "fps_true" in work.columns and len(work):
        try:
            fps_true = float(work["fps_true"].iloc[0])
        except Exception:
            fps_true = float("nan")
    if not np.isfinite(fps_true) or fps_true <= 0:
        fps_true = float(frames_per_minute) / 60.0
    # True seconds for continuity checks and minute metadata.
    work["_t_true"] = frames / float(fps_true)
    work = work.sort_values("frame", kind="mergesort").reset_index(drop=True)
    times = pd.to_numeric(work["_t_true"], errors="coerce").to_numpy(dtype=float)
    speeds = pd.to_numeric(work[speed_col], errors="coerce").to_numpy(dtype=float)
    segments = _continuous_finite_speed_segments(
        times, speeds, max_gap_frames=int(max_gap_frames)
    )

    fpm = max(1, int(frames_per_minute))
    min_leftover = fpm // 2  # half-window threshold for backfill
    out: List[Dict] = []
    minute_in_source = 0

    def _append_slice(rows: np.ndarray, *, backfilled: bool) -> None:
        nonlocal minute_in_source
        slice_df = work.iloc[rows].drop(columns=["_t_true"], errors="ignore")
        slice_df = slice_df.copy().reset_index(drop=True)
        t0 = float(times[rows[0]])
        t1 = float(times[rows[-1]])
        minute_in_source += 1
        out.append(
            {
                "slice_df": slice_df,
                "recording_date": meta.get("recording_date", ""),
                "video_number": meta.get("video_number", ""),
                "source_worm_id": source_worm_id,
                "source_file": source_file,
                "minute_in_source": minute_in_source,
                "time_start": t0,
                "time_end": t1,
                "n_frames": int(len(rows)),
                "backfilled": bool(backfilled),
            }
        )

    for seg in segments:
        n_seg = len(seg)
        n_full = n_seg // fpm
        leftover = n_seg % fpm
        for k in range(n_full):
            _append_slice(seg[k * fpm : (k + 1) * fpm], backfilled=False)
        if (
            bool(use_backfill)
            and n_full >= 1
            and leftover >= min_leftover
            and n_seg >= fpm
        ):
            # Overlapping end window: last fpm samples of the stretch.
            _append_slice(seg[n_seg - fpm : n_seg], backfilled=True)
    return out


def compute_continuous_minute_metrics(
    worms: Union[Dict[str, pd.DataFrame], Sequence[pd.DataFrame]],
    *,
    frames_per_minute: Optional[int] = None,
    max_gap_frames: int = CONTINUOUS_SPEED_GAP_FRAMES,
    require_siblings: bool = True,
    use_backfill: Optional[bool] = None,
    **compute_kwargs,
) -> Tuple[List[MetricsResult], Dict[str, pd.DataFrame]]:
    """
    Compute locomotion metrics for every continuous minute across a dataset.

    Eligible recordings are fixed speed CSVs with a ``*_blob_perimeter.csv``
    sibling (see ``io.CONTINUOUS_MINUTE_REQUIRED_FEATURES``). Minutes are worm-agnostic units labeled ``m00001``, …
    with ``recording_date`` and ``video_number`` taken from the CSV name.

    Window length uses each worm's ``frames_per_minute`` (true FPS × 60)
    unless ``frames_per_minute`` is passed explicitly.

    When ``use_backfill`` / ``USE_CONTINUOUS_MINUTE_BACKFILL`` is on, stretch
    leftovers ≥ half a minute become overlapping end windows.

    Returns ``(results, tables)`` where ``tables`` matches
    ``build_summary_tables`` keys plus ``minute_meta``.
    """
    if use_backfill is None:
        use_backfill = USE_CONTINUOUS_MINUTE_BACKFILL
    if isinstance(worms, dict):
        dfs: List[pd.DataFrame] = list(worms.values())
    else:
        dfs = list(worms)

    results: List[MetricsResult] = []
    meta_rows: List[Dict] = []
    minute_counter = 0

    if require_siblings:
        from .io import missing_continuous_minute_siblings

        skipped = []
        for df in dfs:
            src = str(df["source"].iloc[0] or "") if "source" in df.columns and len(df) else ""
            missing = missing_continuous_minute_siblings(src) if src else ["source file"]
            if missing:
                skipped.append(f"{Path(src).name or '(unknown)'} (missing {', '.join(missing)})")
        if skipped:
            warnings.warn(
                f"{len(skipped)} recording(s) contribute no continuous minutes because "
                "a required sibling CSV is missing: " + "; ".join(skipped),
                stacklevel=2,
            )

    for df in dfs:
        if frames_per_minute is not None:
            fpm = int(frames_per_minute)
        elif "frames_per_minute" in df.columns and len(df):
            try:
                fpm = int(df["frames_per_minute"].iloc[0])
            except Exception:
                fpm = FRAMES_PER_MINUTE
        else:
            fpm = FRAMES_PER_MINUTE
        slices = extract_continuous_minute_slices(
            df,
            frames_per_minute=fpm,
            max_gap_frames=max_gap_frames,
            require_siblings=require_siblings,
            use_backfill=bool(use_backfill),
        )
        for item in slices:
            minute_counter += 1
            minute_id = f"m{minute_counter:05d}"
            slice_df = item["slice_df"].copy()
            # Keep provenance columns for compute_metrics metadata.
            slice_df["recording_date"] = item["recording_date"]
            slice_df["video_number"] = item["video_number"]
            if item["source_file"]:
                slice_df["source"] = item["source_file"]
            # Avoid treating source worm id as the analysis unit.
            slice_df["worm_id"] = minute_id
            slice_df["frames_per_minute"] = fpm
            if "fps_true" in df.columns and len(df):
                slice_df["fps_true"] = df["fps_true"].iloc[0]

            res = compute_metrics(
                slice_df,
                worm_id=minute_id,
                frames_per_minute=fpm,
                **compute_kwargs,
            )
            res.metrics.recording_date = str(item["recording_date"] or "")
            res.metrics.video_number = str(item["video_number"] or "")
            res.metrics.source_worm_id = str(item["source_worm_id"] or "")
            res.metrics.source_file = str(item["source_file"] or "")
            res.metrics.minute_index = int(minute_counter)
            results.append(res)
            meta_rows.append(
                {
                    "minute_id": minute_id,
                    "minute_index": minute_counter,
                    "recording_date": item["recording_date"],
                    "video_number": item["video_number"],
                    "source_worm_id": item["source_worm_id"],
                    "source_file": item["source_file"],
                    "minute_in_source": item["minute_in_source"],
                    "time_start": item["time_start"],
                    "time_end": item["time_end"],
                    "n_frames": item["n_frames"],
                    "frames_per_minute": fpm,
                    "backfilled": bool(item.get("backfilled", False)),
                    "blob_perimeter_median": res.metrics.blob_perimeter_median,
                }
            )

    tables = build_summary_tables(results) if results else {
        "general": pd.DataFrame(),
        "active": pd.DataFrame(),
        "general_per_duration": pd.DataFrame(),
        "active_per_duration": pd.DataFrame(),
        "active_per_active_duration": pd.DataFrame(),
    }
    minute_meta = pd.DataFrame(meta_rows)
    if not minute_meta.empty:
        minute_meta = minute_meta.sort_values(
            ["recording_date", "video_number", "minute_index"]
        ).reset_index(drop=True)
        meta_join = minute_meta.drop(columns=["blob_perimeter_median"], errors="ignore")
        for key, df_tbl in list(tables.items()):
            if df_tbl.empty or "worm_id" not in df_tbl.columns:
                continue
            tbl = df_tbl.drop(
                columns=["recording_date", "video_number"], errors="ignore"
            ).rename(columns={"worm_id": "minute_id"})
            tables[key] = meta_join.merge(tbl, on="minute_id", how="left")
    tables["minute_meta"] = minute_meta
    return results, tables


def summarize_worms(results: Sequence[MetricsResult]) -> pd.DataFrame:
    """Full flat metrics table (all fields). Prefer ``build_summary_tables``."""
    return pd.DataFrame([r.metrics.to_dict() for r in results])


def _rate(value: float, duration: float, ndigits: int = 3) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    return round(value / duration, ndigits) if duration > 0 else 0.0


def _reorient_metrics(active_reversals: int, n_gt: int, n_scored: int) -> dict:
    """
    Reorientation summaries for one worm or minute.

    Units with no active reversals are excluded (all values NaN), including
    counts, so they do not enter means of reorient metrics. Percentage uses
    scored events only (NaN if none were scored). ``reorient_per_reversal_rate``
    is (reorienting reversals per active min) / (active reversals per active
    min), which equals n_reorient / n_active_reversals.
    """
    nan = float("nan")
    n_rev = int(active_reversals or 0)
    if n_rev <= 0:
        return {
            "active_reversals_reorient": nan,
            "active_reversals_reorient_scored": nan,
            "pct_active_reversals_reorient": nan,
            "reorient_per_reversal_rate": nan,
        }
    n_gt = int(n_gt or 0)
    n_scored = int(n_scored or 0)
    return {
        "active_reversals_reorient": float(n_gt),
        "active_reversals_reorient_scored": float(n_scored),
        "pct_active_reversals_reorient": (
            round(100.0 * n_gt / n_scored, 1) if n_scored else nan
        ),
        "reorient_per_reversal_rate": round(float(n_gt) / float(n_rev), 4),
    }


def _reorient_table_value(m: WormMetrics, attr: str) -> float:
    """NaN for never-reversing units so summary tables match compute_metrics."""
    if int(getattr(m, "active_reversals", 0) or 0) <= 0:
        return float("nan")
    if attr == "reorient_per_reversal_rate":
        stored = getattr(m, "reorient_per_reversal_rate", None)
        if stored is not None and np.isfinite(float(stored)):
            return float(stored)
        n_gt = float(getattr(m, "active_reversals_reorient", 0) or 0)
        n_rev = float(m.active_reversals)
        return round(n_gt / n_rev, 4) if n_rev else float("nan")
    val = getattr(m, attr, float("nan"))
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def build_summary_tables(
    results: Sequence[MetricsResult],
) -> Dict[str, pd.DataFrame]:
    """
    Split metrics into five tables:

    - ``general`` — unfiltered locomotion metrics
    - ``active`` — activity-filtered metrics
    - ``general_per_duration`` — general counts/times ÷ captured-speed duration
    - ``active_per_duration`` — active counts/times ÷ captured-speed duration
    - ``active_per_active_duration`` — active counts/times ÷ active duration (per min)
    """
    rows = [r.metrics for r in results]

    general = pd.DataFrame(
        [
            {
                "worm_id": m.worm_id,
                "recording_date": m.recording_date,
                "video_number": m.video_number,
                "fps_true": m.fps_true,
                "frames_per_minute": m.frames_per_minute,
                "duration_s": m.duration_s,
                "duration_min": m.duration_min,
                "total_frames": m.total_frames,
                "pct_captured": m.pct_captured,
                "outliers_removed": m.outliers_removed,
                "avg_speed": m.avg_speed,
                "avg_pos_speed": m.avg_pos_speed,
                "avg_neg_speed": m.avg_neg_speed,
                "reversals": m.reversals,
                "blob_perimeter_median": m.blob_perimeter_median,
                "fwd_frames": m.fwd_frames,
                "rev_frames": m.rev_frames,
                "pct_pos": m.pct_pos,
                "pct_neg": m.pct_neg,
                "time_pos": m.time_pos,
                "time_neg": m.time_neg,
            }
            for m in rows
        ]
    )

    active = pd.DataFrame(
        [
            {
                "worm_id": m.worm_id,
                "active_duration_s": m.active_duration_s,
                "active_duration_min": m.active_duration_min,
                "active_frames": m.active_frames,
                "pct_active": m.pct_active,
                "centroid_threshold_bl_s": m.centroid_threshold_bl_s,
                "centroid_window_s": m.centroid_window_s,
                "use_reversal_sustain_filter": m.use_reversal_sustain_filter,
                "reversal_dwell_frames": m.reversal_dwell_frames,
                "reversal_min_neg_frames": m.reversal_min_neg_frames,
                "reversal_min_pos_frames": m.reversal_min_pos_frames,
                "active_reversals": m.active_reversals,
                "min_active_reversals_per_minute": getattr(
                    m, "min_active_reversals_per_minute", 0
                ),
                "max_active_reversals_per_minute": getattr(
                    m, "max_active_reversals_per_minute", 0
                ),
                "n_minute_bins": getattr(m, "n_minute_bins", 0),
                "active_reversals_reorient": _reorient_table_value(
                    m, "active_reversals_reorient"
                ),
                "active_reversals_reorient_scored": _reorient_table_value(
                    m, "active_reversals_reorient_scored"
                ),
                "pct_active_reversals_reorient": _reorient_table_value(
                    m, "pct_active_reversals_reorient"
                ),
                "reorient_per_reversal_rate": _reorient_table_value(
                    m, "reorient_per_reversal_rate"
                ),
                "reorientation_angle_deg": getattr(
                    m, "reorientation_angle_deg", REORIENTATION_ANGLE_DEG
                ),
                "active_pct_pos": m.active_pct_pos,
                "active_pct_neg": m.active_pct_neg,
                "active_time_pos": m.active_time_pos,
                "active_time_neg": m.active_time_neg,
                "active_fwd_bouts": m.active_fwd_bouts,
                "active_rev_bouts": m.active_rev_bouts,
                "avg_active_fwd_bout_frames": m.avg_active_fwd_bout_frames,
                "avg_active_rev_bout_frames": m.avg_active_rev_bout_frames,
                "avg_active_fwd_bout_s": m.avg_active_fwd_bout_s,
                "avg_active_rev_bout_s": m.avg_active_rev_bout_s,
            }
            for m in rows
        ]
    )

    general_per_duration = pd.DataFrame(
        [
            {
                "worm_id": m.worm_id,
                "duration_min": m.duration_min,
                "reversals_per_min": _rate(m.reversals, m.duration_min),
                "fwd_frames_per_min": _rate(m.fwd_frames, m.duration_min),
                "rev_frames_per_min": _rate(m.rev_frames, m.duration_min),
                "time_pos_per_min": _rate(m.time_pos, m.duration_min),
                "time_neg_per_min": _rate(m.time_neg, m.duration_min),
                "pct_pos": m.pct_pos,
                "pct_neg": m.pct_neg,
            }
            for m in rows
        ]
    )

    active_per_duration = pd.DataFrame(
        [
            {
                "worm_id": m.worm_id,
                "duration_min": m.duration_min,
                "active_duration_min": m.active_duration_min,
                "active_frames_per_min": _rate(
                    m.active_frames, m.duration_min
                ),
                "active_duration_s_per_min": _rate(
                    m.active_duration_s, m.duration_min
                ),
                "active_reversals_per_min": _rate(
                    m.active_reversals, m.duration_min
                ),
                "active_fwd_bouts_per_min": _rate(
                    m.active_fwd_bouts, m.duration_min
                ),
                "active_rev_bouts_per_min": _rate(
                    m.active_rev_bouts, m.duration_min
                ),
                "active_time_pos_per_min": _rate(
                    m.active_time_pos, m.duration_min
                ),
                "active_time_neg_per_min": _rate(
                    m.active_time_neg, m.duration_min
                ),
                "pct_active": m.pct_active,
                "active_pct_pos": m.active_pct_pos,
                "active_pct_neg": m.active_pct_neg,
                "avg_active_fwd_bout_frames": m.avg_active_fwd_bout_frames,
                "avg_active_rev_bout_frames": m.avg_active_rev_bout_frames,
                "avg_active_fwd_bout_s": m.avg_active_fwd_bout_s,
                "avg_active_rev_bout_s": m.avg_active_rev_bout_s,
            }
            for m in rows
        ]
    )

    active_per_active_duration = pd.DataFrame(
        [
            {
                "worm_id": m.worm_id,
                "active_duration_min": m.active_duration_min,
                "active_reversals_per_active_min": _rate(
                    m.active_reversals, m.active_duration_min
                ),
                "active_reversals_reorient_per_active_min": _rate(
                    _reorient_table_value(m, "active_reversals_reorient"),
                    m.active_duration_min,
                ),
                "reorient_per_reversal_rate": _reorient_table_value(
                    m, "reorient_per_reversal_rate"
                ),
                "active_fwd_bouts_per_active_min": _rate(
                    m.active_fwd_bouts, m.active_duration_min
                ),
                "active_rev_bouts_per_active_min": _rate(
                    m.active_rev_bouts, m.active_duration_min
                ),
                "active_time_pos_per_active_min": _rate(
                    m.active_time_pos, m.active_duration_min
                ),
                "active_time_neg_per_active_min": _rate(
                    m.active_time_neg, m.active_duration_min
                ),
                "active_pct_pos": m.active_pct_pos,
                "active_pct_neg": m.active_pct_neg,
                "avg_active_fwd_bout_s": m.avg_active_fwd_bout_s,
                "avg_active_rev_bout_s": m.avg_active_rev_bout_s,
            }
            for m in rows
        ]
    )

    def _sort(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "worm_id" not in df.columns:
            return df.reset_index(drop=True)
        return df.sort_values("worm_id").reset_index(drop=True)

    return {
        "general": _sort(general),
        "active": _sort(active),
        "general_per_duration": _sort(general_per_duration),
        "active_per_duration": _sort(active_per_duration),
        "active_per_active_duration": _sort(active_per_active_duration),
    }

