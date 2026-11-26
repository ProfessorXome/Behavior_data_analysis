"""
Per-video true FPS estimation and lookup.

Videos are treated as exactly ``VIDEO_DURATION_S`` seconds long. For old
Tierpsy exports timed at ``ASSUMED_FPS``, frame indices are recovered as
``round(time * ASSUMED_FPS)``. For native frame-index CSVs (including those
that store frame index in a ``time`` column with step ≈ 1), frames are used
directly. Across all worms in a video, the span of blob_perimeter samples
(preferred over speed) determines ``n_frames``.

  fps_true = n_frames / VIDEO_DURATION_S
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .io import (
    _find_feature_csv,
    _speed_file_base,
    parse_recording_metadata,
)

PathLike = Union[str, Path]

VIDEO_DURATION_S = 600.0  # every recording is exactly 10 minutes
ASSUMED_FPS = 6.0  # integer FPS used when old exports were timed in seconds

# Per-strain CSV folders are discovered under <data root>/Tierpsy_csvs/
# (see worm_analysis.tierpsy_data.strain_csv_dirs).

DEFAULT_FPS_TABLE_NAME = "video_fps.csv"


def video_key(recording_date: str, video_number: str, dataset: str = "") -> str:
    """Stable lookup key: dataset|date|video (dataset optional)."""
    date = str(recording_date or "").strip()
    vid = str(video_number or "").strip()
    ds = str(dataset or "").strip()
    if ds:
        return f"{ds}|{date}|{vid}"
    return f"{date}|{vid}"


def _values_look_like_frames(values: np.ndarray) -> bool:
    """
    True when values behave like integer frame indices (median step ≈ 1).

    Used to distinguish native frame-index exports (time = 0,1,2,…) from
    old second-based exports (time = n/ASSUMED_FPS).
    """
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) < 3:
        return False
    # Prefer consecutive-sample spacing in file order.
    d = np.diff(finite)
    d = d[d > 0]
    if len(d) == 0:
        return False
    med = float(np.median(d))
    if abs(med - 1.0) > 0.05:
        return False
    # Mostly integer-valued.
    frac = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-6)))
    return frac >= 0.9


def recover_frame_index(
    time_vals: np.ndarray,
    *,
    assumed_fps: float = ASSUMED_FPS,
    existing_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Integer video frame index from an export timeline.

    - Frame-indexed ``time`` (step ≈ 1) → ``round(time)``
    - Otherwise old seconds clock → ``round(time * assumed_fps)``
    - Explicit ``existing_frame`` is used only when ``time`` is not
      frame-indexed (so a bad frame rebuild cannot override a frame export).
    """
    t = np.asarray(time_vals, dtype=float)
    if _values_look_like_frames(t):
        out = np.round(t)
        out[~np.isfinite(t)] = np.nan
        return out
    if existing_frame is not None:
        ef = np.asarray(existing_frame, dtype=float)
        if len(ef) == len(t) and np.isfinite(ef).any():
            # Trust existing frames when they look like a unit-step index.
            if _values_look_like_frames(ef):
                return ef
    out = np.round(t * float(assumed_fps))
    out[~np.isfinite(t)] = np.nan
    return out


def _read_timeline_csv(path: Path) -> Tuple[str, np.ndarray]:
    """
    Return (source, values) where source is 'frame' or 'time'.

    Prefers an explicit frame column. A ``time`` column whose values look
    like frame indices is treated as ``frame`` (native frame-index exports).
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "frame" in cols:
        vals = pd.to_numeric(df[cols["frame"]], errors="coerce").to_numpy(dtype=float)
        return "frame", vals
    if "time" in cols:
        vals = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(dtype=float)
        if _values_look_like_frames(vals):
            return "frame", vals
        return "time", vals
    # single non-index numeric column fallback
    numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) or c.lower() in ("time", "frame", "t")
    ]
    if not numeric:
        raise ValueError(f"No time/frame column in {path}")
    vals = pd.to_numeric(df[numeric[0]], errors="coerce").to_numpy(dtype=float)
    if _values_look_like_frames(vals):
        return "frame", vals
    return "time", vals


def n_frames_from_values(
    values: np.ndarray,
    *,
    source: str,
    assumed_fps: float = ASSUMED_FPS,
) -> int:
    """Inclusive frame span from a time or frame timeline."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if source == "frame":
        n0 = int(np.round(lo))
        n1 = int(np.round(hi))
    else:
        n0 = int(np.round(lo * float(assumed_fps)))
        n1 = int(np.round(hi * float(assumed_fps)))
    return max(0, n1 - n0 + 1)


def perimeter_path_for_speed(speed_path: PathLike) -> Optional[Path]:
    """Sibling blob_perimeter CSV for a fixed speed file, if present."""
    return _find_feature_csv(Path(speed_path), "blob_perimeter")


def max_n_frames_for_video(
    speed_paths: Sequence[PathLike],
    *,
    assumed_fps: float = ASSUMED_FPS,
) -> Tuple[int, str, int]:
    """
    Estimate n_frames for one video from perimeter timelines of its worms.

    Returns (n_frames, source, n_worms_used).
    Prefers perimeter CSVs; falls back to the speed CSV timeline if needed.
    """
    best_n = 0
    source_used = ""
    n_used = 0
    for sp in speed_paths:
        sp = Path(sp)
        peri = perimeter_path_for_speed(sp)
        used_peri = peri is not None and peri.exists()
        path = peri if used_peri else sp
        try:
            source, vals = _read_timeline_csv(path)
        except Exception:
            continue
        n = n_frames_from_values(vals, source=source, assumed_fps=assumed_fps)
        if n <= 0:
            continue
        n_used += 1
        if n > best_n:
            best_n = n
            source_used = source if used_peri else f"{source}_from_speed"
    return best_n, source_used, n_used


def _clean_dataset_label(data_dir: Path, explicit: str = "") -> str:
    if explicit:
        return str(explicit)
    for p in reversed(data_dir.resolve().parts):
        if re.match(r"^N2", p, flags=re.IGNORECASE):
            return "N2"
        m = re.match(r"^(HML\d+|VM\d+)", p, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return data_dir.name


def group_speed_files_by_video(
    data_dir: PathLike,
    *,
    pattern: str = "*_fixed.csv",
) -> Dict[Tuple[str, str], List[Path]]:
    """Map (recording_date, video_number) → list of fixed speed paths."""
    data_dir = Path(data_dir)
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for path in sorted(data_dir.glob(pattern)):
        meta = parse_recording_metadata(path)
        date = meta.get("recording_date", "")
        vid = meta.get("video_number", "")
        if not date or not vid:
            continue
        groups.setdefault((date, vid), []).append(path)
    return groups


def estimate_video_fps_rows(
    data_dir: PathLike,
    *,
    dataset: str = "",
    pattern: str = "*_fixed.csv",
    assumed_fps: float = ASSUMED_FPS,
    duration_s: float = VIDEO_DURATION_S,
) -> pd.DataFrame:
    """Build one-row-per-video FPS estimates for a data directory."""
    data_dir = Path(data_dir)
    ds = _clean_dataset_label(data_dir, dataset)
    groups = group_speed_files_by_video(data_dir, pattern=pattern)
    rows = []
    for (date, vid), paths in sorted(groups.items()):
        n_frames, source, n_used = max_n_frames_for_video(
            paths, assumed_fps=assumed_fps
        )
        fps_true = (n_frames / float(duration_s)) if n_frames > 0 else float("nan")
        rows.append(
            {
                "dataset": ds,
                "recording_date": date,
                "video_number": vid,
                "video_key": video_key(date, vid, ds),
                "data_dir": str(data_dir.resolve()),
                "assumed_fps": float(assumed_fps),
                "duration_s": float(duration_s),
                "n_frames": int(n_frames),
                "fps_true": round(float(fps_true), 6) if np.isfinite(fps_true) else float("nan"),
                "frames_per_minute": int(round(fps_true * 60.0))
                if np.isfinite(fps_true)
                else 0,
                "source": source or "none",
                "n_worms_used": int(n_used),
                "n_speed_files": int(len(paths)),
            }
        )
    return pd.DataFrame(rows)


def build_video_fps_table(
    dataset_dirs: Sequence[Tuple[str, PathLike]],
    *,
    pattern: str = "*_fixed.csv",
    assumed_fps: float = ASSUMED_FPS,
    duration_s: float = VIDEO_DURATION_S,
) -> pd.DataFrame:
    """Concatenate FPS rows for many (dataset_label, directory) pairs."""
    frames = [
        estimate_video_fps_rows(
            path,
            dataset=label,
            pattern=pattern,
            assumed_fps=assumed_fps,
            duration_s=duration_s,
        )
        for label, path in dataset_dirs
    ]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["dataset", "recording_date", "video_number"]).reset_index(
        drop=True
    )


def _normalize_date(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        return s.zfill(6)
    return s


def _normalize_video_number(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        return s.zfill(2)
    return s


def load_video_fps_table(path: PathLike) -> pd.DataFrame:
    """Load a previously written video_fps.csv."""
    path = Path(path)
    df = pd.read_csv(path, dtype={"recording_date": str, "video_number": str, "dataset": str})
    required = {"recording_date", "video_number", "fps_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df = df.copy()
    df["recording_date"] = df["recording_date"].map(_normalize_date)
    df["video_number"] = df["video_number"].map(_normalize_video_number)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str)
    if "video_key" not in df.columns:
        if "dataset" in df.columns:
            df["video_key"] = [
                video_key(r, v, d)
                for d, r, v in zip(
                    df["dataset"], df["recording_date"], df["video_number"]
                )
            ]
        else:
            df["video_key"] = [
                video_key(r, v) for r, v in zip(df["recording_date"], df["video_number"])
            ]
    else:
        # rebuild keys with normalized ids
        if "dataset" in df.columns:
            df["video_key"] = [
                video_key(r, v, d)
                for d, r, v in zip(
                    df["dataset"], df["recording_date"], df["video_number"]
                )
            ]
    if "frames_per_minute" not in df.columns:
        df["frames_per_minute"] = [
            int(round(float(f) * 60.0)) if np.isfinite(f) else 0
            for f in df["fps_true"]
        ]
    return df


def lookup_video_fps(
    fps_table: pd.DataFrame,
    *,
    recording_date: str = "",
    video_number: str = "",
    dataset: str = "",
    source_path: Optional[PathLike] = None,
) -> Optional[pd.Series]:
    """
    Find the FPS row for a worm/video.

    Preference order:
      1. dataset + date + video
      2. date + video (unique)
    """
    if source_path is not None:
        meta = parse_recording_metadata(source_path)
        recording_date = recording_date or meta.get("recording_date", "")
        video_number = video_number or meta.get("video_number", "")
        if not dataset:
            dataset = _clean_dataset_label(Path(source_path).parent)

    date = _normalize_date(recording_date or "")
    vid = _normalize_video_number(video_number or "")
    ds = str(dataset or "")
    if fps_table is None or fps_table.empty or not date or not vid:
        return None

    tbl = fps_table
    if ds and "dataset" in tbl.columns:
        hit = tbl[
            (tbl["dataset"].astype(str) == ds)
            & (tbl["recording_date"].astype(str) == date)
            & (tbl["video_number"].astype(str) == vid)
        ]
        if len(hit):
            return hit.iloc[0]

    hit = tbl[
        (tbl["recording_date"].astype(str) == date)
        & (tbl["video_number"].astype(str) == vid)
    ]
    if len(hit) == 1:
        return hit.iloc[0]
    if len(hit) > 1 and ds:
        return None
    if len(hit) > 1:
        return hit.iloc[0]
    return None


def attach_video_fps(
    df: pd.DataFrame,
    fps_table: Optional[pd.DataFrame],
    *,
    dataset: str = "",
    fallback_fps: float = ASSUMED_FPS,
) -> pd.DataFrame:
    """
    Attach ``fps_true`` and ``frames_per_minute`` columns to a worm DataFrame.

    Uses recording_date / video_number / source on ``df`` when present.
    """
    out = df.copy()
    source = str(out["source"].iloc[0]) if "source" in out.columns and len(out) else ""
    date = (
        str(out["recording_date"].iloc[0])
        if "recording_date" in out.columns and len(out)
        else ""
    )
    vid = (
        str(out["video_number"].iloc[0])
        if "video_number" in out.columns and len(out)
        else ""
    )
    row = lookup_video_fps(
        fps_table,
        recording_date=date,
        video_number=vid,
        dataset=dataset,
        source_path=source or None,
    )
    if row is not None and np.isfinite(float(row["fps_true"])):
        fps = float(row["fps_true"])
        fpm = int(row["frames_per_minute"]) if "frames_per_minute" in row.index else int(
            round(fps * 60.0)
        )
    else:
        fps = float(fallback_fps)
        fpm = int(round(fps * 60.0))
    out["fps_true"] = fps
    out["frames_per_minute"] = fpm
    # Recover integer video frames. Raw export timestamps (assumed-6 clock or
    # frame-index disguised as time) are only used here; downstream metrics
    # must use frame + fps_true for any seconds / rates.
    if "time" in out.columns:
        t = pd.to_numeric(out["time"], errors="coerce").to_numpy(dtype=float)
        existing = None
        if "frame" in out.columns:
            existing = pd.to_numeric(out["frame"], errors="coerce").to_numpy(dtype=float)
        frames = recover_frame_index(t, assumed_fps=ASSUMED_FPS, existing_frame=existing)
        out["frame"] = pd.Series(frames, index=out.index).astype("Int64")
    return out
