"""Load worm speed CSVs and Excel flip-range sheets."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .metrics import MIN_FRAMES, MIN_FRAMES_FRAC

PathLike = Union[str, Path]

TIME_CANDIDATES = ("time", "timestamp", "time_s", "t")
SPEED_CANDIDATES = ("speed", "Speed", "velocity")
ANGULAR_CANDIDATES = (
    "angular_velocity_midbody",
    "angular_velocity",
    "midbody_angular_velocity",
)

# Sibling CSVs merged on ``time`` when present beside a *fixed.csv.
# Filenames: <stem>_<feature>.csv  e.g. ..._W8_blob_perimeter.csv
# (``angular_velocity_midbody`` is also merged when present; it is carried
# into the per-frame output but no metric uses it.) Any other sibling CSVs
# in the folder are ignored.
FEATURE_SIBLING_COLUMNS = ("blob_perimeter",)

# Sibling files a recording must have to contribute continuous minutes.
# blob_perimeter feeds the minute's ``blob_perimeter_median``.
CONTINUOUS_MINUTE_REQUIRED_FEATURES = ("blob_perimeter",)


def _pick_column(columns: List[str], candidates: Tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"Could not find a {label} column. Looked for {candidates}; "
        f"available: {list(columns)}"
    )


def load_speed_csv(
    path: PathLike,
    *,
    worm_id: Optional[str] = None,
    fps: Optional[float] = None,
    angular_path: Optional[PathLike] = None,
    fps_table: Optional[pd.DataFrame] = None,
    dataset: str = "",
    fallback_fps: float = 6.0,
) -> pd.DataFrame:
    """
    Load a single worm speed CSV (columns: time, speed).

    If ``angular_path`` is given (or discovered beside the speed file), merges
    midbody angular velocity on ``time``.

    When ``fps_table`` is provided (from ``video_fps.csv``), attaches
    ``fps_true`` / ``frames_per_minute`` for that worm's video and recovers
    an integer ``frame`` index from the export timeline (frame-index ``time``
    or old ``time * 6`` seconds clock). Downstream metrics use ``frame`` +
    ``fps_true`` for durations and rates — never raw timestamps.
    """
    path = Path(path)
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    # Native frame-index exports: synthesize a merge key compatible with siblings.
    if "time" not in cols_lower and "frame" in cols_lower:
        frame_vals = pd.to_numeric(df[cols_lower["frame"]], errors="coerce")
        out = pd.DataFrame(
            {
                "frame": frame_vals.round().astype("Int64"),
                # Placeholder on the assumed-6 clock so sibling CSVs timed the
                # old way still join; analysis uses frame + fps_true.
                "time": frame_vals / 6.0,
                "speed": pd.to_numeric(
                    df[_pick_column(list(df.columns), SPEED_CANDIDATES, "speed")],
                    errors="coerce",
                ),
            }
        )
    else:
        time_col = _pick_column(list(df.columns), TIME_CANDIDATES, "time")
        speed_col = _pick_column(list(df.columns), SPEED_CANDIDATES, "speed")
        out = pd.DataFrame(
            {
                "time": pd.to_numeric(df[time_col], errors="coerce"),
                "speed": pd.to_numeric(df[speed_col], errors="coerce"),
            }
        )
        if "frame" in cols_lower:
            out["frame"] = pd.to_numeric(df[cols_lower["frame"]], errors="coerce").round().astype(
                "Int64"
            )

    if worm_id is None:
        worm_id = _infer_worm_id(path.stem)
    out["worm_id"] = worm_id
    out["source"] = str(path)
    meta = parse_recording_metadata(path)
    out["recording_date"] = meta["recording_date"]
    out["video_number"] = meta["video_number"]

    ang_path = Path(angular_path) if angular_path is not None else _find_angular_csv(path)
    if ang_path is not None and ang_path.exists():
        out = merge_angular_velocity(out, ang_path)

    out = merge_feature_siblings(out, path)

    if fps_table is not None:
        from .video_fps import attach_video_fps

        out = attach_video_fps(
            out,
            fps_table,
            dataset=dataset,
            fallback_fps=float(fps if fps is not None else fallback_fps),
        )
    else:
        from .video_fps import ASSUMED_FPS, recover_frame_index

        use_fps = float(fps) if fps is not None and fps > 0 else float(fallback_fps)
        out["fps_true"] = use_fps
        out["frames_per_minute"] = int(round(use_fps * 60.0))
        existing = (
            pd.to_numeric(out["frame"], errors="coerce").to_numpy(dtype=float)
            if "frame" in out.columns
            else None
        )
        frames = recover_frame_index(
            pd.to_numeric(out["time"], errors="coerce").to_numpy(dtype=float),
            assumed_fps=ASSUMED_FPS,
            existing_frame=existing,
        )
        out["frame"] = pd.Series(frames, index=out.index).astype("Int64")

    return out.reset_index(drop=True)


def _speed_file_base(speed_path: Path) -> str:
    """Strip trailing _fixed / _speed_fixed from a speed CSV stem."""
    stem = speed_path.stem
    return stem.replace("_speed_fixed", "").replace("_fixed", "")


def _find_angular_csv(speed_path: Path) -> Optional[Path]:
    """Look for <stem>_angular_velocity_midbody.csv next to a *fixed.csv speed file."""
    base = _speed_file_base(speed_path)
    candidate = speed_path.with_name(f"{base}_angular_velocity_midbody.csv")
    return candidate if candidate.exists() else None


def _find_feature_csv(speed_path: Path, feature: str) -> Optional[Path]:
    """Look for <stem>_<feature>.csv next to a speed file."""
    base = _speed_file_base(speed_path)
    candidate = speed_path.with_name(f"{base}_{feature}.csv")
    return candidate if candidate.exists() else None


def load_angular_velocity_csv(path: PathLike) -> pd.DataFrame:
    """Load midbody angular velocity CSV (time, angular_velocity_midbody)."""
    path = Path(path)
    df = pd.read_csv(path)
    time_col = _pick_column(list(df.columns), TIME_CANDIDATES, "time")
    ang_col = _pick_column(list(df.columns), ANGULAR_CANDIDATES, "angular velocity")
    return pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_col], errors="coerce"),
            "angular_velocity_midbody": pd.to_numeric(df[ang_col], errors="coerce"),
        }
    )


def load_feature_csv(path: PathLike, feature: str) -> pd.DataFrame:
    """Load a single-feature CSV (time, <feature>)."""
    path = Path(path)
    df = pd.read_csv(path)
    time_col = _pick_column(list(df.columns), TIME_CANDIDATES, "time")
    if feature in df.columns:
        feat_col = feature
    else:
        # allow files whose only non-time column is the feature
        others = [c for c in df.columns if c != time_col]
        if len(others) == 1:
            feat_col = others[0]
        else:
            raise ValueError(
                f"Could not find column {feature!r} in {path}; available: {list(df.columns)}"
            )
    return pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_col], errors="coerce"),
            feature: pd.to_numeric(df[feat_col], errors="coerce"),
        }
    )


def merge_angular_velocity(
    speed_df: pd.DataFrame,
    angular_path: PathLike,
) -> pd.DataFrame:
    """Left-join angular velocity onto speed on ``time``."""
    ang = load_angular_velocity_csv(angular_path)
    out = speed_df.drop(columns=["angular_velocity_midbody"], errors="ignore")
    return out.merge(ang, on="time", how="left")


def merge_feature_siblings(
    speed_df: pd.DataFrame,
    speed_path: PathLike,
    *,
    features: Tuple[str, ...] = FEATURE_SIBLING_COLUMNS,
) -> pd.DataFrame:
    """
    Left-join sibling per-feature CSVs onto ``speed_df`` on ``time``.

    Looks for ``<stem>_<feature>.csv`` beside the speed file. Missing files
    are skipped silently.
    """
    speed_path = Path(speed_path)
    out = speed_df
    for feature in features:
        if feature in out.columns:
            continue
        feat_path = _find_feature_csv(speed_path, feature)
        if feat_path is None:
            continue
        feat = load_feature_csv(feat_path, feature)
        out = out.drop(columns=[feature], errors="ignore").merge(
            feat, on="time", how="left"
        )
    return out


def _infer_worm_id(stem: str) -> str:
    """Best-effort worm id from filenames like 01_W3_speed_fixed or 1630_01_W10_fixed."""
    m = re.search(r"(W\d+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stem


def parse_recording_metadata(path: PathLike) -> Dict[str, str]:
    """
    Parse recording date and video number from a Tierpsy-style CSV name.

    Examples
    --------
    ``070926_N2_demag_01_TS_W10_speed_fixed.csv`` → date ``070926``, video ``01``
    ``071026_HML1641_01_TS_W10_fixed.csv`` → date ``071026``, video ``01``
    """
    stem = _speed_file_base(Path(path))
    date = ""
    m_date = re.match(r"^(\d{6})_", stem)
    if m_date:
        date = m_date.group(1)
    video = ""
    m_vid = re.search(r"_(\d+)_TS_", stem, flags=re.IGNORECASE)
    if m_vid:
        video = m_vid.group(1)
    else:
        m_vid = re.search(r"_(\d+)_W\d+", stem, flags=re.IGNORECASE)
        if m_vid:
            video = m_vid.group(1)
    return {"recording_date": date, "video_number": video}


def has_continuous_minute_siblings(
    speed_path: PathLike,
    *,
    features: Tuple[str, ...] = CONTINUOUS_MINUTE_REQUIRED_FEATURES,
) -> bool:
    """
    True when every required feature CSV exists beside a fixed speed file.

    ``angular_velocity_midbody`` is resolved via the angular helper; other
    names use ``<stem>_<feature>.csv``.
    """
    speed_path = Path(speed_path)
    if not speed_path.exists():
        return False
    for feature in features:
        if feature == "angular_velocity_midbody":
            ang = _find_angular_csv(speed_path)
            if ang is None or not ang.exists():
                return False
            continue
        feat = _find_feature_csv(speed_path, feature)
        if feat is None or not feat.exists():
            return False
    return True


def missing_continuous_minute_siblings(
    speed_path: PathLike,
    *,
    features: Tuple[str, ...] = CONTINUOUS_MINUTE_REQUIRED_FEATURES,
) -> List[str]:
    """Return required feature names whose sibling CSVs are missing."""
    speed_path = Path(speed_path)
    missing: List[str] = []
    for feature in features:
        if feature == "angular_velocity_midbody":
            ang = _find_angular_csv(speed_path)
            if ang is None or not ang.exists():
                missing.append(feature)
            continue
        feat = _find_feature_csv(speed_path, feature)
        if feat is None or not feat.exists():
            missing.append(feature)
    return missing


def _video_max_frames_lookup(
    fps_table: Optional[pd.DataFrame],
    *,
    recording_date: str,
    video_number: str,
    dataset: str = "",
    source: str = "",
) -> Optional[int]:
    """Return video ``n_frames`` from ``fps_table`` when available."""
    if fps_table is None or not len(fps_table):
        return None
    from .video_fps import lookup_video_fps

    row = lookup_video_fps(
        fps_table,
        recording_date=recording_date,
        video_number=video_number,
        dataset=dataset,
        source_path=source or None,
    )
    if row is None or "n_frames" not in row.index:
        return None
    try:
        n = int(row["n_frames"])
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def load_speed_directory(
    directory: PathLike,
    *,
    pattern: str = "*_fixed.csv",
    fps: Optional[float] = None,
    min_frames: Optional[int] = MIN_FRAMES,
    min_frames_frac: float = MIN_FRAMES_FRAC,
    fps_table: Optional[pd.DataFrame] = None,
    dataset: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Load all matching speed CSVs from a directory.

    Inclusion threshold (first match wins):
      * ``min_frames=0`` — keep every file
      * ``min_frames`` is an int ``> 0`` — absolute row floor (legacy)
      * ``min_frames is None`` (default) — require
        ``ceil(min_frames_frac * video_max_frames)`` rows, where
        ``video_max_frames`` is that video's ``n_frames`` from ``fps_table``
        when present, otherwise the max row count among worms sharing the
        same ``(recording_date, video_number)`` in this directory.

    When ``fps_table`` is set, each worm gets ``fps_true`` /
    ``frames_per_minute`` from its video row.

    Returns {worm_id: DataFrame}. If worm_ids collide, suffixes _2, _3, ...
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    loaded: List[pd.DataFrame] = []
    for path in files:
        loaded.append(
            load_speed_csv(
                path,
                fps=fps,
                fps_table=fps_table,
                dataset=dataset,
                fallback_fps=float(fps) if fps is not None else 6.0,
            )
        )

    # Observed max rows per video (fallback when fps_table lacks n_frames).
    observed_max: Dict[Tuple[str, str], int] = {}
    for df in loaded:
        date = str(df["recording_date"].iloc[0]) if "recording_date" in df.columns else ""
        vid = str(df["video_number"].iloc[0]) if "video_number" in df.columns else ""
        key = (date, vid)
        n = len(df)
        if n > observed_max.get(key, 0):
            observed_max[key] = n

    use_frac = min_frames is None
    frac = float(min_frames_frac)
    worms: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []
    for df in loaded:
        wid = str(df["worm_id"].iloc[0])
        date = str(df["recording_date"].iloc[0]) if "recording_date" in df.columns else ""
        vid = str(df["video_number"].iloc[0]) if "video_number" in df.columns else ""
        source = str(df["source"].iloc[0]) if "source" in df.columns else ""
        video_max = _video_max_frames_lookup(
            fps_table,
            recording_date=date,
            video_number=vid,
            dataset=dataset,
            source=source,
        )
        if video_max is None:
            video_max = observed_max.get((date, vid), len(df))

        if min_frames == 0:
            need = 0
        elif use_frac:
            need = max(1, int(video_max * frac))
        else:
            need = int(min_frames)

        n_rows = len(df)
        if need and n_rows < need:
            skipped.append(
                f"{wid} (n={n_rows}, need≥{need}"
                + (f", video_max={video_max}" if use_frac else "")
                + ")"
            )
            continue
        base = wid
        n = 2
        while wid in worms:
            wid = f"{base}_{n}"
            n += 1
        df = df.copy()
        df["worm_id"] = wid
        worms[wid] = df

    if skipped:
        if use_frac:
            rule = f"<{frac:.0%} of video max frames"
        else:
            rule = f"< {min_frames} frames"
        print(f"Skipped {len(skipped)} worm(s) with {rule}: " + ", ".join(skipped))
    if not worms:
        raise FileNotFoundError(
            f"No worms meeting the frame threshold matching {pattern!r} in {directory}"
        )
    return worms


def load_flip_ranges_xlsx(
    path: PathLike,
    *,
    sheet_name: Union[str, int, None] = 0,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Read a 'frames to flip' Excel sheet used for N2 orientation correction.

    Expected layout (as in ``N2 videos frame flips.xlsx`` / ``N2 video 1.xlsx``):
      row 1: headers like ``W1``, ``W2 (366)``, ``W110``, ...
      subsequent rows: ``start,end`` strings per column (spaces allowed)

    ``sheet_name`` is passed to ``pandas.read_excel`` (name, 0-based index, or
    ``None`` for all sheets merged — not used here; default first sheet).
    """
    path = Path(path)
    raw = pd.read_excel(path, header=None, engine="openpyxl", sheet_name=sheet_name)
    if isinstance(raw, dict):
        raise ValueError("Pass a single sheet_name; got multiple sheets")
    if raw.empty:
        return {}

    header = raw.iloc[0].tolist()
    ranges: Dict[str, List[Tuple[int, int]]] = {}

    for col_idx, cell in enumerate(header):
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            continue
        text = str(cell)
        m = re.search(r"(W\d+)", text, flags=re.IGNORECASE)
        if not m:
            continue
        wid = m.group(1).upper()
        col_ranges: List[Tuple[int, int]] = []
        for val in raw.iloc[1:, col_idx]:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val).strip()
            if not s:
                continue
            parsed = _parse_range(s)
            if parsed is not None:
                col_ranges.append(parsed)
            else:
                warnings.warn(
                    f"{path.name} sheet={sheet_name!r} {wid}: "
                    f"skipping unparseable range {s!r}",
                    stacklevel=2,
                )
        ranges[wid] = col_ranges
    return ranges


def load_flip_ranges_csv(
    path: PathLike,
    *,
    video: Optional[str] = None,
) -> Tuple[Dict[str, List[Tuple[int, int]]], str]:
    """
    Read flip ranges written by ``head_tail_flips.flip_range_tool``.

    Expected columns: ``video,worm,frame_start,frame_end`` (extra columns ok).

    Returns ``({ "W1": [(start, end), ...], ... }, video_stem)``.
    If ``video`` is omitted and the file has one unique video, that stem is used;
    otherwise ``video`` is required when multiple videos are present.
    """
    path = Path(path)
    df = pd.read_csv(path)
    need = {"video", "worm", "frame_start", "frame_end"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {sorted(missing)}")

    videos = sorted({str(v) for v in df["video"].dropna().unique()})
    if video is not None:
        stem = str(video)
        df = df[df["video"].astype(str) == stem].copy()
        # Empty after filter is ok — caller may still copy-through all speed files.
    elif len(videos) == 1:
        stem = videos[0]
        df = df[df["video"].astype(str) == stem].copy()
    elif len(videos) == 0:
        return {}, ""
    else:
        raise ValueError(
            f"{path.name} has multiple videos {videos}; pass video=..."
        )

    ranges: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in df.iterrows():
        try:
            worm = int(row["worm"])
            a = int(row["frame_start"])
            b = int(row["frame_end"])
        except (TypeError, ValueError):
            warnings.warn(
                f"{path.name}: skipping unparseable row {row.to_dict()!r}",
                stacklevel=2,
            )
            continue
        if a < 0:
            warnings.warn(
                f"{path.name} W{worm}: clamping frame_start {a} → 0",
                stacklevel=2,
            )
            a = 0
        if b < 0:
            warnings.warn(
                f"{path.name} W{worm}: skipping range with frame_end={b}",
                stacklevel=2,
            )
            continue
        if a > b:
            a, b = b, a
        wid = f"W{worm}"
        ranges.setdefault(wid, []).append((a, b))

    for wid in ranges:
        ranges[wid] = sorted(set(ranges[wid]))
    return ranges, stem


def _parse_range(text: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", text)
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if start > end:
        return None
    return start, end


def invert_speed_ranges(
    df: pd.DataFrame,
    frame_ranges: List[Tuple[int, int]],
    *,
    fps: float,
) -> pd.DataFrame:
    """
    Negate speed over the given frame ranges (orientation / sign correction).

    Uses ``frame`` when present; otherwise recovers frames from the export
    timeline (frame-index or assumed-6 seconds clock). ``fps`` is kept for
    call-site compatibility but is not used to interpret timestamps.
    """
    from .video_fps import ASSUMED_FPS, recover_frame_index

    out = df.copy()
    t = pd.to_numeric(out["time"], errors="coerce").to_numpy(dtype=float)
    existing = None
    if "frame" in out.columns:
        existing = pd.to_numeric(out["frame"], errors="coerce").to_numpy(dtype=float)
    frames = recover_frame_index(
        t, assumed_fps=ASSUMED_FPS, existing_frame=existing
    )
    out["frame"] = pd.Series(frames, index=out.index).astype("Int64")

    mask = pd.Series(False, index=out.index)
    for start, end in frame_ranges:
        mask |= (out["frame"] >= start) & (out["frame"] <= end)
    out.loc[mask, "speed"] = -out.loc[mask, "speed"]
    out["flipped"] = mask
    return out
