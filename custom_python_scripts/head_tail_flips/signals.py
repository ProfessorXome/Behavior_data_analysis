"""
Per-worm, per-frame signal tables assembled from a Tierpsy featuresN file.

Two index spaces exist in these files and they are not interchangeable:

* ``trajectories_data.worm_index_joined`` -- tracker output, always present, and
  the space that ``timeseries_data.worm_index`` refers to.
* ``trajectories_data.worm_index_manual`` -- human curation, present only on
  curated videos, and the numbering the analysis CSVs use.

``blob_features`` rows align 1:1 with ``trajectories_data`` rows, so blob
signals are read positionally. Skeleton-derived signals live in
``timeseries_data`` and are joined on (joined index, frame).
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import pandas as pd

try:
    from .paths import VideoPaths
except ImportError:  # imported as a top-level module from inside head_tail_flips/
    from paths import VideoPaths

# Columns pulled from timeseries_data when present.
_TIMESERIES_COLS = (
    "head_tail_distance",
    "length",
    "turn",
    "curvature_midbody",
    "quirkiness",
    # Tierpsy's signed speed. Sign depends on head/tail assignment, which flips
    # on some tracks, so it is shown to the human as context but never used as
    # ground truth on its own.
    "speed",
    # Skeleton tip / midbody positions. Tierpsy stores these in microns; we
    # convert to pixels (same space as traj ``coord_x``/``coord_y``) after load.
    "coord_x_head",
    "coord_y_head",
    "coord_x_tail",
    "coord_y_tail",
    "coord_x_midbody",
    "coord_y_midbody",
)

_TIP_XY_COLS = (
    "coord_x_head",
    "coord_y_head",
    "coord_x_tail",
    "coord_y_tail",
    "coord_x_midbody",
    "coord_y_midbody",
)


def _microns_per_pixel(vp: VideoPaths) -> float:
    """Read µm/px from the masked video (fallback 15, matching this dataset)."""
    try:
        with h5py.File(vp.masked, "r") as fh:
            val = fh["mask"].attrs.get("microns_per_pixel", None)
            if val is not None and np.isfinite(float(val)) and float(val) > 0:
                return float(val)
    except OSError:
        pass
    return 15.0


def load_video_signals(vp: VideoPaths, *, min_track_frames: int = 300) -> pd.DataFrame:
    """
    Return one row per (worm, frame) with every signal used for proposing
    omega-turn candidates.

    The ``worm`` column carries manual numbering when the video was curated and
    joined numbering otherwise; ``worm_joined`` always carries the tracker index
    so callers can trace back to the raw file.
    """
    with h5py.File(vp.features, "r") as fh:
        traj = fh["trajectories_data"][:]
        blob = fh["blob_features"][:]
        has_ts = "timeseries_data" in fh
        ts = fh["timeseries_data"][:] if has_ts else None

    names = traj.dtype.names
    frame = traj["frame_number"].astype(np.int64)
    joined = traj["worm_index_joined"].astype(np.int64)
    manual = (
        traj["worm_index_manual"].astype(np.int64)
        if "worm_index_manual" in names
        else joined
    )
    was_skel = (
        traj["was_skeletonized"].astype(bool)
        if "was_skeletonized" in names
        else traj["has_skeleton"].astype(bool)
    )

    df = pd.DataFrame(
        {
            "video": vp.name,
            "strain": vp.strain,
            "worm": manual,
            "worm_joined": joined,
            "frame": frame,
            "x": traj["coord_x"].astype(np.float64),
            "y": traj["coord_y"].astype(np.float64),
            "roi_size": traj["roi_size"].astype(np.float64),
            "perimeter": blob["perimeter"].astype(np.float64),
            "area": blob["area"].astype(np.float64),
            "was_skeletonized": was_skel,
        }
    )
    df["curated"] = "worm_index_manual" in names

    if ts is not None:
        ts_names = ts.dtype.names
        keep = {c: c for c in _TIMESERIES_COLS if c in ts_names}
        tsdf = pd.DataFrame(
            {
                "worm_joined": ts["worm_index"].astype(np.int64),
                "frame": ts["timestamp"].astype(np.int64),
                **{dst: ts[src].astype(np.float64) for src, dst in keep.items()},
            }
        )
        # Tierpsy writes timeseries timestamps as frame numbers; guard the join
        # rather than silently producing an all-NaN merge.
        tsdf = tsdf.drop_duplicates(subset=["worm_joined", "frame"])
        df = df.merge(tsdf, on=["worm_joined", "frame"], how="left")

    for col in _TIMESERIES_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Tip XY from timeseries are microns; traj x/y are pixels. Convert tips so
    # overlays and body-axis features share one coordinate frame.
    mpp = _microns_per_pixel(vp)
    for col in _TIP_XY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float64) / mpp

    # Drop untracked rows and very short tracks, which cannot support a
    # temporal window and are dominated by tracking noise.
    df = df[df["worm"] > 0].copy()
    counts = df.groupby("worm")["frame"].transform("size")
    df = df[counts >= int(min_track_frames)].copy()

    df.sort_values(["worm", "frame"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return _add_derived(df)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-worm normalised signals used by the candidate thresholds."""
    peri = df["perimeter"].to_numpy(dtype=float)
    area = df["area"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Compactness is blob-only, so it survives skeletonisation failure --
        # exactly the frames where posture features go missing.
        df["compactness"] = 4.0 * np.pi * area / np.square(peri)
        df["head_tail_ratio"] = df["head_tail_distance"] / df["length"]

    grp = df.groupby("worm", sort=False)
    df["perimeter_frac"] = peri / grp["perimeter"].transform("median").to_numpy()
    df["area_frac"] = area / grp["area"].transform("median").to_numpy()
    df["compactness_rel"] = (
        df["compactness"] / grp["compactness"].transform("median")
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def border_distance(df: pd.DataFrame, *, width: int = 1600, height: int = 1200) -> pd.Series:
    """Distance in pixels from the worm centroid to the nearest frame edge."""
    return pd.concat(
        [df["x"], width - df["x"], df["y"], height - df["y"]], axis=1
    ).min(axis=1)
