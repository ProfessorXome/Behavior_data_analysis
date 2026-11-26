"""
Where the analysis finds its input data, and how it reads Tierpsy HDF5 files.

Data layout
-----------
Everything lives under one *data root* (default: the ``Sample_data/`` folder
next to ``custom_python_scripts/``)::

    <data root>/
        video_fps.csv                    per-video true FPS (build_video_fps_table.py)
        fixed_speed_mapping.json         optional; see "Worm numbering" below
        Videos/                          Tierpsy output, exactly as Tierpsy writes it
            MaskedVideos/<video>.hdf5            (used for the µm/px scale)
            Results/<video>_featuresN.hdf5       (centroid track, skeleton tips, raw speed)
        Tierpsy_csvs/                    per-worm CSVs, one folder per strain
            <strain>/<video>_TS_W<n>_speed.csv          raw export (export_tierpsy_csvs.py)
            <strain>/<video>_TS_W<n>_speed_fixed.csv    head/tail-corrected speed
            <strain>/<video>_TS_W<n>_blob_perimeter.csv blob perimeter

``Videos/`` may hold several Tierpsy output folders (e.g. ``Videos/HML1630/
MaskedVideos`` + ``Videos/HML1630/Results``); every ``Results/*_featuresN.hdf5``
below it is found.

To analyse your own data, either put it in this layout under ``Sample_data/``,
or put it in the same layout somewhere else and point the code at it with the
``WORM_DATA_ROOT`` environment variable, :func:`set_data_root`, or the
``--data-root`` option of the command-line scripts.

Video names
-----------
Video stems look like ``<MMDDYY>_<strain>_<NN>`` (e.g. ``071026_HML1630_01``;
``070926_N2_demag_01`` is also accepted). The strain token is the part after
the date. Per-worm CSVs add ``_TS_W<n>_<feature>.csv`` where ``n`` is Tierpsy's
``worm_index_manual`` (the worm number shown in Tierpsy Tracker Viewer).

Worm numbering
--------------
If your CSV ``W<n>`` numbers are *not* Tierpsy's ``worm_index_manual`` (for
example if worms were renumbered 1..N), the per-worm fixed-speed CSV is
matched to its Tierpsy worm by content (``|fixed speed|`` reproduces
``|raw speed|`` frame for frame). ``fixed_speed_mapping.json`` can list the
mapping explicitly as rows of ``{"video", "csv_worm", "mask_worm",
"reliable"}``; it is not needed when the numbers already match.

The functions here were adapted from code written for the project's
omega-turn detector, which is not part of this repository.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import h5py
import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1]  # custom_python_scripts/
DEFAULT_DATA_ROOT = SCRIPTS_DIR.parent / "Sample_data"
DATA_ROOT_ENV = "WORM_DATA_ROOT"

VIDEOS_DIRNAME = "Videos"
CSV_DIRNAME = "Tierpsy_csvs"
FPS_TABLE_NAME = "video_fps.csv"
MAPPING_JSON_NAME = "fixed_speed_mapping.json"

_data_root_override: Optional[Path] = None


def set_data_root(path: Optional[PathLike]) -> Path:
    """Use ``path`` as the data root for this session (``None`` restores the default)."""
    global _data_root_override
    _data_root_override = Path(path).expanduser().resolve() if path is not None else None
    _clear_caches()
    return data_root()


def data_root() -> Path:
    """Current data root: :func:`set_data_root`, else ``$WORM_DATA_ROOT``, else ``Sample_data/``."""
    if _data_root_override is not None:
        return _data_root_override
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def videos_dir() -> Path:
    """Folder holding Tierpsy output (``MaskedVideos/`` + ``Results/``)."""
    return data_root() / VIDEOS_DIRNAME


def csv_dir() -> Path:
    """Folder holding one sub-folder of per-worm CSVs per strain."""
    return data_root() / CSV_DIRNAME


def strain_csv_dir(strain: str) -> Path:
    """``<data root>/Tierpsy_csvs/<strain>``."""
    return csv_dir() / str(strain)


def strain_csv_dirs() -> List[Tuple[str, Path]]:
    """``[(strain, folder), ...]`` for every strain folder under ``Tierpsy_csvs/``."""
    root = csv_dir()
    if not root.is_dir():
        return []
    return [(p.name, p) for p in sorted(root.iterdir()) if p.is_dir()]


def fps_table_path() -> Path:
    return data_root() / FPS_TABLE_NAME


def mapping_json_path() -> Path:
    return data_root() / MAPPING_JSON_NAME


def strain_from_video(video: str) -> str:
    """
    ``071026_HML1630_01`` → ``HML1630``; ``070926_N2_demag_01`` → ``N2``.

    The strain is the token after the leading date.
    """
    parts = str(video).split("_")
    if len(parts) >= 2 and re.fullmatch(r"\d{6}", parts[0]):
        return parts[1]
    return parts[0]


# ---------------------------------------------------------------------------
# Finding Tierpsy files
# ---------------------------------------------------------------------------

# Collapse letter+digits strain tokens so HML1630 ↔ 1630 and VM487 ↔ 487.
_STRAIN_DIGITS_RE = re.compile(r"^([A-Za-z]+)(\d{3,})$")


def normalize_video_stem(name: str) -> str:
    """``071026_HML1630_01`` and legacy ``071026_1630_01`` map to the same key."""
    parts: List[str] = []
    for part in str(name).split("_"):
        m = _STRAIN_DIGITS_RE.fullmatch(part)
        parts.append(m.group(2) if m else part)
    return "_".join(parts)


@dataclass(frozen=True)
class VideoPaths:
    name: str
    strain: str
    masked: Path
    features: Path


def discover_videos(root: Optional[PathLike] = None) -> List[VideoPaths]:
    """Every ``Results/<video>_featuresN.hdf5`` under ``root`` (default: ``Videos/``)."""
    root = Path(root) if root is not None else videos_dir()
    if not root.is_dir():
        return []
    found: List[VideoPaths] = []
    for features in sorted(root.rglob("*_featuresN.hdf5")):
        name = features.name[: -len("_featuresN.hdf5")]
        tierpsy_dir = features.parent.parent if features.parent.name == "Results" else features.parent
        found.append(
            VideoPaths(
                name=name,
                strain=strain_from_video(name),
                masked=tierpsy_dir / "MaskedVideos" / f"{name}.hdf5",
                features=features,
            )
        )
    return found


def video_index(root: Optional[PathLike] = None) -> Dict[str, VideoPaths]:
    return {vp.name: vp for vp in discover_videos(root)}


def get_video(name: str, root: Optional[PathLike] = None) -> VideoPaths:
    """Look up a video by stem (exact, or legacy strain alias)."""
    idx = video_index(root)
    if name in idx:
        return idx[name]
    target = normalize_video_stem(name)
    for key, vp in idx.items():
        if normalize_video_stem(key) == target:
            return vp
    where = Path(root) if root is not None else videos_dir()
    raise KeyError(
        f"no Results/{name}_featuresN.hdf5 under {where} "
        f"(known videos: {sorted(idx)})"
    )


# ---------------------------------------------------------------------------
# Per-worm signals from featuresN
# ---------------------------------------------------------------------------

_TIMESERIES_COLS = (
    "head_tail_distance",
    "length",
    "turn",
    "curvature_midbody",
    "quirkiness",
    "speed",
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

DEFAULT_MICRONS_PER_PIXEL = 15.0


def microns_per_pixel(vp: VideoPaths) -> float:
    """µm/px from the masked video's ``/mask`` attributes (fallback 15, this rig's value)."""
    try:
        with h5py.File(vp.masked, "r") as fh:
            val = fh["mask"].attrs.get("microns_per_pixel", None)
            if val is not None and np.isfinite(float(val)) and float(val) > 0:
                return float(val)
    except OSError:
        pass
    return DEFAULT_MICRONS_PER_PIXEL


def load_video_signals(vp: VideoPaths, *, min_track_frames: int = 300) -> pd.DataFrame:
    """
    One row per (worm, frame): centroid, ROI size, blob shape and skeleton
    signals from ``featuresN.hdf5``. Skeleton tip coordinates are converted
    from µm to pixels so they share the centroid's coordinate frame.

    ``worm`` is Tierpsy's ``worm_index_manual`` when the video was curated,
    else ``worm_index_joined``.
    """
    with h5py.File(vp.features, "r") as fh:
        traj = fh["trajectories_data"][:]
        blob = fh["blob_features"][:]
        ts = fh["timeseries_data"][:] if "timeseries_data" in fh else None

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
        keep = [c for c in _TIMESERIES_COLS if c in ts_names]
        tsdf = pd.DataFrame(
            {
                "worm_joined": ts["worm_index"].astype(np.int64),
                "frame": ts["timestamp"].astype(np.int64),
                **{c: ts[c].astype(np.float64) for c in keep},
            }
        )
        tsdf = tsdf.drop_duplicates(subset=["worm_joined", "frame"])
        df = df.merge(tsdf, on=["worm_joined", "frame"], how="left")

    for col in _TIMESERIES_COLS:
        if col not in df.columns:
            df[col] = np.nan

    mpp = microns_per_pixel(vp)
    for col in _TIP_XY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float64) / mpp

    df = df[df["worm"] > 0].copy()
    counts = df.groupby("worm")["frame"].transform("size")
    df = df[counts >= int(min_track_frames)].copy()
    df.sort_values(["worm", "frame"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Video / worm ids from CSV file names
# ---------------------------------------------------------------------------


def parse_video_id_from_source(source: PathLike) -> str:
    """``073026_VM487_03_TS_W8_speed_fixed.csv`` → ``073026_VM487_03``."""
    stem = Path(source).name
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    m = re.match(r"^(.+)_TS_W\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^(.+)_W\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def worm_number_from_id(worm_id: str) -> Optional[int]:
    """
    ``W8`` / ``W8_4`` / ``w8`` / ``8`` → 8.

    Collision suffixes from ``load_speed_directory`` (``W3_2``) are ignored.
    """
    s = str(worm_id).strip()
    m = re.match(r"[Ww](\d+)", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def worm_number_from_source(source: PathLike) -> Optional[int]:
    """``…_TS_W3_speed_fixed.csv`` / ``…_W3_fixed.csv`` → 3."""
    stem = Path(source).name
    m = re.search(r"_TS_W(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"_W(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


@lru_cache(maxsize=4)
def _load_mapping_json(path: str) -> Dict[Tuple[str, int], int]:
    """Reliable ``(video, csv_worm) → mask_worm`` rows from a mapping JSON."""
    out: Dict[Tuple[str, int], int] = {}
    p = Path(path)
    if not p.is_file():
        return out
    try:
        rows = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or not row.get("reliable", False):
            continue
        try:
            out[(str(row["video"]), int(row["csv_worm"]))] = int(row["mask_worm"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


@lru_cache(maxsize=4)
def _mapping_json_videos(path: str) -> Set[str]:
    """Normalized names of every video listed in a mapping JSON (reliable or not)."""
    p = Path(path)
    if not p.is_file():
        return set()
    try:
        rows = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(rows, list):
        return set()
    return {
        normalize_video_stem(str(r["video"]))
        for r in rows
        if isinstance(r, dict) and "video" in r
    }


def resolve_mask_worm(video: str, csv_worm: int) -> int:
    """
    Tierpsy ``worm_index_manual`` for CSV worm ``W<csv_worm>``.

    Uses ``fixed_speed_mapping.json`` when it lists this worm; otherwise the
    CSV number is assumed to be the Tierpsy number.
    """
    table = _load_mapping_json(str(mapping_json_path()))
    key = (str(video), int(csv_worm))
    if key in table:
        return int(table[key])
    target = normalize_video_stem(str(video))
    for (v, w), mask in table.items():
        if int(w) == int(csv_worm) and normalize_video_stem(str(v)) == target:
            return int(mask)
    return int(csv_worm)


# ---------------------------------------------------------------------------
# Head/tail inversions implied by fixed-speed curation
# ---------------------------------------------------------------------------

_FIXED_NAME_RE = re.compile(r"^(?P<video>.+?)_TS_W(?P<worm>\d+)_(?:speed_)?fixed\.csv$")
NOMINAL_FPS = 6.0  # fps assumed when a CSV's time column is in seconds
_MATCH_TOL = 1e-3
MIN_COVERAGE = 0.5


def load_fixed_speed(path: PathLike) -> pd.Series:
    """One fixed-speed CSV as a signed-speed series indexed by frame number."""
    df = pd.read_csv(path)
    if "speed" not in df.columns:
        raise ValueError(f"{path} has no 'speed' column")
    time = df["time"].to_numpy(dtype=float)
    step = float(np.nanmedian(np.diff(time))) if len(time) > 1 else 1.0
    frames = np.round(time if abs(step - 1.0) < 1e-6 else time * NOMINAL_FPS)
    series = pd.Series(
        df["speed"].to_numpy(dtype=float), index=frames.astype(np.int64), name="speed_fixed"
    )
    return series[~series.index.duplicated(keep="first")].sort_index()


def raw_speed_by_worm(vp: VideoPaths) -> Dict[int, pd.Series]:
    """Tierpsy's raw signed speed per ``worm_index_manual``, indexed by frame."""
    with h5py.File(vp.features, "r") as fh:
        traj = fh["trajectories_data"][:]
        ts = fh["timeseries_data"][:]
    names = traj.dtype.names
    manual = (
        traj["worm_index_manual"] if "worm_index_manual" in names else traj["worm_index_joined"]
    ).astype(np.int64)
    joined = traj["worm_index_joined"].astype(np.int64)
    joined_to_manual = {int(j): int(m) for j, m in zip(joined, manual) if m > 0}
    df = pd.DataFrame(
        {
            "joined": ts["worm_index"].astype(np.int64),
            "frame": ts["timestamp"].astype(np.int64),
            "speed": ts["speed"].astype(float),
        }
    )
    df["worm"] = df["joined"].map(joined_to_manual)
    out: Dict[int, pd.Series] = {}
    for worm, grp in df.dropna(subset=["worm"]).groupby("worm"):
        s = grp.set_index("frame")["speed"]
        out[int(worm)] = s[~s.index.duplicated(keep="first")].sort_index()
    return out


def _score(fixed: pd.Series, raw: pd.Series) -> Tuple[int, int]:
    """Frames where ``|fixed|`` reproduces ``|raw|``, and the CSV's finite count."""
    csv_finite = int(np.isfinite(fixed.to_numpy()).sum())
    common = fixed.index.intersection(raw.index)
    if len(common) == 0:
        return 0, csv_finite
    a = fixed.reindex(common).to_numpy()
    b = raw.reindex(common).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return 0, csv_finite
    matched = np.abs(np.abs(a[ok]) - np.abs(b[ok])) <= _MATCH_TOL * np.maximum(
        1.0, np.abs(b[ok])
    )
    return int(matched.sum()), csv_finite


def _match_mask_worm(fixed: pd.Series, raw: Dict[int, pd.Series]) -> Optional[int]:
    """Tierpsy worm whose raw speed this fixed series reproduces, or ``None`` if unclear."""
    scored: List[Tuple[float, int, int]] = []
    for mask_worm, rs in raw.items():
        n_matched, n_finite = _score(fixed, rs)
        if n_finite == 0:
            continue
        scored.append((n_matched / n_finite, mask_worm, n_matched))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], -t[2]))
    cov, mask_worm, _ = scored[0]
    runner_cov = scored[1][0] if len(scored) > 1 else None
    reliable = cov >= MIN_COVERAGE and (runner_cov is None or cov > runner_cov + 0.05)
    return int(mask_worm) if reliable else None


def frames_where_ht_inverted(fixed: pd.Series, raw: pd.Series) -> np.ndarray:
    """
    Frames where fixed-speed curation re-signed Tierpsy speed, i.e. where
    Tierpsy's head/tail assignment is backwards.
    """
    common = fixed.index.intersection(raw.index)
    if len(common) == 0:
        return np.array([], dtype=np.int64)
    a = fixed.reindex(common).to_numpy(dtype=float)
    b = raw.reindex(common).to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.array([], dtype=np.int64)
    mag_ok = np.abs(np.abs(a) - np.abs(b)) <= _MATCH_TOL * np.maximum(1.0, np.abs(b))
    sa = np.sign(a)
    sb = np.sign(b)
    inverted = ok & mag_ok & (sa != 0) & (sb != 0) & (sa != sb)
    return np.sort(np.asarray(common, dtype=np.int64)[inverted])


_RAW_SPEED_CACHE: Dict[str, Dict[int, pd.Series]] = {}


def ht_invert_frame_set(fixed_csv: PathLike, mask_worm: int) -> Optional[Set[int]]:
    """
    Frames where Tierpsy's head/tail is backwards for this worm, judged by
    comparing its fixed-speed CSV with Tierpsy's raw speed.

    Returns ``None`` when the fixed CSV cannot be confidently tied to
    ``mask_worm`` (by ``fixed_speed_mapping.json`` or by content), or the
    video's featuresN file is not found.
    """
    fixed_csv = Path(fixed_csv)
    m = _FIXED_NAME_RE.match(fixed_csv.name)
    if not m:
        return None
    video = m.group("video")
    csv_worm = int(m.group("worm"))
    try:
        vp = get_video(video)
    except KeyError:
        return None
    if vp.name not in _RAW_SPEED_CACHE:
        _RAW_SPEED_CACHE[vp.name] = raw_speed_by_worm(vp)
    raw = _RAW_SPEED_CACHE[vp.name]
    fixed = load_fixed_speed(fixed_csv)

    target = normalize_video_stem(video)
    if target in _mapping_json_videos(str(mapping_json_path())):
        # Video listed in the mapping JSON: trust only its reliable rows.
        table = _load_mapping_json(str(mapping_json_path()))
        listed = {
            mask for (v, w), mask in table.items()
            if w == csv_worm and normalize_video_stem(v) == target
        }
        if int(mask_worm) not in listed:
            return None
    elif _match_mask_worm(fixed, raw) != int(mask_worm):
        # Not listed: tie the CSV to its Tierpsy worm by content.
        return None

    rs = raw.get(int(mask_worm))
    if rs is None:
        return set()
    return {int(f) for f in frames_where_ht_inverted(fixed, rs)}


# ---------------------------------------------------------------------------
# Head→tail axis helpers (reorientation angle at reversals)
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else None


def skeleton_dropout_span(valid: np.ndarray, center: int) -> Optional[Tuple[int, int]]:
    """Inclusive ``[lo, hi]`` of the invalid-skeleton run at (or nearest) ``center``."""
    n = len(valid)
    invalid = ~np.asarray(valid, dtype=bool)
    if not invalid.any():
        return None
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not invalid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and invalid[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    for lo, hi in runs:
        if lo <= center <= hi:
            return lo, hi
    return min(runs, key=lambda rh: abs(0.5 * (rh[0] + rh[1]) - center))


def pick_extended_frame(valid_idx: np.ndarray, tip_a: np.ndarray, tip_b: np.ndarray) -> int:
    """Index (from ``valid_idx``) where the two skeleton tips are furthest apart."""
    if len(valid_idx) == 1:
        return int(valid_idx[0])
    sep = np.linalg.norm(tip_a[valid_idx] - tip_b[valid_idx], axis=1)
    return int(valid_idx[int(np.nanargmax(sep))])


def ht_axis(head_xy: np.ndarray, tail_xy: np.ndarray, i: int, flip: int) -> Optional[np.ndarray]:
    """Unit tail→head vector at row ``i`` (head and tail swapped when ``flip``)."""
    h, t = np.asarray(head_xy[i], dtype=float), np.asarray(tail_xy[i], dtype=float)
    if flip:
        h, t = t, h
    return _unit(h - t)


# ---------------------------------------------------------------------------


_WARNED: Set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Emit ``message`` as a warning the first time ``key`` is seen."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, stacklevel=3)


def _clear_caches() -> None:
    _RAW_SPEED_CACHE.clear()
    _load_mapping_json.cache_clear()
    _mapping_json_videos.cache_clear()
    _WARNED.clear()
