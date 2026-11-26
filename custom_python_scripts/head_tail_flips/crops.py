"""
Per-worm ROI crop extraction from Tierpsy masked videos.

The masked video ``/mask`` dataset is chunked one full frame per chunk, so a
single frame read costs one gzip decompression. Worm ROIs in this dataset are
small (``roi_size`` spans roughly 24-89 px), so a fixed 96 px crop centred on
``coord_x``/``coord_y`` contains the whole animal with margin at every frame.

Crops are returned raw (no normalisation, no rotation). Augmentation belongs to
the training pipeline so that the cached pixels stay a faithful record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

CROP_SIZE = 96
CLIP_HALF = 15  # frames either side of centre -> clip length 2*15+1 = 31
# Optional pad on either side of CLIP_HALF for body-axis reorientation: tips
# just outside a long skeleton dropout often sit ~10 frames past the coil.
CONTEXT_EXPAND = 10
# Fixed-position window used to keep translation visible. Worms travel at most
# roughly 40 px across a 31-frame window here, and are about 70 px long, so 224
# px holds the whole excursion with margin.
CONTEXT_SIZE = 224

# Worm index column preference. worm_index_manual reflects human curation and
# matches the numbering used by the analysis CSVs; not every video has it.
_INDEX_FIELDS = ("worm_index_manual", "worm_index_joined")


def _pick_index_field(names: Tuple[str, ...]) -> str:
    for field in _INDEX_FIELDS:
        if field in names:
            return field
    raise KeyError(f"no worm index field among {_INDEX_FIELDS} in {names}")


class WormCropReader:
    """
    Random access to per-worm image crops for one video.

    Usage::

        with WormCropReader(masked_path, features_path) as reader:
            img = reader.crop(worm=1, frame=250)
            clip, present = reader.clip(worm=1, center_frame=250)
    """

    def __init__(
        self,
        masked_path: Path | str,
        features_path: Path | str,
        *,
        crop_size: int = CROP_SIZE,
        index_field: Optional[str] = None,
        min_track_frames: int = 300,
    ) -> None:
        self.masked_path = Path(masked_path)
        self.features_path = Path(features_path)
        self.crop_size = int(crop_size)
        self.min_track_frames = int(min_track_frames)

        with h5py.File(self.features_path, "r") as fh:
            traj = fh["trajectories_data"][:]
            blob = fh["blob_features"][:]

        self.index_field = index_field or _pick_index_field(traj.dtype.names)
        self._worm_ids = traj[self.index_field].astype(np.int64)
        self._frames = traj["frame_number"].astype(np.int64)
        self._x = traj["coord_x"].astype(np.float64)
        self._y = traj["coord_y"].astype(np.float64)
        self._roi = traj["roi_size"].astype(np.float64)
        self._perimeter = blob["perimeter"].astype(np.float64)
        self._area = blob["area"].astype(np.float64)

        # row lookup per worm, ordered by frame
        self._rows_by_worm: Dict[int, np.ndarray] = {}
        for wid in np.unique(self._worm_ids):
            if wid <= 0:
                continue
            rows = np.flatnonzero(self._worm_ids == wid)
            if rows.size < self.min_track_frames:
                continue
            self._rows_by_worm[int(wid)] = rows[np.argsort(self._frames[rows])]

        self._fh: Optional[h5py.File] = None
        # Worm-local crop cache: (worm, frame) -> uint8[crop, crop].
        # HDF5 ``/mask`` is chunked one full frame per gzip chunk, so a region
        # read still decompresses the whole plane; caching the 96 px crop means
        # overlapping clip windows only pay that cost once per frame.
        self._crop_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._crop_cache_enabled = False
        with h5py.File(self.masked_path, "r") as fh:
            self.n_frames, self.height, self.width = fh["mask"].shape
            self.microns_per_pixel = float(
                fh["mask"].attrs.get("microns_per_pixel", np.nan)
            )

    # ---- lifecycle -----------------------------------------------------

    def __enter__(self) -> "WormCropReader":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        if self._fh is None:
            self._fh = h5py.File(self.masked_path, "r")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def _mask(self) -> h5py.Dataset:
        if self._fh is None:
            self.open()
        assert self._fh is not None
        return self._fh["mask"]

    # ---- track metadata ------------------------------------------------

    @property
    def worms(self) -> List[int]:
        return sorted(self._rows_by_worm)

    def track(self, worm: int) -> pd.DataFrame:
        """Per-frame track table for one worm (frame, x, y, roi, perimeter, area)."""
        rows = self._rows_by_worm.get(int(worm))
        if rows is None:
            raise KeyError(f"worm {worm} not tracked in {self.masked_path.stem}")
        return pd.DataFrame(
            {
                "frame": self._frames[rows],
                "x": self._x[rows],
                "y": self._y[rows],
                "roi_size": self._roi[rows],
                "perimeter": self._perimeter[rows],
                "area": self._area[rows],
            }
        )

    def _centers(self, worm: int) -> Dict[int, Tuple[float, float]]:
        rows = self._rows_by_worm.get(int(worm))
        if rows is None:
            raise KeyError(f"worm {worm} not tracked in {self.masked_path.stem}")
        return {
            int(self._frames[r]): (float(self._x[r]), float(self._y[r]))
            for r in rows
        }

    # ---- pixel access --------------------------------------------------

    def enable_crop_cache(self, enabled: bool = True) -> None:
        """Cache worm-local crops across overlapping ``clip`` / batch reads."""
        self._crop_cache_enabled = bool(enabled)
        if not enabled:
            self._crop_cache.clear()

    def clear_crop_cache(self) -> None:
        self._crop_cache.clear()

    def crop(self, worm: int, frame: int, *, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Return one ``crop_size`` square uint8 crop centred on the worm.

        Frames where the worm has no tracked centre, or frames outside the
        recording, come back as zeros (matching the masked-video background).
        Only the worm ROI is copied out of the mask plane — never a full-frame
        tensor for the model. (HDF5 may still decompress the full chunk under
        the hood; enable_crop_cache() avoids repeating that for overlaps.)
        """
        worm = int(worm)
        frame = int(frame)
        if self._crop_cache_enabled:
            cached = self._crop_cache.get((worm, frame))
            if cached is not None:
                return cached

        size = self.crop_size
        out = np.zeros((size, size), dtype=np.uint8)
        if frame < 0 or frame >= self.n_frames:
            return out
        if center is None:
            centers = self._centers(worm)
            if frame not in centers:
                return out
            center = centers[frame]

        cx, cy = center
        if not (np.isfinite(cx) and np.isfinite(cy)):
            return out

        half = size // 2
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        # source window clipped to the frame, with matching destination offsets
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1 = min(self.width, x0 + size)
        sy1 = min(self.height, y0 + size)
        if sx1 <= sx0 or sy1 <= sy0:
            return out
        # Region read: only the worm ROI is materialised into ``out``.
        patch = self._mask[frame, sy0:sy1, sx0:sx1]
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = patch
        if self._crop_cache_enabled:
            self._crop_cache[(worm, frame)] = out
        return out

    def preload_crops(
        self,
        worm: int,
        frames: Sequence[int],
        *,
        centers: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> int:
        """
        Read worm-local crops for ``frames`` once, in ascending order.

        Returns how many new crops were fetched (cache hits excluded).
        """
        self.enable_crop_cache(True)
        worm = int(worm)
        centers = centers or self._centers(worm)
        n_new = 0
        for fr in sorted({int(f) for f in frames}):
            if fr < 0 or fr >= self.n_frames or fr not in centers:
                continue
            if (worm, fr) in self._crop_cache:
                continue
            self.crop(worm, fr, center=centers[fr])
            n_new += 1
        return n_new

    def clips_for_centers(
        self,
        worm: int,
        center_frames: Sequence[int],
        *,
        half_window: int = CLIP_HALF,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Build temporal clips for many centres, reading each frame crop once.

        Far cheaper than calling :meth:`clip` per centre when windows overlap
        (dense scan with stride 2 reuses ~29/31 frames between neighbours).
        """
        worm = int(worm)
        half = int(half_window)
        centers = self._centers(worm)
        needed: List[int] = []
        for c in center_frames:
            c = int(c)
            needed.extend(range(c - half, c + half + 1))
        self.preload_crops(worm, needed, centers=centers)

        size = self.crop_size
        t_len = 2 * half + 1
        clips: List[np.ndarray] = []
        presents: List[np.ndarray] = []
        for c in center_frames:
            c = int(c)
            clip = np.zeros((t_len, size, size), dtype=np.uint8)
            present = np.zeros(t_len, dtype=bool)
            for i, fr in enumerate(range(c - half, c + half + 1)):
                key = (worm, int(fr))
                cached = self._crop_cache.get(key)
                if cached is None:
                    continue
                clip[i] = cached
                present[i] = True
            clips.append(clip)
            presents.append(present)
        return clips, presents

    def clip(
        self,
        worm: int,
        center_frame: int,
        *,
        half_window: int = CLIP_HALF,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(clip, present)`` for a temporal window around ``center_frame``.

        ``clip`` has shape ``(2*half_window+1, crop_size, crop_size)`` uint8.
        ``present`` is a bool array flagging which frames had a tracked centre;
        untracked frames are zero-filled. Reading is done in ascending frame
        order so the underlying gzip chunks are touched once each.
        """
        centers = self._centers(worm)
        frames = np.arange(center_frame - half_window, center_frame + half_window + 1)
        size = self.crop_size
        clip = np.zeros((len(frames), size, size), dtype=np.uint8)
        present = np.zeros(len(frames), dtype=bool)
        for i, fr in enumerate(frames):
            fr = int(fr)
            if fr < 0 or fr >= self.n_frames or fr not in centers:
                continue
            clip[i] = self.crop(worm, fr, center=centers[fr])
            present[i] = True
        return clip, present

    def anchored_clip(
        self,
        worm: int,
        center_frame: int,
        *,
        half_window: int = CLIP_HALF,
        size: int = CONTEXT_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(clip, present, offsets)`` for a window with the crop held at a
        fixed position instead of following the worm.

        Worm-centred crops remove translation, which destroys the evidence for
        whether the animal reversed through a turn. Anchoring every frame to the
        centre frame's position keeps that motion visible.

        ``offsets`` gives each frame's centroid in crop coordinates, so a caller
        can overlay the path.
        """
        centers = self._centers(worm)
        if center_frame not in centers:
            raise KeyError(f"worm {worm} has no tracked centre at frame {center_frame}")
        ax, ay = centers[center_frame]
        half = size // 2
        x0 = int(round(ax)) - half
        y0 = int(round(ay)) - half

        frames = np.arange(center_frame - half_window, center_frame + half_window + 1)
        clip = np.zeros((len(frames), size, size), dtype=np.uint8)
        present = np.zeros(len(frames), dtype=bool)
        offsets = np.full((len(frames), 2), np.nan, dtype=np.float64)

        sx0, sy0 = max(0, x0), max(0, y0)
        sx1 = min(self.width, x0 + size)
        sy1 = min(self.height, y0 + size)
        for i, fr in enumerate(frames):
            fr = int(fr)
            if fr < 0 or fr >= self.n_frames or fr not in centers:
                continue
            if sx1 > sx0 and sy1 > sy0:
                clip[i, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = self._mask[
                    fr, sy0:sy1, sx0:sx1
                ]
            cx, cy = centers[fr]
            offsets[i] = (cx - x0, cy - y0)
            present[i] = True
        return clip, present, offsets

    def frame_range(self, worm: int) -> Tuple[int, int]:
        rows = self._rows_by_worm[int(worm)]
        return int(self._frames[rows].min()), int(self._frames[rows].max())


def normalize_crop(img: np.ndarray, *, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """
    Map a raw crop to float32 in [0, 1] with the worm bright.

    Inside the mask, plate background sits near the top of the intensity range
    (roughly 160-178 here) and the worm body forms a darker tail (roughly
    90-160). Pixels outside the mask are zero, which would otherwise dominate a
    naive rescale, so they are folded to the background level before the range
    is stretched and inverted. The result puts the worm at high values, which is
    both easier to read by eye and a saner input distribution for a CNN.
    """
    img = np.asarray(img)
    out = np.zeros(img.shape, dtype=np.float32)
    inside = img > 0
    if not inside.any():
        return out
    vals = img[inside].astype(np.float32)
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    stretched = img.astype(np.float32)
    stretched[~inside] = hi  # outside the mask reads as plate background
    stretched = np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)
    return (1.0 - stretched).astype(np.float32)


def normalize_clip(clip: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply :func:`normalize_crop` across a clip using one shared intensity range.

    A shared range keeps brightness stable across the clip, so a frame where the
    worm happens to be absent does not rescale differently from its neighbours.
    """
    clip = np.asarray(clip)
    inside = clip > 0
    out = np.zeros(clip.shape, dtype=np.float32)
    if not inside.any():
        return out
    vals = clip[inside].astype(np.float32)
    lo = float(np.percentile(vals, kwargs.get("lo_pct", 1.0)))
    hi = float(np.percentile(vals, kwargs.get("hi_pct", 99.0)))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    stretched = clip.astype(np.float32)
    stretched[~inside] = hi
    stretched = np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)
    out = (1.0 - stretched).astype(np.float32)
    # Frames with no mask at all stay flat zero rather than fully bright.
    empty = ~inside.any(axis=(1, 2))
    out[empty] = 0.0
    return out
