"""
Head/tail flip-range annotation tool.

Proposals are short clips around likely HT breakpoints for worms you name:
every tip appearance and tip disappearance (including calendar gaps in the
track table, plus frame 0 when the worm is present at the start), tip–tip
collapse, and tip-identity jumps gated by bend/coil. You mark the **full**
invert range yourself with an open-range workflow
that persists across clips (so a flip can start in clip 1 and end 200+ frames
later, past another tip dropout).

Output CSV (same invert shape as deploy HT invert ranges)::

    video,worm,frame_start,frame_end,n_frames,source,note

Run (from custom_python_scripts/)::

    python -m head_tail_flips.flip_range_tool \\
        --video 072926_HML1641_03 \\
        --worms 1,2,5 \\
        --out head_tail_flips/data/072926_HML1641_03_ht_flip_ranges.csv

or from inside ``head_tail_flips/``::

    python -m flip_range_tool --video 072926_HML1641_03 --worms 1,2,5

Videos are looked up as Tierpsy writes them, anywhere under ``--root``:
``<folder>/MaskedVideos/<video>.hdf5`` + ``<folder>/Results/<video>_featuresN.hdf5``.
``--root`` defaults to ``../Sample_data/Videos/`` (or ``$WORM_DATA_ROOT/Videos/``);
pass it when your Tierpsy output lives elsewhere::

    python -m flip_range_tool --root /path/to/videos --video ... --worms ...

Keys (also shown in the on-screen help panel)
    s / e     set flip-range START / END at the playhead (absolute frame)
    a         accept/commit the open range to the CSV
    escape    cancel the open range
    x         delete the last committed range for this worm
    d         delete committed range covering the playhead
    t         type start,end in the terminal
    g / h     jump playhead to open start / open end
    j / k     next / previous proposal (open range is kept)
    space     pause / resume
    , / .     step one frame
    r         replay from start of window
    c         recenter window on playhead
    [ / ]     nudge centre −2 / +2 frames
    + / −     expand / shrink context (large max — follow long flips)
    0         reset centre to the proposal centre
    q         save and quit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy  # noqa: F401
import numpy as np
import pandas as pd

try:  # run as part of the package: python -m head_tail_flips.flip_range_tool
    from .crops import CLIP_HALF, CONTEXT_SIZE, CROP_SIZE, WormCropReader, normalize_clip
    from .paths import get_video
    from .signals import load_video_signals
except ImportError:  # run from inside head_tail_flips/: python -m flip_range_tool
    from crops import CLIP_HALF, CONTEXT_SIZE, CROP_SIZE, WormCropReader, normalize_clip
    from paths import get_video
    from signals import load_video_signals

DATA_DIR = Path(__file__).resolve().parent / "data"

LABEL_HALF = CLIP_HALF
HT_HEAD_COLOR = "#4cc9f0"
HT_TAIL_COLOR = "#2ecc71"
OPEN_COLOR = "#e63946"
RANGE_COLOR = "#f4d35e"

OUTPUT_COLUMNS = [
    "video",
    "worm",
    "frame_start",
    "frame_end",
    "n_frames",
    "source",
    "note",
]

HELP_LINES = [
    "RANGE:  s start   e end   a accept   Esc cancel   x del-last   d del-at-playhead   t type start,end",
    "JUMP:   g → open start   h → open end   j/k next/prev proposal (open range KEPT across clips)",
    "VIEW:   space pause   ,/. step   r replay   c recenter   [/] nudge   +/− context   0 reset centre   q quit",
    "CYAN=head  GREEN=tail  GOLD=committed ranges overlapping this window  RED=open range",
]


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive index runs where ``mask`` is True."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(m.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _local_minima(y: np.ndarray, *, order: int = 3) -> np.ndarray:
    """Boolean mask of strict local minima with ``order`` samples either side."""
    n = len(y)
    out = np.zeros(n, dtype=bool)
    if n < 2 * order + 1:
        return out
    for i in range(order, n - order):
        if not np.isfinite(y[i]):
            continue
        win = y[i - order : i + order + 1]
        if not np.isfinite(win).all():
            continue
        if y[i] == np.min(win) and y[i] < y[i - 1] and y[i] < y[i + 1]:
            out[i] = True
    return out


def _tip_swap_score(
    head: np.ndarray, tail: np.ndarray
) -> np.ndarray:
    """
    Per-frame tip-identity evidence (NaN at first frame).

    ``keep = |h_t−h_{t-1}| + |t_t−t_{t-1}|``
    ``swap = |h_t−t_{t-1}| + |t_t−h_{t-1}|``
    Score = ``keep − swap``.

    A true silent HT relabel makes tips jump to each other's previous
    positions → keep large, swap small → **positive** score.

    Straight crawling of an extended worm does the opposite (head stays near
    previous head; head is far from previous tail) → **negative** score ≈
    −2 × body length. Thresholding the old ``swap−keep`` sign therefore
    grabbed straight-line clips.
    """
    n = len(head)
    score = np.full(n, np.nan, dtype=float)
    if n < 2:
        return score
    for i in range(1, n):
        h0, t0 = head[i - 1], tail[i - 1]
        h1, t1 = head[i], tail[i]
        if not (
            np.isfinite(h0).all()
            and np.isfinite(t0).all()
            and np.isfinite(h1).all()
            and np.isfinite(t1).all()
        ):
            continue
        keep = float(np.linalg.norm(h1 - h0) + np.linalg.norm(t1 - t0))
        swap = float(np.linalg.norm(h1 - t0) + np.linalg.norm(t1 - h0))
        score[i] = keep - swap
    return score


@dataclass
class FlipProposalConfig:
    skel_fail_min_run: int = 1  # unused for tip edges; kept for CLI compatibility
    tip_collapse_ratio: float = 0.55
    tip_collapse_order: int = 3
    # Drop tip_collapse if its centre is within this many frames of a
    # tip_appear / tip_disappear centre (collapse only in continuous tip runs).
    tip_collapse_edge_pad: int = 10
    # Tip-jump: identity evidence (keep−swap, px) plus a posture gate so
    # straight crawling does not fire.
    tip_jump_min_score: float = 25.0
    tip_jump_max_ht_ratio: float = 0.70  # require tip–tip collapse / bend
    tip_jump_min_compactness_rel: float = 1.35  # OR coiled blob
    tip_jump_require_skel: bool = True
    half_window: int = LABEL_HALF


def _tips_present(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """True where both tip XY are finite."""
    return (
        np.isfinite(head).all(axis=1)
        & np.isfinite(tail).all(axis=1)
    )


def _tip_edge_proposals(
    worm: int,
    frames: np.ndarray,
    tips_ok: np.ndarray,
) -> List[dict]:
    """
    Every tip appearance and disappearance — no extra gates.

    Appearance: tips present on this row, and either the previous row lacked
    tips or there is a calendar frame gap (worm left the table and returned).

    Disappearance: tips present on the previous row, and either this row lacks
    tips or there is a calendar frame gap.

    Frame 0 is always proposed when the worm has a row there (first existence).
    """
    rows: List[dict] = []
    n = len(frames)
    if n == 0:
        return rows

    if (frames == 0).any():
        rows.append(
            {
                "worm": worm,
                "frame_center": 0,
                "event_frame_start": 0,
                "event_frame_end": 0,
                "sources": "tip_appear",
                "score": 0.0,
            }
        )

    for i in range(n):
        gap = bool(i > 0 and int(frames[i]) - int(frames[i - 1]) > 1)
        prev = bool(tips_ok[i - 1]) if i > 0 else False
        curr = bool(tips_ok[i])

        if i > 0 and prev and (not curr or gap):
            if gap and curr:
                # Tips on both sides of a hole in the track table.
                center = int(frames[i - 1])
                ev_a = center
                ev_b = int(frames[i]) - 1
                score = float(int(frames[i]) - int(frames[i - 1]) - 1)
            else:
                # First table row where tips are missing after being present.
                center = int(frames[i])
                b = i
                while (
                    b + 1 < n
                    and not tips_ok[b + 1]
                    and int(frames[b + 1]) - int(frames[b]) == 1
                ):
                    b += 1
                ev_a = center
                ev_b = int(frames[b])
                score = float(b - i + 1)
            rows.append(
                {
                    "worm": worm,
                    "frame_center": center,
                    "event_frame_start": int(ev_a),
                    "event_frame_end": int(ev_b),
                    "sources": "tip_disappear",
                    "score": score,
                }
            )

        if curr and (i == 0 or not prev or gap):
            center = int(frames[i])
            if center == 0 and i == 0:
                continue  # already added as frame-0 birth
            if i == 0:
                ev_a = ev_b = center
                score = 0.0
            elif gap:
                ev_a = int(frames[i - 1]) + 1
                ev_b = max(ev_a, center - 1)
                score = float(center - int(frames[i - 1]) - 1)
            else:
                a = i - 1
                while (
                    a > 0
                    and not tips_ok[a - 1]
                    and int(frames[a]) - int(frames[a - 1]) == 1
                ):
                    a -= 1
                ev_a = int(frames[a])
                ev_b = int(frames[i - 1])
                score = float(i - a)
            rows.append(
                {
                    "worm": worm,
                    "frame_center": center,
                    "event_frame_start": int(ev_a),
                    "event_frame_end": int(ev_b),
                    "sources": "tip_appear",
                    "score": score,
                }
            )

    return rows


def propose_flip_events(
    track: pd.DataFrame,
    *,
    cfg: Optional[FlipProposalConfig] = None,
) -> pd.DataFrame:
    """
    Build proposal rows for one worm track (must include tip XY when available,
    head_tail_ratio, was_skeletonized for tip_collapse gating).
    """
    cfg = cfg or FlipProposalConfig()
    track = track.sort_values("frame").reset_index(drop=True)
    if track.empty:
        return pd.DataFrame()

    frames = track["frame"].to_numpy(dtype=np.int64)
    worm = int(track["worm"].iloc[0])
    skel = (
        track["was_skeletonized"].to_numpy(dtype=bool)
        if "was_skeletonized" in track.columns
        else np.ones(len(track), dtype=bool)
    )
    tip_cols = (
        "coord_x_head",
        "coord_y_head",
        "coord_x_tail",
        "coord_y_tail",
    )
    has_tips = all(c in track.columns for c in tip_cols)
    head = (
        track[list(tip_cols[:2])].to_numpy(dtype=float)
        if has_tips
        else np.full((len(track), 2), np.nan)
    )
    tail = (
        track[list(tip_cols[2:])].to_numpy(dtype=float)
        if has_tips
        else np.full((len(track), 2), np.nan)
    )
    tips_ok = _tips_present(head, tail) if has_tips else np.zeros(len(track), dtype=bool)
    rows: List[dict] = []

    # --- every tip appearance / disappearance (mechanical; no dedupe) ---
    if has_tips:
        rows.extend(_tip_edge_proposals(worm, frames, tips_ok))
    elif (frames == 0).any():
        rows.append(
            {
                "worm": worm,
                "frame_center": 0,
                "event_frame_start": 0,
                "event_frame_end": 0,
                "sources": "tip_appear",
                "score": 0.0,
            }
        )

    # --- tip collapse: only while tips are continuously present ---
    # tips_ok already required; also require calendar-contiguous neighbours so
    # collapses on track-table holes are excluded. Then drop any collapse whose
    # centre is within tip_collapse_edge_pad of a tip_appear / tip_disappear.
    ht = (
        track["head_tail_ratio"].to_numpy(dtype=float)
        if "head_tail_ratio" in track.columns
        else np.full(len(track), np.nan)
    )
    edge_centers = [
        int(r["frame_center"])
        for r in rows
        if r["sources"] in ("tip_appear", "tip_disappear")
    ]
    if np.isfinite(ht).any() and has_tips:
        continuous = tips_ok.copy()
        for i in range(len(tips_ok)):
            if i > 0 and (
                not tips_ok[i - 1] or int(frames[i]) - int(frames[i - 1]) != 1
            ):
                continuous[i] = False
            if i + 1 < len(tips_ok) and (
                not tips_ok[i + 1] or int(frames[i + 1]) - int(frames[i]) != 1
            ):
                continuous[i] = False
        hot = np.isfinite(ht) & continuous & (ht < cfg.tip_collapse_ratio)
        mins = _local_minima(np.where(hot, ht, np.nan), order=cfg.tip_collapse_order)
        pad = int(cfg.tip_collapse_edge_pad)
        for a, b in _runs(hot):
            local = np.flatnonzero(mins[a : b + 1]) + a
            if local.size == 0:
                local = np.array([(a + b) // 2], dtype=int)
            for ci in local:
                center = int(frames[int(ci)])
                if any(abs(center - ec) <= pad for ec in edge_centers):
                    continue
                rows.append(
                    {
                        "worm": worm,
                        "frame_center": center,
                        "event_frame_start": int(frames[a]),
                        "event_frame_end": int(frames[b]),
                        "sources": "tip_collapse",
                        "score": float(ht[int(ci)]) if np.isfinite(ht[int(ci)]) else np.nan,
                    }
                )

    # --- tip identity jump, gated by bend / tip–tip collapse ---
    if has_tips:
        score = _tip_swap_score(head, tail)  # keep − swap; + ⇒ looks swapped
        compact = (
            track["compactness_rel"].to_numpy(dtype=float)
            if "compactness_rel" in track.columns
            else np.full(len(track), np.nan)
        )
        bent = np.zeros(len(track), dtype=bool)
        if np.isfinite(ht).any():
            bent |= np.isfinite(ht) & (ht <= cfg.tip_jump_max_ht_ratio)
        if np.isfinite(compact).any():
            bent |= np.isfinite(compact) & (
                compact >= cfg.tip_jump_min_compactness_rel
            )
        ok = np.isfinite(score) & (score >= cfg.tip_jump_min_score) & bent & tips_ok
        if cfg.tip_jump_require_skel:
            ok &= skel
        peak = np.zeros(len(score), dtype=bool)
        for i in range(1, len(score) - 1):
            if not ok[i]:
                continue
            if score[i] >= score[i - 1] and score[i] >= score[i + 1]:
                peak[i] = True
        if ok.any() and not peak.any():
            peak[int(np.nanargmax(np.where(ok, score, -np.inf)))] = True
        for ci in np.flatnonzero(peak):
            rows.append(
                {
                    "worm": worm,
                    "frame_center": int(frames[int(ci)]),
                    "event_frame_start": int(frames[max(0, int(ci) - 2)]),
                    "event_frame_end": int(
                        frames[min(len(frames) - 1, int(ci) + 2)]
                    ),
                    "sources": "tip_jump",
                    "score": float(score[int(ci)]),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = (
        pd.DataFrame(rows)
        .sort_values(["worm", "frame_center", "sources"])
        .reset_index(drop=True)
    )
    out.insert(0, "candidate_id", np.arange(1, len(out) + 1))
    return out


def build_flip_tasks(
    video: str,
    worms: Sequence[int],
    *,
    cfg: Optional[FlipProposalConfig] = None,
    root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tasks, signals_for_video) for the named worms.

    ``root`` is the folder holding Tierpsy output (``MaskedVideos/`` +
    ``Results/``; default: ``../Sample_data/Videos/``).
    """
    cfg = cfg or FlipProposalConfig()
    sig = load_video_signals(get_video(video, root=root), min_track_frames=1)
    want = {int(w) for w in worms}
    present = set(int(w) for w in sig["worm"].unique())
    missing = sorted(want - present)
    if missing:
        print(
            f"WARNING: worms not in features track table: {missing} "
            f"(have {sorted(present)[:20]}{'…' if len(present) > 20 else ''})"
        )
    pieces = []
    for w in sorted(want & present):
        track = sig[sig["worm"] == int(w)].copy()
        prop = propose_flip_events(track, cfg=cfg)
        if len(prop):
            prop.insert(0, "video", video)
            pieces.append(prop)
            print(
                f"  worm {w}: {len(prop)} proposals "
                f"({prop['sources'].value_counts().to_dict()})"
            )
        else:
            print(f"  worm {w}: no proposals")
    if not pieces:
        return pd.DataFrame(), sig
    tasks = pd.concat(pieces, ignore_index=True)
    tasks["candidate_id"] = np.arange(1, len(tasks) + 1)
    return tasks, sig


# ---------------------------------------------------------------------------
# Ranges I/O
# ---------------------------------------------------------------------------


def load_ranges(path: Path) -> pd.DataFrame:
    if path.is_file():
        df = pd.read_csv(path)
        for c in OUTPUT_COLUMNS:
            if c not in df.columns:
                df[c] = "" if c in ("source", "note") else np.nan
        return df[OUTPUT_COLUMNS].copy()
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def save_ranges(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if len(out):
        out["n_frames"] = (
            out["frame_end"].astype(int) - out["frame_start"].astype(int) + 1
        )
        out = out.sort_values(["video", "worm", "frame_start"]).reset_index(drop=True)
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Session GUI
# ---------------------------------------------------------------------------


class FlipRangeSession:
    def __init__(
        self,
        tasks: pd.DataFrame,
        *,
        out_path: Path,
        signals: Optional[pd.DataFrame] = None,
        fps: float = 6.0,
        base_half: int = LABEL_HALF,
        max_half: int = 400,
        context_size: int = CONTEXT_SIZE,
        root: Optional[Path] = None,
    ) -> None:
        if tasks.empty:
            raise ValueError("No proposals to review.")
        self.root = root
        self.tasks = tasks.reset_index(drop=True)
        self.out_path = Path(out_path)
        self.ranges = load_ranges(self.out_path)
        self.fps = float(fps)
        self.interval_ms = int(1000 / max(fps, 1.0))
        self.base_half_window = int(base_half)
        self.half_window = int(base_half)
        self.max_half_window = int(max_half)
        self.context_expand_step = 10
        self.context_size = int(context_size)
        self.upscale = 2

        self._sig_tables: Dict[str, pd.DataFrame] = {}
        if signals is not None and len(tasks):
            self._sig_tables[str(tasks.iloc[0]["video"])] = signals
        self._readers: Dict[str, WormCropReader] = {}
        self._speeds: Dict[Tuple[str, int], pd.Series] = {}

        self._clip = None
        self._disp = None
        self._ctx = None
        self._offsets = None
        self._present = None
        self._head_xy = None
        self._tail_xy = None
        self._cent_xy = None
        self._anchor_xy = None
        self._speed_win = None
        self._event_mask = None
        self._centers: Dict[int, int] = {}

        # Open range: absolute frames, persists across proposal navigation.
        self.open_worm: Optional[int] = None
        self.open_video: Optional[str] = None
        self.open_start: Optional[int] = None
        self.open_end: Optional[int] = None
        self.open_source: str = "manual"

        self._play = True
        self._t = 0
        self._play_anchor_t = None
        self._play_anchor_frame = 0
        self._montage_play_i = -1
        self.index = 0

    # ---- data helpers ----

    def _reader(self, video: str) -> WormCropReader:
        if video not in self._readers:
            vp = get_video(video, root=self.root)
            r = WormCropReader(vp.masked, vp.features)
            r.open()
            self._readers[video] = r
        return self._readers[video]

    def _signals(self, video: str) -> pd.DataFrame:
        if video not in self._sig_tables:
            self._sig_tables[video] = load_video_signals(
                get_video(video, root=self.root), min_track_frames=1
            )
        return self._sig_tables[video]

    def _speed_series(self, video: str, worm: int) -> pd.Series:
        key = (video, int(worm))
        if key not in self._speeds:
            sub = self._signals(video)
            sub = sub[sub["worm"] == int(worm)]
            self._speeds[key] = pd.Series(
                sub["speed"].to_numpy(), index=sub["frame"].to_numpy()
            )
        return self._speeds[key]

    def _tip_paths(
        self, video: str, worm: int, frames: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        sub = self._signals(video)
        sub = sub[sub["worm"] == int(worm)].set_index("frame")
        cols_h = ("coord_x_head", "coord_y_head")
        cols_t = ("coord_x_tail", "coord_y_tail")
        if not all(c in sub.columns for c in cols_h + cols_t):
            return None, None, None
        head = sub.reindex(frames)[list(cols_h)].to_numpy(dtype=float)
        tail = sub.reindex(frames)[list(cols_t)].to_numpy(dtype=float)
        if "x" in sub.columns and "y" in sub.columns:
            cent = sub.reindex(frames)[["x", "y"]].to_numpy(dtype=float)
        else:
            cent = np.full_like(head, np.nan)
        return head, tail, cent

    @property
    def n(self) -> int:
        return len(self.tasks)

    def _row(self) -> pd.Series:
        return self.tasks.iloc[self.index]

    def _cid(self) -> int:
        return int(self._row()["candidate_id"])

    def _original_center(self) -> int:
        return int(self._row()["frame_center"])

    def _current_center(self) -> int:
        return int(self._centers.get(self._cid(), self._original_center()))

    def _abs_frame(self) -> int:
        return int(self._current_center() + (self._t - self.half_window))

    def _worm_ranges(self, video: str, worm: int) -> pd.DataFrame:
        if self.ranges.empty:
            return self.ranges
        m = (self.ranges["video"].astype(str) == str(video)) & (
            self.ranges["worm"].astype(int) == int(worm)
        )
        return self.ranges.loc[m].copy()

    # ---- load / draw ----

    def _snap_center(self, new_center: int, worm: int, reader: WormCropReader) -> Optional[int]:
        centers = reader._centers(int(worm))  # noqa: SLF001
        if not centers:
            return None
        frames = np.fromiter(centers.keys(), dtype=int)
        new_center = int(new_center)
        if new_center in centers:
            return new_center
        return int(frames[int(np.argmin(np.abs(frames - new_center)))])

    def _set_center(self, new_center: int, *, keep_playhead_offset: bool = False) -> None:
        row = self._row()
        reader = self._reader(str(row["video"]))
        snapped = self._snap_center(int(new_center), int(row["worm"]), reader)
        if snapped is None:
            return
        play_off = self._t - self.half_window if keep_playhead_offset else 0
        cid = self._cid()
        if snapped == self._original_center():
            self._centers.pop(cid, None)
        else:
            self._centers[cid] = snapped
        self._load()
        if keep_playhead_offset and self._disp is not None:
            self._t = int(np.clip(play_off + self.half_window, 0, self._disp.shape[0] - 1))
        else:
            self._t = self.half_window
        self._play = False
        self._play_anchor_t = None
        # Recentre can keep the same half-window; still refresh montage pixels.
        self._refresh_montage_images()
        self._draw_speed_axis()
        self._draw()

    def _load(self) -> None:
        row = self._row()
        reader = self._reader(str(row["video"]))
        worm = int(row["worm"])
        frame = self._current_center()
        self._clip, self._present = reader.clip(
            worm, frame, half_window=self.half_window
        )
        self._disp = normalize_clip(self._clip)
        try:
            ctx, _, offsets = reader.anchored_clip(
                worm, frame, half_window=self.half_window, size=self.context_size
            )
        except KeyError:
            ctx = np.zeros(
                (2 * self.half_window + 1, self.context_size, self.context_size),
                dtype=np.uint8,
            )
            offsets = np.full((2 * self.half_window + 1, 2), np.nan)
        self._ctx = normalize_clip(ctx)
        self._offsets = offsets
        frames = np.arange(frame - self.half_window, frame + self.half_window + 1)
        self._head_xy, self._tail_xy, self._cent_xy = self._tip_paths(
            str(row["video"]), worm, frames
        )
        if self._cent_xy is not None and np.isfinite(self._cent_xy[self.half_window]).all():
            self._anchor_xy = self._cent_xy[self.half_window].astype(float)
        else:
            self._anchor_xy = None
        spd = self._speed_series(str(row["video"]), worm)
        self._speed_win = np.array(
            [float(spd[f]) if f in spd.index else np.nan for f in frames],
            dtype=float,
        )
        # Mark proposal event band in window offsets.
        a = int(row.get("event_frame_start", frame))
        b = int(row.get("event_frame_end", frame))
        self._event_mask = (frames >= a) & (frames <= b)

    def _zoom(self, img: np.ndarray) -> np.ndarray:
        if self.upscale <= 1:
            return img
        return np.repeat(np.repeat(img, self.upscale, axis=0), self.upscale, axis=1)

    def _ht_at(self, i: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._head_xy is None or self._tail_xy is None:
            return None, None
        if i < 0 or i >= len(self._head_xy):
            return None, None
        h = self._head_xy[i].astype(float)
        t = self._tail_xy[i].astype(float)
        if not (np.isfinite(h).all() and np.isfinite(t).all()):
            return None, None
        return h, t

    def _ht_in_shape(self, i: int):
        h, t = self._ht_at(i)
        if h is None or self._cent_xy is None:
            return None, None
        c = self._cent_xy[i]
        if not np.isfinite(c).all():
            return None, None
        half = CROP_SIZE / 2.0
        k = self.upscale
        return (h - c + half) * k, (t - c + half) * k

    def _ht_in_ctx(self, i: int):
        h, t = self._ht_at(i)
        if h is None or self._anchor_xy is None:
            return None, None
        half = self.context_size / 2.0
        return h - self._anchor_xy + half, t - self._anchor_xy + half

    def _status_text(self) -> str:
        row = self._row()
        video = str(row["video"])
        worm = int(row["worm"])
        abs_f = self._abs_frame()
        n_ranges = len(self._worm_ranges(video, worm))
        half_txt = f"half={self.half_window} ({2 * self.half_window + 1}f)"
        open_txt = "no open range"
        if self.open_start is not None:
            end = "…" if self.open_end is None else str(self.open_end)
            delta = ""
            if self.open_worm == worm and self.open_video == video:
                delta = f"  (playhead {abs_f}, {abs_f - self.open_start:+d} from start)"
            open_txt = (
                f"OPEN W{self.open_worm}  {self.open_start} → {end}{delta}"
            )
        return (
            f"[{self.index + 1}/{self.n}]  {video}  worm {worm}  "
            f"centre {self._current_center()}  playhead F{abs_f}  "
            f"{half_txt}  source={row['sources']}\n"
            f"{open_txt}   |   committed for this worm: {n_ranges}   "
            f"file: {self.out_path.name}"
        )

    def _montage_idxs(self) -> List[int]:
        assert self._disp is not None
        n = self._disp.shape[0]
        max_cells = min(n, 31)
        step = max(1, int(np.ceil(n / max_cells)))
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        return idxs

    def _ensure_montage(self) -> None:
        assert self._disp is not None
        idxs = self._montage_idxs()
        max_cells = 31
        # Clear old montage axes (must wipe the list — a prior remove leaves
        # dead refs that would crash a second ax.remove()).
        for ax in getattr(self, "_montage_axes", []) or []:
            try:
                ax.remove()
            except Exception:
                pass
        self._montage_axes = []
        self._montage_ims = []
        n_show = len(idxs)
        for j, i in enumerate(idxs):
            ax = self._fig.add_subplot(self._gs_montage[0, j])
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(
                self._disp[i], cmap="magma", vmin=0, vmax=1, interpolation="nearest"
            )
            ax._clip_i = i  # noqa: SLF001
            self._montage_axes.append(ax)
            self._montage_ims.append(im)
        # Hide unused montage slots
        for j in range(n_show, max_cells):
            ax = self._fig.add_subplot(self._gs_montage[0, j])
            ax.set_visible(False)
            self._montage_axes.append(ax)

    def _refresh_montage_images(self) -> None:
        """Update montage pixels in place when cell layout is unchanged."""
        if self._disp is None or not getattr(self, "_montage_ims", None):
            return
        idxs = self._montage_idxs()
        visible = [
            ax for ax in self._montage_axes if ax.get_visible() and hasattr(ax, "_clip_i")
        ]
        if len(visible) != len(idxs) or len(self._montage_ims) < len(idxs):
            self._rebuild_montage()
            return
        for im, ax, i in zip(self._montage_ims, visible, idxs):
            ax._clip_i = i  # noqa: SLF001
            im.set_data(self._disp[i])

    def _style_montage(self, ax, i: int, *, is_play: bool) -> None:
        is_event = self._event_mask is not None and bool(self._event_mask[i])
        abs_f = self._current_center() + (i - self.half_window)
        in_open = False
        if (
            self.open_start is not None
            and self.open_worm == int(self._row()["worm"])
            and self.open_video == str(self._row()["video"])
        ):
            lo = self.open_start
            hi = self.open_end if self.open_end is not None else abs_f
            a, b = min(lo, hi), max(lo, hi)
            in_open = a <= abs_f <= b
        if is_play and in_open:
            edge, lw = OPEN_COLOR, 3.0
        elif is_play:
            edge, lw = OPEN_COLOR, 2.5
        elif in_open:
            edge, lw = OPEN_COLOR, 2.0
        elif is_event:
            edge, lw = RANGE_COLOR, 2.0
        else:
            edge, lw = "#adb5bd", 1.0
        for sp in ax.spines.values():
            sp.set_color(edge)
            sp.set_linewidth(lw)

    def _draw_speed_axis(self) -> None:
        ax = self._ax_speed
        ax.clear()
        offs = np.arange(-self.half_window, self.half_window + 1)
        sw = self._speed_win if self._speed_win is not None else np.zeros(len(offs))
        ax.axhline(0, color="#666", lw=1.0)
        # Committed ranges overlapping this window
        row = self._row()
        video, worm = str(row["video"]), int(row["worm"])
        center = self._current_center()
        for _, rr in self._worm_ranges(video, worm).iterrows():
            lo = int(rr.frame_start) - center
            hi = int(rr.frame_end) - center
            ax.axvspan(lo - 0.5, hi + 0.5, color=RANGE_COLOR, alpha=0.35, zorder=0)
        # Proposal event
        if self._event_mask is not None and self._event_mask.any():
            idx = np.flatnonzero(self._event_mask)
            ax.axvspan(
                idx[0] - self.half_window - 0.5,
                idx[-1] - self.half_window + 0.5,
                color="#90e0ef",
                alpha=0.25,
                zorder=0,
            )
        # Open range
        if (
            self.open_start is not None
            and self.open_worm == worm
            and self.open_video == video
        ):
            lo = self.open_start - center
            hi = (self.open_end if self.open_end is not None else self._abs_frame()) - center
            a, b = min(lo, hi), max(lo, hi)
            ax.axvspan(a - 0.5, b + 0.5, color=OPEN_COLOR, alpha=0.25, zorder=1)
            ax.axvline(lo, color=OPEN_COLOR, lw=1.5, ls="--", zorder=3)
            if self.open_end is not None:
                ax.axvline(hi, color=OPEN_COLOR, lw=1.5, ls="--", zorder=3)
        ax.plot(offs, sw, "-o", ms=2.5, color="#3a0ca3", zorder=2)
        self._vline = ax.axvline(
            self._t - self.half_window, color=OPEN_COLOR, lw=1.5, zorder=4
        )
        ax.set_xlim(-self.half_window - 1, self.half_window + 1)
        ax.set_title(
            "Tierpsy signed speed  |  cyan=proposal event  gold=committed  red=open range",
            fontsize=9,
        )
        ax.set_xlabel("frame offset from window centre", fontsize=8)
        ax.tick_params(labelsize=7)

    def _draw_help(self) -> None:
        self._ax_help.clear()
        self._ax_help.set_axis_off()
        self._ax_help.set_xlim(0, 1)
        self._ax_help.set_ylim(0, 1)
        y = 0.82
        for line in HELP_LINES:
            self._ax_help.text(
                0.01,
                y,
                line,
                fontsize=8.5,
                family="monospace",
                va="top",
                ha="left",
                transform=self._ax_help.transAxes,
                clip_on=True,
            )
            y -= 0.22
        # Border so the panel is obviously reserved for controls
        for sp in self._ax_help.spines.values():
            sp.set_visible(True)
            sp.set_color("#495057")
            sp.set_linewidth(1.0)
        self._ax_help.set_facecolor("#f1f3f5")

    def _draw_ranges_list(self) -> None:
        self._ax_list.clear()
        self._ax_list.set_axis_off()
        self._ax_list.set_facecolor("#fff9db")
        row = self._row()
        video, worm = str(row["video"]), int(row["worm"])
        wr = self._worm_ranges(video, worm)
        lines = [f"Committed ranges — {video} W{worm}  (n={len(wr)})"]
        if wr.empty:
            lines.append("  (none yet)")
        else:
            for _, r in wr.tail(12).iterrows():
                lines.append(
                    f"  {int(r.frame_start):>5}–{int(r.frame_end):<5}  "
                    f"n={int(r.n_frames):<4}  {r.source}"
                )
            if len(wr) > 12:
                lines.append(f"  … +{len(wr) - 12} more")
        y = 0.95
        for line in lines:
            self._ax_list.text(
                0.02,
                y,
                line,
                fontsize=8,
                family="monospace",
                va="top",
                ha="left",
                transform=self._ax_list.transAxes,
                clip_on=True,
            )
            y -= 0.07
            if y < 0.02:
                break

    def _draw(self, *, playback: bool = False) -> None:
        assert self._disp is not None and self._ctx is not None
        t = self._t
        self._im_shape.set_data(self._zoom(self._disp[t]))
        self._im_ctx.set_data(self._ctx[t])
        pts = self._offsets[: t + 1]
        good = np.isfinite(pts).all(axis=1)
        self._path.set_data(pts[good, 0], pts[good, 1])
        if np.isfinite(self._offsets[t]).all():
            self._dot.set_data([self._offsets[t, 0]], [self._offsets[t, 1]])
        else:
            self._dot.set_data([], [])

        hs, ts = self._ht_in_shape(t)
        if hs is not None and ts is not None:
            self._ht_shape_h.set_data([hs[0]], [hs[1]])
            self._ht_shape_t.set_data([ts[0]], [ts[1]])
            self._ht_shape_line.set_data([hs[0], ts[0]], [hs[1], ts[1]])
        else:
            self._ht_shape_h.set_data([], [])
            self._ht_shape_t.set_data([], [])
            self._ht_shape_line.set_data([], [])
        hc, tc = self._ht_in_ctx(t)
        if hc is not None and tc is not None:
            self._ht_ctx_h.set_data([hc[0]], [hc[1]])
            self._ht_ctx_t.set_data([tc[0]], [tc[1]])
            self._ht_ctx_line.set_data([hc[0], tc[0]], [hc[1], tc[1]])
        else:
            self._ht_ctx_h.set_data([], [])
            self._ht_ctx_t.set_data([], [])
            self._ht_ctx_line.set_data([], [])

        missing = "" if self._present[t] else "  [absent]"
        self._ax_shape.set_xlabel(
            f"offset {t - self.half_window:+d}  abs F{self._abs_frame()}{missing}",
            fontsize=8,
        )
        if getattr(self, "_vline", None) is not None:
            self._vline.set_xdata([t - self.half_window, t - self.half_window])

        if not playback:
            # Full refresh: keep montage frames in sync with the loaded clip.
            self._refresh_montage_images()

        # Refresh montage borders / playhead ring
        for ax in self._montage_axes:
            if not ax.get_visible() or not hasattr(ax, "_clip_i"):
                continue
            i = int(ax._clip_i)  # noqa: SLF001
            self._style_montage(ax, i, is_play=(i == t))

        self._fig.suptitle(self._status_text(), fontsize=10, ha="left", x=0.01, y=0.995)
        if not playback:
            self._draw_ranges_list()
            self._draw_help()
        self._fig.canvas.draw_idle()

    # ---- range actions ----

    def _set_start(self) -> None:
        row = self._row()
        self.open_video = str(row["video"])
        self.open_worm = int(row["worm"])
        self.open_start = self._abs_frame()
        self.open_end = None
        self.open_source = str(row.get("sources", "manual"))
        self._play = False
        self._draw_speed_axis()
        self._draw()

    def _set_end(self) -> None:
        if self.open_start is None:
            print("No open range — press s to set start first.")
            return
        row = self._row()
        if self.open_worm != int(row["worm"]) or self.open_video != str(row["video"]):
            print(
                f"Open range is on {self.open_video} W{self.open_worm}; "
                f"switch back to that worm to set end (or Esc to cancel)."
            )
            return
        self.open_end = self._abs_frame()
        self._play = False
        self._draw_speed_axis()
        self._draw()

    def _accept_range(self) -> None:
        if self.open_start is None:
            print("No open range to accept.")
            return
        if self.open_end is None:
            # Allow single-frame accept at playhead if end unset? Require end.
            print("Set end with e before accepting (a).")
            return
        a, b = sorted([int(self.open_start), int(self.open_end)])
        rec = {
            "video": self.open_video,
            "worm": int(self.open_worm),
            "frame_start": a,
            "frame_end": b,
            "n_frames": b - a + 1,
            "source": self.open_source or "manual",
            "note": "",
        }
        self.ranges = pd.concat(
            [self.ranges, pd.DataFrame([rec])], ignore_index=True
        )
        self._save()
        print(f"Accepted flip range W{rec['worm']} {a}–{b} (n={b - a + 1})")
        self.open_start = self.open_end = None
        self.open_worm = self.open_video = None
        self._draw_speed_axis()
        self._draw()

    def _cancel_open(self) -> None:
        self.open_start = self.open_end = None
        self.open_worm = self.open_video = None
        self._draw_speed_axis()
        self._draw()

    def _delete_last(self) -> None:
        row = self._row()
        wr = self._worm_ranges(str(row["video"]), int(row["worm"]))
        if wr.empty:
            print("No committed ranges for this worm.")
            return
        idx = wr.index[-1]
        dropped = self.ranges.loc[idx]
        self.ranges = self.ranges.drop(index=idx).reset_index(drop=True)
        self._save()
        print(
            f"Deleted last range W{int(dropped.worm)} "
            f"{int(dropped.frame_start)}–{int(dropped.frame_end)}"
        )
        self._draw_speed_axis()
        self._draw()

    def _delete_at_playhead(self) -> None:
        row = self._row()
        f = self._abs_frame()
        wr = self._worm_ranges(str(row["video"]), int(row["worm"]))
        hit = wr[(wr.frame_start <= f) & (wr.frame_end >= f)]
        if hit.empty:
            print(f"No committed range covers F{f}.")
            return
        idx = hit.index[-1]
        dropped = self.ranges.loc[idx]
        self.ranges = self.ranges.drop(index=idx).reset_index(drop=True)
        self._save()
        print(
            f"Deleted range covering F{f}: "
            f"{int(dropped.frame_start)}–{int(dropped.frame_end)}"
        )
        self._draw_speed_axis()
        self._draw()

    def _type_range(self) -> None:
        row = self._row()
        print(
            f"Type flip range for {row['video']} W{int(row['worm'])} "
            f"as start,end  (or empty to cancel):",
            flush=True,
        )
        try:
            raw = input("> ").strip()
        except EOFError:
            return
        if not raw:
            return
        try:
            parts = raw.replace(" ", "").split(",")
            a, b = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print("Could not parse — expected start,end")
            return
        a, b = sorted([a, b])
        rec = {
            "video": str(row["video"]),
            "worm": int(row["worm"]),
            "frame_start": a,
            "frame_end": b,
            "n_frames": b - a + 1,
            "source": "typed",
            "note": "",
        }
        self.ranges = pd.concat(
            [self.ranges, pd.DataFrame([rec])], ignore_index=True
        )
        self._save()
        print(f"Typed range {a}–{b}")
        self._draw_speed_axis()
        self._draw()

    def _jump_open(self, which: str) -> None:
        if self.open_start is None:
            return
        target = self.open_start if which == "start" else self.open_end
        if target is None:
            return
        # Recentre so target is in view, playhead on target.
        self._set_center(int(target))
        # _set_center puts playhead at centre; good.

    def _save(self) -> None:
        save_ranges(self.out_path, self.ranges)

    def _bump_context(self, delta_steps: int) -> None:
        if self._disp is None:
            return
        step = self.context_expand_step * int(delta_steps)
        new_half = int(
            np.clip(
                self.half_window + step,
                self.base_half_window,
                self.max_half_window,
            )
        )
        if new_half == self.half_window:
            return
        play_off = self._t - self.half_window
        self.half_window = new_half
        self._load()
        self._t = int(np.clip(play_off + self.half_window, 0, self._disp.shape[0] - 1))
        self._play = False
        self._rebuild_montage()
        self._draw_speed_axis()
        self._draw()

    def _rebuild_montage(self) -> None:
        # Wipe refs first so _ensure_montage does not double-remove live axes.
        for ax in getattr(self, "_montage_axes", []) or []:
            try:
                ax.remove()
            except Exception:
                pass
        self._montage_axes = []
        self._montage_ims = []
        self._ensure_montage()

    def _advance(self, step: int) -> None:
        new = self.index + step
        if new < 0 or new >= self.n:
            return
        self.index = new
        self.half_window = self.base_half_window
        self._load()
        self._t = self.half_window
        self._play = True
        self._reset_play_clock()
        # Prefer in-place montage refresh; rebuild only if layout length changed.
        self._refresh_montage_images()
        self._draw_speed_axis()
        self._draw()

    def _reset_play_clock(self) -> None:
        self._play_anchor_t = time.perf_counter()
        self._play_anchor_frame = int(self._t)

    def _tick(self) -> None:
        if not self._play or self._disp is None:
            self._play_anchor_t = None
            return
        n = int(self._disp.shape[0])
        now = time.perf_counter()
        if self._play_anchor_t is None:
            self._reset_play_clock()
        elapsed = now - float(self._play_anchor_t)
        target = int(self._play_anchor_frame + elapsed * self.fps)
        new_t = target % n
        if new_t == self._t and elapsed > 0:
            return
        self._t = new_t
        self._draw(playback=True)

    def _on_key(self, event) -> None:
        key = event.key or ""
        kl = key.lower()
        if kl == "s":
            self._set_start()
        elif kl == "e":
            self._set_end()
        elif kl == "a":
            self._accept_range()
        elif kl == "escape":
            self._cancel_open()
        elif kl == "x":
            self._delete_last()
        elif kl == "d":
            self._delete_at_playhead()
        elif kl == "t":
            self._type_range()
        elif kl == "g":
            self._jump_open("start")
        elif kl == "h":
            self._jump_open("end")
        elif kl in ("j", "right"):
            self._advance(+1)
        elif kl in ("k", "left"):
            self._advance(-1)
        elif key == " ":
            self._play = not self._play
            if self._play:
                self._reset_play_clock()
            else:
                self._play_anchor_t = None
        elif kl == ".":
            self._play = False
            self._t = min(self._disp.shape[0] - 1, self._t + 1)
            self._draw()
        elif kl == ",":
            self._play = False
            self._t = max(0, self._t - 1)
            self._draw()
        elif kl == "r":
            self._t = 0
            self._play = True
            self._reset_play_clock()
        elif kl == "c":
            self._set_center(self._abs_frame())
        elif kl in ("[", "bracketleft"):
            self._set_center(self._current_center() - 2, keep_playhead_offset=True)
        elif kl in ("]", "bracketright"):
            self._set_center(self._current_center() + 2, keep_playhead_offset=True)
        elif kl in ("0", "home"):
            self._set_center(self._original_center())
        elif kl in ("+", "=", "plus"):
            self._bump_context(+1)
        elif kl in ("-", "_", "minus"):
            self._bump_context(-1)
        elif kl == "q":
            self._save()
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _on_click(self, event) -> None:
        if event.inaxes is None:
            return
        for ax in self._montage_axes:
            if event.inaxes is ax and hasattr(ax, "_clip_i"):
                i = int(ax._clip_i)  # noqa: SLF001
                new_c = self._current_center() + (i - self.half_window)
                self._set_center(new_c)
                return

    def run(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Matplotlib's default keymap steals keys we need (s=save, c=back,
        # q=quit, g=grid, etc.). Clear them for this session.
        for param in (
            "keymap.back",
            "keymap.forward",
            "keymap.home",
            "keymap.fullscreen",
            "keymap.grid",
            "keymap.grid_minor",
            "keymap.yscale",
            "keymap.xscale",
            "keymap.quit",
            "keymap.save",
            "keymap.zoom",
            "keymap.pan",
            "keymap.help",
            "keymap.copy",
        ):
            if param in plt.rcParams:
                plt.rcParams[param] = []

        # Large figure; reserved bottom help + right ranges list stay on-canvas.
        self._fig = plt.figure(figsize=(15.5, 10.5), dpi=110)
        # rows: title pad | shape/motion | speed | montage | help
        # cols: main | ranges list
        outer = GridSpec(
            5,
            2,
            figure=self._fig,
            height_ratios=[0.08, 2.2, 0.9, 0.85, 0.7],
            width_ratios=[3.4, 1.15],
            hspace=0.35,
            wspace=0.18,
            left=0.04,
            right=0.98,
            top=0.94,
            bottom=0.04,
        )
        self._ax_shape = self._fig.add_subplot(outer[1, 0])
        # Split shape row: actually use nested for shape|motion
        # Re-do: shape and motion side by side in row 1
        self._fig.delaxes(self._ax_shape)
        gs_top = outer[1, 0].subgridspec(1, 2, wspace=0.15)
        self._ax_shape = self._fig.add_subplot(gs_top[0, 0])
        self._ax_ctx = self._fig.add_subplot(gs_top[0, 1])
        self._ax_speed = self._fig.add_subplot(outer[2, 0])
        self._gs_montage = outer[3, 0].subgridspec(1, 31, wspace=0.05)
        self._ax_help = self._fig.add_subplot(outer[4, :])
        self._ax_list = self._fig.add_subplot(outer[1:4, 1])

        self._load()
        self._ax_shape.set_title("SHAPE — worm-centred + H/T", fontsize=9)
        self._im_shape = self._ax_shape.imshow(
            self._zoom(self._disp[0]),
            cmap="magma",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        self._ax_ctx.set_title("MOTION — fixed crop + path", fontsize=9)
        self._im_ctx = self._ax_ctx.imshow(
            self._ctx[0], cmap="magma", vmin=0, vmax=1, interpolation="nearest"
        )
        (self._path,) = self._ax_ctx.plot(
            [], [], "-", color=HT_HEAD_COLOR, lw=1.6, alpha=0.9
        )
        (self._dot,) = self._ax_ctx.plot([], [], "o", color=HT_HEAD_COLOR, ms=5)
        (self._ht_shape_line,) = self._ax_shape.plot(
            [], [], "-", color="#fff", lw=1.3, alpha=0.85, zorder=4
        )
        (self._ht_shape_h,) = self._ax_shape.plot(
            [], [], "o", color=HT_HEAD_COLOR, ms=7, zorder=5, label="H"
        )
        (self._ht_shape_t,) = self._ax_shape.plot(
            [], [], "o", color=HT_TAIL_COLOR, ms=7, zorder=5, label="T"
        )
        (self._ht_ctx_line,) = self._ax_ctx.plot(
            [], [], "-", color="#fff", lw=1.3, alpha=0.85, zorder=4
        )
        (self._ht_ctx_h,) = self._ax_ctx.plot(
            [], [], "o", color=HT_HEAD_COLOR, ms=6, zorder=5, label="H"
        )
        (self._ht_ctx_t,) = self._ax_ctx.plot(
            [], [], "o", color=HT_TAIL_COLOR, ms=6, zorder=5, label="T"
        )
        self._ax_shape.legend(loc="upper right", fontsize=7, framealpha=0.85)
        self._ax_ctx.legend(loc="upper right", fontsize=7, framealpha=0.85)
        for ax in (self._ax_shape, self._ax_ctx):
            ax.set_xticks([])
            ax.set_yticks([])

        self._ensure_montage()
        self._draw_speed_axis()
        self._draw_help()
        self._draw_ranges_list()

        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        timer = self._fig.canvas.new_timer(interval=self.interval_ms)
        timer.add_callback(self._tick)
        timer.start()
        self._draw()
        plt.show()
        self._save()
        self.close()

    def close(self) -> None:
        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_worms(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--video", required=True, help="Video stem, e.g. 072926_HML1641_03")
    p.add_argument(
        "--worms",
        required=True,
        help="Comma list / ranges of worm ids, e.g. 1,2,5-8",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: head_tail_flips/data/<video>_ht_flip_ranges.csv)",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Folder holding Tierpsy output (MaskedVideos/ + Results/), searched "
        "recursively (default: ../Sample_data/Videos/, or $WORM_DATA_ROOT/Videos/)",
    )
    p.add_argument("--half", type=int, default=LABEL_HALF, help="Base half-window")
    p.add_argument(
        "--max-half",
        type=int,
        default=400,
        help="Max half-window when expanding to follow long flips",
    )
    p.add_argument(
        "--tip-collapse-ratio",
        type=float,
        default=0.55,
        help="head_tail_ratio threshold for tip-collapse proposals",
    )
    p.add_argument(
        "--tip-jump-score",
        type=float,
        default=25.0,
        help="Min keep−swap tip-identity score (px); positive ⇒ looks swapped",
    )
    p.add_argument(
        "--tip-jump-max-ht-ratio",
        type=float,
        default=0.70,
        help="Require head_tail_ratio ≤ this (bend/coil) for tip-jump",
    )
    p.add_argument(
        "--tip-jump-min-compactness",
        type=float,
        default=1.35,
        help="OR compactness_rel ≥ this for tip-jump posture gate",
    )
    p.add_argument(
        "--tasks-out",
        type=Path,
        default=None,
        help="Optional path to write the proposal task CSV",
    )
    args = p.parse_args(argv)

    worms = _parse_worms(args.worms)
    if not worms:
        raise SystemExit("No worms parsed from --worms")
    out = args.out or DATA_DIR / f"{args.video}_ht_flip_ranges.csv"

    cfg = FlipProposalConfig(
        tip_collapse_ratio=float(args.tip_collapse_ratio),
        tip_jump_min_score=float(args.tip_jump_score),
        tip_jump_max_ht_ratio=float(args.tip_jump_max_ht_ratio),
        tip_jump_min_compactness_rel=float(args.tip_jump_min_compactness),
        half_window=int(args.half),
    )
    print(f"Building flip proposals for {args.video} worms={worms} …", flush=True)
    root = args.root.expanduser().resolve() if args.root is not None else None
    if root is not None and not root.is_dir():
        raise SystemExit(f"--root is not a folder: {root}")
    tasks, sig = build_flip_tasks(args.video, worms, cfg=cfg, root=root)
    if tasks.empty:
        raise SystemExit("No proposals found for those worms.")
    if args.tasks_out is not None:
        args.tasks_out.parent.mkdir(parents=True, exist_ok=True)
        tasks.to_csv(args.tasks_out, index=False)
        print(f"wrote tasks {args.tasks_out} ({len(tasks)} rows)")

    print(f"Opening tool → {out}  ({len(tasks)} clips)", flush=True)
    session = FlipRangeSession(
        tasks,
        out_path=out,
        signals=sig,
        base_half=int(args.half),
        max_half=int(args.max_half),
        root=root,
    )
    session.run()
    print(f"Saved {len(session.ranges)} ranges → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
