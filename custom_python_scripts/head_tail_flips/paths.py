"""
Dataset discovery for Tierpsy-tracked videos in this project.

Layout assumed (as written by Tierpsy), anywhere below the video root::

    <folder>/
        MaskedVideos/<video>.hdf5        compressed background-subtracted frames
        Results/<video>_featuresN.hdf5   trajectories + per-frame blob features

The video root defaults to ``Sample_data/Videos/`` (next to
``custom_python_scripts/``), or
``$WORM_DATA_ROOT/Videos/`` when that environment variable is set; the flip
tool's ``--root`` option overrides it. Several Tierpsy output folders can sit
under one root (e.g. ``Videos/HML1630/`` and ``Videos/N2/``).

A "video" is identified by its stem (e.g. ``073026_VM487_04``), which must be
unique under the root. The strain is the token after the date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os
import re

SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # custom_python_scripts/


def default_video_root() -> Path:
    """``$WORM_DATA_ROOT/Videos`` if set, else ``Sample_data/Videos`` next to custom_python_scripts/."""
    env = os.environ.get("WORM_DATA_ROOT", "").strip()
    base = Path(env).expanduser().resolve() if env else SCRIPTS_DIR.parent / "Sample_data"
    return base / "Videos"

# Collapse letter+digits strain tokens so HML1630 ↔ 1630 and VM487 ↔ 487.
_STRAIN_DIGITS_RE = re.compile(r"^([A-Za-z]+)(\d{3,})$")


def normalize_video_stem(name: str) -> str:
    """
    Normalize video stems for alias matching across renames.

    ``071026_HML1630_01`` and legacy ``071026_1630_01`` map to the same key.
    """
    parts: List[str] = []
    for part in str(name).split("_"):
        m = _STRAIN_DIGITS_RE.fullmatch(part)
        parts.append(m.group(2) if m else part)
    return "_".join(parts)


def resolve_video_name(name: str, known: Dict[str, "VideoPaths"]) -> Optional[str]:
    """Return the known-video key matching ``name`` (exact or strain-alias)."""
    if name in known:
        return name
    target = normalize_video_stem(name)
    for key in known:
        if normalize_video_stem(key) == target:
            return key
    return None


@dataclass(frozen=True)
class VideoPaths:
    name: str
    strain: str
    masked: Path
    features: Path

    @property
    def has_all(self) -> bool:
        return self.masked.is_file() and self.features.is_file()


def strain_from_video(name: str) -> str:
    """``071026_HML1630_01`` → ``HML1630``; ``070926_N2_demag_01`` → ``N2``."""
    parts = str(name).split("_")
    if len(parts) >= 2 and re.fullmatch(r"\d{6}", parts[0]):
        return parts[1]
    return parts[0]


def discover_videos(root: Optional[Path] = None) -> List[VideoPaths]:
    """
    Return every video that has both a masked video and a featuresN file.

    Videos missing either file are skipped, since both are required to build
    per-worm crops.
    """
    root = Path(root) if root is not None else default_video_root()
    found: List[VideoPaths] = []
    for masked in sorted(root.rglob("MaskedVideos/*.hdf5")):
        features = masked.parent.parent / "Results" / f"{masked.stem}_featuresN.hdf5"
        vp = VideoPaths(
            name=masked.stem,
            strain=strain_from_video(masked.stem),
            masked=masked,
            features=features,
        )
        if vp.has_all:
            found.append(vp)
    return found


def video_index(root: Optional[Path] = None) -> Dict[str, VideoPaths]:
    return {vp.name: vp for vp in discover_videos(root)}


def get_video(name: str, root: Optional[Path] = None) -> VideoPaths:
    idx = video_index(root)
    key = resolve_video_name(name, idx)
    if key is None:
        raise KeyError(
            f"unknown video {name!r}; known videos: {sorted(idx)}"
        )
    return idx[key]
