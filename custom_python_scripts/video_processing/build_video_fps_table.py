#!/usr/bin/env python3
"""
Estimate true FPS for every video and write video_fps.csv.

Every recording is 10 minutes (600 s), so true FPS = frames in the video /
600. The frame count is the longest timeline among the video's worms, read
from each fixed-speed CSV's sibling ``*_blob_perimeter.csv`` (or the speed
CSV itself if that is missing). Old second-based exports are converted with
assumed_fps=6 via round(time * 6).

Reads every strain folder under ``<data root>/Tierpsy_csvs/`` and writes
``<data root>/video_fps.csv``. The data root is ``../Sample_data/`` unless
``--data-root`` or the ``WORM_DATA_ROOT`` environment variable says otherwise.

Examples (run from custom_python_scripts/)
--------
  conda activate worm-analysis
  python video_processing/build_video_fps_table.py
  python video_processing/build_video_fps_table.py --data-root /path/to/my_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worm_analysis import tierpsy_data  # noqa: E402
from worm_analysis.video_fps import (  # noqa: E402
    ASSUMED_FPS,
    VIDEO_DURATION_S,
    build_video_fps_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-video true-FPS table from the fixed-speed CSVs.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root holding Tierpsy_csvs/ (default: $WORM_DATA_ROOT, else ../Sample_data/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <data root>/video_fps.csv).",
    )
    p.add_argument(
        "--assumed-fps",
        type=float,
        default=ASSUMED_FPS,
        help="FPS assumed when old CSVs store time in seconds (default: 6).",
    )
    p.add_argument(
        "--duration-s",
        type=float,
        default=VIDEO_DURATION_S,
        help="True wall-clock duration of each video in seconds (default: 600).",
    )
    p.add_argument(
        "--pattern",
        default="*_fixed.csv",
        help="Glob for speed CSVs inside each strain folder (default: *_fixed.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root is not None:
        tierpsy_data.set_data_root(args.data_root)
    root = tierpsy_data.data_root()

    dataset_dirs = tierpsy_data.strain_csv_dirs()
    if not dataset_dirs:
        raise SystemExit(f"No strain folders found under {tierpsy_data.csv_dir()}")
    print(f"data root: {root}")
    for label, path in dataset_dirs:
        print(f"  {label}: {path}")

    table = build_video_fps_table(
        dataset_dirs,
        pattern=args.pattern,
        assumed_fps=args.assumed_fps,
        duration_s=args.duration_s,
    )
    out = args.output if args.output is not None else tierpsy_data.fps_table_path()
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    to_write = table.copy()
    if len(to_write):
        # Keep date/video as zero-padded strings in the CSV.
        to_write["recording_date"] = to_write["recording_date"].astype(str)
        to_write["video_number"] = to_write["video_number"].astype(str)
        # Record folders relative to the data root so the table is portable.
        to_write["data_dir"] = [
            str(Path(d).relative_to(root)) if Path(d).is_relative_to(root) else d
            for d in to_write["data_dir"]
        ]
    to_write.to_csv(out, index=False)

    print(f"Wrote {len(table)} video(s) → {out}")
    if len(table):
        cols = [
            "dataset",
            "recording_date",
            "video_number",
            "n_frames",
            "fps_true",
            "frames_per_minute",
            "source",
            "n_worms_used",
        ]
        print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
