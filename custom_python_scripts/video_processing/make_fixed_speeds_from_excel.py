#!/usr/bin/env python3
"""
Build ``*_speed_fixed.csv`` files by inverting signed speed over flip ranges.

Ranges can come from either:

1. Excel workbook sheets (legacy)::

       row 1: ``W1``, ``W2 (366)``, ``W110``, …
       later rows: ``start, end`` inclusive frame ranges

2. Annotator CSV from ``head_tail_flips.flip_range_tool``::

       video,worm,frame_start,frame_end,...

3. ``--no-flips``: no corrections; every fixed file is a copy of the raw
   speed (use when Tierpsy's head/tail assignment needed no fixing).

For each worm with a ``{video}_TS_{Wn}_speed.csv`` in the CSV folder, writes
``{video}_TS_{Wn}_speed_fixed.csv`` beside it. Flip ranges are applied when
present; worms with no listed ranges are copy-through (same ``time,speed``,
no sign flips) so the fixed set stays complete.

The CSV folder defaults to ``<data root>/Tierpsy_csvs/<strain>/`` (where
export_tierpsy_csvs.py writes), with the strain taken from the video stem.
The data root is ``../Sample_data/`` unless ``--data-root`` or the
``WORM_DATA_ROOT`` environment variable says otherwise.

Examples (run from custom_python_scripts/, ``conda activate worm-analysis``)
--------
# No corrections needed (sample data)
python video_processing/make_fixed_speeds_from_excel.py \\
  --no-flips --video 071026_HML1630_01

# Annotator CSV from the flip-range tool
python video_processing/make_fixed_speeds_from_excel.py \\
  --ranges head_tail_flips/data/071026_HML1630_01_ht_flip_ranges.csv

# Excel workbook: one sheet per video, one column per worm (W1, W2, ...),
# cells "start, end" (inclusive frame ranges)
python video_processing/make_fixed_speeds_from_excel.py \\
  --excel my_frame_flips.xlsx --sheet 1630_01 \\
  --video 071026_HML1630_01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worm_analysis import tierpsy_data  # noqa: E402
from worm_analysis.io import (  # noqa: E402
    invert_speed_ranges,
    load_flip_ranges_csv,
    load_flip_ranges_xlsx,
)


def _fixed_path_for_speed(speed_path: Path) -> Path:
    """``…_TS_W3_speed.csv`` → ``…_TS_W3_speed_fixed.csv``."""
    name = speed_path.name
    if name.endswith("_speed_fixed.csv"):
        return speed_path
    if name.endswith("_speed.csv"):
        return speed_path.with_name(name[: -len("_speed.csv")] + "_speed_fixed.csv")
    if name.endswith("_fixed.csv"):
        return speed_path
    stem = speed_path.stem
    return speed_path.with_name(f"{stem}_speed_fixed.csv")


def _speed_worms_on_disk(csv_dir: Path, video: str) -> dict[str, Path]:
    """
    Map ``W{n}`` → raw speed path for every ``{video}_TS_W*_speed.csv`` in
    ``csv_dir`` (excludes already-fixed files).
    """
    import re

    out: dict[str, Path] = {}
    pat = re.compile(
        rf"^{re.escape(video)}_TS_(W\d+)_speed\.csv$",
        flags=re.IGNORECASE,
    )
    for path in sorted(csv_dir.glob(f"{video}_TS_W*_speed.csv")):
        if path.name.endswith("_speed_fixed.csv"):
            continue
        m = pat.match(path.name)
        if not m:
            continue
        out[m.group(1).upper()] = path
    return out


def write_fixed_speed(
    speed_csv: Path,
    frame_ranges: list[tuple[int, int]],
    *,
    out_csv: Path | None = None,
) -> tuple[Path, int]:
    """
    Invert ``speed`` over ``frame_ranges`` and write a fixed CSV.

    Returns ``(output_path, n_rows_inverted)``.
    """
    speed_csv = Path(speed_csv)
    out_csv = Path(out_csv) if out_csv is not None else _fixed_path_for_speed(speed_csv)
    raw = pd.read_csv(speed_csv)
    if "time" not in raw.columns or "speed" not in raw.columns:
        raise ValueError(f"{speed_csv} must have 'time' and 'speed' columns")

    # invert_speed_ranges recovers frames from time (frame-index or assumed-6).
    fixed = invert_speed_ranges(raw[["time", "speed"]].copy(), frame_ranges, fps=6.0)
    n_inv = int(fixed["flipped"].sum()) if "flipped" in fixed.columns else 0
    out = fixed[["time", "speed"]].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv, n_inv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--excel",
        type=Path,
        help="Workbook with per-worm flip ranges (use with --sheet)",
    )
    src.add_argument(
        "--ranges",
        type=Path,
        help="Annotator CSV from flip_range_tool (video,worm,frame_start,frame_end)",
    )
    src.add_argument(
        "--no-flips",
        action="store_true",
        help="No corrections: write every fixed file as a copy of the raw speed "
        "(requires --video)",
    )
    p.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name (required with --excel)",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Video stem used in CSV names (required for --excel and "
        "--no-flips; optional for --ranges when the file has a single video)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root holding Tierpsy_csvs/ "
        "(default: $WORM_DATA_ROOT, else ../Sample_data/)",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory containing raw *_speed.csv exports "
        "(default: <data root>/Tierpsy_csvs/<strain>/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report only; do not write files",
    )
    args = p.parse_args(argv)
    if args.data_root is not None:
        tierpsy_data.set_data_root(args.data_root)

    if args.no_flips:
        if not args.video:
            print("--video is required with --no-flips", file=sys.stderr)
            return 1
        ranges = {}
        video = str(args.video)
        print(f"no flips  video={video}  (copy-through for every worm)")
    elif args.ranges is not None:
        ranges_path = args.ranges.resolve()
        if not ranges_path.is_file():
            print(f"Ranges CSV not found: {ranges_path}", file=sys.stderr)
            return 1
        try:
            ranges, video = load_flip_ranges_csv(ranges_path, video=args.video)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not video:
            if args.video:
                video = str(args.video)
            else:
                print(
                    f"No video in {ranges_path.name}; pass --video",
                    file=sys.stderr,
                )
                return 1
        print(
            f"ranges={ranges_path.name}  video={video}  "
            f"annotated_worms={len(ranges)}"
        )
    else:
        if not args.sheet:
            print("--sheet is required with --excel", file=sys.stderr)
            return 1
        if not args.video:
            print("--video is required with --excel", file=sys.stderr)
            return 1
        excel = args.excel.resolve()
        if not excel.is_file():
            print(f"Excel not found: {excel}", file=sys.stderr)
            return 1
        ranges = load_flip_ranges_xlsx(excel, sheet_name=args.sheet)
        video = str(args.video)
        if not ranges:
            print(
                f"No worm columns found in {excel.name} sheet={args.sheet!r}",
                file=sys.stderr,
            )
            return 1
        print(f"sheet={args.sheet!r}  video={video}  sheet_worms={len(ranges)}")

    if args.csv_dir is not None:
        csv_dir = args.csv_dir.resolve()
    else:
        csv_dir = tierpsy_data.strain_csv_dir(tierpsy_data.strain_from_video(video))
    print(f"csv-dir:  {csv_dir}")
    if not csv_dir.is_dir():
        print(f"CSV dir not found: {csv_dir}", file=sys.stderr)
        return 1

    on_disk = _speed_worms_on_disk(csv_dir, video)
    if not on_disk:
        print(
            f"No {video}_TS_W*_speed.csv files in {csv_dir}",
            file=sys.stderr,
        )
        return 1

    # Every speed file on disk gets a fixed output. Ranges from Excel/CSV are
    # applied when present; otherwise copy-through (empty invert list).
    worms = sorted(on_disk, key=lambda w: int(w[1:]))
    orphan_ranges = sorted(set(ranges) - set(on_disk), key=lambda w: int(w[1:]))
    if orphan_ranges:
        print(
            f"WARNING: ranges for {orphan_ranges} but no matching speed CSV",
            file=sys.stderr,
        )

    wrote = []
    n_copy = 0
    n_flip = 0
    for wid in worms:
        flips = list(ranges.get(wid, []))
        speed_path = on_disk[wid]
        out_path = _fixed_path_for_speed(speed_path)
        if flips:
            n_flip += 1
            print(f"  {wid}: {len(flips)} range(s) → {out_path.name}")
            for a, b in flips:
                print(f"       invert {a}-{b}")
        else:
            n_copy += 1
            print(f"  {wid}: 0 range(s) → {out_path.name}  (copy-through)")
        if args.dry_run:
            wrote.append(out_path)
            continue
        out, n_inv = write_fixed_speed(speed_path, flips, out_csv=out_path)
        wrote.append(out)
        print(f"       wrote {out.name}  ({n_inv} rows inverted)")

    print(
        f"\nDone. {'Would write' if args.dry_run else 'Wrote'} {len(wrote)} "
        f"fixed file(s) ({n_flip} with flips, {n_copy} copy-through)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
