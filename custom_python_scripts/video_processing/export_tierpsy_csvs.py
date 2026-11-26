#!/usr/bin/env python3
"""
Export per-worm Tierpsy timeseries CSVs from *_featuresN.hdf5.

Run from custom_python_scripts/ in the ``worm-analysis`` environment::

    conda activate worm-analysis
    python video_processing/export_tierpsy_csvs.py --video HML1630_01 --list-worms
    python video_processing/export_tierpsy_csvs.py --video HML1630_01 --animals 1,3,8
    python video_processing/export_tierpsy_csvs.py --video HML1630_01 --animals all
    python video_processing/export_tierpsy_csvs.py   # interactive prompts

Where files are read and written (see worm_analysis/tierpsy_data.py)::

    <data root>/Videos/Results/<video>_featuresN.hdf5     input (Tierpsy output)
    <data root>/Tierpsy_csvs/<strain>/                    output CSVs

The data root is ``../Sample_data/`` unless ``--data-root`` or the
``WORM_DATA_ROOT`` environment variable says otherwise. The strain folder
name is taken from the video stem (``071026_HML1630_01`` -> ``HML1630``);
override it with ``--strain`` or pick any folder with ``--output``.

Animal numbers map to Tierpsy ``worm_index_manual`` (W1 -> 1), matching
the worm numbers shown in Tierpsy Tracker Viewer.

Writes the two features the analysis reads, per worm:
  - *_speed.csv           raw whole-body speed (not orientation-corrected);
                          becomes *_speed_fixed.csv after head/tail correction
  - *_blob_perimeter.csv  blob perimeter (perimeter median in the output
                          tables; frame count for video_fps.csv)

Other Tierpsy timeseries can be added to DEFAULT_EXPORTS below if needed;
the analysis ignores them.

Speeds are raw Tierpsy values (no orientation sign flips). Next, make the
head/tail-corrected *_speed_fixed.csv files the analysis reads with
make_fixed_speeds_from_excel.py (``--no-flips`` if nothing needs correcting).

By default the ``time`` column is the video frame index (Tierpsy
``frame_number``), i.e. the same values ``timestamp_time`` would have if
the recording FPS were 1. Use ``--time-unit seconds`` for wall-clock
seconds from the HDF5 ``timestamp_time`` field.

Only needs numpy, pandas and h5py (also runs in a Tierpsy environment).
Does not import worm_analysis.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT.parent  # custom_python_scripts/
DATA_ROOT_ENV = "WORM_DATA_ROOT"


def default_data_root():
    # type: () -> Path
    """``$WORM_DATA_ROOT`` if set, else ``Sample_data/`` next to custom_python_scripts/."""
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return SCRIPTS_DIR.parent / "Sample_data"


def strain_from_stem(stem):
    # type: (str) -> str
    """``071026_HML1630_01`` -> ``HML1630``; ``070926_N2_demag_01`` -> ``N2``."""
    parts = str(stem).split("_")
    if len(parts) >= 2 and re.match(r"^\d{6}$", parts[0]):
        return parts[1]
    return parts[0]

PathLike = Union[str, Path]

# (output_suffix, hdf5_source, column_name)
# hdf5_source: timeseries field, or "blob:<field>" for blob_features.
# Only the features the analysis reads. To export more, add rows such as
#   ("angular_velocity_midbody", "angular_velocity_midbody", "angular_velocity_midbody")
#   ("speed_midbody", "speed_midbody", "speed_midbody")
#   ("blob_area", "blob:area", "blob_area")
DEFAULT_EXPORTS = (
    ("speed", "speed", "speed"),
    ("blob_perimeter", "blob:perimeter", "blob_perimeter"),
)

DEFAULT_FEATURES_GLOBS = (
    "*/Results/*_featuresN.hdf5",
    "*/*_featuresN.hdf5",
)


def parse_animal_ids(spec):
    # type: (Union[str, int, Sequence[Union[str, int]]]) -> List[int]
    """
    Parse animal numbers from ints, ``W3``/``w3``, or comma/space lists
    (``1,2,5`` / ``W1 W3`` / ``1-3``).
    """
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        parts = re.split(r"[\s,;]+", spec.strip())
        out = []  # type: List[int]
        for part in parts:
            if not part:
                continue
            m = re.fullmatch(r"[Ww]?(\d+)\s*-\s*[Ww]?(\d+)", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                lo, hi = (a, b) if a <= b else (b, a)
                out.extend(range(lo, hi + 1))
                continue
            m = re.fullmatch(r"[Ww]?(\d+)", part)
            if not m:
                raise ValueError("Unrecognized animal id: {!r}".format(part))
            out.append(int(m.group(1)))
        return sorted(set(out))
    out = []
    for item in spec:
        out.extend(parse_animal_ids(item))
    return sorted(set(out))


def features_stem(features_path):
    # type: (PathLike) -> str
    """``071026_HML1641_01_featuresN.hdf5`` -> ``071026_HML1641_01``."""
    name = Path(features_path).name
    if name.endswith("_featuresN.hdf5"):
        return name[: -len("_featuresN.hdf5")]
    if name.endswith(".hdf5"):
        return Path(name).stem
    return name


def default_output_dir(features_path, data_root=None, strain=None):
    # type: (PathLike, Optional[PathLike], Optional[str]) -> Path
    """``<data root>/Tierpsy_csvs/<strain>/`` for this video."""
    root = Path(data_root) if data_root is not None else default_data_root()
    strain = strain or strain_from_stem(features_stem(features_path))
    return root / "Tierpsy_csvs" / strain


def find_features_file(video, search_roots=None, globs=DEFAULT_FEATURES_GLOBS):
    # type: (str, Optional[Sequence[PathLike]], Sequence[str]) -> Path
    """
    Resolve a video token to a ``*_featuresN.hdf5`` path.

    ``video`` may be a path, a full stem (``071026_HML1641_01``), or a
    distinctive fragment (``HML1641_01``, ``N2_demag_02``).
    """
    token = video.strip()
    as_path = Path(token).expanduser()
    if as_path.is_file():
        return as_path.resolve()
    candidate = Path("{}_featuresN.hdf5".format(token))
    if candidate.is_file():
        return candidate.resolve()
    if as_path.with_name(as_path.name + "_featuresN.hdf5").is_file():
        return as_path.with_name(as_path.name + "_featuresN.hdf5").resolve()

    if search_roots is None:
        roots = [Path.cwd()]
    else:
        roots = [Path(r) for r in search_roots]

    matches = []  # type: List[Path]
    needle = token.lower()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        candidates = []  # type: List[Path]
        for pattern in globs:
            candidates.extend(root.glob(pattern))
        candidates.extend(root.rglob("*_featuresN.hdf5"))
        for path in candidates:
            stem = features_stem(path).lower()
            name = path.name.lower()
            if needle == stem or needle in stem or needle in name:
                matches.append(path.resolve())

    uniq = sorted(set(matches))
    if not uniq:
        raise FileNotFoundError(
            "No *_featuresN.hdf5 found for video={!r} under {}".format(
                video, list(roots)
            )
        )
    if len(uniq) > 1:
        exact = [p for p in uniq if features_stem(p).lower() == needle]
        if len(exact) == 1:
            return exact[0]
        listing = "\n  ".join(str(p) for p in uniq)
        raise FileNotFoundError(
            "Ambiguous video={!r}; matches:\n  {}\n"
            "Pass a fuller stem or --features path.".format(video, listing)
        )
    return uniq[0]


def list_worms(features_path):
    # type: (PathLike) -> List[Tuple[int, int, float, float, int, int]]
    """
    Summarize animals in a featuresN file (worm_index_manual).

    Returns list of tuples:
      (animal, n_rows, t_min, t_max, frame_min, frame_max)
    """
    with h5py.File(str(Path(features_path)), "r") as f:
        traj = f["trajectories_data"][:]

    manual = traj["worm_index_manual"]
    infos = []
    for animal in sorted(int(x) for x in np.unique(manual) if int(x) > 0):
        m = manual == animal
        infos.append(
            (
                animal,
                int(m.sum()),
                float(traj["timestamp_time"][m].min()),
                float(traj["timestamp_time"][m].max()),
                int(traj["frame_number"][m].min()),
                int(traj["frame_number"][m].max()),
            )
        )
    return infos


def format_worm_table(infos):
    # type: (Sequence[Tuple[int, int, float, float, int, int]]) -> str
    lines = [
        "{:>4}  {:>7}  {:>8}  {:>8}  {:>9}  {:>9}".format(
            "W", "n_rows", "t_min", "t_max", "frame_min", "frame_max"
        )
    ]
    for animal, n_rows, t_min, t_max, frame_min, frame_max in infos:
        lines.append(
            "W{:<3}  {:7d}  {:8.2f}  {:8.2f}  {:9d}  {:9d}".format(
                animal, n_rows, t_min, t_max, frame_min, frame_max
            )
        )
    return "\n".join(lines)


def _series_from_hdf5(source, mask, timeseries, blob):
    # type: (str, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    if source.startswith("blob:"):
        field = source.split(":", 1)[1]
        return np.asarray(blob[field][mask], dtype=float)
    return np.asarray(timeseries[source][mask], dtype=float)


def export_animals(
    features_path,
    animals,
    output_dir=None,
    time_unit="frames",
    include_frame=False,
    speed_suffix="speed",
    stem=None,
    ts_tag="TS",
    exports=DEFAULT_EXPORTS,
    overwrite=True,
):
    # type: (...) -> Dict[int, List[Path]]
    """
    Export CSVs for the given animal numbers.

    time_unit ``frames`` (default) writes ``frame_number`` into the ``time``
    column — equivalent to treating the video as 1 fps. ``seconds`` writes
    HDF5 ``timestamp_time`` (wall-clock seconds at the FPS Tierpsy used).
    """
    features_path = Path(features_path).resolve()
    if time_unit not in ("seconds", "frames"):
        raise ValueError("time_unit must be 'seconds' or 'frames'")
    if speed_suffix not in ("speed", "fixed", "speed_fixed"):
        raise ValueError(
            "speed_suffix must be 'speed', 'fixed', or 'speed_fixed'"
        )

    if output_dir is None:
        out_dir = default_output_dir(features_path)
    else:
        out_dir = Path(output_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    video_stem = stem if stem is not None else features_stem(features_path)
    animals = [int(a) for a in animals]

    with h5py.File(str(features_path), "r") as f:
        timeseries = f["timeseries_data"][:]
        traj = f["trajectories_data"][:]
        blob = f["blob_features"][:]

    if len(timeseries) != len(traj) or len(blob) != len(traj):
        raise RuntimeError(
            "Row-length mismatch in {}: timeseries={} traj={} blob={}".format(
                features_path.name, len(timeseries), len(traj), len(blob)
            )
        )

    written = {}  # type: Dict[int, List[Path]]
    available = set(int(x) for x in np.unique(traj["worm_index_manual"]))

    for animal in animals:
        if animal not in available:
            raise KeyError(
                "Animal W{} not in {}. Available: {}".format(
                    animal,
                    features_path.name,
                    sorted(a for a in available if a > 0),
                )
            )
        mask = traj["worm_index_manual"] == animal
        # frames: index by video frame (as if Tierpsy fps == 1).
        # seconds: wall-clock time stored in the HDF5 at the original fps.
        if time_unit == "seconds":
            time_vals = np.asarray(traj["timestamp_time"][mask], dtype=float)
        else:
            time_vals = np.asarray(traj["frame_number"][mask], dtype=float)
        frame_vals = np.asarray(traj["frame_number"][mask], dtype=int)

        prefix = "{}_{}_W{}".format(video_stem, ts_tag, animal)
        paths = []  # type: List[Path]
        for suffix, source, col in exports:
            file_suffix = speed_suffix if suffix == "speed" else suffix
            path = out_dir / "{}_{}.csv".format(prefix, file_suffix)
            if path.exists() and not overwrite:
                raise FileExistsError(str(path))

            values = _series_from_hdf5(source, mask, timeseries, blob)
            if include_frame:
                df = pd.DataFrame(
                    {
                        "time": time_vals,
                        "frame": frame_vals,
                        col: values,
                    }
                )
            else:
                df = pd.DataFrame({"time": time_vals, col: values})
            df.to_csv(str(path), index=False)
            paths.append(path)
        written[animal] = paths
    return written


def _prompt(msg, default=None):
    # type: (str, Optional[str]) -> str
    if default:
        suffix = " [{}]".format(default)
    else:
        suffix = ""
    value = input("{}{}: ".format(msg, suffix)).strip()
    if not value and default is not None:
        return default
    return value


def parse_args(argv=None):
    # type: (Optional[List[str]]) -> argparse.Namespace
    p = argparse.ArgumentParser(
        description="Export Tierpsy featuresN timeseries to per-worm pipeline CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video",
        help="Video stem or fragment (e.g. 071026_HML1641_01 or HML1641_01).",
    )
    p.add_argument(
        "--features",
        type=Path,
        help="Path to a *_featuresN.hdf5 file (skips video search).",
    )
    p.add_argument(
        "--animals",
        nargs="+",
        help="Animal numbers: 1 2 5, W1,W3, 1-11, or 'all' "
        "(every worm_index_manual in the file).",
    )
    p.add_argument(
        "--list-worms",
        action="store_true",
        help="List worm_index_manual animals in the features file and exit.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root holding Videos/ and Tierpsy_csvs/ "
        "(default: $WORM_DATA_ROOT, else ../Sample_data/).",
    )
    p.add_argument(
        "--strain",
        default=None,
        help="Strain folder name under Tierpsy_csvs/ "
        "(default: taken from the video stem, e.g. HML1630).",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <data root>/Tierpsy_csvs/<strain>/).",
    )
    p.add_argument(
        "--time-unit",
        choices=("frames", "seconds"),
        default="frames",
        help="Value written to the time column. Default: frames "
        "(video frame index, as if fps=1). Use seconds for wall-clock "
        "timestamp_time from the HDF5.",
    )
    p.add_argument(
        "--include-frame",
        action="store_true",
        help="Also write a separate frame column (useful with "
        "--time-unit seconds).",
    )
    p.add_argument(
        "--speed-suffix",
        choices=("speed", "fixed", "speed_fixed"),
        default="speed",
        help="Whole-body speed filename tag (default: speed = raw export).",
    )
    p.add_argument(
        "--search-root",
        type=Path,
        default=None,
        help="Folder to search for featuresN files "
        "(default: <data root>/Videos/).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and animals but do not write files.",
    )
    return p.parse_args(argv)


def interactive_fill(args):
    # type: (argparse.Namespace) -> argparse.Namespace
    if not args.features and not args.video:
        args.video = _prompt("Video stem or fragment (e.g. HML1641_01)")
    if not args.list_worms and not args.animals:
        args.animals = [_prompt("Animals (e.g. 1,3,8 or 1-11 or W1 W2)")]
    return args


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    args = parse_args(argv)
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root is not None
        else default_data_root()
    )
    search_root = (
        Path(args.search_root).expanduser().resolve()
        if args.search_root is not None
        else data_root / "Videos"
    )
    if not any([args.features, args.video, args.animals, args.list_worms]):
        print("Tierpsy -> pipeline CSV export")
        print("Leave blank where a default is shown.\n")
        args = interactive_fill(args)

    if not args.features and not args.video:
        print("error: provide --video or --features", file=sys.stderr)
        return 2

    try:
        if args.features:
            features = Path(args.features).expanduser().resolve()
        else:
            features = find_features_file(
                args.video, search_roots=[search_root]
            )
    except FileNotFoundError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    if not features.is_file():
        print("error: features file not found: {}".format(features), file=sys.stderr)
        return 1

    stem = features_stem(features)
    print("features: {}".format(features))
    print("stem:     {}".format(stem))

    infos = list_worms(features)
    if args.list_worms:
        print(format_worm_table(infos))
        return 0

    if not args.animals:
        print("error: provide --animals (or use --list-worms)", file=sys.stderr)
        return 2

    try:
        if [str(a).strip().lower() for a in args.animals] == ["all"]:
            animals = [info[0] for info in infos]
        else:
            animals = parse_animal_ids(args.animals)
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    if args.output:
        out_dir = Path(args.output).expanduser().resolve()
    else:
        out_dir = default_output_dir(features, data_root=data_root, strain=args.strain)

    speed_suffix = args.speed_suffix

    include_frame = bool(args.include_frame)

    print("animals:  {}".format(", ".join("W{}".format(a) for a in animals)))
    print("output:   {}".format(out_dir))
    if args.time_unit == "frames":
        print("time:     frame index (as if fps=1)")
    elif include_frame:
        print("time:     seconds + frame col")
    else:
        print("time:     seconds")
    print("speed as: *_{}.csv".format(speed_suffix))

    if args.dry_run:
        print("dry-run: no files written")
        return 0

    try:
        written = export_animals(
            features,
            animals,
            out_dir,
            time_unit=args.time_unit,
            include_frame=include_frame,
            speed_suffix=speed_suffix,
        )
    except KeyError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        print("Available animals:")
        print(format_worm_table(infos))
        return 1

    n_files = sum(len(v) for v in written.values())
    print("wrote {} files for {} animals:".format(n_files, len(written)))
    for animal, paths in written.items():
        print(
            "  W{}: {} (+ {} siblings)".format(
                animal, paths[0].name, len(paths) - 1
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
