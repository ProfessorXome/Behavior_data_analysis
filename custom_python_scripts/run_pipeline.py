#!/usr/bin/env python3
"""
Offline worm locomotion analysis pipeline.

Runs the same workflow as worm_analysis/worm_analysis.ipynb without Jupyter:
  1. Load speed CSVs
  2. Compute metrics (activity filter from worm_analysis.metrics)
  3. Write general / active / normalized summary tables
  4. Write per-worm frames, per-minute, and bout tables
  5. Optionally save plots

Input data
----------
Reads the data root (``../Sample_data/`` unless ``--data-root`` or the
``WORM_DATA_ROOT`` environment variable says otherwise)::

    <data root>/Tierpsy_csvs/<strain>/*_speed_fixed.csv   (+ sibling feature CSVs)
    <data root>/Videos/Results/<video>_featuresN.hdf5     centroid track for activity
    <data root>/Videos/MaskedVideos/<video>.hdf5          µm/px scale
    <data root>/video_fps.csv                             per-video true FPS

See ../Sample_data/README.md for how to lay out your own data.

Examples (run from custom_python_scripts/)
--------
  conda activate worm-analysis

  # every strain folder in ../Sample_data/Tierpsy_csvs → output/<strain>/
  python run_pipeline.py

  # one strain folder, with plots
  python run_pipeline.py \\
      --data-dir ../Sample_data/Tierpsy_csvs/HML1630 \\
      --plot-worms W4 W7 \\
      --plot-summary

  # your own data, laid out like ../Sample_data/
  python run_pipeline.py --data-root /path/to/my_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worm_analysis import (  # noqa: E402
    load_speed_directory,
    compute_metrics,
    compute_continuous_minute_metrics,
    summarize_worms,
    build_summary_tables,
    MIN_FRAMES,
    MIN_FRAMES_FRAC,
)
from worm_analysis import tierpsy_data  # noqa: E402
from worm_analysis.video_fps import load_video_fps_table  # noqa: E402
from worm_analysis import metrics as metrics_mod  # noqa: E402
from worm_analysis.plotting import plot_metrics_board, plot_worm_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline worm locomotion analysis (notebook pipeline).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root holding Tierpsy_csvs/, Videos/ and video_fps.csv "
        "(default: $WORM_DATA_ROOT, else ../Sample_data/).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="One folder of worm speed CSVs (sibling feature CSVs auto-merged). "
        "Default: run every strain folder under <data root>/Tierpsy_csvs/.",
    )
    p.add_argument(
        "--pattern",
        default="*_fixed.csv",
        help="Glob for speed CSVs inside --data-dir (default: *_fixed.csv).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Fallback FPS if video_fps.csv has no row (default: use table / 6.0).",
    )
    p.add_argument(
        "--fps-table",
        type=Path,
        default=None,
        help="Per-video FPS CSV (default: <data root>/video_fps.csv if present).",
    )
    p.add_argument(
        "--dataset-label",
        default="",
        help="Dataset label for FPS lookup (e.g. HML1641). Inferred from path if empty.",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=None,
        help="Skip worms with fewer rows than this absolute count. "
        # trailing "%" doubles the percent sign so argparse prints a literal "%"
        f"Default (omit flag): require ≥{MIN_FRAMES_FRAC:.0%}% of that video's "
        "max frames (video_fps.csv n_frames). Use 0 to keep all.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/<strain>/ in the repository).",
    )
    p.add_argument(
        "--plot-worms",
        nargs="*",
        default=None,
        help="Worm IDs to save individual metric-board plots for "
        "(e.g. W4 W7). Omit flag for none; pass with no IDs to plot all.",
    )
    p.add_argument(
        "--plot-summary",
        action="store_true",
        help="Save cross-worm summary bar chart.",
    )
    return p.parse_args()



def run(args: argparse.Namespace) -> Path:
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    min_frames = MIN_FRAMES if args.min_frames is None else args.min_frames
    if min_frames is None:
        print(f"min-frames:  ≥{MIN_FRAMES_FRAC:.0%} of each video's max frames")
    else:
        print(f"min-frames:  {min_frames}")
    leaf = data_dir.name
    parent = data_dir.parent.name if data_dir.parent != data_dir else ""
    generic = {"FIXED", "fixed", "csv_files", "Results", "output"}
    run_name = f"{parent}_{leaf}" if leaf in generic and parent else leaf

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = ROOT / "output" / run_name
    elif not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"data root:   {tierpsy_data.data_root()}")
    print(f"data-dir:    {data_dir}")
    print(f"pattern:     {args.pattern}")

    fps_table_path = args.fps_table
    if fps_table_path is None:
        candidate = tierpsy_data.fps_table_path()
        fps_table_path = candidate if candidate.is_file() else None
    elif not fps_table_path.is_absolute():
        fps_table_path = (ROOT / fps_table_path).resolve()

    fps_table = None
    if fps_table_path is not None and Path(fps_table_path).is_file():
        fps_table = load_video_fps_table(fps_table_path)
        print(f"fps-table:   {fps_table_path} ({len(fps_table)} videos)")
    else:
        print("fps-table:   (none — using fallback FPS)")

    fallback_fps = 6.0 if args.fps is None else float(args.fps)
    print(f"fps fallback:{fallback_fps}")
    print(
        f"activity:    centroid path-rate > {metrics_mod.CENTROID_ACTIVITY_THRESHOLD_BL_S:g} BL/s "
        f"over {metrics_mod.CENTROID_ACTIVITY_WINDOW_S:g}s"
    )
    sustain = (
        f"on (sign dwell ≥{metrics_mod.REVERSAL_DWELL_FRAMES} frames)"
        if metrics_mod.USE_REVERSAL_SUSTAIN_FILTER
        else "off (single +→−)"
    )
    print(f"reversals:   dwell filter {sustain}")
    print(
        f"cont. minutes: gaps ≤{metrics_mod.CONTINUOUS_SPEED_GAP_FRAMES} frames, "
        f"{metrics_mod.FRAMES_PER_MINUTE} speed samples / minute"
        + (
            "; backfill leftovers ≥½ window"
            if metrics_mod.USE_CONTINUOUS_MINUTE_BACKFILL
            else "; no backfill"
        )
        + " (recordings need a *_blob_perimeter.csv)"
    )
    print(f"output-dir:  {out_dir}")

    from worm_analysis.video_fps import _clean_dataset_label

    dataset_label = args.dataset_label or _clean_dataset_label(data_dir)
    worms = load_speed_directory(
        data_dir,
        pattern=args.pattern,
        fps=fallback_fps,
        min_frames=min_frames,
        fps_table=fps_table,
        dataset=dataset_label,
    )
    print(f"\nLoaded {len(worms)} worm(s) [dataset={dataset_label}]:")
    for wid, df in worms.items():
        fps_v = df["fps_true"].iloc[0] if "fps_true" in df.columns else float("nan")
        fpm_v = df["frames_per_minute"].iloc[0] if "frames_per_minute" in df.columns else ""
        print(
            f"  {wid:6s}  n={len(df):5d}  "
            f"fps_true={fps_v:.4f}  fpm={fpm_v}"
        )
    results = {}
    for wid, wdf in worms.items():
        print(f"computing {wid} ...")
        results[wid] = compute_metrics(wdf)

    if not results:
        raise SystemExit("No worms left to analyze.")

    tables = build_summary_tables(list(results.values()))
    general = tables["general"]
    active = tables["active"]
    general_per_duration = tables["general_per_duration"]
    active_per_duration = tables["active_per_duration"]
    active_per_active_duration = tables["active_per_active_duration"]
    summary = summarize_worms(list(results.values())).sort_values("worm_id").reset_index(
        drop=True
    )

    prefix = run_name
    paths = {
        "general": out_dir / f"{prefix}_general.csv",
        "active": out_dir / f"{prefix}_active.csv",
        "general_per_duration": out_dir / f"{prefix}_general_per_duration.csv",
        "active_per_duration": out_dir / f"{prefix}_active_per_duration.csv",
        "active_per_active_duration": out_dir / f"{prefix}_active_per_active_duration.csv",
        "summary": out_dir / f"{prefix}_summary.csv",
    }
    general.to_csv(paths["general"], index=False)
    active.to_csv(paths["active"], index=False)
    general_per_duration.to_csv(paths["general_per_duration"], index=False)
    active_per_duration.to_csv(paths["active_per_duration"], index=False)
    active_per_active_duration.to_csv(paths["active_per_active_duration"], index=False)
    summary.to_csv(paths["summary"], index=False)

    print("\ncomputing continuous minutes (worm-agnostic) ...")
    _minute_results, minute_tables = compute_continuous_minute_metrics(worms)
    minute_paths = {
        "continuous_minutes_meta": out_dir / f"{prefix}_continuous_minutes_meta.csv",
        "continuous_minutes_general": out_dir / f"{prefix}_continuous_minutes_general.csv",
        "continuous_minutes_active": out_dir / f"{prefix}_continuous_minutes_active.csv",
        "continuous_minutes_general_per_duration": (
            out_dir / f"{prefix}_continuous_minutes_general_per_duration.csv"
        ),
        "continuous_minutes_active_per_duration": (
            out_dir / f"{prefix}_continuous_minutes_active_per_duration.csv"
        ),
        "continuous_minutes_active_per_active_duration": (
            out_dir / f"{prefix}_continuous_minutes_active_per_active_duration.csv"
        ),
    }
    minute_tables["minute_meta"].to_csv(
        minute_paths["continuous_minutes_meta"], index=False
    )
    minute_tables["general"].to_csv(
        minute_paths["continuous_minutes_general"], index=False
    )
    minute_tables["active"].to_csv(
        minute_paths["continuous_minutes_active"], index=False
    )
    minute_tables["general_per_duration"].to_csv(
        minute_paths["continuous_minutes_general_per_duration"], index=False
    )
    minute_tables["active_per_duration"].to_csv(
        minute_paths["continuous_minutes_active_per_duration"], index=False
    )
    minute_tables["active_per_active_duration"].to_csv(
        minute_paths["continuous_minutes_active_per_active_duration"], index=False
    )
    n_minutes = len(minute_tables["minute_meta"])
    print(f"continuous minutes extracted: {n_minutes}")

    for wid, res in results.items():
        res.frames.to_csv(out_dir / f"{wid}_frames.csv", index=False)
        res.per_minute.to_csv(out_dir / f"{wid}_per_minute.csv", index=False)
        res.bouts.to_csv(out_dir / f"{wid}_bouts.csv", index=False)

    print("\nGENERAL")
    print(general.to_string(index=False))
    print("\nACTIVE")
    print(active.to_string(index=False))
    print("\nGENERAL / CAPTURED-SPEED DURATION (per min)")
    print(general_per_duration.to_string(index=False))
    print("\nACTIVE / CAPTURED-SPEED DURATION (per min)")
    print(active_per_duration.to_string(index=False))
    print("\nACTIVE / ACTIVE DURATION (per min of active time)")
    print(active_per_active_duration.to_string(index=False))
    if n_minutes:
        print("\nCONTINUOUS MINUTES (general, head)")
        print(minute_tables["general"].head(12).to_string(index=False))

    print("\nWrote tables:")
    for label, path in paths.items():
        print(f"  {label:28s} {path}")
    for label, path in minute_paths.items():
        print(f"  {label:28s} {path}")
    print(f"  per-worm frames/per_minute/bouts → {out_dir}")

    # Individual worm plots
    if args.plot_worms is not None:
        plot_ids = list(args.plot_worms) if args.plot_worms else list(results.keys())
        for wid in plot_ids:
            if wid not in results:
                print(f"plot skip {wid}: not in results")
                continue
            fig = plot_metrics_board(results[wid])
            dest = plots_dir / f"{wid}_metrics_board.png"
            fig.savefig(dest, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  plot {dest}")

    if args.plot_summary:
        fig = plot_worm_summary(summary, figsize=(12, 4))
        dest = plots_dir / f"{prefix}_summary.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot {dest}")

    return out_dir


def main() -> None:
    args = parse_args()
    if args.data_root is not None:
        tierpsy_data.set_data_root(args.data_root)
    if args.data_dir is not None:
        run(args)
        return
    strains = tierpsy_data.strain_csv_dirs()
    if not strains:
        raise SystemExit(
            f"No strain folders under {tierpsy_data.csv_dir()}. "
            "Pass --data-dir, or see ../Sample_data/README.md for the data layout."
        )
    base_out = args.output_dir
    for strain, folder in strains:
        print(f"\n===== {strain} =====")
        args.data_dir = folder
        if base_out is not None:
            args.output_dir = Path(base_out) / strain
        run(args)


if __name__ == "__main__":
    main()
