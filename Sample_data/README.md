# Sample data

One real recording, `071026_HML1630_01` (strain HML1630, 10 tracked worms),
laid out the way every script in this repository expects its input. Use it to
try the pipeline, and copy its layout for your own data.

## Layout

```
Sample_data/                         <- the "data root" (next to custom_python_scripts/)
├── video_fps.csv                    per-video true FPS
├── Videos/                          Tierpsy output, unchanged from how Tierpsy writes it
│   ├── 071026_HML1630_01.avi                raw video (download from Zenodo; see below)
│   ├── MaskedVideos/
│   │   └── 071026_HML1630_01.hdf5           masked video (µm/px scale; flip-range tool)
│   └── Results/
│       └── 071026_HML1630_01_featuresN.hdf5 tracks, skeletons, raw speed
└── Tierpsy_csvs/                    per-worm CSVs, one folder per strain
    └── HML1630/
        ├── 071026_HML1630_01_TS_W1_speed.csv           raw Tierpsy speed
        ├── 071026_HML1630_01_TS_W1_speed_fixed.csv     head/tail-corrected speed
        ├── 071026_HML1630_01_TS_W1_blob_perimeter.csv  blob perimeter
        └── … the same 3 files for W2–W8, W10, W26
```

Not included: Tierpsy's `_skeletons.hdf5` / `_intensities.hdf5`, which no
script here reads.

### Raw video (download separately)

The raw recording, `071026_HML1630_01.avi` (6.5 GB), is too large for GitHub,
so it is not in the repository. It is published on Zenodo:
<https://doi.org/10.5281/zenodo.24546189> (record 24546189, CC BY 4.0). No
script here reads it; download it only if you want to watch it or run Tierpsy
on it. From `custom_python_scripts/`:

```bash
python video_processing/download_sample_video.py
```

This saves it to `Sample_data/Videos/` and checks it against the MD5 checksum
listed on Zenodo. If the download is interrupted, run the same command again
to resume. Use `--dest /path/to/folder` to save it somewhere else. The script
needs only the Python standard library, so any conda environment works.
The video is listed in the repository's `.gitignore`, so it is not committed.

To process it with Tierpsy (`TierpsyTracker_modified/INSTALL_INSTRUCTIONS.txt`,
step 5), put a copy in a folder of its own. Tierpsy writes `MaskedVideos/`
and `Results/` next to the video, and `Sample_data/Videos/` already holds the
Tierpsy output used by the analysis.

### What each file is for

| File | Made by | Used for |
|---|---|---|
| `Videos/Results/<video>_featuresN.hdf5` | Tierpsy | Centroid track, which decides when a worm is **active**; skeleton head/tail positions for **reorientation angles**; raw speed, to tell which frames the fixed speed re-signed |
| `Videos/MaskedVideos/<video>.hdf5` | Tierpsy | Microns-per-pixel scale (falls back to 15 µm/px if missing); frames for the flip-range tool |
| `Tierpsy_csvs/<strain>/*_speed.csv` | `video_processing/export_tierpsy_csvs.py` | Raw Tierpsy speed; input for the fixed speed |
| `Tierpsy_csvs/<strain>/*_speed_fixed.csv` | `video_processing/make_fixed_speeds_from_excel.py` | The signed speed the analysis reads (forward > 0, reverse < 0) |
| `Tierpsy_csvs/<strain>/*_blob_perimeter.csv` | `video_processing/export_tierpsy_csvs.py` | `blob_perimeter_median` in the output tables; frame count for `video_fps.csv`. **Required** for a recording to contribute continuous minutes |
| `video_fps.csv` | `video_processing/build_video_fps_table.py` | True frame rate of each video |

**Activity and reorientation need the featuresN file.** If a video's
`featuresN.hdf5` is missing, the analysis warns and treats its worms as never
active, so every active-only metric comes out empty.

**Continuous minutes need the perimeter CSV.** A recording without its
`*_blob_perimeter.csv` is analysed as a whole worm but contributes no
continuous minutes; the analysis warns and names it. Other feature CSVs
(e.g. `angular_velocity_midbody`, regional speeds, `blob_area`) are not used.
They are ignored if present.

**About the sample's fixed speeds:** they were made with `--no-flips`, so they
are exact copies of the raw speed. In the published analysis, frames where
Tierpsy had swapped head and tail were re-signed by hand, so results for this
video will differ somewhat from the paper.

## Naming rules

- **Video stem:** `<MMDDYY>_<strain>_<NN>`, e.g. `071026_HML1630_01`
  (`070926_N2_demag_01` also works). The strain is the token after the date.
  Stems must be unique across your data.
- **Per-worm CSVs:** `<video>_TS_W<n>_<feature>.csv`, where `n` is Tierpsy's
  `worm_index_manual`, the worm number shown in Tierpsy Tracker Viewer.
  All of a worm's CSVs sit in the same folder.
- **Strain folder:** `Tierpsy_csvs/<strain>/` must be named the same as the
  strain token in the video stems (`HML1630`).
- **CSV columns:** `time` plus the feature (e.g. `time,speed`). `time` is the
  video frame index (0, 1, 2, …), the export default. Older exports with time in
  seconds at 6 fps are also accepted.

## Assumptions to check for your own recordings

- **Recording length:** every video is assumed to be exactly 10 minutes.
  True FPS = frames ÷ 600 s. Change `VIDEO_DURATION_S` in
  `worm_analysis/video_fps.py`, or pass `--duration-s` to
  `build_video_fps_table.py`, if yours differ.
- **Track length:** worms tracked for under half of their video's frames are
  skipped by default. Use `--min-frames` in `run_pipeline.py` to change this.
- **Worm numbers:** if your CSV `W<n>` numbers are not Tierpsy's
  `worm_index_manual` (for example, renumbered 1..N), each fixed-speed CSV is
  matched to its Tierpsy worm by content. You can also list the mapping in an
  optional `fixed_speed_mapping.json` in the data root (see
  `worm_analysis/tierpsy_data.py`).

## Processing your own data

1. **Track with Tierpsy.** Put Tierpsy's `MaskedVideos/` and `Results/` folders
   under `<data root>/Videos/`. Several Tierpsy folders can sit there side by
   side, e.g. `Videos/N2/MaskedVideos` + `Videos/N2/Results`; every
   `Results/*_featuresN.hdf5` below `Videos/` is found.
2. **Export per-worm CSVs** into `Tierpsy_csvs/<strain>/`:
   ```bash
   python video_processing/export_tierpsy_csvs.py --data-root /path/to/my_data --video 071026_HML1630_01 --list-worms
   python video_processing/export_tierpsy_csvs.py --data-root /path/to/my_data --video 071026_HML1630_01 --animals 1-8,10,26
   ```
3. **Correct head/tail swaps** and write `*_speed_fixed.csv`. Mark swapped
   frame ranges with the flip-range tool (`head_tail_flips/`) or in an Excel
   sheet, then:
   ```bash
   python video_processing/make_fixed_speeds_from_excel.py --data-root /path/to/my_data --ranges head_tail_flips/data/071026_HML1630_01_ht_flip_ranges.csv
   ```
   If nothing needs correcting, use `--no-flips --video <video>` instead.
4. **Build the FPS table:**
   ```bash
   python video_processing/build_video_fps_table.py --data-root /path/to/my_data
   ```
5. **Analyse:**
   ```bash
   python run_pipeline.py --data-root /path/to/my_data
   ```
   Or set `DATA_ROOT` in the setup cell of `worm_analysis/worm_analysis.ipynb`.

You can also set the environment variable `WORM_DATA_ROOT=/path/to/my_data`
once instead of passing `--data-root` to each script. Or put your data
directly in `Sample_data/`, and then no option is needed.

All commands run from `custom_python_scripts/` in the `worm-analysis` conda
environment (`custom_python_scripts/worm_analysis/environment.yml`). The flip-range tool runs in
`ht_flip` (`custom_python_scripts/head_tail_flips/environment.yml`).
