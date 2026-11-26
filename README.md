# Worm locomotion analysis with a modified Tierpsy Tracker

Code and sample data accompanying *[paper title, to be added]*.

Worm videos are tracked with Tierpsy Tracker (a copy with local changes is
included). Per-worm features are exported to CSV, frames where Tierpsy swapped
head and tail are corrected, and a Python pipeline turns each worm's speed and
track into general, active-only and normalized summary tables, per-worm
frame, per-minute and bout tables, and plots.

## Repository layout

```
custom_python_scripts/       analysis code (run commands from this folder)
├── run_pipeline.py          batch analysis: CSVs → tables and plots
├── worm_analysis/           analysis package, notebook (worm_analysis.ipynb), environment.yml
├── video_processing/        Tierpsy CSV export, head/tail-corrected speeds,
│                            FPS table, sample-video download
├── head_tail_flips/         interactive tool for marking head/tail-swap frame ranges
└── USEFUL_COMMANDS.txt      every processing step, with commands
Sample_data/                 one real recording (071026_HML1630_01) in the layout
                             the code expects; its README explains the layout
TierpsyTracker_modified/     Tierpsy Tracker source with local changes, its conda
                             environment and INSTALL_INSTRUCTIONS.txt
```

## Quick start (sample data)

Needs conda (Miniforge or Anaconda).

```bash
git clone <this repository's URL>
cd <repository folder>/custom_python_scripts
conda env create -f worm_analysis/environment.yml
conda activate worm-analysis
python run_pipeline.py
```

This analyses the sample recording and writes its tables to
`custom_python_scripts/output/HML1630/`. The same analysis is available
interactively in `worm_analysis/worm_analysis.ipynb`.

The sample's head/tail-corrected speeds were made without any corrections
(`--no-flips`), so its results will differ somewhat from the paper; see
`Sample_data/README.md`.

## Using your own videos

1. Track the videos with Tierpsy Tracker:
   `TierpsyTracker_modified/INSTALL_INSTRUCTIONS.txt`.
2. Lay out the Tierpsy output and CSVs like `Sample_data/`, then export
   features, correct head/tail swaps, build the FPS table and run the
   analysis: `Sample_data/README.md` and `custom_python_scripts/USEFUL_COMMANDS.txt`.

## Large files (on Zenodo, not in this repository)

| What | Where | How to get it |
|---|---|---|
| Raw sample video `071026_HML1630_01.avi` (6.5 GB) | [doi:10.5281/zenodo.24546189](https://doi.org/10.5281/zenodo.24546189) | `python video_processing/download_sample_video.py` (from `custom_python_scripts/`) |
| Tierpsy Tracker test data (901 MB) | [Zenodo record 3837679](https://zenodo.org/records/3837679), published by the Tierpsy developers | `tierpsy_tests --download_files` (`TierpsyTracker_modified/INSTALL_INSTRUCTIONS.txt`, step 6) |

Neither is needed to run the analysis on the sample data.

## Environments

Three conda environments, each with an exact-version `environment.yml`:

| Environment | File | Used for |
|---|---|---|
| `worm-analysis` | `custom_python_scripts/worm_analysis/environment.yml` | analysis, notebook, CSV export, fixed speeds, FPS table |
| `ht_flip` | `custom_python_scripts/head_tail_flips/environment.yml` | head/tail flip-range tool |
| `tierpsy` | `TierpsyTracker_modified/environment.yml` | Tierpsy Tracker |

All three were recorded on macOS on Apple silicon (osx-arm64). Other platforms
were not tested; comments in each file say which entries to delete if conda
cannot resolve them.

## Tierpsy Tracker

`TierpsyTracker_modified/tierpsy-tracker/` is
[Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) at upstream
commit `2653ecb`, with local changes to four files:

- `tierpsy/features/open_worm_analysis_toolbox/prefeatures/basic_worm.py`
- `tierpsy/gui/GetAllParameters.py`
- `tierpsy/gui/MWTrackerViewer.py`
- `tierpsy/gui/PlotFeatures.py`

Tierpsy Tracker was developed by the MRC-LMS Behavioural Phenomics Lab; its
publications are listed in `TierpsyTracker_modified/tierpsy-tracker/README.md`.

## License

The code in this repository is released under the MIT License (`LICENSE`).
`TierpsyTracker_modified/tierpsy-tracker/` keeps Tierpsy Tracker's own MIT
License (`TierpsyTracker_modified/tierpsy-tracker/LICENSE`). The raw sample
video on Zenodo is licensed CC BY 4.0.

## Citation

<!-- Add the paper's citation (authors, title, journal, year, DOI) here. -->
*To be added.*
