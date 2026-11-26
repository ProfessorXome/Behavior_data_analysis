# Worm locomotion analysis with a modified Tierpsy Tracker

Code and sample data accompanying *[paper title, to be added]*.

Worm videos are tracked with a modified version of Tierpsy Tracker (included
in `TierpsyTracker_modified/`). Per-worm features are exported to CSV, frames
where Tierpsy swapped head and tail are corrected, and a Python pipeline turns
each worm's speed and track into general, active-only and normalized summary
tables, per-worm frame, per-minute and bout tables, and plots.

> **About Tierpsy Tracker.** Tierpsy Tracker is not our software. It was
> developed by the MRC-LMS Behavioural Phenomics Lab at Imperial College
> London and is published at <https://github.com/Tierpsy/tierpsy-tracker>
> (Javer et al., *Nature Methods* 2018). The copy in
> `TierpsyTracker_modified/tierpsy-tracker/` is a **modified version** of the
> original, not an official Tierpsy release: it starts from the original
> code at commit `2653ecb`, with four files changed. See
> [Tierpsy Tracker: original and modified version](#tierpsy-tracker-original-and-modified-version)
> for what changed, and [Citation](#citation) for the original papers. For the
> unmodified software, use the original repository.

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
                             the code expects, with the paper's head/tail-corrected
                             speeds (and a backup copy); its README explains it
TierpsyTracker_modified/     modified version of Tierpsy Tracker (original:
                             github.com/Tierpsy/tierpsy-tracker), its conda
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

The sample's head/tail-corrected speeds (`*_speed_fixed.csv`) are the
hand-corrected files used for this video in the paper, so the analysis gives
the same data as the paper for this recording. If you redo the whole process
yourself, including marking head/tail swaps with the flip-range tool, frame
choices that differ from the paper's can give very slightly different results.
A backup of the paper's files is in `Sample_data/Fixed_speeds_backup/`; see
`Sample_data/README.md`.

## Using your own videos

1. Track the videos with the modified Tierpsy Tracker:
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
| `tierpsy` | `TierpsyTracker_modified/environment.yml` | modified Tierpsy Tracker |

All three were recorded on macOS on Apple silicon (osx-arm64). Other platforms
were not tested; comments in each file say which entries to delete if conda
cannot resolve them.

## Tierpsy Tracker: original and modified version

### Original

Tierpsy Tracker was developed by the MRC-LMS Behavioural Phenomics Lab at
Imperial College London.

- Source code: <https://github.com/Tierpsy/tierpsy-tracker>
- Website: <https://tierpsy.com>
- License: MIT, Copyright (c) 2017 Medical Research Council
- Papers: see [Citation](#citation)

### Modified version in this repository

`TierpsyTracker_modified/tierpsy-tracker/` is a modified version of the
original. It was copied from the original repository at commit
[`2653ecb`](https://github.com/Tierpsy/tierpsy-tracker/commit/2653ecb3697602e3d6d763c92bc5fb1864c60f15)
(`development` branch, 18 May 2026). Compared with that commit:

- 298 files are unchanged.
- 4 files were modified:
  - `tierpsy/features/open_worm_analysis_toolbox/prefeatures/basic_worm.py`
  - `tierpsy/gui/GetAllParameters.py`
  - `tierpsy/gui/MWTrackerViewer.py`
  - `tierpsy/gui/PlotFeatures.py`
- 1 file was added: `requirements-macos_fixed.txt`.

To see every change, compare the original commit with the modified copy (from
this repository's top folder):

```bash
git clone https://github.com/Tierpsy/tierpsy-tracker.git tierpsy-original
git -C tierpsy-original checkout 2653ecb3697602e3d6d763c92bc5fb1864c60f15
diff -ru tierpsy-original/tierpsy TierpsyTracker_modified/tierpsy-tracker/tierpsy
```

The modified version keeps the original's MIT license and copyright notice
(`TierpsyTracker_modified/tierpsy-tracker/LICENSE`) and its README
(`TierpsyTracker_modified/tierpsy-tracker/README.md`). Install it with
`TierpsyTracker_modified/INSTALL_INSTRUCTIONS.txt`.

## License

The code written for this repository is released under the MIT License
(`LICENSE`). The modified Tierpsy Tracker in
`TierpsyTracker_modified/tierpsy-tracker/` remains under Tierpsy Tracker's
original MIT License, Copyright (c) 2017 Medical Research Council
(`TierpsyTracker_modified/tierpsy-tracker/LICENSE`). The raw sample video on
Zenodo is licensed CC BY 4.0.

## Citation

### This work

<!-- Add the paper's citation (authors, title, journal, year, DOI) here. -->
*To be added.*

### Tierpsy Tracker

If you use Tierpsy Tracker, including the modified version in this
repository, please cite the original Tierpsy Tracker papers:

- Javer, A., Currie, M., Lee, C. W., Hokanson, J., Li, K., Martineau, C. N.,
  Yemini, E., Grundy, L. J., Li, C., Ch'ng, Q., Schafer, W. R., Nollen, E. A. A.,
  Kerr, R. & Brown, A. E. X. An open-source platform for analyzing and sharing
  worm-behavior data. *Nature Methods* **15**, 645–646 (2018).
  <https://doi.org/10.1038/s41592-018-0112-1>
- Javer, A., Ripoll-Sánchez, L. & Brown, A. E. X. Powerful and interpretable
  behavioural features for quantitative phenotyping of *Caenorhabditis
  elegans*. *Philosophical Transactions of the Royal Society B* **373**,
  20170375 (2018). <https://doi.org/10.1098/rstb.2017.0375>
