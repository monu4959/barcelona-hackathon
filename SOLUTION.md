# How to run

This repository holds the pipeline, plus the challenge's starter code and scorer (`starter/`,
`evaluation/`, by Metrica Sports, MIT). The method, the data I
used and dropped, the possession rules and the limitations are in
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). This page covers how to run it.

## Reproduce the submission

```bash
# Python 3.12 (3.12.6 was used), CPU only
python -m venv venv
venv/Scripts/pip install -r requirements-pipeline.txt        # Windows (venv/bin/pip elsewhere)

# the match bundle, unpacked as in data/README.md (or point MTAH_DATA_DIR at it)
export MTAH_DATA_DIR=/path/to/mtah2027                       # PowerShell: $env:MTAH_DATA_DIR = "..."

python reproduce.py
```

Output: `evaluation/results/monu4959.csv`, validated by `evaluation/validate_submission.py`.
From a fresh clone, a fresh virtual environment and an empty work directory, this reproduces the
submitted file byte for byte. It takes about 10 minutes on a laptop CPU, plus a 2.5 GB download of the external data.

Requirements: `requirements-pipeline.txt` is everything the pipeline needs. `requirements.txt`
(numpy and pandas) is enough for the scorer alone, and `requirements-starter.txt` is only for the
notebooks in `starter/`.

Environment variables (all optional):

| variable | default | what |
|---|---|---|
| `MTAH_DATA_DIR` | `./data` | the provided match data (never committed) |
| `MTAH_WORK_DIR` | `./work` (gitignored) | intermediate files: ball track, dead-ball intervals, models |
| `MTAH_DFL_RAW` | `$MTAH_WORK_DIR/external/dfl_idsse` | where the DFL files are, or are downloaded to |

## The steps `reproduce.py` runs

| # | script | output (in `MTAH_WORK_DIR` unless noted) |
|---|---|---|
| 1 | `dead_ball/fetch_dfl.py` | the 21 DFL / IDSSE files, pinned revision, sizes checked |
| 2 | `dead_ball/dfl_prepare.py` | `external/dfl_idsse/npz/*.npz` — players, ball, ball status, possession |
| 3 | `ball_recovery/link_ball.py` | `ball_track_v2/track.npz` — ball track from the tracking data alone |
| 4 | `dead_ball/stoppage_model.py` | `stoppage_model/` — model, DFL cross-validation, threshold, this match's dead probability |
| 5 | `calibration/dfl_check.py` | `stoppage_model/edges.json`, `calibration/possession.json` (in the repo), `calibration/dfl_check.txt` |
| 6 | `dead_ball/detect_dead.py` (`MTAH_VERSION=v2 MTAH_TAGS=1 MTAH_STOPMODEL=1`) | `dead_ball_v2_tags_model/` — dead-ball intervals and per-frame state |
| 7 | `possession/estimate_possession.py --name monu4959` (same variables) | `evaluation/results/monu4959.csv` |
| 8 | `evaluation/validate_submission.py` | format check |

Every script has a docstring with its rules and constants. `analysis/eda.py` holds the data
loaders (and the exploratory analysis I started from).

## What is not in this repository

- The match data and anything that would reconstruct it: no tracking exports or frame-level
  files. The one exception is the 2.5-minute demo video in `media/`, published with permission. `data/`, `work/`, `analysis/out/` and `evaluation/results/*.csv` are
  gitignored.
- The dropped ball detectors (one trained on the provided video, removed after the organisers'
  clarification; one trained on external images, which transferred poorly). Neither affects the
  submission. See the technical report.

## Credits

External data: IDSSE — Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset
of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195.
doi:10.1038/s41597-025-04505-y. Data owner: Deutsche Fußball Liga (DFL). Licence: CC BY 4.0.
