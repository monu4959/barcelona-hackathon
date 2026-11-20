"""Regenerate the submission file from the provided data, in one command.

    python reproduce.py --team <team_name>

Writes evaluation/results/<team_name>.csv and validates it. Every step is a script of its own and
can be run by hand in the same order (see SOLUTION.md):

  1. dead_ball/fetch_dfl.py          external DFL / IDSSE matches (CC BY 4.0), pinned revision
  2. dead_ball/dfl_prepare.py        ...as compact arrays
  3. ball_recovery/link_ball.py      ball track from the provided tracking data alone
  4. dead_ball/stoppage_model.py     stoppage model: trained, thresholded on DFL; applied here once
  5. calibration/dfl_check.py        the pipeline run on DFL: dead-ball edges, possession settings
  6. dead_ball/detect_dead.py        dead ball on this match (rules + model + Smart Tagging)
  7. possession/estimate_possession.py   the possession file
  8. evaluation/validate_submission.py   the organisers' format check

Environment: MTAH_DATA_DIR = the unpacked match data (default ./data), MTAH_WORK_DIR = scratch
space for intermediate files (default ./work, gitignored), MTAH_DFL_RAW = where the DFL files are
or go (default MTAH_WORK_DIR/external/dfl_idsse). About 10 minutes on a laptop CPU, plus the 2.5 GB
download; no GPU is needed.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = {"MTAH_VERSION": "v2", "MTAH_TAGS": "1", "MTAH_STOPMODEL": "1"}


def run(script, *args, env=None, isolated=False):
    command = [sys.executable] + (["-I"] if isolated else []) + [os.path.join(ROOT, script)] + list(args)
    print("\n>>> " + " ".join(command[1:]), flush=True)
    started = time.time()
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, **(env or {})), check=True)
    print("    ({:.0f} s)".format(time.time() - started), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True, help="team name: the file is evaluation/results/<team>.csv")
    args = parser.parse_args()
    os.environ.setdefault("MTAH_WORK_DIR", os.path.join(ROOT, "work"))

    run("dead_ball/fetch_dfl.py")
    run("dead_ball/dfl_prepare.py", isolated=True)   # external XML: parsed in isolated mode
    run("ball_recovery/link_ball.py")
    run("dead_ball/stoppage_model.py")
    run("calibration/dfl_check.py")
    run("dead_ball/detect_dead.py", env=FINAL)
    run("possession/estimate_possession.py", "--name", args.team, env=FINAL)
    run("evaluation/validate_submission.py", os.path.join("evaluation", "results", args.team + ".csv"))


if __name__ == "__main__":
    main()
