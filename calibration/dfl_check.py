"""Run the whole pipeline on EXTERNAL matches and choose its free settings there.

    python calibration/dfl_check.py                 # dead-ball variants, then possession settings
    python calibration/dfl_check.py --dead-only

The organisers' rules: anything tuned must be tuned on external data. The DFL / IDSSE open matches
(dead_ball/dfl_prepare.py; CC BY 4.0, data owner DFL) carry an official per-frame ball status
(in play / dead) and ball possession (home / away), which is what our output predicts. So the
pipeline's own functions -- dead_ball/detect_dead.py and possession/estimate_possession.py,
unchanged -- are run on them, scored with the competition's own metric
(evaluation/possession_metrics.py: spell F1, macro F1), and the settings that do best there are the
ones used on our match. Nothing from the provided game's labels or video is involved.

Making DFL look like our input
------------------------------
- players go into our 46-slot layout (home 0-23, away 24-45, goalkeepers in slots 23 and 45), as
  pitch fractions, with speeds computed from their positions;
- the DFL ball is always tracked, ours is not: the ball is HIDDEN on exactly the frames where our
  tracking data has no ball (our match's own gap pattern, half by half, repeated if needed). Only
  the pattern of gaps is borrowed -- whether the ball is tracked, not where it is or who has it.
  The gaps then fall at random moments of the DFL match, whereas ours cluster at stoppages, so the
  check is somewhat harsh on live play and lenient on dead balls;
- the stoppage model's flag for each DFL match comes from the model that did not see that match
  (stoppage_model.py, dfl_oof.npz);
- there is no Smart Tagging for DFL, so the tag rules are not part of this check.

What is chosen
--------------
1. dead ball: rules alone (with the NO BALL rule) vs the rules merged with the stoppage model, each
   edge of a rule interval kept, taken from the model, or the wider of the two (9 combinations).
   The best mean spell F1 wins and is written to stoppage_model/edges.json.
2. possession: HOLD_S, CONTROL_M, PASS_GAP and BACKTRACK, searched one at a time around the current
   values (two passes), with the chosen dead ball. The best goes to calibration/possession.json,
   which estimate_possession.py reads.

Output: MTAH_WORK_DIR/calibration/dfl_check.txt, stoppage_model/edges.json, calibration/possession.json
"""

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "analysis", "dead_ball", "possession", "evaluation", "ball_recovery"):
    sys.path.insert(0, os.path.join(ROOT, sub))
os.environ["MTAH_CALIBRATED"] = "0"   # search around the hand-set values, not an earlier result
import detect_dead as dd  # noqa: E402
import estimate_possession as ep  # noqa: E402
import possession_metrics as pm  # noqa: E402
import stoppage_model as sm  # noqa: E402
import eda  # noqa: E402

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(ROOT, "work")
DFL = os.path.join(WORK_DIR, "external", "dfl_idsse", "npz")
OOF = os.path.join(WORK_DIR, "stoppage_model", "dfl_oof.npz")
OUR_TRACK = os.path.join(WORK_DIR, "ball_track_v2", "track.npz")
EDGES_OUT = os.path.join(WORK_DIR, "stoppage_model", "edges.json")
POSSESSION_OUT = os.path.join(ROOT, "calibration", "possession.json")
REPORT = os.path.join(WORK_DIR, "calibration", "dfl_check.txt")
FPS = 25
MODES = ("rule", "model", "outer")
SEARCH = {"HOLD_S": [0.5, 1.0, 1.5, 2.0, 3.0],
          "CONTROL_M": [1.0, 1.5, 2.0, 2.5, 3.0],
          "PASS_GAP": [1 * FPS, 2 * FPS, 3 * FPS, 5 * FPS],
          "BACKTRACK": [0, 2 * FPS, 4 * FPS, 6 * FPS, 10 * FPS]}
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def our_gap_pattern():
    """{half: bool[n]} -- True where our tracking data has no ball, over the graded frames."""
    graded, half = eda.graded_mask()
    missing = np.isnan(np.load(OUR_TRACK)["x"])
    return {h: missing[graded & (half == h)] for h in (1, 2)}


def dfl_halves(gaps):
    """[(key, players[n, 46, 3], x, y, dead_true, possession_true)] in our formats."""
    out = []
    for path in sorted(glob.glob(os.path.join(DFL, "*.npz"))):
        data = np.load(path)
        for h in (1, 2):
            xy, team, keeper = data["xy%d" % h], data["team%d" % h], data["keeper%d" % h]
            n = len(xy)
            players = np.full((n, 46, 3), np.nan, np.float32)
            for side, first, gk_slot in ((0, 0, dd.HOME_GK), (1, 24, dd.AWAY_GK)):
                members = np.flatnonzero(team == side)
                keepers = [p for p in members if keeper[p]]
                others = [p for p in members if not keeper[p]]
                order = keepers[1:] + others  # a second keeper (a substitute) goes in an outfield slot
                slots = {keepers[0]: gk_slot} if keepers else {}
                slots.update({p: first + i for i, p in enumerate(order[:23])})
                for p, slot in slots.items():
                    players[:, slot, 0] = (xy[:, p, 0] + dd.LENGTH / 2) / dd.LENGTH
                    players[:, slot, 1] = (xy[:, p, 1] + dd.WIDTH / 2) / dd.WIDTH
            metres = players[:, :, :2] * np.array([dd.LENGTH, dd.WIDTH], np.float32)
            ahead = np.concatenate([metres[6:], np.full((6, 46, 2), np.nan, np.float32)])
            back = np.concatenate([np.full((6, 46, 2), np.nan, np.float32), metres[:-6]])
            players[:, :, 2] = np.linalg.norm(ahead - back, axis=2) / (12 / FPS)
            ball = data["ball%d" % h].astype(float)
            x, y = ball[:, 0] + dd.LENGTH / 2, ball[:, 1] + dd.WIDTH / 2
            hide = np.resize(gaps[h], n)
            x, y = np.where(hide, np.nan, x), np.where(hide, np.nan, y)
            key = "%s_%d" % (os.path.basename(path)[:-4], h)
            out.append((key, h, players, x, y, data["status%d" % h] == 0, data["possession%d" % h]))
    return out


def rule_intervals(players, x, y, h):
    n = len(x)
    graded, half = np.ones(n, bool), np.full(n, h, np.int8)
    speed = ep_speed(x, y)
    valid = ~np.isnan(x)
    intervals = dd.build_intervals(x, y, speed, valid, graded, half, players, [])
    dead, _ = dd.to_state(intervals, n)
    extra = pd.DataFrame(dd.whistles(x, y, speed, valid, graded, players, dead))
    if len(extra):
        extra["half"] = h
        intervals = pd.concat([intervals, extra]).sort_values("start").reset_index(drop=True)
    return intervals


def ep_speed(x, y):
    """Ball speed as the linker computes it (positions 2 frames either side)."""
    speed = np.full(len(x), np.nan)
    speed[2:-2] = np.hypot(x[4:] - x[:-4], y[4:] - y[:-4]) / (4 / FPS)
    return speed


def labels_for(players, x, y, intervals, params):
    n = len(x)
    for name, value in params.items():
        setattr(ep, name, value)
    graded = np.ones(n, bool)
    dead, _ = dd.to_state(intervals, n)
    speed = ep_speed(x, y)
    series, d_home, d_away = ep.control_series(players, x, y, speed, graded & ~dead)
    touches = ep.touch_series(x, y, d_home, d_away, graded & ~dead)
    hold = int(round(ep.HOLD_S * FPS))
    labels, _ = ep.possession(graded, dead, series, touches, intervals, players, (x, y), hold)
    return labels


def score(halves, labels):
    gt, pred = [], []
    for (_, _, _, _, _, dead, possession), lab in zip(halves, labels):
        truth = np.where(dead, pm.D, np.where(possession == 1, pm.H, pm.A))
        gt += [truth, [pm.X]]
        pred += [np.array([CODE[c] for c in lab]), [pm.X]]
    result = pm.evaluate(np.concatenate(gt).astype(np.int8), np.concatenate(pred).astype(np.int8))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]


def dead_f1(halves, interval_sets):
    return np.mean([sm.stretch_f1(h[5], dd.to_state(iv, len(h[3]))[0]) for h, iv in zip(halves, interval_sets)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dead-only", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    halves = dfl_halves(our_gap_pattern())
    oof = np.load(OOF)
    defaults = {name: getattr(ep, name) for name in SEARCH}
    lines = ["External check on {} DFL halves (ball hidden on our match's gap pattern)".format(len(halves)), ""]

    rules = [rule_intervals(p, x, y, h) for _, h, p, x, y, _, _ in halves]
    variants = {"rules only (with NO BALL)": rules}
    for start_mode, end_mode in itertools.product(MODES, MODES):
        variants["rules + model, start {}, end {}".format(start_mode, end_mode)] = [
            dd.merge_model(iv, oof[key], np.full(len(x), h, np.int8), start_mode, end_mode)
            for iv, (key, h, _, x, _, _, _) in zip(rules, halves)]
    lines.append("dead ball variants (possession settings at their current values):")
    lines.append("  {:42s} {:>9s} {:>9s} {:>9s} {:>13s}".format("", "spell F1", "macro F1", "dead F1", "spells/true"))
    results = {}
    for name, sets in variants.items():
        labels = [labels_for(p, x, y, iv, defaults) for (_, _, p, x, y, _, _), iv in zip(halves, sets)]
        spell, macro, n_pred, n_gt = score(halves, labels)
        results[name] = (spell, macro)
        lines.append("  {:42s} {:9.3f} {:9.3f} {:9.3f} {:>6d}/{:<6d}".format(
            name, spell, macro, dead_f1(halves, sets), n_pred, n_gt))
        print(lines[-1], flush=True)
    best = max(results, key=lambda k: results[k])
    lines.append("chosen: " + best)
    if best.startswith("rules + model"):
        start_mode, end_mode = best.split("start ")[1].split(", end ")
        with open(EDGES_OUT, "w") as handle:
            json.dump({"start": start_mode, "end": end_mode, "chosen_on": "DFL/IDSSE (calibration/dfl_check.py)"}, handle)
    sets = variants[best]

    if not args.dead_only:
        lines += ["", "possession settings, one at a time around the current values (spell F1, macro F1):"]
        current = dict(defaults)
        best_score = results[best]
        for sweep in (1, 2):
            for name, values in SEARCH.items():
                row = []
                for value in values:
                    trial = dict(current, **{name: value})
                    labels = [labels_for(p, x, y, iv, trial) for (_, _, p, x, y, _, _), iv in zip(halves, sets)]
                    spell, macro, _, _ = score(halves, labels)
                    row.append((spell, macro, value))
                    if (spell, macro) > best_score:
                        best_score, current = (spell, macro), trial
                lines.append("  pass {} {:10s} ".format(sweep, name) + "  ".join(
                    "{}: {:.3f}/{:.3f}".format(v, s, m) for s, m, v in row))
                print(lines[-1], flush=True)
        lines.append("chosen: " + ", ".join("{} = {}".format(k, v) for k, v in current.items()) +
                     "  (spell F1 {:.3f}, macro F1 {:.3f}; at the old values {:.3f} / {:.3f})".format(
                         *best_score, *results[best]))
        with open(POSSESSION_OUT, "w") as handle:
            json.dump(dict(current, chosen_on="DFL/IDSSE (calibration/dfl_check.py)"), handle, indent=2)
    text = "\n".join(lines)
    print(text)
    with open(REPORT, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
