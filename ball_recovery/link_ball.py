"""Ball recovery, step 4: one ball position per frame, from the ATD ball (and optionally video detections).

    python ball_recovery/link_ball.py                    # tracking data only -> ball_track_v2
    python ball_recovery/link_ball.py --detector v3      # + the externally trained detector -> ball_track_v3
    python ball_recovery/link_ball.py --detector v1      # + the v1 detector (trained on this match) -> ball_track_v1

The submission uses the tracking data only. The v1 detector was trained on tiles of this match's
video, which the organisers' clarification of the rules does not allow (no training, fitting or
calibrating on the provided video; no training data from the provided game). --detector v1 is kept
only to reproduce the earlier experiments; nothing built on ball_track_v1 is submitted. v3 was
trained and thresholded on external data only (train_external.py); with it MIN_CONF is its
threshold and MISS_COST = -log(threshold), so "no ball" costs exactly as much as a detection at
the threshold -- the same relation v1 had (0.3 and 1.2).
With the ATD as the only candidate the Viterbi below reduces to "the ATD ball where it has one";
the hole filling and the speed estimate are what remains of the linker.

Why a linker and not "take the detector, else the ATD"
-------------------------------------------------------
Both sources are wrong in their own way. The ATD drifts or lags around restarts (at the 2nd-half
kickoff its ball travels 37 m while the real one sits on the centre spot). The detector is precise
when it is on the ball, but also fires on spare balls, white litter and the warm-up balls beside the
pitch, sometimes for minutes without moving. So per half, a Viterbi pass picks one candidate per
frame -- a detection, the ATD ball, or "no ball" -- minimising

    emission    -log(conf) for a detection, a fixed cost for the ATD ball, very little when the two
                agree within AGREE_M; extra cost for a STATIC candidate far from every player (a ball
                lying still with nobody near it is a spare ball, not the match ball) and for one deep
                in the stands
    transition  (metres moved per frame / STEP_M)^2 -- a ball cannot teleport -- or a flat
                re-entry cost after a "no ball" stretch

Holes of up to FILL_MAX frames between chosen positions are then filled by straight lines, and speed
comes from the positions (the ATD's speed field is empty for the ball). Positions are on the ground
plane, so a high ball reads further from the camera than it is.

Output: MTAH_WORK_DIR/ball_track_v2/track.npz (v1 with --detections), arrays over every video frame:
    x, y      metres (NaN = no ball)        px, py    pixels, for drawing
    speed     m/s, smoothed                 conf      0-1 of the chosen candidate
    source    0 none, 1 ATD, 2 detector, 3 both agree, 4 filled
Frame-level data derived from the match: work dir only, never publish.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from games import GAME, GAMES  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
import eda  # noqa: E402

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(ROOT, "work")
DETECTOR_RUNS = {"none": (None, "ball_track_v2"), "v1": ("ball_gaps_v1", "ball_track_v1"),
                 "v3": ("ball_gaps_v3", "ball_track_v3")}
V3_THRESHOLD = os.path.join(WORK_DIR, "ball_detector_ext", "v3", "threshold.json")
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
MAX_DET = 5
MIN_CONF = 0.3              # weaker detections are not candidates
ATD_CONF = 0.6              # the ATD ball costs as much as a 0.6 detection...
AGREE_CONF = 0.95           # ...and an ATD ball with a detection on it as much as a 0.95 one
AGREE_M = 3.0
MISS_COST = 1.2             # per frame with no ball: a detection under ~0.3 conf is worth less than nothing
REENTRY_COST = 2.0          # picking a ball up again after a "no ball" stretch
STEP_M = 1.2                # metres per frame that cost 1 (a 30 m/s ball moves 1.2 m per frame)
STATIC_LAG, STATIC_M = FPS, 0.5        # static = within 0.5 m of a candidate 1 s before AND after
LONELY_M, LONELY_COST = 12.0, 3.0      # ...and further than this from every player: spare ball
STANDS_M, STANDS_COST = 6.0, 3.0       # this far outside the lines is the stands or the bench area
FILL_MAX = FPS              # straight-line fill of holes up to 1 s
SPEED_HALF = 2              # speed from positions 2 frames either side
SOURCE_NAMES = {0: "none", 1: "ATD", 2: "detector", 3: "both", 4: "filled"}


def candidates(n_frames, ball, detections):
    """cands[n, K, 2] metres and cost[n, K] (inf = absent); K = MAX_DET detections + the ATD ball.
    Without detections the K - 1 detection slots stay empty."""
    K = MAX_DET + 1
    cands = np.full((n_frames, K, 2), np.nan)
    pixels = np.full((n_frames, K, 2), np.nan)
    conf = np.zeros((n_frames, K))
    if detections:
        table = pd.read_csv(detections, on_bad_lines="skip")  # run_gaps.py may still be appending
        table = table.dropna(subset=["frame"]).astype({"frame": int, "rank": int})
        table = table[(table.conf >= MIN_CONF) & (table["rank"] < MAX_DET)]
        rows, ranks = table.frame.values, table["rank"].values.astype(int)
        cands[rows, ranks] = table[["pitch_x", "pitch_y"]].values
        pixels[rows, ranks] = table[["px", "py"]].values
        conf[rows, ranks] = table.conf.values

    cost = np.where(conf > 0, -np.log(np.maximum(conf, 1e-6)), np.inf)
    atd = np.stack([ball[:, 0] * LENGTH, ball[:, 1] * WIDTH], axis=1)
    has_atd = ~np.isnan(atd[:, 0])
    cands[:, MAX_DET] = atd
    cost[has_atd, MAX_DET] = -np.log(ATD_CONF)

    # A detection on the ATD ball: keep the detection (pixel-precise), make it cheap, drop the ATD copy.
    distance = np.linalg.norm(cands[:, :MAX_DET] - atd[:, None], axis=2)
    agree = (distance < AGREE_M) & np.isfinite(cost[:, :MAX_DET])
    best = np.where(agree.any(axis=1), np.argmax(np.where(agree, conf[:, :MAX_DET], -1), axis=1), -1)
    hit = best >= 0
    cost[hit, best[hit]] = -np.log(AGREE_CONF)
    cost[hit, MAX_DET] = np.inf
    agreed = np.zeros((n_frames, K), bool)
    agreed[hit, best[hit]] = True
    return cands, pixels, cost, agreed


def static_mask(cands, cost):
    """Candidates with a detection within STATIC_M both STATIC_LAG frames before and after."""
    dets, present = cands[:, :MAX_DET], np.isfinite(cost[:, :MAX_DET])
    static = np.zeros(present.shape, bool)
    for shift in (STATIC_LAG, -STATIC_LAG):
        other = np.roll(dets, shift, axis=0)
        other_present = np.roll(present, shift, axis=0)
        gap = np.linalg.norm(dets[:, :, None] - other[:, None], axis=3)
        near = ((gap < STATIC_M) & other_present[:, None]).any(axis=2)
        static = near if shift == STATIC_LAG else static & near
    out = np.zeros(cost.shape, bool)
    out[:, :MAX_DET] = static & present
    return out


def nearest_player(cands, players, chunk=20000):
    xy = players[:, :, :2] * np.array([LENGTH, WIDTH])
    out = np.full(cands.shape[:2], np.inf)
    for start in range(0, len(cands), chunk):
        part = np.linalg.norm(cands[start:start + chunk, :, None] - xy[start:start + chunk, None], axis=3)
        out[start:start + chunk] = np.nanmin(np.where(np.isnan(part), np.inf, part), axis=2)
    return out


def viterbi(cands, cost):
    """Best path over K candidates + a 'no ball' state. Returns the chosen index per frame (K = none)."""
    T, K = cost.shape
    back = np.zeros((T, K + 1), np.int16)
    score = np.append(cost[0], MISS_COST)
    for t in range(1, T):
        step = np.linalg.norm(cands[t - 1][:, None] - cands[t][None], axis=2) / STEP_M
        moves = score[:K, None] + np.where(np.isnan(step), np.inf, step ** 2)
        moves = np.vstack([moves, np.full((1, K), score[K] + REENTRY_COST)])
        prev = np.argmin(moves, axis=0)
        to_none = int(np.argmin(score))
        back[t, :K], back[t, K] = prev, to_none
        score = np.append(moves[prev, np.arange(K)] + cost[t], score[to_none] + MISS_COST)
    path = np.empty(T, np.int16)
    path[-1] = int(np.argmin(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def fill_holes(x, y, px, py, source):
    starts, lengths = eda.runs(np.isnan(x))
    for start, length in zip(starts, lengths):
        before, after = start - 1, start + length
        if length > FILL_MAX or before < 0 or after >= len(x) or np.isnan(x[before]) or np.isnan(x[after]):
            continue
        if np.hypot(x[after] - x[before], y[after] - y[before]) / (length + 1) > STEP_M:
            continue  # the two ends are not the same ball
        weights = np.arange(1, length + 1) / (length + 1)
        for array in (x, y, px, py):
            array[start:after] = array[before] + weights * (array[after] - array[before])
        source[start:after] = 4


def speed_of(x, y):
    """|p(t+2) - p(t-2)| / 4 frames, in m/s; NaN where either end is missing."""
    speed = np.full(len(x), np.nan)
    h = SPEED_HALF
    speed[h:-h] = np.hypot(x[2 * h:] - x[:-2 * h], y[2 * h:] - y[:-2 * h]) / (2 * h / FPS)
    return speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=sorted(DETECTOR_RUNS), default="none",
                        help="video detections to add (v1 was trained on the match video: not for the submission)")
    args = parser.parse_args()
    global MIN_CONF, MISS_COST
    run, out_name = DETECTOR_RUNS[args.detector]
    out_dir = os.path.join(WORK_DIR, out_name)
    detections = os.path.join(WORK_DIR, run, "detections.csv") if run else None
    if args.detector == "v3":
        with open(V3_THRESHOLD) as handle:
            MIN_CONF = json.load(handle)["conf"]
        MISS_COST = -np.log(MIN_CONF)
    players, ball = eda.load_atd()
    graded, _ = eda.graded_mask()
    n = len(ball)
    matrices, _, _ = load_homography()

    cands, pixels, cost, agreed = candidates(n, ball, detections)
    static = static_mask(cands, cost)
    lonely = nearest_player(cands, players) > LONELY_M
    cost = cost + np.where(static & lonely, LONELY_COST, 0.0)
    outside = np.maximum(np.maximum(-cands[..., 0], cands[..., 0] - LENGTH),
                         np.maximum(-cands[..., 1], cands[..., 1] - WIDTH))
    cost = cost + np.where(np.nan_to_num(outside) > STANDS_M, STANDS_COST, 0.0)

    x, y, px, py = (np.full(n, np.nan) for _ in range(4))
    conf, source = np.zeros(n), np.zeros(n, np.int8)
    starts, lengths = eda.runs(graded)
    for start, length in zip(starts, lengths):  # each half on its own
        frames = np.arange(start, start + length)
        path = viterbi(cands[frames], cost[frames])
        chosen = path < cands.shape[1]
        f, k = frames[chosen], path[chosen]
        x[f], y[f] = cands[f, k, 0], cands[f, k, 1]
        px[f], py[f] = pixels[f, k, 0], pixels[f, k, 1]
        conf[f] = np.exp(-np.minimum(cost[f, k], 20))
        source[f] = np.where(k == cands.shape[1] - 1, 1, np.where(agreed[f, k], 3, 2))
    atd_pick = source == 1  # ATD candidates have no pixel position yet
    for frame in np.flatnonzero(atd_pick):
        px[frame], py[frame] = project(matrices[frame], ball[frame, 0], ball[frame, 1])
    fill_holes(x, y, px, py, source)
    speed = speed_of(x, y)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "track.npz"), x=x, y=y, px=px, py=py, speed=speed, conf=conf,
                        source=source)
    report(graded, source, static, lonely, cands, cost, ball, x, y, out_dir)


def report(graded, source, static, lonely, cands, cost, ball, x, y, out_dir):
    lines = ["Ball track v1 over {} graded frames".format(graded.sum())]
    for code, name in SOURCE_NAMES.items():
        lines.append("  {:9s} {:6.1%}".format(name, np.mean(source[graded] == code)))
    has_atd = ~np.isnan(ball[:, 0]) & graded
    moved = np.hypot(x - ball[:, 0] * LENGTH, y - ball[:, 1] * WIDTH)
    lines.append("frames where the ATD has a ball but the track chose a position > {} m away: {:.1%}".format(
        AGREE_M, np.mean(moved[has_atd] > AGREE_M)))
    lines.append("frames where the ATD has a ball but the track says no ball: {:.1%}".format(
        np.mean(source[has_atd] == 0)))
    spare = (static & lonely & np.isfinite(cost)).any(axis=1) & graded
    lines.append("frames with a static, lonely candidate (likely spare ball) penalised: {:.1%}".format(
        spare.sum() / graded.sum()))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "report.txt"), "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
