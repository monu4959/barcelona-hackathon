"""Stoppage model: is the ball dead, judged from the players alone? Trained on EXTERNAL data only.

    python dead_ball/stoppage_model.py            # cross-validate on DFL, fit, apply to this match
    python dead_ball/stoppage_model.py --cv-only  # just the DFL cross-validation

Why: our tracking data loses the ball for long stretches, and most whistle stoppages (fouls,
offsides, injuries) happen with the ball inside the pitch, so the line-crossing rules never see
them. But when the referee whistles, all the players react together -- they slow down, the shape
loosens, a taker walks to the ball -- and that needs no ball at all.

Training data: the DFL / IDSSE open matches (dfl_prepare.py; CC BY 4.0, data owner DFL), whose
ball frames carry the official BallStatus flag (in play / dead). Seven matches, 25 Hz, the same kind
of optical tracking as ours. Nothing from the provided game is used to train, choose or tune: the
model, its smoothing, its threshold and its minimum stretch length are all chosen by leave-one-
match-out cross-validation on the DFL matches, and only then applied here, once.

Features (per frame, the same code for both datasets, positions in metres from the pitch centre):
the players' speeds from their own positions (not the provider's speed field, which differs between
providers), as mean / median / top-5 / shares under 1 and 2 m/s and over 4 m/s; how spread out
they are and how fast their centroid moves; players beyond the lines, in a penalty box, near a
corner flag or around the centre spot; and moving means of the speed features over windows before
and after the frame (a stoppage is running, then a sudden slowdown, then walking).

Selection on DFL (held-out match each time): the smoothed probability is thresholded and stretches
shorter than MIN_S are dropped (both ways); the (threshold, MIN_S) pair with the best mean
stretch-level F1 (a dead stretch counts if IoU >= 0.5 with a true one) over the held-out matches is
kept, then the model is refitted on all seven matches.

Output: MTAH_WORK_DIR/stoppage_model/{model.joblib, choice.json, cv.txt, prob.npz, dfl_oof.npz}
(dfl_oof.npz: the thresholded held-out flag for every DFL half, from the model that never saw it)
(prob.npz: per video frame of this match, the smoothed dead probability and the thresholded flag).
"""

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(ROOT, "work")
DFL = os.path.join(WORK_DIR, "external", "dfl_idsse", "npz")
OUT = os.path.join(WORK_DIR, "stoppage_model")
FPS = 25
LENGTH, WIDTH = 105.0, 68.0
SPEED_LAG = 6               # speed from positions 6 frames either side (0.48 s)
WINDOWS_S = (1, 3, 8)       # context windows before and after the frame
TRAIN_STEP = 5              # every 5th frame for training (neighbours are near-duplicates)
SMOOTH = FPS                # 1 s centred mean of the probability
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_S = (1, 2, 3, 5)
IOU = 0.5
SEED = 2027


def rolling_mean(values, start, stop):
    """Mean of values[t+start : t+stop] per t (NaN-aware, edges shrink)."""
    n = len(values)
    ok = ~np.isnan(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    count = np.concatenate([[0.0], np.cumsum(ok)])
    lo = np.clip(np.arange(n) + start, 0, n)
    hi = np.clip(np.arange(n) + stop, 0, n)
    c = count[hi] - count[lo]
    return np.where(c > 0, (total[hi] - total[lo]) / np.maximum(c, 1), np.nan)


def features(xy):
    """xy[n, P, 2] metres from the pitch centre (NaN = off the pitch) -> X[n, F]."""
    n = len(xy)
    shifted_ahead = np.concatenate([xy[SPEED_LAG:], np.full((SPEED_LAG,) + xy.shape[1:], np.nan)])
    shifted_back = np.concatenate([np.full((SPEED_LAG,) + xy.shape[1:], np.nan), xy[:-SPEED_LAG]])
    speed = np.linalg.norm(shifted_ahead - shifted_back, axis=2) / (2 * SPEED_LAG / FPS)
    speed = np.where(speed > 12, np.nan, speed)  # tracking jumps, not running
    on = ~np.isnan(speed)
    count = np.maximum(on.sum(axis=1), 1)
    with np.errstate(all="ignore"):
        mean = np.nanmean(speed, axis=1)
        median = np.nanmedian(speed, axis=1)
        top5 = np.nanmean(-np.sort(-np.nan_to_num(speed, nan=-1), axis=1)[:, :5], axis=1)
        slow1 = np.sum(on & (speed < 1.0), axis=1) / count
        slow2 = np.sum(on & (speed < 2.0), axis=1) / count
        fast4 = np.sum(on & (speed > 4.0), axis=1) / count
        x, y = xy[..., 0], xy[..., 1]
        spread_x, spread_y = np.nanstd(x, axis=1), np.nanstd(y, axis=1)
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
    centroid_speed = np.hypot(np.gradient(cx), np.gradient(cy)) * FPS
    outside = np.maximum(np.abs(x) - LENGTH / 2, np.abs(y) - WIDTH / 2)
    max_outside = np.nanmax(np.where(np.isnan(outside), -99, outside), axis=1)
    n_outside = np.sum(np.nan_to_num(outside, nan=-99) > 0.5, axis=1)
    in_box = (np.abs(np.abs(x) - (LENGTH / 2 - 8.25)) < 8.25) & (np.abs(y) < 20.16)
    box_left = np.sum(in_box & (x < 0), axis=1)
    box_right = np.sum(in_box & (x > 0), axis=1)
    corner = np.hypot(np.abs(x) - LENGTH / 2, np.abs(y) - WIDTH / 2)
    near_corner = np.nanmin(np.where(np.isnan(corner), 99, corner), axis=1)
    centre = np.sum(np.nan_to_num(np.hypot(x, y), nan=99) < 9.15, axis=1)

    columns = [mean, median, top5, slow1, slow2, fast4, spread_x, spread_y, centroid_speed,
               max_outside, n_outside, np.maximum(box_left, box_right), near_corner, centre]
    for series in (mean, top5, slow1, centroid_speed):
        for w in WINDOWS_S:
            columns.append(rolling_mean(series, -w * FPS, 0))      # before
            columns.append(rolling_mean(series, 1, w * FPS + 1))    # after
    return np.stack(columns, axis=1).astype(np.float32)


def load_dfl():
    """[(match, half, X, dead)]"""
    out = []
    for path in sorted(glob.glob(os.path.join(DFL, "*.npz"))):
        data = np.load(path)
        for half in (1, 2):
            out.append((os.path.basename(path)[:-4], half, features(data["xy%d" % half]),
                        data["status%d" % half] == 0))
    return out


def stretches(flag):
    change = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(np.int8), [0]])))
    return list(zip(change[::2], change[1::2]))


def postprocess(prob, threshold, min_frames):
    smooth = rolling_mean(prob, -(SMOOTH // 2), SMOOTH - SMOOTH // 2)
    flag = smooth > threshold
    for value in (False, True):  # fill short live gaps, then drop short dead blips
        for start, stop in stretches(flag == value):
            if stop - start < min_frames and start > 0 and stop < len(flag):
                flag[start:stop] = not value
    return smooth, flag


def stretch_f1(true, pred):
    t, p = stretches(true), stretches(pred)
    used, hits = set(), 0
    for a, b in t:
        for j, (c, d) in enumerate(p):
            if j in used or d <= a or b <= c:
                continue
            if (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) >= IOU:
                used.add(j)
                hits += 1
                break
    precision = hits / len(p) if p else 0.0
    recall = hits / len(t) if t else 0.0
    return 2 * precision * recall / (precision + recall) if hits else 0.0


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=SEED)


def cross_validate(halves):
    matches = sorted({m for m, _, _, _ in halves})
    scores = {(t, s): [] for t in THRESHOLDS for s in MIN_S}
    lines, aucs, oof = [], [], {}
    for held in matches:
        train = [h for h in halves if h[0] != held]
        X = np.concatenate([h[2][::TRAIN_STEP] for h in train])
        y = np.concatenate([h[3][::TRAIN_STEP] for h in train])
        clf = model().fit(X, y)
        for _, half, X_test, dead in [h for h in halves if h[0] == held]:
            prob = clf.predict_proba(X_test)[:, 1]
            oof["%s_%d" % (held, half)] = prob.astype(np.float32)
            aucs.append(roc_auc_score(dead, prob))
            for t in THRESHOLDS:
                for s in MIN_S:
                    scores[(t, s)].append(stretch_f1(dead, postprocess(prob, t, s * FPS)[1]))
        lines.append("held out {}: AUC {:.3f} / {:.3f}".format(held, *aucs[-2:]))
    choice = max(scores, key=lambda k: (np.mean(scores[k]), -k[1]))
    lines.append("")
    lines.append("frame AUC over held-out halves: mean {:.3f}, min {:.3f}".format(np.mean(aucs), np.min(aucs)))
    lines.append("stretch F1 (IoU >= {}) by threshold x min length, mean over held-out halves:".format(IOU))
    lines.append("        " + "".join("{:>7d}s".format(s) for s in MIN_S))
    for t in THRESHOLDS:
        lines.append("  {:.1f}  ".format(t) + "".join("{:8.3f}".format(np.mean(scores[(t, s)])) for s in MIN_S))
    lines.append("chosen: threshold {}, min {} s (stretch F1 {:.3f})".format(choice[0], choice[1], np.mean(scores[choice])))
    return choice, "\n".join(lines), oof


def match_xy():
    """This match's players as metres from the pitch centre, NaN off the pitch."""
    from games import GAME, GAMES  # noqa: E402
    import eda  # noqa: E402
    length, width = GAMES[GAME]["pitch"]
    players, _ = eda.load_atd()
    xy = players[:, :, :2].astype(np.float32) * np.array([length, width], np.float32)
    return xy - np.array([length / 2, width / 2], np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    halves = load_dfl()
    print("DFL: {} halves, {} frames, dead {:.1%}".format(
        len(halves), sum(len(h[3]) for h in halves), np.mean(np.concatenate([h[3] for h in halves]))))
    (threshold, min_s), text, oof = cross_validate(halves)
    # held-out flags for every DFL half, for the external check of the whole pipeline
    np.savez_compressed(os.path.join(OUT, "dfl_oof.npz"), **{
        key: postprocess(prob, threshold, min_s * FPS)[1] for key, prob in oof.items()})
    print(text)
    with open(os.path.join(OUT, "cv.txt"), "w") as handle:
        handle.write(text + "\n")
    if args.cv_only:
        return

    clf = model().fit(np.concatenate([h[2][::TRAIN_STEP] for h in halves]),
                      np.concatenate([h[3][::TRAIN_STEP] for h in halves]))
    joblib.dump(clf, os.path.join(OUT, "model.joblib"))
    with open(os.path.join(OUT, "choice.json"), "w") as handle:
        json.dump({"threshold": threshold, "min_s": min_s, "chosen_on": "DFL/IDSSE leave-one-match-out"}, handle)

    prob = clf.predict_proba(features(match_xy()))[:, 1]
    smooth, flag = postprocess(prob, threshold, min_s * FPS)
    np.savez_compressed(os.path.join(OUT, "prob.npz"), prob=smooth.astype(np.float32), dead=flag)
    import eda  # noqa: E402
    graded, _ = eda.graded_mask()
    print("this match (graded frames): model says dead on {:.1%}, {} dead stretches".format(
        flag[graded].mean(), len([s for s in stretches(flag & graded)])))


if __name__ == "__main__":
    main()
