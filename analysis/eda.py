"""Exploratory analysis of the BAR-ATH tactical recording.

Reads the raw ATD, homography and Smart Tagging, and writes a text summary plus figures to
analysis/out/. Everything is computed on graded frames unless a line says otherwise.

    python analysis/eda.py

The first run parses the 196 MB ATD into analysis/out/atd_cache.npz (about 110 MB) so later runs
start in seconds. analysis/out/ is gitignored: it holds frame-level derived data, which the data
terms do not allow us to publish.

The public ground truth is used in section 8 only, descriptively, to see how the label definition
plays out against the tracking. It is not used to fit or choose anything.
"""

import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

# Params
OUT_DIR = os.path.join(HERE, "out")
FIG_DIR = os.path.join(OUT_DIR, "figures")
CACHE = os.path.join(OUT_DIR, "atd_cache.npz")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BASELINE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_FRAMES = source_frames("tactical")
IMAGE_CENTRE = (960.0, 540.0)
LONG_GAP = 2 * FPS          # a ball gap at least this long is "long"
CONTROL_RADIUS = 1.5        # metres: nearest player this close plausibly has the ball
CONTESTED_RADIUS = 2.0      # metres: both teams this close is a duel
LINE_MARGIN = 2.0           # metres from a line counts as "at the line"
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
HOME, AWAY = 0, 1
TEAM_COLOURS = {"Barcelona": "#a50044", "Atletic Club": "#2f6db5", "N/A": "#888888"}

summary_lines = []


def say(text=""):
    print(text)
    summary_lines.append(text)


def runs(mask):
    """(starts, lengths) of each run of True in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def load_atd():
    """(players[n, 46, 3], ball[n, 3]) as float32 in raw units, row = video frame."""
    if os.path.isfile(CACHE):
        cached = np.load(CACHE)
        return cached["players"], cached["ball"]
    raw_path = require(["atd_raw"])[0]
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).values
    del text
    if not np.array_equal(table[:, 0], np.arange(1, N_FRAMES + 1)):
        raise ValueError("ATD frame ids are not 1..{}".format(N_FRAMES))
    players = table[:, 1:-3].reshape(N_FRAMES, -1, 3)
    ball = table[:, -3:]
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(CACHE, players=players, ball=ball)
    return players, ball


def graded_mask():
    mask = np.zeros(N_FRAMES, dtype=bool)
    half = np.zeros(N_FRAMES, dtype=np.int8)
    for number, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
        half[first:last + 1] = number
    return mask, half


def to_metres(xy):
    return xy * np.array([LENGTH, WIDTH], dtype=np.float32)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=110)
    plt.close(fig)


def draw_pitch(ax):
    ax.set_xlim(0, LENGTH)
    ax.set_ylim(0, WIDTH)
    ax.set_aspect("equal")
    ax.plot([LENGTH / 2] * 2, [0, WIDTH], color="k", lw=0.6)
    for x0, sign in ((0, 1), (LENGTH, -1)):
        ax.add_patch(plt.Rectangle((x0 if sign > 0 else x0 - BOX_DEPTH, WIDTH / 2 - BOX_HALF_WIDTH),
                                   BOX_DEPTH, 2 * BOX_HALF_WIDTH, fill=False, lw=0.6))
    ax.set_xticks([])
    ax.set_yticks([])


# -------------------------------------------------------------------------------------------
# 1. Coverage
# -------------------------------------------------------------------------------------------
def coverage(players, ball, team, graded, half):
    say("== 1. Coverage ==")
    ball_seen = ~np.isnan(ball[:, 0])
    seen = ~np.isnan(players[:, :, 0])
    home_n = seen[:, team == HOME].sum(axis=1)
    away_n = seen[:, team == AWAY].sum(axis=1)
    say("ball visible: all frames {:.1%}, graded {:.1%}, 1st half {:.1%}, 2nd half {:.1%}".format(
        ball_seen.mean(), ball_seen[graded].mean(), ball_seen[half == 1].mean(),
        ball_seen[half == 2].mean()))
    say("tracked players per graded frame: home mean {:.1f}, away mean {:.1f}, total mean {:.1f}"
        .format(home_n[graded].mean(), away_n[graded].mean(), (home_n + away_n)[graded].mean()))
    total = (home_n + away_n)[graded]
    say("  total tracked: p5 {:.0f}, median {:.0f}, p95 {:.0f}; frames with >22 tracked {:.2%}; "
        "frames with a team >11 {:.2%}".format(
            *np.percentile(total, [5, 50, 95]), (total > 22).mean(),
            ((home_n > 11) | (away_n > 11))[graded].mean()))
    goalkeeper = [i for i in (23, 45)]
    for slot in goalkeeper:
        say("  goalkeeper slot {} ({}) seen on {:.1%} of graded frames".format(
            slot + 1, "home" if team[slot] == HOME else "away", seen[graded, slot].mean()))

    minute = np.arange(N_FRAMES) // (60 * FPS)
    frame = pd.DataFrame({"minute": minute, "ball": ball_seen, "home": home_n, "away": away_n,
                          "graded": graded})
    per_minute = frame.groupby("minute").mean()
    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.fill_between(per_minute.index, 0, 100, where=per_minute["graded"] > 0.5,
                     color="#eeeeee", step="mid", label="graded")
    ax1.plot(per_minute.index, 100 * per_minute["ball"], color="k", label="ball visible %")
    ax1.set_ylabel("ball visible (% of frames)")
    ax1.set_xlabel("video minute")
    ax2 = ax1.twinx()
    ax2.plot(per_minute.index, per_minute["home"], color=TEAM_COLOURS["Barcelona"], label="home tracked")
    ax2.plot(per_minute.index, per_minute["away"], color=TEAM_COLOURS["Atletic Club"], label="away tracked")
    ax2.set_ylabel("mean tracked players")
    ax2.set_ylim(0, 14)
    fig.legend(loc="lower center", ncol=4, fontsize=8)
    ax1.set_title("Coverage per minute of video")
    savefig(fig, "01_coverage_timeline.png")
    return ball_seen, seen


# -------------------------------------------------------------------------------------------
# 2. Ball gaps
# -------------------------------------------------------------------------------------------
def ball_gaps(ball, ball_seen, graded):
    say("\n== 2. Ball gaps (graded frames) ==")
    starts, lengths = runs(~ball_seen & graded)
    missing = lengths.sum()
    say("{} gaps covering {} frames; length median {:.1f} s, p90 {:.1f} s, max {:.1f} s".format(
        len(lengths), missing, np.median(lengths) / FPS, np.percentile(lengths, 90) / FPS,
        lengths.max() / FPS))
    for threshold in (1, 2, 5, 10):
        long_ = lengths >= threshold * FPS
        say("  gaps >= {:>2} s: {:>4} gaps, {:.1%} of missing frames".format(
            threshold, long_.sum(), lengths[long_].sum() / missing))

    long_starts = starts[lengths >= LONG_GAP]
    long_ends = long_starts + lengths[lengths >= LONG_GAP]
    valid = (long_starts > 0) & (long_ends < N_FRAMES)
    before = to_metres(ball[long_starts[valid] - 1, :2])
    after = to_metres(ball[long_ends[valid], :2])

    def where(xy):
        x, y = xy[:, 0], xy[:, 1]
        at_line = ((x < LINE_MARGIN) | (x > LENGTH - LINE_MARGIN)
                   | (y < LINE_MARGIN) | (y > WIDTH - LINE_MARGIN))
        in_box = ((x < BOX_DEPTH) | (x > LENGTH - BOX_DEPTH)) & (np.abs(y - WIDTH / 2) < BOX_HALF_WIDTH)
        return at_line, in_box & ~at_line

    line_b, box_b = where(before)
    line_a, box_a = where(after)
    jump = np.hypot(*(after - before).T)
    say("long gaps (>= {:.0f} s): last-seen ball within {:.0f} m of a line {:.0%}, in a box {:.0%}"
        .format(LONG_GAP / FPS, LINE_MARGIN, line_b.mean(), box_b.mean()))
    say("  reappears at a line {:.0%}, in a box {:.0%}; jump between last and next sighting "
        "median {:.1f} m, >20 m in {:.0%}".format(line_a.mean(), box_a.mean(), np.nanmedian(jump),
                                                    (jump > 20).mean()))
    same = np.zeros(N_FRAMES, dtype=bool)
    same[1:] = (ball[1:, 0] == ball[:-1, 0]) & (ball[1:, 1] == ball[:-1, 1])
    say("ball position repeated exactly from the previous frame on {:.2%} of graded frames: gaps "
        "are real losses, not a held last position".format(same[graded].mean()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(lengths / FPS, bins=np.logspace(np.log10(0.04), np.log10(lengths.max() / FPS), 40),
                 color="#555555")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("gap length (s, log)")
    axes[0].set_ylabel("gaps")
    axes[0].set_title("Ball-missing gap lengths")
    seen_xy = to_metres(ball[ball_seen & graded, :2])
    axes[1].hexbin(seen_xy[:, 0], seen_xy[:, 1], gridsize=40, cmap="Greys", bins="log",
                   extent=(0, LENGTH, 0, WIDTH))
    draw_pitch(axes[1])
    axes[1].set_title("Ball position when visible")
    draw_pitch(axes[2])
    axes[2].scatter(before[:, 0], before[:, 1], s=5, color="#d62728", alpha=0.5, label="last seen")
    axes[2].scatter(after[:, 0], after[:, 1], s=5, color="#2ca02c", alpha=0.5, label="seen again")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].set_title("Ball either side of gaps >= 2 s")
    savefig(fig, "02_ball_gaps.png")
    return starts, lengths


# -------------------------------------------------------------------------------------------
# 3. Speed units and track fragmentation
# -------------------------------------------------------------------------------------------
def speeds_and_tracks(players, ball, seen, team, graded):
    say("\n== 3. Speed field and track fragmentation ==")
    xy = to_metres(players[:, :, :2])
    step = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1)) * FPS          # m/s from positions
    field = players[1:, :, 2]
    ok = np.isfinite(step) & np.isfinite(field) & (step > 1) & (step < 9) & graded[1:, None]
    ratio = np.median(field[ok] / step[ok])
    say("player speed field: p50 {:.2f}, p99 {:.2f}, max {:.2f}; median field / (m/s from positions)"
        " = {:.2f}  ->  {}".format(*np.nanpercentile(field[graded[1:]], [50, 99]),
                                   np.nanmax(field), ratio,
                                   "field is m/s" if 0.8 < ratio < 1.25 else
                                   "field is km/h" if 3.0 < ratio < 4.3 else "units unclear"))
    say("ball speed field present on {:.1%} of frames where the ball is visible".format(
        np.isfinite(ball[:, 2])[~np.isnan(ball[:, 0])].mean()))
    ball_xy = to_metres(ball[:, :2])
    ball_step = np.hypot(*(ball_xy[1:] - ball_xy[:-1]).T) * FPS
    ball_step = ball_step[np.isfinite(ball_step) & graded[1:]]
    say("ball speed from positions: p50 {:.1f}, p90 {:.1f}, p99 {:.1f} m/s; steps > 45 m/s "
        "(implausible, i.e. jumps) {:.2%}".format(*np.percentile(ball_step, [50, 90, 99]),
                                                (ball_step > 45).mean()))

    lifetimes = []
    for slot in range(seen.shape[1]):
        _, lengths = runs(seen[:, slot] & graded)
        lifetimes.append(lengths)
    lifetimes = np.concatenate(lifetimes)
    say("tracklets in graded frames: {} (vs 22 players); lifetime median {:.1f} s, p90 {:.0f} s; "
        "{:.0%} of tracked player-frames are in tracklets >= 60 s".format(
            len(lifetimes), np.median(lifetimes) / FPS, np.percentile(lifetimes, 90) / FPS,
            lifetimes[lifetimes >= 60 * FPS].sum() / lifetimes.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(lifetimes / FPS, bins=np.logspace(np.log10(0.04), np.log10(lifetimes.max() / FPS), 40),
                 color="#555555")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("tracklet lifetime (s, log)")
    axes[0].set_title("A slot is a tracklet, not a player")
    axes[1].hist(ball_step, bins=np.linspace(0, 40, 81), color="#555555")
    axes[1].set_xlabel("ball speed from positions (m/s)")
    axes[1].set_title("Ball speed, graded frames")
    savefig(fig, "03_tracks_and_speed.png")
    return np.concatenate(([np.nan], np.nanmean(np.where(np.isfinite(step), step, np.nan), axis=1)))


# -------------------------------------------------------------------------------------------
# 4. Ball-player geometry
# -------------------------------------------------------------------------------------------
def nearest_geometry(players, ball, ball_seen, team, graded):
    say("\n== 4. Ball-player geometry (graded frames with a ball) ==")
    rows = np.flatnonzero(ball_seen & graded)
    xy = to_metres(players[rows, :, :2])
    distance = np.hypot(*(xy - to_metres(ball[rows, None, :2])).transpose(2, 0, 1))
    with np.errstate(all="ignore"):
        home_d = np.nanmin(np.where(team == HOME, distance, np.nan), axis=1)
        away_d = np.nanmin(np.where(team == AWAY, distance, np.nan), axis=1)
    nearest = np.fmin(home_d, away_d)
    say("nearest player: median {:.1f} m, p90 {:.1f} m; within {} m on {:.0%} of frames".format(
        np.nanmedian(nearest), np.nanpercentile(nearest, 90), CONTROL_RADIUS,
        (nearest < CONTROL_RADIUS).mean()))
    say("both teams within {} m (a duel) on {:.1%}; |home - away nearest| < 1 m on {:.0%}".format(
        CONTESTED_RADIUS, ((home_d < CONTESTED_RADIUS) & (away_d < CONTESTED_RADIUS)).mean(),
        (np.abs(home_d - away_d) < 1).mean()))

    nearest_team = np.where(home_d <= away_d, HOME, AWAY)
    contiguous = np.diff(rows) == 1
    flips = (np.diff(nearest_team) != 0) & contiguous
    minutes = graded.sum() / FPS / 60
    say("raw nearest-team flips: {} ({:.0f} per minute of play); a real match has roughly 3-4 "
        "possession changes per minute".format(flips.sum(), flips.sum() / minutes))
    runs_team = np.diff(np.flatnonzero(np.concatenate(([True], np.diff(nearest_team) != 0, [True]))))
    say("  runs of the same nearest team: median {:.2f} s, {:.0%} shorter than 1 s".format(
        np.median(runs_team) / FPS, (runs_team < FPS).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(nearest, bins=np.linspace(0, 20, 81), color="#555555")
    axes[0].axvline(CONTROL_RADIUS, color="#d62728", ls="--", lw=1)
    axes[0].set_xlabel("distance ball -> nearest tracked player (m)")
    axes[0].set_title("How close is anyone to the ball?")
    margin = np.clip(away_d - home_d, -15, 15)
    axes[1].hist(margin[np.isfinite(margin)], bins=np.linspace(-15, 15, 121), color="#555555")
    axes[1].set_xlabel("nearest away - nearest home (m); > 0 favours home")
    axes[1].set_title("Team margin: most frames are close calls")
    savefig(fig, "04_nearest_geometry.png")


# -------------------------------------------------------------------------------------------
# 5. Homography: the camera as a ball proxy
# -------------------------------------------------------------------------------------------
def camera_proxy(players, ball, ball_seen, seen, graded):
    say("\n== 5. Homography and camera aim ==")
    matrices, error, valid = load_homography.load()
    say("valid {:.1%}; reprojection error graded median {:.2f}, p99 {:.2f}".format(
        valid.mean(), np.median(error[graded]), np.percentile(error[graded], 99)))
    inverse = np.linalg.inv(matrices)
    centre = inverse @ np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
    camera = to_metres((centre[:, :2] / centre[:, 2:3]).astype(np.float32))

    xy = to_metres(players[:, :, :2])
    centroid = np.nanmean(xy, axis=1)
    rows = ball_seen & graded
    ball_xy = to_metres(ball[:, :2])
    camera_err = np.hypot(*(camera[rows] - ball_xy[rows]).T)
    centroid_err = np.hypot(*(centroid[rows] - ball_xy[rows]).T)
    corr_x = np.corrcoef(camera[rows, 0], ball_xy[rows, 0])[0, 1]
    say("image-centre ground point vs ball: median {:.1f} m, p90 {:.1f} m, x corr {:.2f}".format(
        np.median(camera_err), np.percentile(camera_err, 90), corr_x))
    say("player centroid vs ball:          median {:.1f} m, p90 {:.1f} m".format(
        np.median(centroid_err), np.percentile(centroid_err, 90)))
    x_only = np.abs(camera[rows, 0] - ball_xy[rows, 0])
    say("  camera x vs ball x alone: median {:.1f} m, p90 {:.1f} m -- the camera pans with play, "
        "so it locates the ball along the pitch even when the ball is not tracked".format(
            np.median(x_only), np.percentile(x_only, 90)))

    pan = np.abs(np.diff(camera[:, 0])) * FPS
    say("camera pan speed (graded): median {:.1f} m/s, p95 {:.1f} m/s".format(
        np.nanmedian(pan[graded[1:]]), np.nanpercentile(pan[graded[1:]], 95)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sample = np.flatnonzero(rows)[::20]
    axes[0].scatter(ball_xy[sample, 0], camera[sample, 0], s=2, alpha=0.3, color="#333333")
    axes[0].plot([0, LENGTH], [0, LENGTH], color="#d62728", lw=1)
    axes[0].set_xlabel("ball x (m)")
    axes[0].set_ylabel("camera-centre x (m)")
    axes[0].set_title("Camera aim follows the ball (r = {:.2f})".format(corr_x))
    window = slice(11185 + 60 * FPS, 11185 + 4 * 60 * FPS)
    t = np.arange(N_FRAMES)[window] / FPS / 60
    axes[1].plot(t, camera[window, 0], color="#1f77b4", lw=1, label="camera-centre x")
    axes[1].plot(t, np.where(ball_seen[window], ball_xy[window, 0], np.nan), color="k", lw=1,
                 label="ball x (gaps = not tracked)")
    axes[1].set_xlabel("video minute")
    axes[1].set_ylabel("x (m)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Three minutes of play")
    savefig(fig, "05_camera_proxy.png")
    return camera


# -------------------------------------------------------------------------------------------
# 6. Smart Tagging
# -------------------------------------------------------------------------------------------
# Team tag = the team in possession for these codes...
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL"}
# ...and the team out of possession for these. PLAYERS IN THE BOX, SET PIECES and GOALS say
# nothing about who has the ball and are left out.
DEFENDING_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}
OTHER_TEAM = {"Barcelona": "Atletic Club", "Atletic Club": "Barcelona"}


def implied_possessor(events):
    """{team: bool[n_frames]} -- frames inside a phase whose tag implies that team has the ball."""
    cover = {name: np.zeros(N_FRAMES, dtype=bool) for name in OTHER_TEAM}
    for event in events.itertuples():
        if event.team not in OTHER_TEAM:
            continue
        if event.code in ATTACKING_CODES:
            cover[event.team][event.start_frame:event.end_frame] = True
        elif event.code in DEFENDING_CODES:
            cover[OTHER_TEAM[event.team]][event.start_frame:event.end_frame] = True
    return cover


def smart_tagging(ball_seen, graded, starts, lengths):
    say("\n== 6. Smart Tagging ==")
    events = pd.DataFrame(load_smart_tagging.load())
    events["team"] = events["tags"].map(lambda tags: tags.get("Team", "N/A"))
    events["type"] = events["tags"].map(lambda tags: tags.get("Type"))
    events["duration"] = events["end_s"] - events["start_s"]
    table = events.groupby("code").agg(n=("code", "size"), mean_s=("duration", "mean"),
                                       barca=("team", lambda t: (t == "Barcelona").sum()),
                                       athletic=("team", lambda t: (t == "Atletic Club").sum()))
    say(table.sort_values("n", ascending=False).round(1).to_string())

    # Defensive codes are the same interval as an attacking code, tagged with the other team:
    # PROGRESSION + DEFENDING IN MIDDLE THIRD, CREATING CHANCES + DEFENDING IN DEFENSIVE THIRD,
    # BUILD UP + DEFENDING IN ATTACKING THIRD, ATTACKING + DEFENSIVE TRANSITION.
    pairs = events.groupby(["start_frame", "end_frame"])["code"].apply(lambda c: " + ".join(sorted(c)))
    paired = pairs[pairs.str.contains(r"\+")]
    say("{} intervals carry two codes, always an attacking/defending pair: {}".format(
        len(paired), ", ".join(sorted(set(paired)))))
    possessor = implied_possessor(events)
    barca, athletic = possessor["Barcelona"][graded], possessor["Atletic Club"][graded]
    say("graded frames inside a phase implying possession: Barcelona {:.1%}, Athletic {:.1%}, "
        "conflicting {:.1%}, untagged {:.1%}".format((barca & ~athletic).mean(),
                                                     (athletic & ~barca).mean(),
                                                     (barca & athletic).mean(),
                                                     (~barca & ~athletic).mean()))
    baseline = pd.read_csv(BASELINE)["label"].values
    for name, letter in (("Barcelona", "H"), ("Atletic Club", "A")):
        only = possessor[name] & ~possessor["Atletic Club" if letter == "H" else "Barcelona"] & graded
        alive = only & np.isin(baseline, ["H", "A"])
        say("  inside {}-implied phases the smooth50 baseline says {} on {:.0%} of its alive frames"
            .format(name, letter, (baseline[alive] == letter).mean()))

    pieces = events[events["code"] == "SET PIECES"]
    say("set pieces by type: " + ", ".join("{} {}".format(k, v)
                                           for k, v in pieces["type"].value_counts().items()))
    gap_end = starts + lengths
    near = []
    for event in pieces.itertuples():
        before = (gap_end <= event.start_frame + 2 * FPS) & (gap_end >= event.start_frame - 10 * FPS)
        near.append(lengths[before].max() / FPS if before.any() else 0.0)
    near = np.array(near)
    say("ball gap ending within [-10 s, +2 s] of a set-piece tag start: {:.0%} of set pieces; "
        "longest such gap median {:.1f} s".format((near > 0).mean(), np.median(near[near > 0])))
    say("  (does the ball vanish before restarts? if so, the gap itself is a dead-ball signal)")

    fig, ax = plt.subplots(figsize=(13, 5))
    codes = table.sort_values("n").index.tolist()
    for event in events.itertuples():
        ax.barh(codes.index(event.code), event.duration / 60, left=event.start_s / 60, height=0.7,
                color=TEAM_COLOURS.get(event.team, "#888888"))
    for _, first, last in half_video_blocks(GAME):
        ax.axvspan(first / FPS / 60, last / FPS / 60, color="#f4f4f4", zorder=-1)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=8)
    ax.set_xlabel("video minute  (red Barcelona, blue Athletic, grey N/A; shaded = graded)")
    ax.set_title("Smart Tagging timeline (inferred, not ground truth)")
    savefig(fig, "06_smart_tagging.png")
    return events


# -------------------------------------------------------------------------------------------
# 7. Baseline on the full match
# -------------------------------------------------------------------------------------------
def baseline_shape(graded):
    say("\n== 7. What the published floor (smooth50) predicts over the full match ==")
    labels = pd.read_csv(BASELINE)["label"].values[graded]
    change = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1], [True])))
    segment_labels = labels[change[:-1]]
    durations = np.diff(change) / FPS
    for name in "HAD":
        chosen = segment_labels == name
        say("  {}: {:>4} segments, mean {:.1f} s, {:.1%} of graded frames".format(
            name, chosen.sum(), durations[chosen].mean(), (labels == name).mean()))
    possessions = np.isin(segment_labels, ["H", "A"]).sum()
    expected = 17 / 5.0 * graded.sum() / FPS / 60
    say("  {} possessions; at the public window's rate (17 in 5 min) the match would have about "
        "{:.0f}, so the floor over-segments by ~{:.1f}x".format(possessions, expected,
                                                               possessions / expected))


# -------------------------------------------------------------------------------------------
# 8. Public window, descriptive only
# -------------------------------------------------------------------------------------------
def public_window(ball_seen, mean_speed, camera, ball):
    say("\n== 8. Public 5 minutes, descriptive only (not used to fit anything) ==")
    truth = pd.read_csv(PUBLIC_GT)["label"].values
    window = truth != "X"
    rows = np.flatnonzero(window)
    say("ball visible on {:.0%} of the public window, against 59% of the whole graded match: the "
        "window is not representative".format(ball_seen[window].mean()))
    for name in "HAD":
        chosen = truth == name
        say("  {}: {:>5} frames, ball visible {:.0%}, mean player speed {:.2f} m/s".format(
            name, chosen.sum(), ball_seen[chosen].mean(), np.nanmean(mean_speed[chosen])))
    labels = truth[rows]
    change = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1], [True])))
    for start, end in zip(change[:-1], change[1:]):
        if labels[start] == "D":
            frames = rows[start:end]
            say("  D spell {:>6}-{:>6} ({:4.1f} s): ball visible {:.0%}, players {:.2f} m/s".format(
                frames[0], frames[-1], len(frames) / FPS, ball_seen[frames].mean(),
                np.nanmean(mean_speed[frames])))

    baseline = pd.read_csv(BASELINE)["label"].values
    colour = {"H": TEAM_COLOURS["Barcelona"], "A": TEAM_COLOURS["Atletic Club"], "D": "#bbbbbb",
              "X": "white"}
    t = rows / FPS / 60
    fig, axes = plt.subplots(4, 1, figsize=(13, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.5, 1.6]})
    for ax, series, title in ((axes[0], truth, "ground truth"), (axes[1], baseline, "baseline smooth50")):
        ax.scatter(t, np.zeros_like(t), c=[colour[v] for v in series[rows]], marker="|", s=400)
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", fontsize=8)
    axes[2].fill_between(t, 0, ball_seen[rows], color="k", step="mid")
    axes[2].set_yticks([])
    axes[2].set_ylabel("ball tracked", rotation=0, ha="right", fontsize=8)
    axes[3].plot(t, np.where(ball_seen[rows], ball[rows, 0] * LENGTH, np.nan), color="k", lw=1,
                 label="ball x")
    axes[3].plot(t, camera[rows, 0], color="#1f77b4", lw=1, label="camera-centre x")
    speed_axis = axes[3].twinx()
    speed_axis.plot(t, pd.Series(mean_speed[rows]).rolling(25, center=True).mean(),
                    color="#ff7f0e", lw=1, label="mean player speed")
    speed_axis.set_ylabel("m/s", fontsize=8)
    axes[3].set_ylabel("x (m)", fontsize=8)
    axes[3].set_xlabel("video minute")
    axes[3].legend(loc="upper left", fontsize=7)
    speed_axis.legend(loc="upper right", fontsize=7)
    axes[0].set_title("Public window: red H (Barcelona), blue A (Athletic), grey D")
    savefig(fig, "08_public_window.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    players, ball = load_atd()
    meta_path = require(["atd_meta"])[0]
    team = team_of_each_track(meta_path)
    graded, half = graded_mask()
    say("BAR-ATH tactical: {} frames, {} graded ({:.1f} min), {} track slots ({} home, {} away)"
        .format(N_FRAMES, graded.sum(), graded.sum() / FPS / 60, len(team),
                (team == HOME).sum(), (team == AWAY).sum()))
    say()

    ball_seen, seen = coverage(players, ball, team, graded, half)
    starts, lengths = ball_gaps(ball, ball_seen, graded)
    mean_speed = speeds_and_tracks(players, ball, seen, team, graded)
    nearest_geometry(players, ball, ball_seen, team, graded)
    camera = camera_proxy(players, ball, ball_seen, seen, graded)
    smart_tagging(ball_seen, graded, starts, lengths)
    baseline_shape(graded)
    public_window(ball_seen, mean_speed, camera, ball)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as handle:
        handle.write("\n".join(summary_lines) + "\n")
    print("\nwrote {} and figures in {}".format(os.path.join(OUT_DIR, "summary.txt"), FIG_DIR))
