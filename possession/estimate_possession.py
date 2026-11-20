"""Possession v1: who has the ball on every graded frame, from the ball track, players and dead balls.

    python possession/estimate_possession.py            # HOLD_S = 1.0
    python possession/estimate_possession.py --hold 1.5 --name possession_v1_hold15

The rules (written down from how the game works, not learned or tuned on the public answers)
---------------------------------------------------------------------------------------------
CONTROL     a player is within CONTROL_M of the ball and the ball is slower than CONTROL_SPEED
            (a ball flying past a player is not his). Both teams that close, within
            CONTEST_MARGIN of each other: contested, nobody's.
TEAM RUN    a team "has the ball" from its first control frame to its last one, while no opponent
            controls it; flights between teammates of up to PASS_GAP count as still theirs (passes).
            A run is CONFIRMED once it spans HOLD_S and holds MIN_CONTROL control frames: a team
            that only touches the ball for a moment (a deflection, a lost duel) does not get it.
TOUCH       the ball changes direction by more than TURN_DEG, or its speed jumps or drops sharply,
            within TOUCH_M of a player: a one-touch pass, header or deflection, which is never
            "control" because the ball is too fast.
SWITCH      possession changes only when the other team has a confirmed run. The switch is put at
            that team's FIRST TOUCH after the opponent last had the ball (up to BACKTRACK before the
            run): an interception played on first time and received by a teammate belongs to the
            intercepting team from the interception, not from the teammate's reception.
RESTART     after a dead ball, the team of the player nearest the ball when play restarts has it.
HOLD        everything else -- passes in the air, contested balls, frames with no ball -- keeps
            the current team.
DEAD        frames dead_ball/detect_dead.py calls dead are D. Frames outside the graded halves are X.
TAGS        (MTAH_TAGS=1) Smart Tagging as input, which the organisers allow:
            - a restart matched to a set-piece tag goes to the team the tag names as taking it;
            - where the ball is untracked for at least NO_BALL_MIN inside a live stretch, the team
              the phase tags imply (attacking codes for their team, defending codes for the other)
              takes possession, when exactly one team is implied -- otherwise the current team is
              held as before. Ball evidence on the same frame always wins.
            With the tags as input the Smart Tagging agreement in the report is not an independent
            check any more; it is printed anyway, marked as such.

Output
------
evaluation/results/<name>.csv   the submission file (frame,label; gitignored: frame-level labels)
MTAH_WORK_DIR/possession/<name>_report.txt
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
from starter import load_smart_tagging  # noqa: E402
import eda  # noqa: E402

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(ROOT, "work")
# MTAH_VERSION picks (ball track for control and touches, ball track + dead balls for restarts):
#   v2      tracking data only
#   v3      + the externally trained detector everywhere
#   hybrid  the detector only for dead balls and restarts (it sees the ball cross the lines and come
#           back in), the tracking-data ball for who controls it in play -- on control frames the
#           detector's top pick is off the ATD ball 66% of the time (agreement_check.py), mostly on
#           players' heads and socks, exactly where control is decided.
VERSION = os.environ.get("MTAH_VERSION", "v2")
SOURCES = {"v1": ("v1", "v1"),  # v1: detector trained on this match -- reference only, never submitted
           "v2": ("v2", "v2"), "v3": ("v3", "v3"), "hybrid": ("v2", "v3")}
PLAY_SOURCE, DEAD_SOURCE = SOURCES[VERSION]
USE_TAGS = os.environ.get("MTAH_TAGS", "0") == "1"
TAG_SUFFIX = ("_tags" if USE_TAGS else "") + ("_model" if os.environ.get("MTAH_STOPMODEL", "0") == "1" else "")
TRACK = os.path.join(WORK_DIR, "ball_track_" + PLAY_SOURCE, "track.npz")
RESTART_TRACK = os.path.join(WORK_DIR, "ball_track_" + DEAD_SOURCE, "track.npz")
DEAD = os.path.join(WORK_DIR, "dead_ball_" + DEAD_SOURCE + TAG_SUFFIX, "state.npz")
INTERVALS = os.path.join(WORK_DIR, "dead_ball_" + DEAD_SOURCE + TAG_SUFFIX, "intervals.csv")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
REPORT_DIR = os.path.join(WORK_DIR, "possession")
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
HOME_SLOTS, AWAY_SLOTS = slice(0, 24), slice(24, 46)
NONE, HOME, AWAY, CONTESTED = 0, 1, 2, 3

CONTROL_M = 1.5             # slow balls sit a median 1.2 m from the nearest player (p75 ~1.5 m)
CONTROL_SPEED = 8.0         # m/s: faster than a dribble is a pass or shot going past
CONTEST_MARGIN = 0.5        # both teams within CONTROL_M and this close to each other: contested
PASS_GAP = 3 * FPS          # a flight between teammates of up to 3 s keeps the run going
HOLD_S = 1.0                # a team must have the ball this long to take possession (user: 1-2 s)
MIN_CONTROL = 8             # ...with at least this many frames of actual control
NO_BALL_MIN = FPS           # untracked this long (the linker's fill limit) before the phase tags decide
TAG_TEAMS = {"Barcelona": HOME, "Atletic Club": AWAY}
RESTART_SEARCH = 10         # frames either side of a restart to find the ball and its taker
TOUCH_M = 2.0               # a touch is within this of the player (the ball leaves the foot)
TURN_DEG = 30.0             # direction change that counts as a touch...
TOUCH_SPEED = 3.0           # ...when the ball moves at least this fast on one side of it
TOUCH_LAG = 3               # frames either side used for the ball's direction
BACKTRACK = 6 * FPS         # a switch moves back to a touch at most this far
# Settings chosen on the external DFL matches (calibration/dfl_check.py) replace the values above.
# MTAH_CALIBRATED=0 keeps the hand-set ones (for comparison).
CALIBRATION = os.path.join(ROOT, "calibration", "possession.json")
if os.path.exists(CALIBRATION) and os.environ.get("MTAH_CALIBRATED", "1") == "1":
    with open(CALIBRATION) as _handle:
        _chosen = json.load(_handle)
    HOLD_S, CONTROL_M = float(_chosen["HOLD_S"]), float(_chosen["CONTROL_M"])
    PASS_GAP, BACKTRACK = int(_chosen["PASS_GAP"]), int(_chosen["BACKTRACK"])


def control_series(players, x, y, speed, alive):
    """Per frame: NONE / HOME / AWAY / CONTESTED, and the nearest distance of each team."""
    xy = players[:, :, :2] * np.array([LENGTH, WIDTH])
    ball = np.stack([x, y], axis=1)
    distance = np.linalg.norm(xy - ball[:, None], axis=2)
    d_home = np.nanmin(np.where(np.isnan(distance[:, HOME_SLOTS]), np.inf, distance[:, HOME_SLOTS]), axis=1)
    d_away = np.nanmin(np.where(np.isnan(distance[:, AWAY_SLOTS]), np.inf, distance[:, AWAY_SLOTS]), axis=1)
    slow = np.nan_to_num(speed, nan=0.0) < CONTROL_SPEED
    usable = alive & ~np.isnan(x) & slow
    home_near, away_near = usable & (d_home < CONTROL_M), usable & (d_away < CONTROL_M)
    series = np.full(len(x), NONE, np.int8)
    series[home_near & ~away_near] = HOME
    series[away_near & ~home_near] = AWAY
    both = home_near & away_near
    series[both & (d_home < d_away - CONTEST_MARGIN)] = HOME
    series[both & (d_away < d_home - CONTEST_MARGIN)] = AWAY
    series[both & (np.abs(d_home - d_away) <= CONTEST_MARGIN)] = CONTESTED
    return series, d_home, d_away


def touch_series(x, y, d_home, d_away, alive):
    """Per frame: HOME / AWAY where that team's player touches the ball (direction or speed change)."""
    n, lag = len(x), TOUCH_LAG
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[lag:] = np.stack([x[lag:] - x[:-lag], y[lag:] - y[:-lag]], axis=1)
    after[:-lag] = np.stack([x[lag:] - x[:-lag], y[lag:] - y[:-lag]], axis=1)
    s1 = np.linalg.norm(before, axis=1) * FPS / lag
    s2 = np.linalg.norm(after, axis=1) * FPS / lag
    cosine = np.sum(before * after, axis=1) / np.maximum(s1 * s2 * (lag / FPS) ** 2, 1e-9)
    moving = np.fmax(s1, s2) > TOUCH_SPEED
    turned = moving & (cosine < np.cos(np.radians(TURN_DEG)))
    kicked = (s2 > 1.5 * s1 + 2.0) | ((s1 > 4.0) & (s2 < 0.5 * s1))
    event = alive & np.nan_to_num(turned | kicked, nan=0).astype(bool)
    series = np.full(n, NONE, np.int8)
    series[event & (d_home < TOUCH_M) & (d_home < d_away)] = HOME
    series[event & (d_away < TOUCH_M) & (d_away < d_home)] = AWAY
    return series


def backtrack(runs, series, touches):
    """Move each confirmed run's start back to its team's first touch since the opponent last had it."""
    moved = []
    for start, team in runs:
        other = AWAY if team == HOME else HOME
        lo = max(0, start - BACKTRACK)
        window_control, window_touch = series[lo:start], touches[lo:start]
        opponent = np.flatnonzero((window_control == other) | (window_touch == other))
        first = lo + (int(opponent[-1]) + 1 if len(opponent) else 0)
        ours = np.flatnonzero(touches[first:start] == team)
        moved.append((first + int(ours[0]) if len(ours) else start, team))
    return moved


def confirmed_runs(series, alive, hold):
    """[(start, team)] for every confirmed team run, in time order. Runs never cross dead frames."""
    runs = []
    for team, other in ((HOME, AWAY), (AWAY, HOME)):
        frames = np.flatnonzero(series == team)
        if not len(frames):
            continue
        breaks = np.flatnonzero(series == other)
        dead_frames = np.flatnonzero(~alive)
        start = prev = frames[0]
        count = 1
        for frame in list(frames[1:]) + [None]:
            interrupted = (frame is None or frame - prev > PASS_GAP or
                           _any_between(breaks, prev, frame) or _any_between(dead_frames, prev, frame))
            if interrupted:
                if prev - start + 1 >= hold and count >= MIN_CONTROL:
                    runs.append((int(start), team))
                if frame is not None:
                    start, count = frame, 0
            if frame is not None:
                prev, count = frame, count + 1
    return sorted(runs)


def _any_between(sorted_frames, lo, hi):
    """True if sorted_frames has an element strictly between lo and hi."""
    i = np.searchsorted(sorted_frames, lo, side="right")
    return i < len(sorted_frames) and sorted_frames[i] < hi


def restart_teams(intervals, players, x, y):
    """{restart frame: HOME / AWAY}: team of the player nearest the ball as play restarts."""
    xy = players[:, :, :2] * np.array([LENGTH, WIDTH])
    out = {}
    for frame in intervals.end.values:
        window = range(frame - RESTART_SEARCH, frame + RESTART_SEARCH + 1)
        best = None
        for f in sorted(window, key=lambda g: abs(g - frame)):  # closest frame with a ball first
            if 0 <= f < len(x) and not np.isnan(x[f]):
                distance = np.linalg.norm(xy[f] - np.array([x[f], y[f]]), axis=1)
                if np.all(np.isnan(distance)):
                    continue
                best = HOME if np.nanargmin(distance) < 24 else AWAY
                break
        if best is not None:
            out[int(frame)] = best
    return out


def tag_phase_events(alive, x, implied):
    """{frame: team} where the ball is untracked for NO_BALL_MIN+ and the phase tags imply one team."""
    out = {}
    starts, lengths = eda.runs(alive & np.isnan(x))
    for start, length in zip(starts, lengths):
        if length < NO_BALL_MIN:
            continue
        segment = implied[start:start + length]
        changes = np.flatnonzero(np.diff(segment, prepend=-1) != 0)
        for offset in changes:
            if segment[offset] != NONE:
                out[int(start + offset)] = int(segment[offset])
    return out


def possession(graded, dead, series, touches, intervals, players, restart_xy, hold, x=None, implied=None):
    alive = graded & ~dead
    events = {}
    for start, team in backtrack(confirmed_runs(series, alive, hold), series, touches):
        events[start] = team
    restarts = restart_teams(intervals, players, *restart_xy)
    if USE_TAGS and "tag_team" in intervals:
        for end, team in zip(intervals.end.values, intervals.tag_team.fillna("").values):
            if team in TAG_TEAMS:
                restarts[int(end)] = TAG_TEAMS[team]
    for frame, team in restarts.items():
        if alive[frame] if frame < len(alive) else False:
            events.setdefault(frame, team)  # a confirmed run starting on the same frame wins
    if USE_TAGS:
        for frame, team in tag_phase_events(alive, x, implied).items():
            events.setdefault(frame, team)
    owner = np.zeros(len(graded), np.int8)
    for frame, team in events.items():
        owner[frame] = team
    starts, lengths = eda.runs(alive)
    for start, length in zip(starts, lengths):  # carry each team forward inside a live stretch
        segment = owner[start:start + length]
        known = np.flatnonzero(segment)
        if not len(known):
            continue
        filled = segment[known[np.maximum(np.searchsorted(known, np.arange(length), side="right") - 1, 0)]]
        filled[:known[0]] = segment[known[0]]  # before the first event: the first team seen
        owner[start:start + length] = filled
    labels = np.full(len(graded), "X", dtype="<U1")
    labels[graded] = "D"
    labels[alive & (owner == HOME)] = "H"
    labels[alive & (owner == AWAY)] = "A"
    labels[alive & (owner == NONE)] = "H"  # a live stretch with no evidence at all: rare, guess home
    return labels, len(events)


def spells(labels, graded):
    """[(start, end, label)] maximal runs of H or A inside the graded frames."""
    out = []
    for team in ("H", "A"):
        starts, lengths = eda.runs((labels == team) & graded)
        out += [(int(s), int(s + n), team) for s, n in zip(starts, lengths)]
    return sorted(out)


def report(labels, graded, dead, series, touches, events, name, hold):
    alive = graded & ~dead
    found = spells(labels, graded)
    durations = np.array([(e - s) / FPS for s, e, _ in found])
    alive_minutes = alive.sum() / FPS / 60
    lines = ["{}: possession v1, HOLD_S = {:.1f} s".format(name, hold / FPS), ""]
    lines.append("in play {:.1f} min; control frames: home {:.1%}, away {:.1%}, contested {:.1%}".format(
        alive_minutes, np.mean(series[alive] == HOME), np.mean(series[alive] == AWAY),
        np.mean(series[alive] == CONTESTED)))
    lines.append("touch frames: home {}, away {}".format(int(np.sum(touches == HOME)), int(np.sum(touches == AWAY))))
    lines.append("possession share of in-play frames: Barcelona {:.1%}, Athletic {:.1%}".format(
        np.mean(labels[alive] == "H"), np.mean(labels[alive] == "A")))
    lines.append("possession spells: {} ({:.1f} per in-play minute), median {:.1f} s, p10 {:.1f} s, p90 {:.1f} s".format(
        len(found), len(found) / alive_minutes, np.median(durations), np.percentile(durations, 10),
        np.percentile(durations, 90)))
    lines.append("spells under 2 s: {}".format(int(np.sum(durations < 2))))

    events_table = pd.DataFrame(load_smart_tagging.load())
    events_table["team"] = events_table["tags"].map(lambda tags: tags.get("Team", "N/A"))
    cover = eda.implied_possessor(events_table)
    barca, athletic = cover["Barcelona"], cover["Atletic Club"]
    one = alive & (barca ^ athletic)
    tag_label = np.where(barca, "H", "A")
    lines.append("")
    if USE_TAGS:
        lines.append("(MTAH_TAGS=1: the tags are an INPUT here, so this agreement is not an independent check)")
    lines.append("Smart Tagging agreement (in-play frames where the tags imply exactly one team, {:.0%} of them):".format(
        one.sum() / alive.sum()))
    lines.append("  this file      {:.1%}".format(np.mean(labels[one] == tag_label[one])))
    for baseline in ("baseline_smooth50", "baseline_smooth125"):
        path = os.path.join(RESULTS_DIR, baseline + ".csv")
        if os.path.exists(path):
            other = pd.read_csv(path).label.values.astype("<U1")
            lines.append("  {:14s} {:.1%}".format(baseline, np.mean(other[one] == tag_label[one])))
    text = "\n".join(lines)
    print(text)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, name + "_report.txt"), "w") as handle:
        handle.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", type=float, default=HOLD_S, help="seconds a team must have the ball to take it")
    parser.add_argument("--name", default="possession_" + VERSION + TAG_SUFFIX)
    args = parser.parse_args()
    hold = int(round(args.hold * FPS))

    players, _ = eda.load_atd()
    graded, _ = eda.graded_mask()
    track, state = np.load(TRACK), np.load(DEAD)
    x, y, speed, dead = track["x"], track["y"], track["speed"], state["dead"]
    intervals = pd.read_csv(INTERVALS)
    restart = np.load(RESTART_TRACK)

    series, d_home, d_away = control_series(players, x, y, speed, graded & ~dead)
    touches = touch_series(x, y, d_home, d_away, graded & ~dead)
    implied = None
    if USE_TAGS:
        table = pd.DataFrame(load_smart_tagging.load())
        table["team"] = table["tags"].map(lambda tags: tags.get("Team", "N/A"))
        cover = eda.implied_possessor(table)
        barca, athletic = cover["Barcelona"], cover["Atletic Club"]
        implied = np.where(barca & ~athletic, HOME, np.where(athletic & ~barca, AWAY, NONE)).astype(np.int8)
    labels, events = possession(graded, dead, series, touches, intervals, players, (restart["x"], restart["y"]), hold,
                                x, implied)
    path = os.path.join(RESULTS_DIR, args.name + ".csv")
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
    print("wrote", path)
    report(labels, graded, dead, series, touches, events, args.name, hold)


if __name__ == "__main__":
    main()
