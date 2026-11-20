"""Dead ball vs in play, from the linked ball track and the players.

    python dead_ball/detect_dead.py

Rules
-----
OUT       the ball is more than MARGIN_M beyond a touchline or goal line for OUT_MIN frames, or the
          track vanishes for VANISH_MIN while the ball is at a line and heading out. MARGIN_M is
          there because the whole ball must cross, positions are noisy, and a high ball near the
          line projects outside (the homography assumes the ground). For the same reason an OUT is
          cancelled if the ball is seen moving inside the pitch within CONFIRM frames (it was in
          the air), and a vanishing ball in the AIRBORNE frames after a set piece is a cross, not
          an exit.
RESTART   depends on the line the ball left by:
          touchline  -> THROW-IN: the ball comes back into the pitch near that touchline, moving,
                        after being outside or untracked (the thrown ball is often not tracked)
          goal line  -> CORNER or GOAL KICK: a PLACED ball at that end -- still for at least
                        PLACED_RUN at a corner arc (within CORNER_SPOT_M of a flag) or in the goal
                        area (not on the goal line itself) -- leaves its spot. The departure is
                        the restart whether or not the ball is tracked after it: corners and goal
                        kicks usually go into the air, where the track loses the ball. Placement
                        matters because during a dead ball the ball is thrown back, passed to the
                        taker or swapped for a new one, and those movements look like kicks.
                        Without a placed ball, any kick near that goal line ends it ("unplaced").
          A touchline exit within GOAL_END_M of a goal line may also be restarted by a corner:
          near the flag the crossing point is ambiguous.
RESUMED   whatever the line, play has restarted once the ball is tracked moving inside the pitch
          for RESUMED_MOVING of RESUMED_WINDOW frames AND the players speed up with it (mean
          player speed over that window above RESUMED_PLAYERS). This catches restarts the track
          did not see -- e.g. a corner placed while the ball was untracked -- which otherwise
          run on until the next restart that is seen, swallowing the play in between.
          A stop inside a stretch that is already dead (a goal, an earlier stop) starts nothing.
GOAL      an OUT through the goal mouth (or a GOALS tag) followed by a KICKOFF: dead until then.
KICKOFF   both teams in their own halves (at most one player of each across the halfway line,
          at least 8 of each on the pitch) for KICKOFF_MIN; play starts with the first fast ball
          movement near the centre spot at the end of that shape. Also opens each half: the
          stretch from the start of the graded half to its kickoff is dead (half 2 is graded from
          ~2 min before its kickoff).
WHISTLE   fouls, offsides, injuries: the ball never leaves the pitch, but it stops. A stoppage is
          the ball still (slower than FOUL_STILL_SPEED) for FOUL_STILL_MIN, more than FOUL_INSIDE_M
          inside the lines, with the players slow (mean under FOUL_PLAYERS), and either
          FOUL_LONG of it or a free-kick shape: nobody of the other team within FOUL_SPACE_M of the
          ball (the wall stands 9.15 m back) after the players had been running (1 s mean speed
          above FOUL_DECEL in the FOUL_DECEL_WINDOW before). A goalkeeper holding the ball in his box
          is in play and never counts.
          The ball only goes still once it is placed for the free kick; between the whistle and
          that moment it is dribbled back, picked up or passed to the taker, which looks like play.
          So the stop is TRACED BACK from the placed ball, at most FOUL_TRACE and never into an
          earlier dead ball:
            1. a free kick is taken where the foul was: the last challenge (both teams within
               CHALLENGE_M of the ball) within SPOT_M of the placed ball while the players were
               still active (1 s mean speed above FOUL_ACTIVE), as long as the players stopped
               running within SPOT_SETTLE of it (otherwise play went on after that challenge);
            2. else the whistle reaction: the last frame the players were running (1 s mean above
               FOUL_DECEL) -- after it they slow down together (offside free kicks are not taken
               where the ball was, so 1 does not apply to them);
            3. else the last active challenge in the last FOUL_LOOKBACK;
            4. else, the players having been slow for the whole window, a long stoppage (injury,
               VAR): the last frame they were running up to LONG_TRACE back; else the ball going still.
          The restart is the free kick: the first kick, resumed play or departure from the spot
          after the ball starts to move.
NO BALL   stoppages with the ball untracked: the players average under NOBALL_PLAYERS for at least
          NOBALL_MIN while the ball is missing from most frames; dead until they speed up again.
          Lower confidence than everything above, so it is a reason of its own.

TAGS      (MTAH_TAGS=1) the Smart Tagging set pieces as input -- the organisers allow Smart Tagging
          as an input. Each tag names its type and the team taking it, and nothing new is fitted:
            - a FREE KICK tag sits at the foul: if no dead stretch covers it, a foul starts there and
              runs to the free kick (the first kick or resumed play after FOUL_STILL_MIN, else the
              players moving again);
            - a THROW-IN / CORNER / GOAL KICK tag starts at the restart: if no dead stretch ends
              within TAG_TOLERANCE of it, one is added. Its restart is the candidate of that kind
              closest to the tag (throw-in / placed-ball departure / kick / resumed play), else the
              tag itself; its stop is the last exit over a line before it, else the last frame the
              players ran (as in WHISTLE rule 2), never reaching back into an earlier dead stretch.
              A dead stretch that runs on through the tagged restart is cut there;
            - a dead stretch that does end near a tag takes the tag's type as its reason;
            - every dead stretch whose end matches a tag records the tag's team (tag_team), which
              the possession step uses for the restart.
          With the tags as input, the Smart Tagging recall in the report is no longer an independent
          check (it is printed anyway, marked as such).

MODEL     (MTAH_STOPMODEL=1) the stoppage model (stoppage_model.py: players only, trained and
          thresholded on the external DFL matches) replaces the NO BALL rule, which is the same idea
          with hand-set thresholds, and adds every stretch it calls dead that no rule above has
          touched (reason "whistle (player model)"). Where a ball-based rule found a stoppage and
          the model overlaps it, each edge is the rule's, the model's or the wider of the two, as
          chosen on the DFL matches by calibration/dfl_check.py (stoppage_model/edges.json; the
          wider of the two won there). Without edges.json the rule's edges are kept.

Every threshold is physics or this match's own data distributions, never the public ground truth.
Smart Tagging is used for the goal start when the ball track cannot see it, and for the recall
check in the report (and with MTAH_TAGS=1 as described under TAGS).

Output: MTAH_WORK_DIR/dead_ball_v1/
    intervals.csv   start, end (exclusive), reason, how the stop and restart were found
    state.npz       dead[n] bool, reason[n] index into REASONS, over every video frame
    report.txt
"""

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
from starter.load_smart_tagging import load as load_tags  # noqa: E402
import eda  # noqa: E402

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(ROOT, "work")
VERSION = os.environ.get("MTAH_VERSION", "v2")   # v2 = tracking data only, v3 = + external detector
USE_TAGS = os.environ.get("MTAH_TAGS", "0") == "1"   # Smart Tagging set pieces as input (see TAGS)
USE_MODEL = os.environ.get("MTAH_STOPMODEL", "0") == "1"   # external stoppage model (see MODEL)
TRACK = os.path.join(WORK_DIR, "ball_track_" + VERSION, "track.npz")
OUT_DIR = os.path.join(WORK_DIR, "dead_ball_" + VERSION + ("_tags" if USE_TAGS else "") + ("_model" if USE_MODEL else ""))
STOP_MODEL = os.path.join(WORK_DIR, "stoppage_model", "prob.npz")
EDGE_CHOICE = os.path.join(WORK_DIR, "stoppage_model", "edges.json")   # written by dfl_check.py
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
GOAL_HALF_WIDTH = 7.32 / 2
CORNER_M, GOAL_AREA_DEPTH, GOAL_AREA_HALF = 3.0, 5.5, 9.16   # restart spots, metres
HOME_GK, AWAY_GK = 23, 45                                     # 0-based player slots

MARGIN_M = 1.0              # ball centre this far beyond the line = out
OUT_MIN = 15                # frames outside (0.6 s) before it counts
OUT_HOLE = 5                # frames of no ball / back inside tolerated within an OUT run
VANISH_MIN = 2 * FPS        # a track that vanishes for 2 s...
VANISH_LINE_M = 2.0         # ...within 2 m of a line...
VANISH_LOOKAHEAD = 5        # ...and would be out 5 frames later at its last velocity
MIN_DEAD = FPS              # a restart is at least 1 s after the stop
MAX_DEAD = 90 * FPS         # longest throw-in / corner / goal kick we search for a restart in
CONFIRM = 2 * FPS           # an OUT seen back in play this soon was a high ball...
CONFIRM_FRAMES = 10         # ...if this many frames are inside by 1 m and moving
AIRBORNE = 4 * FPS          # after a set piece, a vanishing ball is a cross for this long
KICK_SPEED = 5.0            # m/s: a pass or throw
SLOW_SPEED = 2.5            # m/s: the ball is placed or held before a set piece
KICK_PRE = KICK_POST = FPS // 2
KICK_OUT_TOL = 1.5          # metres outside the line the kick may start from (throw-ins)
INSIDE_SHARE = 0.6          # share of the next half second the ball must be inside the pitch
KICKOFF_TOL = 1.0           # metres across halfway still counted as own half
KICKOFF_MIN = 2 * FPS
KICKOFF_HOLE = 10
KICKOFF_SEARCH = 2 * FPS    # look for the kick this far either side of the end of the shape
KICKOFF_BALL_M = 8.0
KICKOFF_SPEED = 3.0
KICKOFF_SIT_MIN = 5         # frames in the second before the kick with the ball still near the centre
KICKOFF_MERGE = 60 * FPS    # kickoff shapes this close are one kickoff (the shape flickers)
MIN_THROW = 2 * FPS         # a throw-in is taken at least 2 s after the ball goes out
THROW_INSIDE_M = 0.5        # back in the pitch by this much...
THROW_SPEED = 2.0           # ...moving at least this fast...
THROW_NEAR_M = 8.0          # ...within this of the touchline...
THROW_PRE = FPS             # ...after being outside or untracked in the second before
CORNER_NEAR_M = 8.0
GOAL_END_M = 12.0           # a corner or goal kick is restarted within this of the goal line it left by
CORNER_SPOT_M = 3.0         # placed at a corner: this close to the flag (positions are a few m off at worst)
GOAL_AREA_X = (0.5, 7.5)    # placed for a goal kick: this far out from the goal line...
GOAL_AREA_Y = 11.0          # ...and within this of the centre line of the goal (goal area +-9.16 m)
PLACED_WINDOW = 20          # frames before a kick inspected for placement...
PLACED_MIN = 10             # ...of which this many must have the ball still on the spot
PLACED_RUN = FPS            # a placed ball sits on its spot at least this long...
PLACED_HOLE = 2 * FPS       # ...counting flickers of the track this short as still there
GOAL_GUARD = 10 * FPS       # stops this soon before a goal belong to the goal
GOAL_LOOKBACK = 6 * 60 * FPS   # a kickoff's goal happened at most 6 min before it
HALF_START_SEARCH = 6 * 60 * FPS
RESUMED_WINDOW = 2 * FPS
RESUMED_MOVING = int(1.5 * FPS)
RESUMED_SPEED = 3.0         # m/s ball, inside the pitch by more than 1 m
RESUMED_PLAYERS = 1.6       # m/s mean player speed (in play ~1.9, dead ~1.2 on this match)
FOUL_STILL_SPEED = 0.7      # m/s: a placed ball (a ball under a dribbling foot still moves)
FOUL_STILL_MIN = 3 * FPS
FOUL_STILL_HOLE = 12        # track flickers tolerated inside a still stretch
FOUL_LONG = 5 * FPS         # this long still with the players slow is a stoppage whatever the shape
FOUL_INSIDE_M = 3.0
FOUL_PLAYERS = 1.3          # m/s mean player speed while the ball is still (in play ~1.9, dead ~1.1)
FOUL_SPACE_M = 5.0
FOUL_DECEL = 1.8            # m/s, 1 s mean player speed...
FOUL_DECEL_WINDOW = 6 * FPS # ...reached in this window before the ball went still
FOUL_LOOKBACK = 8 * FPS     # (rule 3 only)
FOUL_TRACE = 20 * FPS       # how far back from the placed ball the whistle is searched for
SPOT_M = 5.0                # "at the free-kick spot" (positions are a few metres off at worst)
FOUL_ACTIVE = 1.6           # m/s, 1 s mean player speed: play was live at the challenge
SPOT_SETTLE = 4 * FPS       # players stopped running at most this long after the spot challenge
LONG_TRACE = 60 * FPS       # rule 4: how far back a long stoppage is traced
CHALLENGE_M = 2.0
GK_HOLD_M = 2.5
BOX_DEPTH, BOX_HALF = 16.5, 20.16
FOUL_RESTART_SEARCH = 30 * FPS
NOBALL_PLAYERS = 1.0        # m/s, 2 s mean
NOBALL_MIN = 5 * FPS
NOBALL_MISSING = 0.8        # share of frames without a ball
NOBALL_RESUME = 1.5         # m/s, 1 s mean: the players are moving again
TAG_TOLERANCE = 10 * FPS    # a restart this close to a set-piece tag is that tag's restart
TAG_KINDS = {"Throw-in": "throw-in", "Corner Kick": "corner", "Goal Kick": "goal kick",
             "Free Kick": "foul / whistle", "Kick Off (Start)": "half start", "Kick Off (After Goal)": "goal"}

REASONS = ["in play", "throw-in", "corner", "goal kick", "out (other)", "goal", "half start",
           "foul / whistle", "stoppage (no ball)", "whistle (player model)"]


def closed_runs(mask, hole):
    """Runs of mask after closing holes of up to `hole` frames."""
    starts, lengths = eda.runs(~mask)
    filled = mask.copy()
    for start, length in zip(starts, lengths):
        if 0 < start and start + length < len(mask) and length <= hole:
            filled[start:start + length] = True
    return eda.runs(filled)


def outside_distance(x, y):
    """Metres beyond the nearest line (negative = inside the pitch by that much)."""
    return np.maximum(np.maximum(-x, x - LENGTH), np.maximum(-y, y - WIDTH))


def find_stops(x, y, valid, graded):
    """[(frame, how)] where the ball goes out of play over a line."""
    out = outside_distance(x, y)
    stops = []
    starts, lengths = closed_runs(valid & (np.nan_to_num(out, nan=-99) > MARGIN_M) & graded, OUT_HOLE)
    for start, length in zip(starts, lengths):
        if np.sum(valid[start:start + length]) >= OUT_MIN:
            stops.append((int(start), "crossed"))
    starts, lengths = eda.runs(~valid & graded)
    for start, length in zip(starts, lengths):
        last = start - 1
        if length < VANISH_MIN or last < 3 or not valid[last] or not valid[last - 3]:
            continue
        line_distance = -out[last]
        if line_distance > VANISH_LINE_M:
            continue
        vx, vy = (x[last] - x[last - 3]) / 3, (y[last] - y[last - 3]) / 3
        ahead = outside_distance(x[last] + VANISH_LOOKAHEAD * vx, y[last] + VANISH_LOOKAHEAD * vy)
        if ahead > 0:
            stops.append((int(start), "vanished"))
    return sorted(stops)


def line_of(x, y):
    if x < 0 or x > LENGTH:
        return "goal mouth" if abs(y - WIDTH / 2) < GOAL_HALF_WIDTH + 0.5 else "goal line"
    return "touchline"


def kick_frames(x, y, speed, valid):
    """Boolean per frame: a set-piece kick or throw starts here."""
    n = len(x)
    out = np.nan_to_num(outside_distance(x, y), nan=99)
    fast = valid & (np.nan_to_num(speed) > KICK_SPEED) & (out < KICK_OUT_TOL)
    slow_or_out = valid & ((np.nan_to_num(speed, nan=99) < SLOW_SPEED) | (out > 0))
    inside = valid & (out < 0.3)
    window = np.ones(KICK_PRE)
    pre_count = np.convolve(valid.astype(float), window)[:n]               # frames t-KICK_PRE+1..t
    pre_slow = np.convolve(slow_or_out.astype(float), window)[:n]
    pre_count, pre_slow = np.roll(pre_count, 1), np.roll(pre_slow, 1)      # ...shifted to end at t-1
    post_count = np.convolve(valid.astype(float), window[::-1])[KICK_POST - 1:][:n]   # t..t+KICK_POST-1
    post_inside = np.convolve(inside.astype(float), window[::-1])[KICK_POST - 1:][:n]
    return (fast & (pre_count > 0) & (pre_slow >= 0.5 * pre_count) &
            (post_count > 0) & (post_inside >= INSIDE_SHARE * post_count))


def corner_distance(x, y):
    return np.min([np.hypot(x - cx, y - cy) for cx in (0, LENGTH) for cy in (0, WIDTH)], axis=0)


def placed_spots(x, y, speed, valid):
    """Per frame: 'corner' / 'goal kick' where a ball was placed and still in the frames before."""
    n = len(x)
    still = valid & (np.nan_to_num(speed, nan=99) < SLOW_SPEED)
    depth = np.minimum(np.nan_to_num(x, nan=-99), LENGTH - np.nan_to_num(x, nan=-99))
    at_corner = still & (corner_distance(np.nan_to_num(x, nan=-99), np.nan_to_num(y, nan=-99)) < CORNER_SPOT_M)
    in_area = (still & (depth > GOAL_AREA_X[0]) & (depth < GOAL_AREA_X[1]) &
               (np.abs(np.nan_to_num(y, nan=-99) - WIDTH / 2) < GOAL_AREA_Y))
    window = np.ones(PLACED_WINDOW)
    counts = {name: np.roll(np.convolve(mask.astype(float), window)[:n], 1)
              for name, mask in (("corner", at_corner), ("goal kick", in_area))}
    placed = np.full(n, "", dtype=object)
    placed[counts["goal kick"] >= PLACED_MIN] = "goal kick"
    placed[counts["corner"] >= PLACED_MIN] = "corner"
    return placed


def departures(x, y, speed, valid):
    """[(frame, 'corner' | 'goal kick', x)] where a placed ball leaves its spot."""
    still = valid & (np.nan_to_num(speed, nan=99) < SLOW_SPEED)
    xs, ys = np.nan_to_num(x, nan=-99), np.nan_to_num(y, nan=-99)
    depth = np.minimum(xs, LENGTH - xs)
    spots = {"corner": still & (corner_distance(xs, ys) < CORNER_SPOT_M),
             "goal kick": still & (depth > GOAL_AREA_X[0]) & (depth < GOAL_AREA_X[1]) &
             (np.abs(ys - WIDTH / 2) < GOAL_AREA_Y)}
    out = []
    for kind, mask in spots.items():
        starts, lengths = closed_runs(mask, PLACED_HOLE)
        for start, length in zip(starts, lengths):
            if np.sum(mask[start:start + length]) >= PLACED_RUN:
                last = start + length - 1
                out.append((int(start + length), kind, float(x[last])))
    return sorted(out)


def throw_ins(x, y, speed, valid):
    """Boolean per frame: the ball has just come back into the pitch near a touchline."""
    n = len(x)
    out = np.nan_to_num(outside_distance(x, y), nan=99)
    near_line = (np.nan_to_num(y, nan=-99) < THROW_NEAR_M) | (np.nan_to_num(y, nan=-99) > WIDTH - THROW_NEAR_M)
    entering = valid & (out < -THROW_INSIDE_M) & (np.nan_to_num(speed) > THROW_SPEED) & near_line
    before = (~valid | (out > -0.3)).astype(float)
    recent = np.roll(np.convolve(before, np.ones(THROW_PRE))[:n], 1)  # frames t-THROW_PRE..t-1
    return entering & (recent > 0)


def kickoffs(players, x, y, speed, valid, graded, half):
    """[(kick frame, shape start)] for every kickoff shape in the graded halves."""
    xs = players[:, :, 0] * LENGTH
    found = []
    for h in np.unique(half[graded]):
        frames = graded & (half == h)
        home_left = np.nanmean(xs[frames, HOME_GK]) < LENGTH / 2
        mid = LENGTH / 2
        home, away = xs[:, :24], xs[:, 24:]
        home_on, away_on = np.sum(~np.isnan(home), axis=1), np.sum(~np.isnan(away), axis=1)
        if home_left:
            home_across, away_across = np.sum(home > mid + KICKOFF_TOL, axis=1), np.sum(away < mid - KICKOFF_TOL, axis=1)
        else:
            home_across, away_across = np.sum(home < mid - KICKOFF_TOL, axis=1), np.sum(away > mid + KICKOFF_TOL, axis=1)
        shape = frames & (home_on >= 8) & (away_on >= 8) & (home_across <= 1) & (away_across <= 1)
        starts, lengths = closed_runs(shape, KICKOFF_HOLE)
        for start, length in zip(starts, lengths):
            if length < KICKOFF_MIN:
                continue
            end = start + length
            centre = valid & (np.hypot(x - mid, y - WIDTH / 2) < KICKOFF_BALL_M)
            sitting = centre & (np.nan_to_num(speed, nan=99) < SLOW_SPEED)
            sat = np.roll(np.convolve(sitting.astype(float), np.ones(FPS))[:len(x)], 1)
            near = centre & (np.nan_to_num(speed) > KICKOFF_SPEED) & (sat >= KICKOFF_SIT_MIN)
            lo = max(start, end - KICKOFF_SEARCH)
            window = np.flatnonzero(near[lo:end + KICKOFF_SEARCH])
            kick = lo + int(window[0]) if len(window) else end
            found.append((kick, int(start)))
    merged = []
    for kick, start in sorted(found):  # a flickering shape is one kickoff: keep its last kick
        if merged and kick - merged[-1][0] < KICKOFF_MERGE:
            merged[-1] = (kick, merged[-1][1])
        else:
            merged.append((kick, start))
    return merged


def confirmed(stops, x, y, speed, valid):
    """Drop OUTs after which the ball is quickly tracked moving well inside the pitch."""
    inside_moving = (valid & (np.nan_to_num(outside_distance(x, y), nan=99) < -1.0) &
                     (np.nan_to_num(speed) > THROW_SPEED))
    return [(f, how) for f, how in stops if inside_moving[f + 5:f + CONFIRM].sum() < CONFIRM_FRAMES]


def resumed_frames(x, y, speed, valid, players):
    """Boolean per frame: sustained ball movement inside the pitch starts here, with the players moving."""
    n = len(x)
    moving = (valid & (np.nan_to_num(outside_distance(x, y), nan=99) < -1.0) &
              (np.nan_to_num(speed) > RESUMED_SPEED))
    ahead = np.convolve(moving.astype(float), np.ones(RESUMED_WINDOW)[::-1])[RESUMED_WINDOW - 1:][:n]
    player_speed = np.nan_to_num(np.nanmean(players[:, :, 2], axis=1))
    player_ahead = np.convolve(player_speed, np.ones(RESUMED_WINDOW)[::-1])[RESUMED_WINDOW - 1:][:n] / RESUMED_WINDOW
    return moving & (ahead >= RESUMED_MOVING) & (player_ahead > RESUMED_PLAYERS)


def build_intervals(x, y, speed, valid, graded, half, players, tags):
    stops = confirmed(find_stops(x, y, valid, graded), x, y, speed, valid)
    resumed = np.flatnonzero(resumed_frames(x, y, speed, valid, players) & graded)
    kicks = np.flatnonzero(kick_frames(x, y, speed, valid) & graded)
    throws = np.flatnonzero(throw_ins(x, y, speed, valid) & graded)
    placed = placed_spots(x, y, speed, valid)
    leaving = departures(x, y, speed, valid)
    kickoff_list = kickoffs(players, x, y, speed, valid, graded, half)
    goal_tags = sorted(t["start_frame"] for t in tags if t["code"] == "GOALS")
    intervals = []

    # Half starts and goals end at a kickoff.
    for kick, shape_start in kickoff_list:
        first = int(np.flatnonzero(graded & (half == half[kick]))[0])
        if kick - first < HALF_START_SEARCH and not any(i["reason"] == "half start" and i["half"] == half[kick]
                                                        for i in intervals):
            intervals.append({"start": first, "end": kick, "reason": "half start", "half": half[kick],
                              "stop_how": "half begins", "restart_how": "kickoff"})
            continue
        mouth = [f for f, _ in stops if kick - GOAL_LOOKBACK < f < shape_start and
                 line_of(*stop_position(x, y, valid, f)) == "goal mouth"]
        tag_before = [f for f in goal_tags if kick - GOAL_LOOKBACK < f < shape_start]
        after_tag = [f for f, _ in stops if tag_before and tag_before[-1] < f < shape_start]
        if mouth:
            start, how = mouth[-1], "ball out through the goal mouth"
        elif after_tag:
            start, how = after_tag[0], "first stop after the GOALS tag start"
        elif tag_before:
            start, how = tag_before[-1], "GOALS tag start"
        else:
            start, how = shape_start, "kickoff shape start"
        intervals.append({"start": int(start), "end": kick, "reason": "goal", "half": half[kick],
                          "stop_how": how, "restart_how": "kickoff"})

    # Every other stop ends at the restart that fits the line it left by.
    for stop, how in stops:
        if any(i["start"] - (GOAL_GUARD if i["reason"] == "goal" else 0) <= stop < i["end"] for i in intervals):
            continue
        if how == "vanished" and any(i["end"] <= stop < i["end"] + AIRBORNE for i in intervals
                                     if i["reason"] in ("corner", "goal kick", "throw-in", "out (other)")):
            continue
        sx, sy = stop_position(x, y, valid, stop)
        end_x = 0.0 if sx < LENGTH / 2 else LENGTH
        near_goal_line = abs(sx - end_x) < GOAL_END_M
        options = []  # (frame, reason, how): the earliest fitting restart wins
        if line_of(sx, sy) == "touchline":
            window = throws[(throws >= stop + MIN_THROW) & (throws < stop + MAX_DEAD)]
            if len(window):
                options.append((int(window[0]), "throw-in", "ball back in near the touchline"))
        if line_of(sx, sy) != "touchline" or near_goal_line:
            allowed = ("corner", "goal kick") if line_of(sx, sy) != "touchline" else ("corner",)
            options += [(f, kind, "placed ball left its spot") for f, kind, dx in leaving
                        if stop + MIN_DEAD <= f < stop + MAX_DEAD and abs(dx - end_x) < GOAL_END_M and kind in allowed]
            window = kicks[(kicks >= stop + MIN_DEAD) & (kicks < stop + MAX_DEAD)]
            window = window[np.abs(x[window] - end_x) < GOAL_END_M]
            options += [(int(f), placed[f], "kick of a placed ball") for f in window if placed[f] in allowed]
        restart = None
        if options:
            restart, reason, restart_how = min(options)
        elif line_of(sx, sy) != "touchline":
            window = kicks[(kicks >= stop + MIN_DEAD) & (kicks < stop + MAX_DEAD)]
            window = window[np.abs(x[window] - end_x) < GOAL_END_M]
            if len(window):
                restart, restart_how = int(window[0]), "kick near that goal line (unplaced)"
                reason = "corner" if corner_distance(x[restart], y[restart]) < CORNER_NEAR_M else "goal kick"
        if restart is None:  # fall back to any set-piece kick
            window = kicks[(kicks >= stop + MIN_DEAD) & (kicks < stop + MAX_DEAD)]
            if not len(window):
                continue
            restart, reason, restart_how = int(window[0]), "out (other)", "any kick (no matching restart)"
        early = resumed[(resumed >= stop + MIN_DEAD) & (resumed < restart)]
        if len(early):  # play visibly restarted before the restart we found
            restart, restart_how = int(early[0]), "play resumed (ball and players moving)"
            if line_of(sx, sy) == "touchline" and not near_goal_line:
                reason = "throw-in"
            else:
                reason = "corner" if corner_distance(x[restart], y[restart]) < 2 * GOAL_END_M else "goal kick"
        if half[restart] != half[stop]:
            continue
        intervals.append({"start": stop, "end": restart, "reason": reason, "half": half[stop],
                          "stop_how": how, "restart_how": restart_how})
    return pd.DataFrame(intervals).sort_values("start").reset_index(drop=True)


def stop_position(x, y, valid, frame):
    """Ball position at a stop, or at the last tracked frame before it (a vanished ball)."""
    lo = max(0, frame - 2 * FPS)
    tracked = np.flatnonzero(valid[lo:frame + 1])
    at = lo + int(tracked[-1]) if len(tracked) else frame
    return x[at], y[at]


def moving_mean(values, window):
    """Centred moving mean of a per-frame series, NaN-aware."""
    ok = ~np.isnan(values)
    total = np.convolve(np.where(ok, values, 0.0), np.ones(window), "same")
    count = np.convolve(ok.astype(float), np.ones(window), "same")
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def trace_back(start, end, x, y, valid, alive, challenge, speed_1s):
    """(stop frame, how) for a stoppage whose ball went still at `start` (see WHISTLE)."""
    lo = max(0, start - FOUL_TRACE)
    earlier_dead = np.flatnonzero(~alive[lo:start])
    if len(earlier_dead):
        lo += int(earlier_dead[-1]) + 1
    if lo >= start:
        return int(start), "ball still"
    placed = valid[start:end]
    spot_x, spot_y = np.median(x[start:end][placed]), np.median(y[start:end][placed])
    window = slice(lo, start)
    active = np.nan_to_num(speed_1s[window]) >= FOUL_ACTIVE
    running = np.flatnonzero(np.nan_to_num(speed_1s[window]) >= FOUL_DECEL)
    near_spot = valid[window] & (np.hypot(x[window] - spot_x, y[window] - spot_y) < SPOT_M)
    at_spot = np.flatnonzero(near_spot & challenge[window] & active)
    if len(at_spot) and (not len(running) or running[-1] - at_spot[-1] <= SPOT_SETTLE):
        return lo + int(at_spot[-1]), "last challenge at the free-kick spot"
    if len(running):
        return lo + int(running[-1]), "players stopped running (whistle reaction)"
    near = max(lo, start - FOUL_LOOKBACK) - lo
    fought = np.flatnonzero(challenge[window][near:] & active[near:])
    if len(fought):
        return lo + near + int(fought[-1]), "last challenge before a still ball"
    far = max(0, start - LONG_TRACE)
    earlier_dead = np.flatnonzero(~alive[far:start])
    if len(earlier_dead):
        far += int(earlier_dead[-1]) + 1
    running = np.flatnonzero(np.nan_to_num(speed_1s[far:start]) >= FOUL_DECEL)
    if len(running):
        return far + int(running[-1]), "long stoppage: players last running"
    return int(start), "ball still"


def whistles(x, y, speed, valid, graded, players, dead):
    """Stoppages with the ball inside the pitch: fouls, offsides, injuries. Returns interval dicts."""
    n = len(x)
    alive = graded & ~dead
    xy = players[:, :, :2] * np.array([LENGTH, WIDTH])
    distance = np.linalg.norm(xy - np.stack([x, y], axis=1)[:, None], axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    d_home, d_away = distance[:, :24].min(axis=1), distance[:, 24:].min(axis=1)
    player_speed = np.nanmean(players[:, :, 2], axis=1)
    speed_1s, speed_2s = moving_mean(player_speed, FPS), moving_mean(player_speed, 2 * FPS)

    xs, ys = np.nan_to_num(x, nan=-99), np.nan_to_num(y, nan=-99)
    in_box = (np.minimum(xs, LENGTH - xs) < BOX_DEPTH) & (np.abs(ys - WIDTH / 2) < BOX_HALF)
    keeper_holds = in_box & (np.minimum(distance[:, HOME_GK], distance[:, AWAY_GK]) < GK_HOLD_M)
    still = (alive & valid & (np.nan_to_num(outside_distance(x, y), nan=99) < -FOUL_INSIDE_M) &
             (np.nan_to_num(speed, nan=99) < FOUL_STILL_SPEED) & ~keeper_holds)
    kicks = np.flatnonzero(kick_frames(x, y, speed, valid))
    resumed = np.flatnonzero(resumed_frames(x, y, speed, valid, players))
    challenge = valid & (d_home < CHALLENGE_M) & (d_away < CHALLENGE_M)

    found = []
    starts, lengths = closed_runs(still, FOUL_STILL_HOLE)
    for start, length in zip(starts, lengths):
        end = start + length
        if length < FOUL_STILL_MIN or np.nanmean(player_speed[start:end]) >= FOUL_PLAYERS:
            continue
        home_takes = np.median(d_home[start:end]) < np.median(d_away[start:end])
        space = np.median((d_away if home_takes else d_home)[start:end]) > FOUL_SPACE_M
        decel = np.nanmax(speed_1s[max(0, start - FOUL_DECEL_WINDOW):start + 1]) >= FOUL_DECEL
        if not (length >= FOUL_LONG or (space and decel)):
            continue
        stop, how = trace_back(start, end, x, y, valid, alive, challenge, speed_1s)
        if found and stop < found[-1]["end"]:  # never reach back across the previous stoppage
            stop, how = max(found[-1]["end"], min(stop, start)), how + " (after previous stoppage)"
            if stop >= start:
                stop = int(start)
        after = [a[(a >= end - 5) & (a < end + FOUL_RESTART_SEARCH)] for a in (kicks, resumed)]
        options = [int(a[0]) for a in after if len(a)]
        restart, restart_how = (min(options), "free kick taken") if options else (int(end), "ball moved off its spot")
        found.append({"start": int(stop), "end": restart, "reason": "foul / whistle", "half": 0,
                      "stop_how": how, "restart_how": restart_how})

    covered = dead.copy()
    for item in found:
        covered[item["start"]:item["end"]] = True
    missing = moving_mean((~valid).astype(float), 2 * FPS) > NOBALL_MISSING
    quiet = graded & ~covered & missing & (np.nan_to_num(speed_2s, nan=9) < NOBALL_PLAYERS)
    starts, lengths = eda.runs(quiet)
    for start, length in zip(starts, lengths):
        if length < NOBALL_MIN:
            continue
        moving_again = np.flatnonzero(np.nan_to_num(speed_1s[start + length:start + length + FOUL_RESTART_SEARCH])
                                      > NOBALL_RESUME)
        end = start + length + (int(moving_again[0]) if len(moving_again) else 0)
        found.append({"start": int(start), "end": int(end), "reason": "stoppage (no ball)", "half": 0,
                      "stop_how": "players slow, ball untracked", "restart_how": "players moving again"})
    return found


def tag_moment(tag):
    """The frame a set-piece tag refers to: kickoff tags end at the kick, free-kick tags sit at the
    foul, the others start at the restart."""
    return tag["end_frame"] if tag["tags"].get("Type", "").startswith("Kick Off") else tag["start_frame"]


def apply_tags(intervals, tags, x, y, speed, valid, graded, half, players):
    """Smart Tagging set pieces as input (see TAGS). Returns the updated intervals."""
    pieces = [t for t in tags if t["code"] == "SET PIECES" and t["tags"].get("Type") in TAG_KINDS]
    rows = intervals.to_dict("records")
    for row in rows:
        row.setdefault("tag_team", "")
    player_speed = np.nanmean(players[:, :, 2], axis=1)
    speed_1s = moving_mean(player_speed, FPS)
    kicks = np.flatnonzero(kick_frames(x, y, speed, valid) & graded)
    resumed = np.flatnonzero(resumed_frames(x, y, speed, valid, players) & graded)
    throws = np.flatnonzero(throw_ins(x, y, speed, valid) & graded)
    leaving = departures(x, y, speed, valid)
    exits = [f for f, _ in confirmed(find_stops(x, y, valid, graded), x, y, speed, valid)]

    def dead_before(frame):  # end of the latest dead stretch that ends at or before frame
        ends = [r["end"] for r in rows if r["end"] <= frame]
        return max(ends) if ends else 0

    for tag in sorted(pieces, key=tag_moment):
        kind, team, moment = TAG_KINDS[tag["tags"]["Type"]], tag["tags"].get("Team", ""), tag_moment(tag)
        if not graded[min(moment, len(graded) - 1)]:
            continue
        if kind == "foul / whistle":
            if any(r["start"] <= moment < r["end"] for r in rows):
                continue
            lo, hi = moment + FOUL_STILL_MIN, moment + FOUL_RESTART_SEARCH
            options = [int(a[(a >= lo) & (a < hi)][0]) for a in (kicks, resumed) if np.any((a >= lo) & (a < hi))]
            if options:
                restart, how = min(options), "free kick taken"
            else:
                moving = np.flatnonzero(np.nan_to_num(speed_1s[lo:hi]) > NOBALL_RESUME)
                restart, how = (lo + int(moving[0]), "players moving again") if len(moving) else (lo, "tag + 3 s")
            rows.append({"start": int(moment), "end": int(restart), "reason": kind, "half": int(half[moment]),
                         "stop_how": "Smart Tagging free kick", "restart_how": how, "tag_team": team})
            continue
        if kind in ("goal", "half start"):
            continue  # kickoffs are found from the players' shape; their team is recorded below
        near = [r for r in rows if abs(r["end"] - moment) <= TAG_TOLERANCE]
        if near:
            min(near, key=lambda r: abs(r["end"] - moment))["reason"] = kind
            continue
        lo, hi = moment - TAG_TOLERANCE, moment + TAG_TOLERANCE
        if kind == "throw-in":
            candidates = list(throws[(throws >= lo) & (throws <= hi)])
        else:
            candidates = [f for f, k, _ in leaving if lo <= f <= hi and k == kind]
            candidates += list(kicks[(kicks >= lo) & (kicks <= hi)])
        candidates += list(resumed[(resumed >= lo) & (resumed <= hi)])
        restart = int(min(candidates, key=lambda f: abs(f - moment))) if candidates else int(moment)
        restart_how = "tagged {} ({})".format(kind, "track" if candidates else "tag time")
        if any(r["start"] < restart < r["end"] for r in rows if r["reason"] in ("goal", "half start")):
            continue  # goals and half starts end at their kickoff, which is found from the players
        for r in rows:  # a dead stretch running on through the tagged restart ends there
            if r["start"] < restart < r["end"]:
                r["end"] = restart
        floor = max(dead_before(restart), restart - MAX_DEAD)
        before = [f for f in exits if floor <= f <= restart - MIN_DEAD]
        if before:
            stop, stop_how = before[-1], "crossed (tagged restart)"
        else:
            running = np.flatnonzero(np.nan_to_num(speed_1s[floor:max(floor, restart - MIN_DEAD)]) >= FOUL_DECEL)
            stop = floor + int(running[-1]) if len(running) else restart - MIN_DEAD
            stop_how = "players stopped running (tagged restart)"
        rows.append({"start": int(stop), "end": restart, "reason": kind, "half": int(half[restart]),
                     "stop_how": stop_how, "restart_how": restart_how, "tag_team": team})

    for tag in pieces:  # the team that takes each restart
        if TAG_KINDS[tag["tags"]["Type"]] == "foul / whistle":
            continue
        moment = tag_moment(tag)
        near = [r for r in rows if abs(r["end"] - moment) <= TAG_TOLERANCE]
        if near:
            min(near, key=lambda r: abs(r["end"] - moment))["tag_team"] = tag["tags"].get("Team", "")
    out = pd.DataFrame(rows)
    out = out[out.end > out.start]
    return out.sort_values("start").reset_index(drop=True)


def merge_model(intervals, flag, half, start_mode="rule", end_mode="rule"):
    """Combine the rule intervals with the stoppage model's dead flag (see MODEL).

    The NO BALL rule is dropped; model stretches touching no rule interval are added. For a rule
    interval that a model stretch overlaps, each edge comes from start_mode / end_mode:
    "rule" keeps the rule's edge, "model" takes the overlapping model stretch's edge, "outer"
    takes whichever is wider. The modes are chosen on the external DFL matches (dfl_check.py).
    """
    intervals = intervals[intervals.reason != "stoppage (no ball)"].reset_index(drop=True)
    change = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(np.int8), [0]])))
    model = list(zip(change[::2], change[1::2]))
    covered, _ = to_state(intervals, len(flag))
    rows = intervals.to_dict("records")
    for row in rows:
        over = [(a, b) for a, b in model if a < row["end"] and row["start"] < b]
        if not over:
            continue
        ms, me = min(a for a, _ in over), max(b for _, b in over)
        row["start"] = int({"rule": row["start"], "model": ms, "outer": min(ms, row["start"])}[start_mode])
        row["end"] = int({"rule": row["end"], "model": me, "outer": max(me, row["end"])}[end_mode])
    rows += [{"start": int(a), "end": int(b), "reason": "whistle (player model)", "half": int(half[a]),
              "stop_how": "stoppage model", "restart_how": "stoppage model"}
             for a, b in model if not covered[a:b].any()]
    out = pd.DataFrame(rows)
    return out[out.end > out.start].sort_values("start").reset_index(drop=True)


def to_state(intervals, n):
    """Per-frame dead flag and reason; on overlaps the earlier-starting interval's reason wins,
    except that goal and half-start intervals override everything inside them."""
    dead, reason = np.zeros(n, bool), np.zeros(n, np.int8)
    priority = {"goal": 2, "half start": 2}
    for row in sorted(intervals.itertuples(), key=lambda r: priority.get(r.reason, 1)):
        span = slice(row.start, row.end)
        code = REASONS.index(row.reason)
        if priority.get(row.reason, 1) == 2:
            reason[span] = code
        else:
            reason[span] = np.where(dead[span], reason[span], code)
        dead[span] = True
    return dead, reason


def report(intervals, dead, reason, graded, players, tags):
    lines = ["Dead ball v1 over {} graded frames: dead {:.1%}".format(
        graded.sum(), dead[graded].mean()), ""]
    lines.append("intervals by reason (count, median and total seconds):")
    durations = (intervals.end - intervals.start) / FPS
    table = intervals.assign(seconds=durations).groupby("reason").seconds.agg(["count", "median", "sum"])
    lines += [table.round(1).to_string(), ""]
    lines.append("how stops were found:   " + intervals.stop_how.value_counts().to_dict().__repr__())
    lines.append("how restarts were found: " + intervals.restart_how.value_counts().to_dict().__repr__())

    mean_speed = np.nanmean(players[:, :, 2], axis=1)
    lines.append("")
    lines.append("player mean speed: in play {:.2f} m/s, dead {:.2f} m/s".format(
        np.nanmean(mean_speed[graded & ~dead]), np.nanmean(mean_speed[graded & dead])))

    lines.append("")
    if USE_TAGS:
        lines.append("(MTAH_TAGS=1: the set-piece tags are an INPUT here, so this recall is not an independent check)")
    lines.append("Smart Tagging recall. A set-piece tag starts at the restart (a kickoff tag ends at it), so")
    lines.append("a tag is found when a dead stretch ENDS within {} s of that moment; 'same reason' also".format(
        TAG_TOLERANCE // FPS))
    lines.append("needs the reason to match; offset = median (restart - tag moment) in seconds. Free-kick")
    lines.append("tags sit at the foul, so they are found when a dead stretch covers them")
    kinds = {"Throw-in": "throw-in", "Corner Kick": "corner", "Goal Kick": "goal kick",
             "Free Kick": "foul / whistle", "Kick Off (Start)": "half start", "Kick Off (After Goal)": "goal"}
    ends = intervals.end.values
    for kind, expected in kinds.items():
        windows = [t for t in tags if t["code"] == "SET PIECES" and t["tags"].get("Type") == kind]
        found, same, offsets = 0, 0, []
        for t in windows:
            moment = t["end_frame"] if kind.startswith("Kick Off") else t["start_frame"]
            if kind == "Free Kick":  # these tags sit at the foul, not the restart: is it inside a dead stretch?
                inside = intervals[(intervals.start <= moment) & (intervals.end > moment)]
                found += len(inside) > 0
                same += bool((inside.reason == expected).any())
                continue
            near = intervals[np.abs(ends - moment) <= TAG_TOLERANCE]
            if len(near):
                found += 1
                best = near.iloc[np.argmin(np.abs(near.end.values - moment))]
                offsets.append((best.end - moment) / FPS)
                same += expected is None or best.reason == expected
        lines.append("  {:22s} {:2d} tagged, {:2d} found, {:2d} with the same reason, offset {}".format(
            kind, len(windows), found, same, "{:+.1f} s".format(np.median(offsets)) if offsets else "-"))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as handle:
        handle.write(text + "\n")


def main():
    players, _ = eda.load_atd()
    graded, half = eda.graded_mask()
    track = np.load(TRACK)
    x, y, speed = track["x"], track["y"], track["speed"]
    valid = ~np.isnan(x)
    tags = load_tags()

    intervals = build_intervals(x, y, speed, valid, graded, half, players, tags)
    dead, _ = to_state(intervals, len(x))
    extra = pd.DataFrame(whistles(x, y, speed, valid, graded, players, dead & graded))
    if len(extra):
        extra["half"] = half[extra.start.values]
        intervals = pd.concat([intervals, extra]).sort_values("start").reset_index(drop=True)
    if USE_MODEL:
        edges = {"start": "rule", "end": "rule"}
        if os.path.exists(EDGE_CHOICE):
            with open(EDGE_CHOICE) as handle:
                edges = json.load(handle)
        intervals = merge_model(intervals, np.load(STOP_MODEL)["dead"] & graded, half, edges["start"], edges["end"])
    if USE_TAGS:
        intervals = apply_tags(intervals, tags, x, y, speed, valid, graded, half, players)
    dead, reason = to_state(intervals, len(x))
    dead &= graded
    reason[~dead] = 0

    os.makedirs(OUT_DIR, exist_ok=True)
    intervals.to_csv(os.path.join(OUT_DIR, "intervals.csv"), index=False)
    np.savez_compressed(os.path.join(OUT_DIR, "state.npz"), dead=dead, reason=reason, reasons=np.array(REASONS))
    report(intervals, dead, reason, graded, players, tags)


if __name__ == "__main__":
    main()
