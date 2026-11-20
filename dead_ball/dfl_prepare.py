"""External data for the stoppage model: the DFL / IDSSE open matches, turned into compact arrays.

    python -I dead_ball/dfl_prepare.py

Source: "An integrated dataset of spatiotemporal and event data in elite soccer" (Bassek, Rein,
Weber, Memmert; Scientific Data 12, 195, 2025; doi 10.1038/s41597-025-04505-y). Seven Bundesliga
2022/23 matches, official DFL data (TRACAB optical tracking, 25 Hz), CC BY 4.0, data owner the
Deutsche Fussball Liga (DFL). Downloaded by fetch_dfl.py (Hugging Face mirror of the Figshare
record, pinned revision). Nothing from the provided game is involved.

What is kept, per match and half: every player's position (metres, centre origin, as DFL gives it)
on every frame he is on the pitch, and the ball's BallStatus (1 = in play, 0 = dead) and
BallPossession (1 = home, 2 = away) flags -- the labels the stoppage model learns from. Referees are
dropped (our tracking data has no referee).

Output: MTAH_WORK_DIR/external/dfl_idsse/npz/<match>.npz with, per half h in {1, 2}:
    xy<h>[n, P, 2] (NaN when a player is not on the pitch), team<h>[P] (0 home, 1 away),
    keeper<h>[P] (listed as goalkeeper, "TW"), ball<h>[n, 2] (metres, centre origin),
    status<h>[n], possession<h>[n], frame<h>[n] (DFL frame numbers)
"""

import argparse
import glob
import os
import re

import numpy as np

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "work")
SRC = os.environ.get("MTAH_DFL_RAW") or os.path.join(WORK_DIR, "external", "dfl_idsse")   # fetch_dfl.py
OUT = os.path.join(WORK_DIR, "external", "dfl_idsse", "npz")
HALVES = {"firstHalf": 1, "secondHalf": 2}

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
BALL_FLAGS = re.compile(r'BallPossession="(\d)"[^>]*BallStatus="(\d)"|BallStatus="(\d)"[^>]*BallPossession="(\d)"')


def match_info(match_id):
    """(home team id, set of goalkeeper person ids)"""
    path = glob.glob(os.path.join(SRC, "DFL_02_01_matchinformation_*{}.xml".format(match_id)))[0]
    text = open(path, encoding="utf-8").read()
    keepers = set(re.findall(r'<Player PersonId="([^"]+)"[^>]*PlayingPosition="TW"', text))
    return re.search(r'HomeTeamId="([^"]+)"', text).group(1), keepers


def parse(path, home):
    """{half: {"players": {person: (team, frames, xy)}, "ball": (frames, status, possession)}}"""
    out = {h: {"players": {}, "ball": None} for h in HALVES.values()}
    current, frames, values = None, [], []

    def flush():
        if current is None or not frames:
            return
        half, team, person = current
        n = np.array(frames, np.int64)
        if team == "BALL":
            v = np.array(values, np.float32)
            out[half]["ball"] = (n, v[:, 0].astype(np.int8), v[:, 1].astype(np.int8), v[:, 2:4])
        elif team != "referee":
            out[half]["players"][person] = (0 if team == home else 1, n, np.array(values, np.float32))

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "<FrameSet" in line:
                flush()
                match = FRAMESET.search(line)
                current = (HALVES[match.group(1)], match.group(2), match.group(3))
                frames, values = [], []
            elif "<Frame " in line and current is not None:
                match = FRAME.search(line)
                if not match:
                    continue
                frames.append(int(match.group(1)))
                if current[1] == "BALL":
                    flags = BALL_FLAGS.search(line)
                    possession, status = (flags.group(1), flags.group(2)) if flags.group(1) else (flags.group(4), flags.group(3))
                    values.append((int(status), int(possession), float(match.group(2)), float(match.group(3))))
                else:
                    values.append((float(match.group(2)), float(match.group(3))))
    flush()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", help="only this match id (e.g. DFL-MAT-J03WMX)")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(SRC, "DFL_04_03_positions_raw_observed_*.xml"))):
        match_id = path.rsplit("_", 1)[-1][:-4]
        if args.match and match_id != args.match:
            continue
        target = os.path.join(OUT, match_id + ".npz")
        if os.path.exists(target):
            print(match_id, "already done")
            continue
        home, keepers = match_info(match_id)
        parsed = parse(path, home)
        arrays = {}
        for half, data in parsed.items():
            n_frames, status, possession, ball_xy = data["ball"]
            index = {f: i for i, f in enumerate(n_frames)}
            people = sorted(data["players"])
            xy = np.full((len(n_frames), len(people), 2), np.nan, np.float32)
            team = np.zeros(len(people), np.int8)
            keeper = np.array([person in keepers for person in people])
            for p, person in enumerate(people):
                team[p], frames, values = data["players"][person]
                rows = np.array([index.get(f, -1) for f in frames])
                keep = rows >= 0
                xy[rows[keep], p] = values[keep]
            arrays.update({"xy%d" % half: xy, "team%d" % half: team, "keeper%d" % half: keeper,
                           "ball%d" % half: ball_xy, "status%d" % half: status,
                           "possession%d" % half: possession, "frame%d" % half: n_frames})
            print("{} half {}: {} frames, {} players, ball dead {:.1%}, home possession {:.1%}".format(
                match_id, half, len(n_frames), len(people), np.mean(status == 0), np.mean(possession[status == 1] == 1)))
        np.savez_compressed(target, **arrays)


if __name__ == "__main__":
    main()
