"""Demo video: the pipeline's per-frame possession labels drawn on the tactical camera.

The clip is taken from the 5 minutes that have public labels (video frames 11185-18684), so the
scores on the title card are measured on the same stretch of play.

    python media/make_video.py                     # -> media/possession_demo.mp4 (720p, under 10 MB)
    python media/make_video.py --start 0 --end 300 --pred evaluation/results/possession_final.csv

Needs the match data (MTAH_DATA_DIR, see data/README.md) and `pip install opencv-python imageio-ffmpeg`.
The labels come from the CSV only; nothing here feeds back into the pipeline.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402

WINDOW = 11185                         # first video frame of the public window
FPS = 25
W, H = 1280, 720
TOP, BOTTOM = 64, 92                   # header and timeline panel heights
INTRO_S, OUTRO_S = 4, 5
TARGET_MB = 9.5                        # GitHub's limit for a video dropped into a README is 10 MB

# BGR
COL = {"H": (176, 86, 22), "A": (45, 45, 205), "D": (120, 120, 120), "X": (60, 60, 60)}
NAME = {"H": "BARCELONA BALL", "A": "ATHLETIC BALL", "D": "DEAD BALL", "X": "NO LABEL"}
PANEL, WHITE, GREY = (24, 24, 24), (240, 240, 240), (170, 170, 170)
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, xy, scale, colour=WHITE, thick=1):
    cv2.putText(img, s, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def text_w(s, scale, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def clock(seconds):
    return "%d:%02d" % (seconds // 60, seconds % 60)


def card(lines, stats=None, note=None):
    img = np.full((H, W, 3), 16, np.uint8)
    y = 210
    for s, scale, colour in lines:
        text(img, s, ((W - text_w(s, scale, 2)) // 2, y), scale, colour, 2 if scale > 0.9 else 1)
        y += int(70 * scale) + 18
    if stats:
        bw, gap = 230, 20
        x0 = (W - len(stats) * bw - (len(stats) - 1) * gap) // 2
        for i, (value, label) in enumerate(stats):
            x = x0 + i * (bw + gap)
            cv2.rectangle(img, (x, y + 10), (x + bw, y + 120), (40, 40, 40), -1)
            text(img, value, (x + (bw - text_w(value, 1.4, 2)) // 2, y + 70), 1.4, WHITE, 2)
            text(img, label, (x + (bw - text_w(label, 0.5)) // 2, y + 102), 0.5, GREY)
        y += 120
    if note:
        text(img, note, ((W - text_w(note, 0.45)) // 2, y + 45), 0.45, GREY)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(ROOT, "evaluation", "results", "monu4959.csv"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    ap.add_argument("--video", default=games.asset_paths()["video"])
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "possession_demo.mp4"))
    ap.add_argument("--start", type=float, default=160, help="clip start, seconds into the public window")
    ap.add_argument("--end", type=float, default=300, help="clip end, seconds into the public window")
    ap.add_argument("--still", type=float, help="write one PNG at this many seconds into the clip, no video")
    args = ap.parse_args()

    # scores on the whole public window, for the title card
    gt_all = pd.read_csv(args.gt)["label"].values[WINDOW:WINDOW + 7500]
    pr_all = pd.read_csv(args.pred)["label"].values[WINDOW:WINDOW + 7500]
    acc = (gt_all == pr_all).mean()

    first = WINDOW + int(args.start * FPS)
    n = int((args.end - args.start) * FPS)
    pr = pd.read_csv(args.pred)["label"].values[first:first + n]

    tl_x, tl_w = 120, W - 120 - 24
    idx = np.linspace(0, n - 1, tl_w).astype(int)
    row = np.repeat(np.array([COL[c] for c in pr[idx]], np.uint8)[None], 22, axis=0)

    intro = card([("BARCELONA HACKATHON", 1.6, WHITE),
                  ("Who has the ball? FC Barcelona vs Athletic Club, labelled frame by frame", 0.7, GREY),
                  ("from player and ball tracking data alone -- no video model.", 0.7, GREY)],
                 stats=[("%.1f%%" % (100 * acc), "frame accuracy"),
                        ("0.97", "in-play / dead F1"),
                        ("0.867", "macro F1"),
                        ("0.500", "possession-spell F1")],
                 note="Scores on the 5 minutes with official labels. The labels in this clip are the pipeline's output.")
    outro = card([("BARCELONA HACKATHON", 1.4, WHITE),
                  ("Tracking data -> ball track -> dead-ball rules + stoppage model", 0.65, GREY),
                  ("(trained on open Bundesliga data) -> possession rules", 0.65, GREY),
                  ("github.com/monu4959/barcelona-hackathon", 0.75, WHITE)])

    tmp = os.path.join(tempfile.gettempdir(), "possession_demo_master.mp4")
    writer = None
    if args.still is None:
        writer = imageio_ffmpeg.write_frames(tmp, (W, H), fps=FPS, codec="libx264", quality=None,
                                             output_params=["-crf", "14", "-preset", "fast"])
        writer.send(None)

    def emit(img, frames=1):
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        for _ in range(frames):
            writer.send(rgb)

    if writer:
        emit(intro, INTRO_S * FPS)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    vid_h = H - TOP - BOTTOM
    vid_w = int(vid_h * 16 / 9)
    x0 = (W - vid_w) // 2
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if args.still is not None and i != int(args.still * FPS):
            continue
        img = np.full((H, W, 3), 16, np.uint8)
        img[TOP:TOP + vid_h, x0:x0 + vid_w] = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)

        # header: title, the pipeline's label, clip clock
        cv2.rectangle(img, (0, 0), (W, TOP), PANEL, -1)
        text(img, "BARCELONA HACKATHON", (18, 28), 0.62, WHITE)
        text(img, "possession from tracking data", (18, 50), 0.45, GREY)
        cw = 440
        cx0 = (W - cw) // 2
        cv2.rectangle(img, (cx0, 12), (cx0 + cw, 52), COL[pr[i]], -1)
        text(img, "PIPELINE", (cx0 + 14, 39), 0.5, (230, 230, 230))
        text(img, NAME[pr[i]], (cx0 + cw - 16 - text_w(NAME[pr[i]], 0.75, 2), 41), 0.75, WHITE, 2)
        s = "%s / %s" % (clock(i / FPS), clock(n / FPS))
        text(img, s, (W - 20 - text_w(s, 0.6), 40), 0.6, WHITE)

        # timeline of the clip
        y0 = H - BOTTOM
        cv2.rectangle(img, (0, y0), (W, H), PANEL, -1)
        text(img, "PIPELINE", (24, y0 + 28), 0.48, GREY)
        img[y0 + 12:y0 + 34, tl_x:tl_x + tl_w] = row
        cx = tl_x + int(i / (n - 1) * (tl_w - 1))
        cv2.line(img, (cx, y0 + 6), (cx, y0 + 40), WHITE, 2)
        for k in range(1, int(n / FPS // 30) + 1):
            x = tl_x + int(k * 30 * FPS / (n - 1) * (tl_w - 1))
            if x < tl_x + tl_w - 10:
                cv2.line(img, (x, y0 + 34), (x, y0 + 38), GREY, 1)
                text(img, clock(k * 30), (x - 14, y0 + 54), 0.38, GREY)
        lx = tl_x
        for code in "HAD":
            cv2.rectangle(img, (lx, y0 + 66), (lx + 14, y0 + 80), COL[code], -1)
            text(img, NAME[code].title(), (lx + 20, y0 + 79), 0.45, GREY)
            lx += 200

        if args.still is not None:
            cv2.imwrite(args.out.replace(".mp4", "_still.png"), img)
            cv2.imwrite(args.out.replace(".mp4", "_intro.png"), intro)
            return
        emit(img)
        if i % 500 == 0:
            print("frame %d / %d" % (i, n), flush=True)

    emit(outro, OUTRO_S * FPS)
    writer.close()

    # two-pass encode of the master to just under the size limit
    seconds = INTRO_S + n / FPS + OUTRO_S
    kbps = int(TARGET_MB * 8e3 / seconds * 0.97)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    log = os.path.join(tempfile.gettempdir(), "possession_demo_pass")
    common = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "slow",
              "-b:v", "%dk" % kbps, "-passlogfile", log, "-an"]
    subprocess.run(common + ["-pass", "1", "-f", "mp4", os.devnull], check=True)
    subprocess.run(common + ["-pass", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], check=True)
    os.remove(tmp)
    print("wrote %s: %.0f s, %d kb/s, %.1f MB" % (args.out, seconds, kbps, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
