"""Demo video: the pipeline's per-frame possession labels next to the official ground truth, drawn on
the tactical camera, for the 5 minutes that have public labels (video frames 11185-18684).

    python media/make_video.py                         # -> media/possession_demo.mp4 (720p, H.264)
    python media/make_video.py --pred evaluation/results/possession_final.csv --out other.mp4

Needs the match data (MTAH_DATA_DIR, see data/README.md) and `pip install opencv-python imageio-ffmpeg`.
The labels come from the CSVs only; nothing here feeds back into the pipeline.
"""

import argparse
import os
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402

FIRST, LAST = 11185, 18684             # the public ground-truth window (5 min at 25 fps)
FPS = 25
W, H = 1280, 720
TOP, BOTTOM = 64, 118                  # header and timeline panel heights
INTRO_S, OUTRO_S = 5, 6

# BGR
COL = {"H": (176, 86, 22), "A": (45, 45, 205), "D": (120, 120, 120), "X": (60, 60, 60)}
NAME = {"H": "BARCELONA BALL", "A": "ATHLETIC BALL", "D": "DEAD BALL", "X": "NO LABEL"}
GOOD, BAD = (90, 200, 90), (70, 70, 235)
PANEL, WHITE, GREY = (24, 24, 24), (240, 240, 240), (170, 170, 170)
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, xy, scale, colour=WHITE, thick=1, font=FONT):
    cv2.putText(img, s, xy, font, scale, colour, thick, cv2.LINE_AA)


def text_w(s, scale, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def chip(img, x, y, label, code, w=250):
    cv2.rectangle(img, (x, y), (x + w, y + 36), COL[code], -1)
    text(img, label, (x + 10, y + 24), 0.5, (230, 230, 230))
    s = NAME[code]
    text(img, s, (x + w - 12 - text_w(s, 0.62, 1), y + 25), 0.62, WHITE, 1)


def strip(labels, width):
    """One row of the timeline: label colours resampled to `width` pixels."""
    idx = np.linspace(0, len(labels) - 1, width).astype(int)
    row = np.array([COL[c] for c in labels[idx]], np.uint8)
    return np.repeat(row[None], 16, axis=0)


def card(lines, stats=None):
    img = np.full((H, W, 3), 16, np.uint8)
    y = 200
    for s, scale, colour in lines:
        text(img, s, ((W - text_w(s, scale, 2)) // 2, y), scale, colour, 2 if scale > 0.9 else 1)
        y += int(70 * scale) + 18
    if stats:
        n = len(stats)
        bw = 230
        x0 = (W - n * bw - (n - 1) * 20) // 2
        for i, (value, label) in enumerate(stats):
            x = x0 + i * (bw + 20)
            cv2.rectangle(img, (x, y + 10), (x + bw, y + 120), (40, 40, 40), -1)
            text(img, value, (x + (bw - text_w(value, 1.4, 2)) // 2, y + 70), 1.4, WHITE, 2)
            text(img, label, (x + (bw - text_w(label, 0.5)) // 2, y + 102), 0.5, GREY)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(ROOT, "evaluation", "results", "monu4959.csv"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    ap.add_argument("--video", default=games.asset_paths()["video"])
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "possession_demo.mp4"))
    ap.add_argument("--still", type=int, help="write one PNG at this frame offset instead of the video")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)["label"].values[FIRST:LAST + 1]
    pr = pd.read_csv(args.pred)["label"].values[FIRST:LAST + 1]
    n = len(gt)
    correct = np.cumsum(gt == pr)
    acc = correct[-1] / n

    tl_x, tl_w = 120, W - 120 - 24
    gt_row, pr_row = strip(gt, tl_w), strip(pr, tl_w)

    writer = imageio_ffmpeg.write_frames(args.out, (W, H), fps=FPS, codec="libx264", quality=None,
                                         output_params=["-crf", "26", "-preset", "slow",
                                                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer.send(None)

    def emit(img, frames=1):
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        for _ in range(frames):
            writer.send(rgb)

    intro = card([("BARCELONA HACKATHON", 1.6, WHITE),
                  ("Who has the ball? Every frame of FC Barcelona vs Athletic Club,", 0.7, GREY),
                  ("labelled from player and ball tracking data alone -- no video model.", 0.7, GREY),
                  ("Below: 5 minutes with official labels. Top chip = my pipeline, second = truth.", 0.6, GREY)],
                 stats=[("%.1f%%" % (100 * acc), "frame accuracy"),
                        ("0.97", "in-play / dead F1"),
                        ("0.867", "macro F1"),
                        ("0.500", "possession-spell F1")])
    emit(intro, INTRO_S * FPS)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FIRST)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        img = np.full((H, W, 3), 16, np.uint8)
        vid_h = H - TOP - BOTTOM
        vid_w = int(vid_h * 16 / 9)
        x0 = (W - vid_w) // 2
        img[TOP:TOP + vid_h, x0:x0 + vid_w] = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)

        # header
        cv2.rectangle(img, (0, 0), (W, TOP), PANEL, -1)
        text(img, "BARCELONA HACKATHON", (18, 26), 0.62, WHITE, 1)
        text(img, "possession from tracking data", (18, 48), 0.45, GREY)
        chip(img, 300, 14, "PIPELINE", pr[i], 290)
        chip(img, 610, 14, "TRUTH", gt[i], 290)
        match = gt[i] == pr[i]
        cv2.circle(img, (936, 32), 13, GOOD if match else BAD, -1)
        text(img, "agree" if match else "differ", (956, 38), 0.55, GOOD if match else BAD)
        t = i / FPS
        text(img, "%d:%04.1f" % (t // 60, t % 60), (1050, 30), 0.6, WHITE)
        text(img, "accuracy so far %.1f%%" % (100 * correct[i] / (i + 1)), (1050, 52), 0.42, GREY)

        # timeline
        y0 = H - BOTTOM
        cv2.rectangle(img, (0, y0), (W, H), PANEL, -1)
        text(img, "TRUTH", (24, y0 + 30), 0.48, GREY)
        text(img, "PIPELINE", (24, y0 + 58), 0.48, GREY)
        img[y0 + 17:y0 + 33, tl_x:tl_x + tl_w] = gt_row
        img[y0 + 45:y0 + 61, tl_x:tl_x + tl_w] = pr_row
        cx = tl_x + int(i / (n - 1) * (tl_w - 1))
        cv2.line(img, (cx, y0 + 10), (cx, y0 + 68), WHITE, 2)
        lx = tl_x
        for code in "HAD":
            cv2.rectangle(img, (lx, y0 + 92), (lx + 14, y0 + 106), COL[code], -1)
            text(img, NAME[code].title(), (lx + 20, y0 + 105), 0.45, GREY)
            lx += 200
        for m in range(6):
            x = tl_x + int(m * 60 * FPS / (n - 1) * (tl_w - 1))
            if m:
                cv2.line(img, (x, y0 + 62), (x, y0 + 66), GREY, 1)
                text(img, "%d:00" % m, (min(x - 14, W - 44), y0 + 80), 0.38, GREY)
        if args.still is not None:
            if i == args.still:
                cv2.imwrite(args.out.replace(".mp4", "_still.png"), img)
                cv2.imwrite(args.out.replace(".mp4", "_intro.png"), intro)
                return
            continue
        emit(img)
        if i % 500 == 0:
            print("frame %d / %d" % (i, n), flush=True)

    outro = card([("BARCELONA HACKATHON", 1.4, WHITE),
                  ("Tracking data -> ball reconstruction -> dead-ball detection", 0.65, GREY),
                  ("-> stoppage model trained on open Bundesliga data -> possession rules", 0.65, GREY),
                  ("github.com/monu4959/barcelona-hackathon", 0.75, WHITE)])
    emit(outro, OUTRO_S * FPS)
    writer.close()
    print("wrote", args.out, "%.1f MB" % (os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
