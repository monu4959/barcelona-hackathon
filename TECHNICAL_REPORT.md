# Per-frame possession from automatic tracking — technical report

*More than a Hack 2027, Phase 1 (BAR–ATH). Team: **monu4959**.*

## 1. Summary

The submission labels every frame H / A / D / X with a pipeline of explicit football rules on
the provided automatic tracking (ATD) and Smart Tagging. On top of the rules sits **one learned
component, a stoppage model that judges from the players alone whether the ball is dead**. It is
trained, thresholded and combined with the rules **only on external data**: the seven open DFL
Bundesliga matches of the IDSSE dataset (CC BY 4.0). The possession settings (how long and how
close a team must have the ball) are also chosen on those matches, by running the whole pipeline
on them and scoring it with the competition's own metric.

No video frame and no label of the provided game is used for training, fitting, thresholding or
choosing anything. The 5-minute public ground truth was used only as a final check of the frozen
file (section 7).

| Measure | Value |
|---|---|
| External check (7 DFL matches, ball hidden as in our data), spell F1 / macro F1 | rules only 0.258 / 0.657 → final pipeline **0.387 / 0.800** |
| Stoppage model, leave-one-match-out on DFL | frame AUC 0.983, dead-stretch F1 0.837 |
| Public 5 minutes (one check, after freezing), spell F1 / macro F1 | **0.500 / 0.867** (best baseline 0.424 / 0.581) |

## 2. Approach and why

The task is a segmentation: possessions are maximal runs of one team with the ball alive, split
by stoppages, and the ranking metric (spell F1) matches them by temporal IoU. A missed stoppage
merges two possessions into one and costs two misses plus a false spell. So **getting dead balls
right matters as much as getting the team right**, and our early versions were weakest there.

The provided ATD has no ball on 41% of graded frames, in long blocks (median 10.7 s), mostly
around restarts and stoppages. We therefore built, in order:

1. **Ball track** (`ball_recovery/link_ball.py`): ATD ball positions linked with a Viterbi pass,
   holes of ≤ 1 s filled linearly, speed from positions. Tracking data only.
2. **Dead ball by rules** (`dead_ball/detect_dead.py`): out over a line → restart matched to the
   line type (throw-in near the touchline, a placed ball at a corner arc or in the goal area);
   goals and half starts end at a kickoff found from the players' shape; whistles inside the
   pitch from a still ball, slow players and a free-kick shape, with the stop traced back to the
   last challenge. Every rule is written from how football works (section 5).
3. **Smart Tagging as input**: set-piece tags (type, team, time) add restarts the track did not
   see; free-kick tags mark fouls; the tag's team takes the restart; when the ball is untracked
   for ≥ 1 s, the team the phase tags imply takes possession.
4. **Stoppage model** (`dead_ball/stoppage_model.py`): when the referee whistles, all 22
   players react together (slow down, loosen their shape, a taker walks to the ball), which needs
   no ball at all. A gradient-boosted classifier on player-only features learns this on DFL,
   where every frame carries the official ball-in-play flag.
5. **Possession** (`possession/estimate_possession.py`): a team "controls" the ball when its
   nearest player is within CONTROL_M and the ball is slower than a pass going by; possession
   switches only when the other team holds control for HOLD_S (section 5).
6. **External calibration** (`calibration/dfl_check.py`): the functions above, unchanged, run on
   the DFL matches converted to our formats, with the DFL ball **hidden on exactly the frames
   where our tracking data has no ball**. They are scored with `evaluation/possession_metrics.py`
   against DFL's official ball status and possession flags. The competition's ground truth is
   derived the same way ("from corrected tracking ... the ball dead, and otherwise the team the
   tracking names as owning it", [CHALLENGE.md](docs/challenge/CHALLENGE.md)), so DFL's flags are the closest external proxy.

Why rules plus one small model rather than end-to-end learning: there is no labelled data on
this game, and external data with frame-level possession labels is scarce (seven matches). Rules
are transparent, and the external check lets us measure them where answers exist.

## 3. Assets used — and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (players + ball) | **yes** | all player positions; the ATD ball is the only ball source |
| Smart Tagging | **yes** | GOALS start; SET PIECES type / team / time; phase codes → implied team where the ball is untracked |
| Homography | marginal | only to store pixel positions of the track; no effect on the labels |
| Tactical video | **no** (final) | see the dropped detectors below; watched to understand possession, never used as labels or for checks |
| Broadcast video / ATD | no | not synchronised to the tactical timeline |
| Audio of the tactical video | dropped | a simple whistle detector was too weak (crowd whistling) |
| DFL / IDSSE open data | **yes** | training and thresholding the stoppage model; choosing edges and possession settings |
| Public 5-minute ground truth | check only | one score of the frozen file (section 7) |

**Dropped: a ball detector trained on this match's video (v1).** YOLOv8s-P2 trained on tiles of
the tactical video with pseudo-labels from the ATD ball. It worked well, but the organisers'
clarification forbids training on the provided video, so it was removed completely. Nothing from
it is in this repository or in the submission.

**Dropped: a ball detector trained on external data only (v3).** The same architecture,
COCO-pretrained, trained on the Roboflow Universe dataset *football-players-detection* v20 (CC BY
4.0, 372 broadcast images labelled by its Roboflow publisher, ball class only), threshold chosen
on its own validation split (external test F1 0.83). On our tactical camera it transferred
poorly: its top detection was within 3 m of the ATD ball on only 32% of frames where both exist.
Dead-ball detection improved with it, but possession got worse, so it was dropped. It is not used
by the submission and its code is not included.

## 4. Training data

| Dataset | Source | Labelled by | Amount | Used for |
|---|---|---|---|---|
| IDSSE — integrated spatiotemporal and event data in elite soccer | Bassek, Rein, Weber, Memmert, *Scientific Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y; Figshare doi:10.6084/m9.figshare.28196177 (mirror: Hugging Face `pysport/idsse-data`, revision `a715a38`) | official DFL data (Sportec Solutions, TRACAB optical tracking); per-frame `BallStatus` and `BallPossession` flags | 7 Bundesliga 2022/23 matches, 14 halves, 1,002,644 frames at 25 Hz | stoppage model; all external calibration |

Licence CC BY 4.0; data owner Deutsche Fußball Liga (DFL). No data of any kind from the provided
game is used for training. The dropped v3 detector used the Roboflow dataset above and COCO
weights. Neither is part of the submission.

## 5. Possession rules

**Possession (in play).**
- *Control:* a player within **CONTROL_M = 3.0 m** of the ball while the ball is slower than
  8 m/s; a ball flying past is nobody's. Both teams within that distance and within 0.5 m of each
  other: contested.
- *A team run:* from a team's first control to its last, while the opponent never controls; flights
  between teammates of up to **PASS_GAP = 3 s** keep it going (passes).
- *Switch:* possession changes only when the other team has a run of at least **HOLD_S = 0.5 s**
  with at least 8 frames of control. A touch without control, a deflection or a lost aerial duel is
  therefore not a change. That follows CHALLENGE.md: "a change of possession occurs when a player
  from the opposing team gains controlled possession".
- *Loose and contested balls* (and frames with no ball): possession stays with the current team. If
  the ball is untracked for ≥ 1 s and the Smart Tagging phases imply exactly one team, that team
  has it.
- *Restarts:* the team the set-piece tag names, otherwise the team of the player nearest the ball
  as play restarts.

HOLD_S, CONTROL_M, PASS_GAP and the "switch back to the first touch" window (BACKTRACK, chosen 0)
were searched on DFL around our hand-set values (1.0 s, 1.5 m, 3 s, 6 s). On DFL, spell F1 went
from 0.353 to 0.387 (`calibration/possession.json`).

**Dead ball.** Out of play when the ball is > 1 m beyond a line for 0.6 s (or vanishes heading out
at a line), unless it is soon seen moving inside the pitch (it was in the air). The restart
follows the line: a throw-in is the ball coming back in near that touchline; a corner or goal
kick is a placed, still ball leaving its spot at that end; after a goal, the kickoff from the
players' shape. Fouls and whistles are a still ball inside the pitch with slow players, either
for 5 s or with a free-kick shape. Their start is traced back to the last challenge at the spot,
or to the moment the players stopped running.

The stoppage model then **replaces** the rule for a stoppage with no ball (the same idea with
hand-set thresholds), **adds** every stretch it calls dead that no rule touched, and **widens**
rule stoppages it overlaps to the wider of the two edges. That last choice won on DFL among 9
edge combinations, though all were within noise of each other. Model settings: a threshold of 0.6
on the 1 s smoothed probability, stretches of ≥ 3 s, both chosen by leave-one-match-out on DFL.

**Where the constants come from.** The rule constants (distances, speeds, durations) were written
from football and physics: ball size, goal-area and box dimensions, running and walking speeds.
Some were sanity-checked against the distributions of this match's own tracking data (for example,
mean player speed in play versus at stoppages). None was fitted to labels of this game, and the
rules as a whole are measured on DFL (section 6). The model, its threshold, the edge rule and the
four possession settings are all chosen on DFL alone.

## 6. External validation (DFL)

Pipeline run on the 7 DFL matches, ball hidden on our gap pattern. Each match's stoppage flags
come from the model that never saw that match.

| Variant (on DFL) | Spell F1 | Macro F1 | Dead-stretch F1 |
|---|---|---|---|
| Rules only (no-ball rule, hand-set settings) | 0.258 | 0.657 | 0.452 |
| + stoppage model, rule edges | 0.337 | 0.742 | 0.725 |
| + stoppage model, wider edges (chosen) | 0.353 | 0.797 | 0.792 |
| + possession settings chosen on DFL (**final**) | **0.387** | **0.800** | 0.792 |

With the hand-set possession settings we predicted 1,258 possessions on DFL against 3,061 true
ones: possession changes were under-detected. The shorter hold time chosen on DFL works against
that, but under-segmentation is still the main weakness (section 8).

## 7. Result on the provided game

Frozen file (`evaluation/results/monu4959.csv`): the ball is dead on 46.5% of graded frames,
374 possessions, Barcelona 58.9% of in-play frames.

**The public 5 minutes.** The file was scored once, after the method and the submission were
fixed, to confirm the pipeline is wired correctly. No change was made afterwards.

| Public 5 minutes | Spell F1 | Macro F1 | Possessions (truth: 17) |
|---|---|---|---|
| this submission | 0.500 | 0.867 | 19 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 | 0.326 | 0.621 | 26 |

The sample is 17 possessions, and the metric's own noise band on spell F1 is about ±0.04.
Disclosure: one earlier pipeline (the dropped v1) was also scored once on this window, before the
organisers' clarification. It scored 0.497 / 0.731. Nothing was tuned on either score.

## 8. Limitations and what we would do next

- **When possession changes.** Only 6 of the 20 public-window changes land within 1 s, and on DFL
  possession changes are under-detected. Where the ATD has no ball (41% of frames),
  possession comes from Smart Tagging phases, which are coarse. A model of the ball's position
  from the players, trained on DFL, is the next step.
- **Domain shift.** DFL is TRACAB data from the Bundesliga; ours is automatic tracking from La Liga.
  The external check hides the ball at random moments, whereas ours goes missing mostly at
  stoppages, so the check is somewhat harsh on live play and lenient on dead balls. The model may
  call too much dead time: 46.5% here, against about 45% on average in DFL.
- **Few external matches.** Seven matches; two chosen settings sit at the edge of their search
  range, where the score was flat.
- **A tactical-camera ball detector** trained on external fixed-camera data would recover the
  ball in the ATD gaps. Our external detector failed mainly on domain (broadcast versus tactical).
- **Whistle audio** with a pretrained audio tagger, or broadcast audio, to time fouls.

## 9. Reproduction

```bash
python -m venv venv && venv/Scripts/pip install -r requirements-pipeline.txt   # Python 3.12
export MTAH_DATA_DIR=/path/to/match/data         # the unpacked bundle (default ./data)
python reproduce.py --team monu4959             # → evaluation/results/monu4959.csv
```

`reproduce.py` downloads the DFL files (2.5 GB, pinned revision, sizes checked) and runs the
eight steps listed in it, ending with the organisers' validator. CPU only; about 10 minutes on a
laptop. Run from a fresh clone, a fresh virtual environment and an empty work directory, it
reproduces the submitted file **byte for byte** (checked before submission). See SOLUTION.md for
the individual steps.
