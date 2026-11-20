import numpy as np
import pandas as pd
import pytest

import detect_dead as dd


def test_outside_distance():
    assert dd.outside_distance(50.0, 30.0) < 0
    assert dd.outside_distance(50.0, dd.WIDTH + 2.0) == pytest.approx(2.0)
    assert dd.outside_distance(-3.0, 30.0) == pytest.approx(3.0)


def test_line_of():
    assert dd.line_of(-1.0, dd.WIDTH / 2) == "goal mouth"
    assert dd.line_of(-1.0, 5.0) == "goal line"
    assert dd.line_of(dd.LENGTH + 1.0, dd.WIDTH - 5.0) == "goal line"
    assert dd.line_of(40.0, dd.WIDTH + 1.0) == "touchline"


def test_closed_runs_fills_short_holes_only():
    mask = np.zeros(20, bool)
    mask[2:6] = True
    mask[8:12] = True                       # 2-frame hole between the runs
    starts, lengths = dd.closed_runs(mask, hole=3)
    assert list(zip(starts, lengths)) == [(2, 10)]
    starts, lengths = dd.closed_runs(mask, hole=1)
    assert list(zip(starts, lengths)) == [(2, 4), (8, 4)]


def test_ball_over_the_touchline_is_a_stop():
    n = 200
    x = np.full(n, 50.0)
    y = np.concatenate([np.linspace(30.0, dd.WIDTH + 3.0, 100), np.full(100, dd.WIDTH + 3.0)])
    stops = dd.find_stops(x, y, np.ones(n, bool), np.ones(n, bool))
    assert len(stops) == 1
    frame, how = stops[0]
    assert how == "crossed"
    assert y[frame] > dd.WIDTH + dd.MARGIN_M


def test_ball_that_stays_in_play_is_not_a_stop():
    n = 200
    x = np.linspace(10.0, 90.0, n)
    y = np.full(n, 30.0)
    assert dd.find_stops(x, y, np.ones(n, bool), np.ones(n, bool)) == []


def test_ball_vanishing_while_heading_out_is_a_stop():
    n = 200
    x = np.full(n, 50.0)
    y = np.concatenate([np.linspace(dd.WIDTH - 6.0, dd.WIDTH - 0.5, 50), np.full(150, np.nan)])
    valid = ~np.isnan(y)
    stops = dd.find_stops(x, y, valid, np.ones(n, bool))
    assert stops == [(50, "vanished")]


def intervals(rows):
    return pd.DataFrame(rows, columns=["start", "end", "reason", "half", "stop_how", "restart_how"])


def test_to_state_goal_overrides_other_reasons():
    table = intervals([(10, 30, "throw-in", 1, "", ""), (20, 40, "goal", 1, "", "")])
    dead, reason = dd.to_state(table, 50)
    assert dead[10:40].all() and not dead[:10].any() and not dead[40:].any()
    assert reason[15] == dd.REASONS.index("throw-in")
    assert reason[25] == dd.REASONS.index("goal")


def test_merge_model_widens_adds_and_drops():
    n = 500
    flag = np.zeros(n, bool)
    flag[90:160] = True                     # overlaps the throw-in
    flag[300:350] = True                    # touches no rule interval
    table = intervals([(100, 150, "throw-in", 1, "crossed", "throw-in"),
                       (400, 450, "stoppage (no ball)", 1, "", "")])
    merged = dd.merge_model(table, flag, np.ones(n, int), start_mode="outer", end_mode="outer")
    rows = list(merged[["start", "end", "reason"]].itertuples(index=False, name=None))
    assert rows == [(90, 160, "throw-in"), (300, 350, "whistle (player model)")]
