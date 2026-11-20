import numpy as np
import pandas as pd
import pytest

import estimate_possession as ep

BALL = (50.0, 30.0)


def players_at(frames):
    """Player array for control_series: one list of (slot, metres from the ball) per frame."""
    out = np.full((len(frames), 46, 2), np.nan)
    for i, placed in enumerate(frames):
        for slot, metres in placed:
            out[i, slot] = [(BALL[0] + metres) / ep.LENGTH, BALL[1] / ep.WIDTH]
    return out


def test_control_goes_to_the_nearest_team():
    far = ep.CONTROL_M + 2
    home, away = 0, 30
    frames = [
        [(home, 1.0), (away, far)],   # home close
        [(home, far), (away, 1.0)],   # away close
        [(home, 1.0), (away, 1.2)],   # both close, within the margin: contested
        [(home, 1.0), (away, 2.0)],   # both close, home clearly closer
        [(home, far), (away, far)],   # nobody close
    ]
    n = len(frames)
    series, d_home, d_away = ep.control_series(players_at(frames), np.full(n, BALL[0]), np.full(n, BALL[1]),
                                               np.zeros(n), np.ones(n, bool))
    assert list(series) == [ep.HOME, ep.AWAY, ep.CONTESTED, ep.HOME, ep.NONE]
    assert d_home[0] == pytest.approx(1.0)
    assert d_away[1] == pytest.approx(1.0)


def test_no_control_of_a_fast_ball_or_a_dead_ball():
    frames = [[(0, 1.0)]] * 3
    speed = np.array([0.0, ep.CONTROL_SPEED + 5, 0.0])
    alive = np.array([True, True, False])
    series, _, _ = ep.control_series(players_at(frames), np.full(3, BALL[0]), np.full(3, BALL[1]), speed, alive)
    assert list(series) == [ep.HOME, ep.NONE, ep.NONE]


def test_no_control_without_a_ball():
    series, _, _ = ep.control_series(players_at([[(0, 1.0)]]), np.array([np.nan]), np.array([np.nan]),
                                     np.zeros(1), np.ones(1, bool))
    assert series[0] == ep.NONE


def hold():
    return int(ep.HOLD_S * ep.FPS)


def test_a_long_run_of_control_is_confirmed():
    series = np.zeros(200, np.int8)
    series[50:100] = ep.HOME
    assert ep.confirmed_runs(series, np.ones(200, bool), hold()) == [(50, ep.HOME)]


def test_a_short_touch_is_not_a_run():
    series = np.zeros(200, np.int8)
    series[50:54] = ep.AWAY
    assert ep.confirmed_runs(series, np.ones(200, bool), hold()) == []


def test_runs_stop_at_the_opponent_and_at_dead_frames():
    series = np.zeros(300, np.int8)
    series[0:40] = ep.HOME
    series[40:80] = ep.AWAY
    series[150:180] = ep.HOME
    series[190:230] = ep.HOME
    alive = np.ones(300, bool)
    alive[184:186] = False
    runs = ep.confirmed_runs(series, alive, hold())
    assert runs == [(0, ep.HOME), (40, ep.AWAY), (150, ep.HOME), (190, ep.HOME)]


def label(series, dead):
    n = len(series)
    labels, _ = ep.possession(np.ones(n, bool), dead, series, np.zeros(n, np.int8),
                              pd.DataFrame({"end": pd.Series([], dtype=int)}),
                              np.full((n, 46, 2), np.nan), (np.full(n, np.nan), np.full(n, np.nan)), hold())
    return "".join(labels)


def test_possession_labels_end_to_end():
    series = np.zeros(300, np.int8)
    series[0:80] = ep.HOME        # home keeps it through the loose ball that follows
    series[100:160] = ep.AWAY
    series[250:300] = ep.AWAY
    dead = np.zeros(300, bool)
    dead[200:240] = True
    assert label(series, dead) == "H" * 100 + "A" * 100 + "D" * 40 + "A" * 60


def test_a_brief_opponent_touch_does_not_change_possession():
    series = np.zeros(200, np.int8)
    series[0:200] = ep.HOME
    series[100:104] = ep.AWAY
    assert label(series, np.zeros(200, bool)) == "H" * 200


def test_ungraded_frames_are_x():
    n = 50
    graded = np.ones(n, bool)
    graded[:10] = False
    series = np.full(n, ep.HOME, np.int8)
    labels, _ = ep.possession(graded, np.zeros(n, bool), series, np.zeros(n, np.int8),
                              pd.DataFrame({"end": pd.Series([], dtype=int)}),
                              np.full((n, 46, 2), np.nan), (np.full(n, np.nan), np.full(n, np.nan)), hold())
    assert "".join(labels) == "X" * 10 + "H" * 40


def test_spells_are_maximal_team_runs():
    labels = np.array(list("HHHDDAAAHH"))
    assert ep.spells(labels, np.ones(10, bool)) == [(0, 3, "H"), (5, 8, "A"), (8, 10, "H")]
