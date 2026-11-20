import numpy as np
import pytest

import link_ball as lb
import stoppage_model as sm


def test_speed_from_positions():
    x = np.arange(50) * 0.4                 # 0.4 m per frame = 10 m/s at 25 fps
    speed = lb.speed_of(x, np.zeros(50))
    assert np.isnan(speed[0]) and np.isnan(speed[-1])
    assert speed[10:40] == pytest.approx(10.0)


def test_short_holes_are_filled_in_a_straight_line():
    x = np.array([0.0, 1.0, np.nan, np.nan, np.nan, 5.0, 6.0])
    y = np.zeros(7)
    px, py = x.copy(), y.copy()
    source = np.ones(7, np.int8)
    lb.fill_holes(x, y, px, py, source)
    assert x == pytest.approx([0, 1, 2, 3, 4, 5, 6])
    assert list(source) == [1, 1, 4, 4, 4, 1, 1]


def test_long_holes_and_jumps_are_left_alone():
    long_hole = np.concatenate([[0.0], np.full(lb.FILL_MAX + 1, np.nan), [1.0]])
    x = long_hole.copy()
    lb.fill_holes(x, np.zeros_like(x), x.copy(), np.zeros_like(x), np.ones(len(x), np.int8))
    assert np.isnan(x[1:-1]).all()

    jump = np.array([0.0, np.nan, 40.0])     # 40 m in two frames is not the same ball
    lb.fill_holes(jump, np.zeros(3), jump.copy(), np.zeros(3), np.ones(3, np.int8))
    assert np.isnan(jump[1])


def test_stretches():
    assert sm.stretches(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_stretch_f1():
    true = np.zeros(100, bool)
    true[10:30] = True
    true[60:80] = True
    assert sm.stretch_f1(true, true) == 1.0
    assert sm.stretch_f1(true, np.zeros(100, bool)) == 0.0
    half = np.zeros(100, bool)
    half[10:30] = True                      # one of the two stoppages found
    assert sm.stretch_f1(true, half) == pytest.approx(2 / 3)


def test_postprocess_keeps_long_stoppages_and_drops_blips():
    prob = np.zeros(1000)
    prob[300:600] = 1.0
    prob[800:805] = 1.0
    _, flag = sm.postprocess(prob, threshold=0.5, min_frames=3 * sm.FPS)
    assert flag[320:580].all()
    assert not flag[700:].any()
    assert not flag[:250].any()
