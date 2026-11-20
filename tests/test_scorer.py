"""The competition scorer on the public 5-minute labels and the shipped baselines (no private data)."""

import glob
import os

import numpy as np
import pandas as pd
import pytest

import possession_metrics as pm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BASELINES = sorted(glob.glob(os.path.join(ROOT, "evaluation", "results", "baseline_*.csv")))


@pytest.fixture(scope="module")
def gt():
    return pm.label_codes(pm.read_label_csv(GT))


def test_perfect_prediction_scores_one(gt):
    result = pm.evaluate(gt, gt)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_always_dead_scores_zero_on_alive_dead(gt):
    result = pm.evaluate(gt, np.full(len(gt), pm.D, np.int8))
    assert result["alive_dead_f1"] == 0.0
    assert result["spell_f1_mean"] == 0.0


def test_best_baseline_matches_the_published_scoreboard(gt):
    pred = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "results", "baseline_smooth125.csv")))
    result = pm.evaluate(gt, pred)
    assert result["spell_f1_mean"] == pytest.approx(0.424, abs=5e-4)
    assert result["macro_f1"] == pytest.approx(0.581, abs=5e-4)


@pytest.mark.parametrize("path", BASELINES, ids=os.path.basename)
def test_shipped_baselines_are_valid_submissions(path, gt):
    assert pm.validate_submission(pm.read_label_csv(path), len(gt)) is None


def test_validator_rejects_broken_files(gt):
    good = pm.read_label_csv(BASELINES[0])
    assert "row count" in pm.validate_submission(good.iloc[:-1], len(gt))
    bad_label = good.copy()
    bad_label.loc[5, "label"] = "Q"
    assert "labels outside" in pm.validate_submission(bad_label, len(gt))
    blank = good.copy()
    blank["label"] = blank["label"].astype(object)
    blank.loc[5, "label"] = np.nan
    assert "empty label" in pm.validate_submission(blank, len(gt))
    shuffled = good.copy()
    shuffled.loc[[0, 1], "frame"] = [1, 0]
    assert "not 0.." in pm.validate_submission(pd.DataFrame(shuffled), len(gt))
