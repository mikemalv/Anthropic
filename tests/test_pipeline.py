"""Offline tests on synthetic data: no network, no FRED key needed."""

import numpy as np
import pandas as pd
import pytest

from fintechco_demo import analysis as an
from fintechco_demo import cleaning as cl


@pytest.fixture
def target() -> pd.Series:
    old = pd.Series(1.00, index=pd.date_range("2008-12-01", "2008-12-15"))
    new = pd.Series(0.25, index=pd.date_range("2008-12-16", "2008-12-31"))
    return cl.splice_policy_target(old.rename("DFEDTAR"), new.rename("DFEDTARU"))


def test_splice_counts_move_to_range_as_cut(target):
    events = cl.policy_events(target)
    assert list(events.index) == [pd.Timestamp("2008-12-16")]
    assert events["change_bp"].iloc[0] == -75
    assert events["direction"].iloc[0] == "cut"


def test_policy_events_sizes_and_directions():
    s = pd.Series([1.0, 1.0, 1.25, 1.25, 0.75], index=pd.date_range("2020-01-01", periods=5))
    events = cl.policy_events(s)
    assert events["change_bp"].tolist() == [25, -50]
    assert events["direction"].tolist() == ["hike", "cut"]


def test_align_daily_fills_at_most_one_day():
    base = pd.Series(1.0, index=pd.bdate_range("2024-01-01", periods=6), name="BASE")
    other = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-04"]), name="X")
    out = cl.align_daily({"BASE": base, "X": other}, base="BASE", max_ffill=1)
    # Jan 2 filled from Jan 1; Jan 3 left missing; Jan 5 filled from Jan 4
    x = out["X"]
    assert x.iloc[:2].tolist() == [1.0, 1.0]
    assert np.isnan(x.iloc[2])
    assert x.iloc[3:5].tolist() == [2.0, 2.0]


def test_events_map_to_next_trading_day():
    idx = pd.bdate_range("2024-01-01", periods=10)
    events = pd.DataFrame({"change_bp": [25]}, index=pd.to_datetime(["2024-01-06"]))  # Saturday
    out = cl.events_to_trading_days(events, idx)
    assert out["trading_day"].iloc[0] == pd.Timestamp("2024-01-08")


def test_event_paths_measure_change_from_day_before():
    series = pd.Series(np.arange(30, dtype=float))
    paths = an.event_paths(series, pd.Series([10]), pre=2, post=3)
    assert paths.columns.tolist() == [-2, -1, 0, 1, 2, 3]
    assert paths.iloc[0].tolist() == [-1, 0, 1, 2, 3, 4]


def test_event_paths_drop_incomplete_windows():
    series = pd.Series(np.zeros(20))
    paths = an.event_paths(series, pd.Series([2, 10, 18]), pre=5, post=5)
    assert paths.index.tolist() == [1]


def test_placebo_is_centered_on_zero_for_random_walk_noise():
    rng = np.random.default_rng(0)
    series = pd.Series(20 + rng.normal(0, 1, 3000))
    null = an.placebo_mean_changes(series, pd.Series([500, 1500, 2500]), horizon=5, n_draws=300)
    assert abs(null.mean()) < 0.5
    assert 0.0 <= an.empirical_p_value(0.0, null) <= 1.0


def test_surprise_response_recovers_known_slope():
    rng = np.random.default_rng(1)
    pos = np.arange(10, 410, 2)  # spaced so day -1 is never another event
    surprise = rng.normal(0, 10, len(pos))
    series = np.cumsum(rng.normal(0, 0.01, 420))
    series[pos] += 0.1 * surprise  # 0.1 points per bp => 2.5 points per 25 bp
    events = pd.DataFrame({"pos": pos, "surprise_bp": surprise})
    lp = an.surprise_response(pd.Series(series), events, range(0, 1))
    assert lp.loc[0, "effect"] == pytest.approx(2.5, abs=0.2)


def test_policy_regime_labels():
    s = pd.Series([1.0] * 5 + [1.25] * 5 + [1.0] * 5)
    reg = an.policy_regime(s, lookback=3)
    assert set(reg) == {"tightening", "easing", "on hold"}


def test_ols_slope_recovers_known_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 300)
    frame = pd.DataFrame({"x": x, "y": 1.0 - 0.5 * x + rng.normal(0, 0.1, 300)})
    out = an.ols_slope(frame, "y", "x")
    assert out["slope"] == pytest.approx(-0.5, abs=0.05)
    assert out["ci_low"] < out["slope"] < out["ci_high"]
    assert out["n"] == 300


def test_rolling_slope_detects_flattening():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    slope = np.where(np.arange(200) < 100, -0.6, 0.0)  # steep, then flat
    frame = pd.DataFrame({"x": x, "y": slope * x + rng.normal(0, 0.1, 200)})
    roll = an.rolling_slope(frame, "y", "x", window=40)
    assert len(roll) == 200 - 40 + 1
    assert roll["slope"].iloc[0] == pytest.approx(-0.6, abs=0.1)
    assert roll["slope"].iloc[-1] == pytest.approx(0.0, abs=0.1)
