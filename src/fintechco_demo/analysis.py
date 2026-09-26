"""Event studies, placebo tests, policy regimes, and surprise regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TRADING_DAYS_PER_YEAR = 252


def realized_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling realized volatility (%) from a daily price index, e.g. NASDAQCOM."""
    log_ret = np.log(prices.dropna()).diff()
    rv = log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    rv.name = f"realized_vol_{window}d_pct"
    return rv


def event_paths(
    series: pd.Series, positions: pd.Series, pre: int = 5, post: int = 20
) -> pd.DataFrame:
    """Change in `series` relative to the close the day before each event (day -1).

    Rows are events (indexed like `positions`), columns are trading days relative to the event.
    Events without a full window are dropped.
    """
    values = series.to_numpy()
    rel_days = np.arange(-pre, post + 1)
    rows, keep = [], []
    for idx, pos in positions.items():
        if pos - pre < 1 or pos + post >= len(values):
            continue
        rows.append(values[pos + rel_days] - values[pos - 1])
        keep.append(idx)
    return pd.DataFrame(rows, index=pd.Index(keep, name=positions.index.name), columns=rel_days)


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    """Mean path with a 95% confidence band (normal approximation) across events."""
    n = paths.notna().sum()
    mean = paths.mean()
    se = paths.std(ddof=1) / np.sqrt(n)
    return pd.DataFrame(
        {"mean": mean, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se, "n": n}
    ).rename_axis("rel_day")


def placebo_mean_changes(
    series: pd.Series,
    event_positions: pd.Series,
    horizon: int,
    n_draws: int = 2000,
    n_bins: int = 10,
    exclusion: int = 20,
    seed: int = 7,
) -> np.ndarray:
    """Null distribution for the mean event-window change, from random non-event days.

    Each draw picks one placebo day per real event, matched on the decile of the series level
    at day -1. Matching matters for VIX: it mean-reverts, and the Fed tends to cut when VIX is
    high, so unmatched placebos would credit the Fed with ordinary mean reversion.
    """
    values = series.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)

    valid = np.zeros(n, dtype=bool)
    valid[1 : n - horizon] = True
    for pos in event_positions:
        valid[max(pos - exclusion, 0) : pos + exclusion + 1] = False

    level_prev = pd.Series(np.r_[np.nan, values[:-1]])
    bins = pd.qcut(level_prev, n_bins, labels=False).to_numpy()
    pools = {b: np.flatnonzero(valid & (bins == b)) for b in range(n_bins)}

    event_bins = [int(bins[p]) for p in event_positions if 1 <= p < n - horizon]
    draws = np.empty(n_draws)
    for i in range(n_draws):
        picks = np.array([rng.choice(pools[b]) for b in event_bins])
        draws[i] = np.mean(values[picks + horizon] - values[picks - 1])
    return draws


def empirical_p_value(observed: float, null_draws: np.ndarray) -> float:
    """Two-sided p-value: share of placebo draws at least as far from the placebo mean."""
    center = null_draws.mean()
    return float(np.mean(np.abs(null_draws - center) >= abs(observed - center)))


def policy_regime(target_on_index: pd.Series, lookback: int = 126) -> pd.Series:
    """Label days tightening/easing/on hold by the target change over the past `lookback` days."""
    change = target_on_index.diff(lookback)
    regime = pd.Series("on hold", index=target_on_index.index, name="regime")
    regime[change > 0] = "tightening"
    regime[change < 0] = "easing"
    return regime[change.notna()]


def surprise_response(
    series: pd.Series,
    events: pd.DataFrame,
    horizons: range,
    surprise_col: str = "surprise_bp",
    scale_bp: float = 25.0,
) -> pd.DataFrame:
    """Local-projection regressions of the series' change from day -1 to day h on the policy
    surprise, one OLS per horizon with HC1 robust errors. Coefficients are scaled to the
    response to a `scale_bp` hawkish surprise.
    """
    values = series.to_numpy()
    rows = []
    for h in horizons:
        ok = (events["pos"] >= 1) & (events["pos"] + h < len(values))
        df = events.loc[ok, [surprise_col]].copy()
        df["dy"] = values[events.loc[ok, "pos"] + h] - values[events.loc[ok, "pos"] - 1]
        fit = smf.ols(f"dy ~ {surprise_col}", data=df).fit(cov_type="HC1")
        beta, se = fit.params[surprise_col], fit.bse[surprise_col]
        rows.append(
            {
                "horizon": h,
                "effect": beta * scale_bp,
                "ci_low": (beta - 1.96 * se) * scale_bp,
                "ci_high": (beta + 1.96 * se) * scale_bp,
                "p_value": fit.pvalues[surprise_col],
                "n": int(fit.nobs),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")
