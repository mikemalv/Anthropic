"""Frequency alignment, missing-data handling, and construction of the Fed policy event list."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# FRED discontinued the single target rate (DFEDTAR) when the FOMC moved to a target range.
TARGET_RANGE_START = pd.Timestamp("2008-12-16")


def splice_policy_target(dfedtar: pd.Series, dfedtaru: pd.Series) -> pd.Series:
    """Splice DFEDTAR (to 2008-12-15) with DFEDTARU (upper bound, from 2008-12-16), in percent."""
    old = dfedtar.loc[: TARGET_RANGE_START - pd.Timedelta(days=1)]
    new = dfedtaru.loc[TARGET_RANGE_START:]
    target = pd.concat([old, new]).dropna().sort_index()
    target.name = "fed_target_pct"
    return target


def policy_events(target: pd.Series) -> pd.DataFrame:
    """List every change in the policy target (DFEDTAR/DFEDTARU) with its size in basis points."""
    change_bp = (target.diff() * 100).round(0)
    events = change_bp[change_bp.ne(0) & change_bp.notna()].to_frame("change_bp")
    events["direction"] = np.where(events["change_bp"] > 0, "hike", "cut")
    events["target_after_pct"] = target.loc[events.index]
    events.index.name = "date"
    return events


def align_daily(series: dict[str, pd.Series], base: str, max_ffill: int = 1) -> pd.DataFrame:
    """Align series onto the trading calendar of `base` (e.g. VIXCLS), forward-filling at most
    `max_ffill` days so holiday mismatches between markets don't create gaps.
    """
    base_index = series[base].dropna().index
    frame = {}
    for sid, s in series.items():
        s = s.sort_index()
        # Reindex on the union so a value from a non-trading day can carry forward one step.
        aligned = s.reindex(s.index.union(base_index)).ffill(limit=max_ffill).reindex(base_index)
        filled = int(aligned.notna().sum() - s.reindex(base_index).notna().sum())
        missing = int(aligned.isna().sum())
        log.info("%s: %d values forward-filled, %d still missing", sid, filled, missing)
        frame[sid] = aligned
    return pd.DataFrame(frame, index=base_index)


def events_to_trading_days(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each event date to the first trading day on or after it (column `pos` = row number)."""
    pos = index.searchsorted(events.index, side="left")
    out = events.copy()
    in_range = pos < len(index)
    out = out.loc[in_range]
    out["pos"] = pos[in_range]
    out["trading_day"] = index[out["pos"]]
    return out
