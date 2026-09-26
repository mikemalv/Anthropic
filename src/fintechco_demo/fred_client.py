"""FRED fetch + on-disk cache. The only module in this package that touches the network."""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_START = "1990-01-01"
TIMEOUT_S = 30

log = logging.getLogger(__name__)


def _api_key() -> str:
    """Read FRED_API_KEY from the environment (.env supported). Never log the return value."""
    load_dotenv()
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def _get(url: str, params: dict) -> dict:
    """GET a FRED endpoint. Errors are re-raised without the URL, which would contain the key."""
    try:
        resp = requests.get(
            url, params={**params, "api_key": _api_key(), "file_type": "json"}, timeout=TIMEOUT_S
        )
    except requests.RequestException as exc:
        raise ConnectionError(f"FRED request failed ({type(exc).__name__})") from None
    if resp.status_code != 200:
        raise ConnectionError(f"FRED returned HTTP {resp.status_code} for {params}")
    return resp.json()


def _read_cache(path: Path, series_id: str) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = series_id
    return s


def fetch_series(
    series_id: str,
    start: str = DEFAULT_START,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Return one FRED series (any ID) as a float Series, served from cache when available.

    Cache files are keyed by series and start date (`{id}_from{start}_{pulled}.csv`), so a
    request for a longer history never silently returns a shorter cached pull.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(cache_dir.glob(f"{series_id}_from{start}_*.csv"))
    if cached and not refresh:
        log.info("%s: using cache %s", series_id, cached[-1].name)
        return _read_cache(cached[-1], series_id)

    try:
        payload = _get(FRED_OBSERVATIONS_URL, {"series_id": series_id, "observation_start": start})
    except ConnectionError as exc:
        if cached:
            log.warning("%s: %s; falling back to cache %s", series_id, exc, cached[-1].name)
            return _read_cache(cached[-1], series_id)
        raise

    obs = pd.DataFrame(payload["observations"])
    # FRED encodes missing values as "."
    s = pd.Series(
        pd.to_numeric(obs["value"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(obs["date"]), name="date"),
        name=series_id,
    )
    path = cache_dir / f"{series_id}_from{start}_{date.today():%Y-%m-%d}.csv"
    s.to_csv(path)
    log.info("%s: downloaded %d observations -> %s", series_id, len(s), path.name)
    return s


def fetch_many(series_ids: list[str], **kwargs) -> dict[str, pd.Series]:
    """Fetch several FRED series. Returns a dict keyed by series ID (frequencies may differ)."""
    return {sid: fetch_series(sid, **kwargs) for sid in series_ids}


def series_info(series_id: str) -> dict:
    """Return FRED metadata (title, units, frequency, last_updated) for a series ID."""
    return _get(FRED_SERIES_URL, {"series_id": series_id})["seriess"][0]
