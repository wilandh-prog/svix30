"""Download pre-trade derivatives data from Nasdaq Nordic trade reports."""

import io
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

_BASE = "https://tradereports.nasdaq.com"
_LIST_URL = f"{_BASE}/api/regulatory/trade-reports?type=PRE_TRADE&assetClass=DERIVATIVES"
_DL_URL = (
    f"{_BASE}/api/regulatory/trade-report/download"
    "?type=PRE_TRADE&assetClass=DERIVATIVES&fileName={name}"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nordic-options-research/1.0)"}


class NordicFetcher:
    """Client for Nasdaq Nordic pre-trade derivatives files (15-min delayed, free)."""

    def list_available(self) -> list[str]:
        """Return all available file names (last 48 h, one per minute)."""
        r = requests.get(_LIST_URL, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()["reports"]

    def latest_name(self) -> str:
        """Return the most recent file name."""
        return self.list_available()[0]

    def fetch(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Download a pre-trade snapshot and return a tidy DataFrame.

        Parameters
        ----------
        file_name : str, optional
            Specific file name from :meth:`list_available`.  Defaults to the
            most recent file.
        """
        if file_name is None:
            file_name = self.latest_name()

        url = _DL_URL.format(name=file_name)
        r = requests.get(url, headers=_HEADERS, timeout=60)
        r.raise_for_status()

        # Files start with '"sep=;"' hint line — skip it, use semicolons
        text = r.text
        if text.startswith('"sep='):
            text = "\n".join(text.splitlines()[1:])

        df = pd.read_csv(
            io.StringIO(text),
            sep=";",
            dtype=str,
            na_values=["", "null"],
        )
        df = _clean(df, file_name)
        return df

    def fetch_range(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch and concatenate multiple files covering [start, end] (inclusive).

        Parameters
        ----------
        start, end : str
            Timestamps matching the file name suffix, e.g. ``"2026-05-20T0900"``.
        """
        available = self.list_available()
        chosen = [n for n in available if start <= n[-16:] <= end]
        frames = [self.fetch(n) for n in chosen]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COL_MAP = {
    "Publication Time": "pub_time",
    "Name": "name",
    "ISIN": "isin",
    "CCY": "ccy",
    "Bid price at level 1": "bid",
    "Bid volume at level 1": "bid_vol",
    "Bid orders at level 1": "bid_orders",
    "Ask price at level 1": "ask",
    "Ask volume at level 1": "ask_vol",
    "Ask orders at level 1": "ask_orders",
    "Underlying": "underlying_isin",
}


def _clean(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    df = df.rename(columns=_COL_MAP)
    df = df[[c for c in _COL_MAP.values() if c in df.columns]].copy()

    df["pub_time"] = pd.to_datetime(df["pub_time"], utc=True, errors="coerce")
    for col in ("bid", "bid_vol", "bid_orders", "ask", "ask_vol", "ask_orders"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Replace zero prices with NaN for clarity
    df["bid"] = df["bid"].replace(0, float("nan"))
    df["ask"] = df["ask"].replace(0, float("nan"))

    df["file"] = file_name
    return df
