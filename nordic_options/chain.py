"""Build option chains from parsed instrument data."""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from .parser import parse_instrument, Instrument


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse each instrument name and add structured columns to *df*.

    Added columns: ``underlying``, ``instrument_type``, ``expiry_year``,
    ``expiry_month``, ``expiry_day``, ``expiry_label``, ``strike``,
    ``series_no``, ``series_code``.
    """
    parsed: list[Instrument] = [
        parse_instrument(
            row["name"],
            isin=row.get("isin", ""),
            underlying_isin=row.get("underlying_isin", ""),
            ccy=row.get("ccy", ""),
        )
        for _, row in df.iterrows()
    ]

    df = df.copy()
    df["underlying"]      = [p.underlying      for p in parsed]
    df["instrument_type"] = [p.instrument_type for p in parsed]
    df["expiry_year"]     = [p.expiry_year      for p in parsed]
    df["expiry_month"]    = [p.expiry_month     for p in parsed]
    df["expiry_day"]      = [p.expiry_day       for p in parsed]
    df["expiry_label"]    = [p.expiry_label     for p in parsed]
    df["strike"]          = [p.strike           for p in parsed]
    df["series_no"]       = [p.series_no        for p in parsed]
    df["series_code"]     = [p.series_code      for p in parsed]
    return df


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple intra-minute rows to one row per instrument (last quote).
    """
    return (
        df.sort_values("pub_time")
        .groupby("name", sort=False)
        .last()
        .reset_index()
    )


def build_chain(
    df: pd.DataFrame,
    underlying: str,
    instrument_type: str = "both",
    expiry_label: Optional[str] = None,
    weekly: bool = True,
    r: float = 0.0225,
    as_of: Optional[date] = None,
    forward_override: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build an option chain table for *underlying*, with implied volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`enrich`.
    underlying : str
        Underlying ticker, e.g. ``"OMXS30"`` or ``"SAND"``.
    instrument_type : str
        ``"call"``, ``"put"``, or ``"both"`` (default).
    expiry_label : str, optional
        Filter to a single expiry, e.g. ``"Jun-26"``.
    weekly : bool
        Include weekly expirations.  Default True.
    r : float
        Risk-free rate (continuously compounded).  Default 2.25 %.
    as_of : date, optional
        Pricing date.  Defaults to today.
    forward_override : dict, optional
        ``{expiry_label: forward_price}`` to bypass parity estimation.

    Returns
    -------
    pd.DataFrame
        Rows are individual call/put instruments; columns include
        strike, expiry, type, bid, ask, mid, spread, bid_vol, ask_vol,
        c_iv, p_iv (IV as fractions, i.e. 0.18 = 18 %).
    """
    mask = df["underlying"] == underlying
    if not weekly:
        mask &= df["expiry_day"].isna()
    if expiry_label:
        mask &= df["expiry_label"] == expiry_label

    sub = df[mask].copy()

    if instrument_type != "both":
        sub = sub[sub["instrument_type"] == instrument_type]
    else:
        sub = sub[sub["instrument_type"].isin(["call", "put"])]

    if sub.empty:
        return pd.DataFrame()

    both = sub["bid"].notna() & sub["ask"].notna()
    sub["mid"]    = float("nan")
    sub["spread"] = float("nan")
    sub.loc[both, "mid"]    = (sub.loc[both, "bid"] + sub.loc[both, "ask"]) / 2
    sub.loc[both, "spread"] = sub.loc[both, "ask"] - sub.loc[both, "bid"]

    sub = sub.sort_values(["expiry_year", "expiry_month", "expiry_day", "strike"])

    records = []
    for _, row in sub.iterrows():
        records.append({
            "strike":   row["strike"],
            "expiry":   row["expiry_label"],
            "type":     row["instrument_type"],
            "bid":      row["bid"],
            "ask":      row["ask"],
            "mid":      row["mid"],
            "spread":   row["spread"],
            "bid_vol":  row.get("bid_vol", 0),
            "ask_vol":  row.get("ask_vol", 0),
        })

    chain = pd.DataFrame(records)

    # Add implied volatility
    from .iv import add_iv
    chain = add_iv(chain, r=r, as_of=as_of, forward_override=forward_override)

    return chain


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a coverage summary: instrument counts per underlying and type."""
    grp = (
        df.groupby(["underlying", "instrument_type"])
        .agg(
            count       = ("name", "count"),
            expirations = ("expiry_label", "nunique"),
        )
        .reset_index()
        .sort_values(["underlying", "instrument_type"])
    )
    return grp
