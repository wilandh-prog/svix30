"""
Black-76 implied volatility for Nasdaq Nordic derivatives.

Uses the Black-76 (Black model) rather than Black-Scholes because the data
provides futures prices directly, avoiding the need for dividends or repo rates.

Put-call parity is used to estimate the forward price from quoted call/put
mid-prices at the same strikes, which is more robust than any single price
source.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Normal distribution (no scipy dependency)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


# ---------------------------------------------------------------------------
# Black-76 pricer and greeks
# ---------------------------------------------------------------------------

def black76_price(F: float, K: float, T: float, r: float,
                  sigma: float, opt: str) -> float:
    """European option price under the Black-76 model."""
    if T <= 0.0:
        intrinsic = F - K if opt == "call" else K - F
        return max(0.0, intrinsic)
    if sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return float("nan")
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / sq
    d2 = d1 - sq
    df = math.exp(-r * T)
    if opt == "call":
        return df * (F * _ncdf(d1) - K * _ncdf(d2))
    return df * (K * _ncdf(-d2) - F * _ncdf(-d1))


def _vega76(F: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / sq
    return F * math.exp(-r * T) * _npdf(d1) * math.sqrt(T)


# ---------------------------------------------------------------------------
# IV solver — Newton-Raphson with bisection fallback
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    opt: str,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Return the implied volatility (annualised) that matches *market_price*.

    Returns ``nan`` when no solution exists (deep ITM/OTM with no time value,
    or degenerate inputs).
    """
    if (math.isnan(market_price) or market_price <= 0.0
            or math.isnan(F) or F <= 0.0
            or math.isnan(K) or K <= 0.0
            or T <= 0.0):
        return float("nan")

    # Intrinsic value check
    df = math.exp(-r * T)
    intrinsic = df * max(0.0, F - K if opt == "call" else K - F)
    if market_price < intrinsic - tol:
        return float("nan")

    # Brenner-Subrahmanyam initial guess
    sigma = math.sqrt(2.0 * math.pi / T) * market_price / F
    sigma = max(0.02, min(sigma, 4.0))

    # Newton-Raphson
    for _ in range(max_iter):
        price = black76_price(F, K, T, r, sigma, opt)
        vega = _vega76(F, K, T, r, sigma)
        if vega < 1e-10:
            break
        diff = market_price - price
        if abs(diff) < tol:
            return sigma
        sigma += diff / vega
        sigma = max(1e-4, min(sigma, 5.0))

    # Bisection fallback
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = black76_price(F, K, T, r, mid, opt) - market_price
        if abs(val) < tol or (hi - lo) < 1e-7:
            return mid
        if val > 0:
            hi = mid
        else:
            lo = mid

    return float("nan")


# ---------------------------------------------------------------------------
# Expiry date helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month (Nasdaq Nordic standard expiry)."""
    d = date(year, month, 1)
    days_to_fri = (4 - d.weekday()) % 7   # Monday=0, Friday=4
    return d + timedelta(days=days_to_fri + 14)


def expiry_date(expiry_label: str) -> Optional[date]:
    """
    Convert an expiry label (e.g. ``"Jun-26"`` or ``"22-May-26"``) to a date.

    Monthly options → third Friday of the month.
    Weekly options  → the explicit calendar date.
    """
    _MONTHS = {m: i + 1 for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )}
    parts = expiry_label.split("-")
    try:
        if len(parts) == 2:                         # "Jun-26"
            mo = _MONTHS[parts[0]]
            yr = 2000 + int(parts[1])
            return _third_friday(yr, mo)
        if len(parts) == 3:                         # "22-May-26"
            day = int(parts[0])
            mo = _MONTHS[parts[1]]
            yr = 2000 + int(parts[2])
            return date(yr, mo, day)
    except (KeyError, ValueError):
        return None
    return None


def time_to_expiry(expiry_label: str, as_of: Optional[date] = None) -> float:
    """Return years to expiry from *as_of* (default: today)."""
    exp = expiry_date(expiry_label)
    if exp is None:
        return float("nan")
    today = as_of or date.today()
    days = (exp - today).days
    return max(days, 0) / 365.25


# ---------------------------------------------------------------------------
# Forward price estimation via put-call parity
# ---------------------------------------------------------------------------

def estimate_forward(chain: pd.DataFrame, expiry: str, r: float) -> float:
    """
    Estimate the forward price for *expiry* using put-call parity.

    C - P = e^{-rT}(F - K)  →  F = K + (C - P) × e^{rT}

    Uses the median across all strikes that have both a call and put mid-price,
    which is robust to outliers and missing quotes.

    Returns ``nan`` if fewer than two valid pairs exist.
    """
    T = time_to_expiry(expiry)
    if math.isnan(T) or T <= 0:
        return float("nan")

    # Group by strike: duplicate listings at the same strike (different
    # series codes) collapse to their mean mid-quote.
    calls = (
        chain[(chain["expiry"] == expiry) & (chain["type"] == "call")]
        .dropna(subset=["mid"])
        .groupby("strike")["mid"].mean()
    )
    puts = (
        chain[(chain["expiry"] == expiry) & (chain["type"] == "put")]
        .dropna(subset=["mid"])
        .groupby("strike")["mid"].mean()
    )
    common = calls.index.intersection(puts.index)
    if len(common) < 2:
        return float("nan")

    df_factor = math.exp(r * T)
    forwards = [
        strike + (calls[strike] - puts[strike]) * df_factor
        for strike in common
    ]
    return float(pd.Series(forwards).median())


# ---------------------------------------------------------------------------
# Public helper: add IV columns to a chain DataFrame
# ---------------------------------------------------------------------------

def add_iv(
    chain: pd.DataFrame,
    r: float = 0.0225,
    as_of: Optional[date] = None,
    forward_override: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Add ``c_iv`` and ``p_iv`` columns (annualised, as fractions) to *chain*.

    Parameters
    ----------
    chain : pd.DataFrame
        Output of :func:`~nordic_options.chain.build_chain`.
    r : float
        Risk-free rate (continuously compounded).  Default 2.25 % (Riksbank).
    as_of : date, optional
        Calculation date.  Defaults to today.
    forward_override : dict, optional
        ``{expiry_label: forward_price}`` to skip parity estimation.
    """
    chain = chain.copy()
    chain["c_iv"] = float("nan")
    chain["p_iv"] = float("nan")

    for expiry in chain["expiry"].unique():
        T = time_to_expiry(expiry, as_of)
        if math.isnan(T) or T <= 0:
            continue

        # Forward price
        if forward_override and expiry in forward_override:
            F = forward_override[expiry]
        else:
            F = estimate_forward(chain, expiry, r)
        if math.isnan(F):
            continue

        exp_mask = chain["expiry"] == expiry

        for idx, row in chain[exp_mask].iterrows():
            K = row["strike"]
            if math.isnan(K):
                continue

            # Call IV
            if not math.isnan(row.get("mid", float("nan"))) and row["type"] == "call":
                chain.at[idx, "c_iv"] = implied_vol(
                    row["mid"], F, K, T, r, "call"
                )
            # Put IV
            if not math.isnan(row.get("mid", float("nan"))) and row["type"] == "put":
                chain.at[idx, "p_iv"] = implied_vol(
                    row["mid"], F, K, T, r, "put"
                )

    return chain
