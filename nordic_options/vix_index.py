"""
SVIX30 — a VIX-style 30-day implied volatility index for OMXS30.

Methodology (CBOE VIX white paper, adapted to Nasdaq Nordic data):

1.  For each expiration, estimate the forward F via put-call parity
    (median across strikes with both call and put mid-quotes).
2.  K0 = largest strike <= F.  Out-of-the-money options are puts with
    K < K0 and calls with K > K0; at K0 the call/put mid average is used.
3.  Model-free variance for the expiration:

        sigma^2 = (2/T) * sum_i  dK_i / K_i^2 * e^{rT} * Q(K_i)
                  - (1/T) * (F/K0 - 1)^2

    where dK_i is half the distance between adjacent strikes (one-sided
    at the edges) and Q(K_i) the OTM mid-quote.  Each wing is truncated
    after two consecutive strikes without a valid quote, per the VIX rule.
4.  The variances of the two expirations bracketing 30 days are linearly
    interpolated in total variance to a constant 30-day horizon:

        sigma30^2 = [ T1*s1^2*(T2-T30) + T2*s2^2*(T30-T1) ] / (T2-T1) / T30

    SVIX30 = 100 * sqrt(sigma30^2)

Data-quality guards: expirations need >= MIN_OTM_QUOTES valid OTM quotes
and T >= MIN_DAYS, otherwise they are skipped.  If no pair brackets
30 days the two nearest usable expirations are used instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from .iv import estimate_forward, time_to_expiry, implied_vol

TARGET_DAYS = 30
MIN_DAYS = 5            # skip expirations closer than this (noisy)
MIN_OTM_QUOTES = 6      # minimum valid OTM quotes per expiration
MAX_CONSECUTIVE_MISSING = 2


@dataclass
class ExpirySlice:
    """Per-expiration intermediate results."""
    expiry: str
    T: float                      # years
    days: float
    forward: float
    k0: float
    variance: float               # annualised sigma^2
    vol: float                    # 100 * sqrt(variance)
    n_options: int
    atm_iv: float = float("nan")  # Black-76 IV at K0, in %


@dataclass
class IndexResult:
    value: float                          # SVIX30 in vol points
    as_of: date
    near: Optional[ExpirySlice] = None
    next: Optional[ExpirySlice] = None
    slices: list[ExpirySlice] = field(default_factory=list)
    spot_estimate: float = float("nan")   # forward of nearest expiry
    atm30: float = float("nan")           # 30-day interpolated ATM IV, in %

    @property
    def ok(self) -> bool:
        return not math.isnan(self.value)


def _strike_variance(
    sub: pd.DataFrame, F: float, T: float, r: float
) -> tuple[float, float, int]:
    """Return (variance, K0, n_options) for one expiration's chain rows."""
    # Mean over duplicate listings at the same strike (different series codes)
    calls = sub[sub["type"] == "call"].groupby("strike")["mid"].mean()
    puts = sub[sub["type"] == "put"].groupby("strike")["mid"].mean()
    strikes = sorted(set(calls.index) | set(puts.index))
    strikes = [k for k in strikes if not math.isnan(k)]
    if len(strikes) < 3:
        return float("nan"), float("nan"), 0

    below = [k for k in strikes if k <= F]
    if not below:
        return float("nan"), float("nan"), 0
    k0 = max(below)

    def quote(k: float) -> float:
        """OTM quote at strike k (call/put average at K0)."""
        c = calls.get(k, float("nan"))
        p = puts.get(k, float("nan"))
        if k == k0:
            if not math.isnan(c) and not math.isnan(p):
                return 0.5 * (c + p)
            return c if not math.isnan(c) else p
        return p if k < k0 else c

    # Truncate each wing after MAX_CONSECUTIVE_MISSING missing quotes
    i0 = strikes.index(k0)
    selected: dict[float, float] = {}
    for direction in (-1, +1):
        missing = 0
        i = i0 + (0 if direction == -1 else 1)
        while 0 <= i < len(strikes):
            q = quote(strikes[i])
            if math.isnan(q) or q <= 0:
                missing += 1
                if missing >= MAX_CONSECUTIVE_MISSING:
                    break
            else:
                missing = 0
                selected[strikes[i]] = q
            i += direction

    if len(selected) < MIN_OTM_QUOTES:
        return float("nan"), k0, len(selected)

    ks = sorted(selected)
    disc = math.exp(r * T)
    total = 0.0
    for j, k in enumerate(ks):
        if j == 0:
            dk = ks[1] - ks[0]
        elif j == len(ks) - 1:
            dk = ks[-1] - ks[-2]
        else:
            dk = 0.5 * (ks[j + 1] - ks[j - 1])
        total += dk / (k * k) * disc * selected[k]

    variance = (2.0 / T) * total - (1.0 / T) * (F / k0 - 1.0) ** 2
    return variance, k0, len(selected)


def compute_slices(
    chain: pd.DataFrame,
    r: float = 0.0225,
    as_of: Optional[date] = None,
) -> list[ExpirySlice]:
    """Compute per-expiration variance slices from a build_chain() DataFrame."""
    slices: list[ExpirySlice] = []
    for expiry in chain["expiry"].unique():
        T = time_to_expiry(expiry, as_of)
        if math.isnan(T) or T * 365.25 < MIN_DAYS:
            continue
        F = estimate_forward(chain, expiry, r)
        if math.isnan(F) or F <= 0:
            continue
        sub = chain[chain["expiry"] == expiry].dropna(subset=["strike"])
        variance, k0, n = _strike_variance(sub, F, T, r)
        if math.isnan(variance) or variance <= 0:
            continue

        # ATM IV at K0 (secondary metric): average of call and put Black-76
        # IVs at K0, using whichever mid-quotes exist.
        ivs = []
        for opt in ("call", "put"):
            mids = sub[(sub["strike"] == k0) & (sub["type"] == opt)]["mid"]
            if len(mids) and not math.isnan(mids.iloc[0]):
                iv = implied_vol(mids.iloc[0], F, k0, T, r, opt)
                if not math.isnan(iv):
                    ivs.append(iv)
        atm = 100.0 * sum(ivs) / len(ivs) if ivs else float("nan")

        slices.append(ExpirySlice(
            expiry=expiry, T=T, days=T * 365.25, forward=F, k0=k0,
            variance=variance, vol=100.0 * math.sqrt(variance),
            n_options=n, atm_iv=atm,
        ))
    slices.sort(key=lambda s: s.T)
    return slices


def compute_index(
    chain: pd.DataFrame,
    r: float = 0.0225,
    as_of: Optional[date] = None,
    target_days: float = TARGET_DAYS,
) -> IndexResult:
    """Compute the SVIX30 index from an OMXS30 option chain."""
    as_of = as_of or date.today()
    slices = compute_slices(chain, r=r, as_of=as_of)
    result = IndexResult(value=float("nan"), as_of=as_of, slices=slices)
    if slices:
        result.spot_estimate = slices[0].forward
    if len(slices) < 1:
        return result

    t_star = target_days / 365.25

    if len(slices) == 1:
        s = slices[0]
        result.near = s
        result.value = s.vol
        return result

    # Pick the pair bracketing the target, else the two nearest
    below = [s for s in slices if s.T <= t_star]
    above = [s for s in slices if s.T > t_star]
    if below and above:
        near, nxt = below[-1], above[0]
    elif above:
        near, nxt = above[0], above[1]
    else:
        near, nxt = below[-2], below[-1]

    w = (nxt.T - t_star) / (nxt.T - near.T)
    total_var = (near.T * near.variance * w
                 + nxt.T * nxt.variance * (1.0 - w))
    sigma30_sq = total_var / t_star
    if sigma30_sq <= 0:
        return result

    result.near, result.next = near, nxt
    result.value = 100.0 * math.sqrt(sigma30_sq)

    # Secondary series: 30-day ATM IV, same total-variance interpolation.
    # More robust to sparse strike coverage than the variance strip.
    if not math.isnan(near.atm_iv) and not math.isnan(nxt.atm_iv):
        v1 = (near.atm_iv / 100.0) ** 2
        v2 = (nxt.atm_iv / 100.0) ** 2
        atm_var = (near.T * v1 * w + nxt.T * v2 * (1.0 - w)) / t_star
        if atm_var > 0:
            result.atm30 = 100.0 * math.sqrt(atm_var)
    return result
