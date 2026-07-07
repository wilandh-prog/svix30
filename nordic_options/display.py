"""Rich terminal display for option chains and summaries."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(width=132)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def display_summary(df: pd.DataFrame, title: str = "Nordic Derivatives Coverage") -> None:
    """Print an instrument coverage table by underlying and type."""
    from .chain import summary_table

    grp = summary_table(df)
    tbl = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    tbl.add_column("Underlying",   style="bold white", min_width=12)
    tbl.add_column("Type",         style="yellow",     min_width=8)
    tbl.add_column("Instruments",  justify="right",    style="green")
    tbl.add_column("Expirations",  justify="right",    style="blue")

    for _, row in grp.iterrows():
        tbl.add_row(
            row["underlying"],
            row["instrument_type"],
            str(int(row["count"])),
            str(int(row["expirations"])),
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------

def display_chain(
    chain: pd.DataFrame,
    underlying: str,
    expiry: Optional[str] = None,
    max_strikes: int = 40,
    forward: Optional[float] = None,
) -> None:
    """
    Display an option chain: IV · calls | strike | puts · IV.

    Parameters
    ----------
    chain : pd.DataFrame
        Output of :func:`~nordic_options.chain.build_chain` (must include
        ``c_iv`` / ``p_iv`` columns from :func:`~nordic_options.iv.add_iv`).
    underlying : str
        Label for the panel header.
    expiry : str, optional
        Expiry label shown in the header.
    max_strikes : int
        Limit displayed rows, centred on ATM.
    forward : float, optional
        Forward price used to centre the ATM row (highlighted in bold).
    """
    if chain.empty:
        console.print(f"[red]No chain data for {underlying}[/red]")
        return

    calls = chain[chain["type"] == "call"].set_index("strike")
    puts  = chain[chain["type"] == "put"].set_index("strike")

    all_strikes = sorted(set(calls.index.tolist()) | set(puts.index.tolist()))
    if not all_strikes:
        return

    # Centre on ATM: pick the strike closest to forward (or midpoint of range)
    atm_ref = forward if forward and not math.isnan(forward) else _estimate_atm(calls, puts)
    atm_strike = min(all_strikes, key=lambda k: abs(k - atm_ref))

    if len(all_strikes) > max_strikes:
        atm_idx = all_strikes.index(atm_strike)
        half = max_strikes // 2
        lo = max(0, atm_idx - half)
        hi = min(len(all_strikes), lo + max_strikes)
        all_strikes = all_strikes[lo:hi]

    expiry_str = f"  [{expiry}]" if expiry else ""
    fwd_str = f"  F={atm_ref:,.1f}" if forward and not math.isnan(forward) else ""
    title = f"Option Chain — {underlying}{expiry_str}{fwd_str}"

    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        header_style="bold white on dark_blue",
        border_style="blue",
    )

    # ── Call columns (left → centre) ──────────────────────────────────────
    tbl.add_column("C-IV %",  justify="right", style="bright_green", min_width=7)
    tbl.add_column("C-Vol",   justify="right", style="dim green",    min_width=6)
    tbl.add_column("C-Bid",   justify="right", style="green",        min_width=8)
    tbl.add_column("C-Ask",   justify="right", style="bright_green", min_width=8)
    tbl.add_column("C-Sprd",  justify="right", style="dim green",    min_width=6)
    # ── Strike ────────────────────────────────────────────────────────────
    tbl.add_column("Strike",  justify="center", style="bold yellow",  min_width=9)
    # ── Put columns (centre → right) ──────────────────────────────────────
    tbl.add_column("P-Sprd",  justify="right", style="dim red",      min_width=6)
    tbl.add_column("P-Bid",   justify="right", style="red",          min_width=8)
    tbl.add_column("P-Ask",   justify="right", style="bright_red",   min_width=8)
    tbl.add_column("P-Vol",   justify="right", style="dim red",      min_width=6)
    tbl.add_column("P-IV %",  justify="right", style="bright_red",   min_width=7)

    for strike in all_strikes:
        c = _get_row(calls, strike)
        p = _get_row(puts,  strike)

        is_atm = (strike == atm_strike)
        row_style = "bold" if is_atm else ""

        strike_str = f"{strike:,.2f}" if strike % 1 != 0 else f"{int(strike):,}"

        tbl.add_row(
            _fmt_iv(c["c_iv"] if c is not None else None),
            _fmt_vol(c, "bid_vol", "ask_vol") if c is not None else "—",
            _fmt_px(c["bid"]    if c is not None else None),
            _fmt_px(c["ask"]    if c is not None else None),
            _fmt_px(c["spread"] if c is not None else None),
            strike_str,
            _fmt_px(p["spread"] if p is not None else None),
            _fmt_px(p["bid"]    if p is not None else None),
            _fmt_px(p["ask"]    if p is not None else None),
            _fmt_vol(p, "bid_vol", "ask_vol") if p is not None else "—",
            _fmt_iv(p["p_iv"]   if p is not None else None),
            style=row_style,
        )

    console.print(tbl)


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------

def display_futures(df: pd.DataFrame, underlying: str) -> None:
    """Display futures contracts for an underlying."""
    fut = df[
        (df["underlying"] == underlying) & (df["instrument_type"] == "future")
    ].copy().sort_values(["expiry_year", "expiry_month"])

    if fut.empty:
        console.print(f"[red]No futures for {underlying}[/red]")
        return

    tbl = Table(title=f"Futures — {underlying}", box=box.ROUNDED,
                header_style="bold cyan")
    tbl.add_column("Expiry",  style="yellow")
    tbl.add_column("Name",    style="dim white")
    tbl.add_column("Bid",     justify="right", style="green")
    tbl.add_column("Ask",     justify="right", style="bright_green")
    tbl.add_column("Mid",     justify="right", style="white")
    tbl.add_column("Spread",  justify="right", style="dim white")
    tbl.add_column("Bid Vol", justify="right", style="dim green")
    tbl.add_column("Ask Vol", justify="right", style="dim red")

    for _, row in fut.iterrows():
        bid, ask = row["bid"], row["ask"]
        has_both = not pd.isna(bid) and not pd.isna(ask)
        mid  = (bid + ask) / 2 if has_both else float("nan")
        sprd = ask - bid       if has_both else float("nan")
        tbl.add_row(
            row["expiry_label"],
            row["name"],
            _fmt_px(bid), _fmt_px(ask), _fmt_px(mid), _fmt_px(sprd),
            str(int(row["bid_vol"])) if row["bid_vol"] > 0 else "—",
            str(int(row["ask_vol"])) if row["ask_vol"] > 0 else "—",
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_row(df_side: pd.DataFrame, strike: float):
    """Return a single row (Series) for *strike*, or None."""
    if strike not in df_side.index:
        return None
    row = df_side.loc[strike]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _fmt_px(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.3f}" if val < 10 else f"{val:.2f}"


def _fmt_vol(row, bid_col: str, ask_col: str) -> str:
    try:
        v = int((row[bid_col] + row[ask_col]) / 2)
        return str(v) if v > 0 else "—"
    except Exception:
        return "—"


def _fmt_iv(val) -> str:
    """Format IV as a percentage string, e.g. '18.3'."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    pct = val * 100
    if pct < 0 or pct > 300:
        return "—"
    return f"{pct:.1f}"


def _estimate_atm(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """ATM proxy: median of all available strikes."""
    all_k = list(calls.index) + list(puts.index)
    if not all_k:
        return 0.0
    return float(pd.Series(all_k).median())
