#!/usr/bin/env python3
"""
Daily SVIX30 updater — fetch, compute, persist, publish.

Run daily after the Stockholm close (data is 15-min delayed):

    python update_index.py            # fetch latest snapshot
    python update_index.py --file NordicDerivatives-pretrade-2026-07-07T1730

Steps
-----
1. Download the latest Nasdaq Nordic pre-trade derivatives snapshot.
2. Build the OMXS30 option chain and compute the SVIX30 index.
3. Append the result to data/history.csv (keyed by trade date, idempotent —
   re-running on the same date replaces that date's row).
4. Regenerate docs/index.html from docs/template.html with the full history
   and the latest term structure inlined as JSON.

Designed to be safe under Task Scheduler: never raises out of main(),
logs to logs/update.log, exit code 0 on success / 1 on failure.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SITE_DIR = ROOT / "docs"
LOG_DIR = ROOT / "logs"
HISTORY_CSV = DATA_DIR / "history.csv"
TEMPLATE = SITE_DIR / "template.html"
OUTPUT = SITE_DIR / "index.html"

UNDERLYING = "OMXS30"
RISK_FREE = 0.0225  # Riksbank policy rate, update when it changes

for d in (DATA_DIR, SITE_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "update.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("svix30")


def data_date_from_filename(file_name: str) -> date:
    """Extract the trade date from e.g. '...pretrade-2026-07-07T1730'."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})T\d{4}$", file_name)
    if m:
        return date.fromisoformat(m.group(1))
    return date.today()


# Stockholm close: continuous trading ends 17:25, quotes are pulled from the
# pre-trade files after that.  Aggregate the last few pre-close minute-files
# for the fullest order book.
CLOSE_CUTOFF = "1724"
CLOSE_WINDOW_START = "1700"
N_CLOSE_FILES = 3


def select_close_files(available: list[str]) -> list[str]:
    """
    Pick the last N_CLOSE_FILES minute-files at or before the market close on
    the most recent trading day present in *available*.

    Falls back to the newest file overall when no pre-close file exists
    (e.g. intraday runs on a day that has not reached the close yet: the
    previous day's close files may already have rolled out of the 48 h
    window — in that case the latest intraday snapshot is the best we have).
    """
    dated: dict[str, list[str]] = {}
    for n in available:
        m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{4})$", n)
        if m:
            dated.setdefault(m.group(1), []).append(n)

    for day in sorted(dated, reverse=True):
        pre_close = sorted(
            n for n in dated[day]
            if CLOSE_WINDOW_START <= n[-4:] <= CLOSE_CUTOFF
        )
        if pre_close:
            return pre_close[-N_CLOSE_FILES:]
    return available[:1]


def load_history() -> list[dict]:
    if not HISTORY_CSV.exists():
        return []
    import csv
    with HISTORY_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_history(rows: list[dict]) -> None:
    import csv
    fields = ["date", "svix30", "atm30", "spot",
              "near_expiry", "near_days", "near_vol",
              "next_expiry", "next_days", "next_vol", "n_options", "file"]
    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x: float, nd: int = 2) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_site(history: list[dict], term_structure: list[dict],
                meta: dict) -> None:
    payload = json.dumps({
        "history": history,
        "term_structure": term_structure,
        "meta": meta,
    }, ensure_ascii=False)
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", payload)
    OUTPUT.write_text(html, encoding="utf-8")
    log.info("Site written to %s", OUTPUT)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Update the SVIX30 index")
    parser.add_argument("--file", default=None,
                        help="Specific snapshot file name (default: latest)")
    parser.add_argument("--rate", type=float, default=RISK_FREE * 100,
                        help="Risk-free rate in %% (default: 2.25)")
    args = parser.parse_args()
    r = args.rate / 100.0

    from nordic_options.fetcher import NordicFetcher
    from nordic_options.chain import enrich, latest_snapshot, build_chain
    from nordic_options.vix_index import compute_index

    fetcher = NordicFetcher()
    if args.file:
        files = [args.file]
    else:
        files = select_close_files(fetcher.list_available())
    file_name = files[-1]
    trade_date = data_date_from_filename(file_name)
    log.info("Fetching %d file(s), last %s (trade date %s)",
             len(files), file_name, trade_date)

    import pandas as pd
    df = pd.concat([fetcher.fetch(n) for n in files], ignore_index=True)
    df = latest_snapshot(df)
    df = enrich(df)
    log.info("Loaded %d instruments", len(df))

    chain = build_chain(df, underlying=UNDERLYING, r=r, as_of=trade_date)
    if chain.empty:
        log.error("No %s options in snapshot — aborting", UNDERLYING)
        return 1
    log.info("%s chain: %d rows, %d expiries",
             UNDERLYING, len(chain), chain["expiry"].nunique())

    result = compute_index(chain, r=r, as_of=trade_date)
    if not result.ok:
        log.error("Index computation failed (insufficient quotes)")
        return 1
    log.info("SVIX30 = %.2f (near %s %.2f / next %s %.2f)",
             result.value,
             result.near.expiry if result.near else "-",
             result.near.vol if result.near else float("nan"),
             result.next.expiry if result.next else "-",
             result.next.vol if result.next else float("nan"))

    # --- persist history (idempotent per trade date) ---------------------
    row = {
        "date": trade_date.isoformat(),
        "svix30": fmt(result.value),
        "atm30": fmt(result.atm30),
        "spot": fmt(result.spot_estimate),
        "near_expiry": result.near.expiry if result.near else "",
        "near_days": fmt(result.near.days, 1) if result.near else "",
        "near_vol": fmt(result.near.vol) if result.near else "",
        "next_expiry": result.next.expiry if result.next else "",
        "next_days": fmt(result.next.days, 1) if result.next else "",
        "next_vol": fmt(result.next.vol) if result.next else "",
        "n_options": str(sum(s.n_options for s in result.slices)),
        "file": file_name,
    }
    history = [h for h in load_history() if h["date"] != row["date"]]
    history.append(row)
    history.sort(key=lambda h: h["date"])
    save_history(history)
    log.info("History: %d observations", len(history))

    # --- regenerate website ----------------------------------------------
    term_structure = [
        {
            "expiry": s.expiry,
            "days": round(s.days, 1),
            "forward": round(s.forward, 2),
            "vol": round(s.vol, 2),
            "atm_iv": None if math.isnan(s.atm_iv) else round(s.atm_iv, 2),
            "n_options": s.n_options,
        }
        for s in result.slices
    ]
    meta = {
        "underlying": UNDERLYING,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "trade_date": trade_date.isoformat(),
        "source_file": file_name,
        "rate_pct": args.rate,
        "spot": None if math.isnan(result.spot_estimate)
                else round(result.spot_estimate, 2),
        "atm30": None if math.isnan(result.atm30) else round(result.atm30, 2),
        "near": None if not result.near else {
            "expiry": result.near.expiry, "days": round(result.near.days, 1),
            "vol": round(result.near.vol, 2)},
        "next": None if not result.next else {
            "expiry": result.next.expiry, "days": round(result.next.days, 1),
            "vol": round(result.next.vol, 2)},
    }
    render_site(
        [{"date": h["date"], "svix30": float(h["svix30"]),
          "atm30": float(h["atm30"]) if h.get("atm30") else None}
         for h in history if h.get("svix30")],
        term_structure, meta,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.exception("Update failed")
        sys.exit(1)
