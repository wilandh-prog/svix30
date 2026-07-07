# SVIX30 — Swedish Implied Volatility Index

A VIX-style 30-day model-free implied volatility index for **OMXS30**, computed
daily from Nasdaq Nordic pre-trade derivatives data (15-minute delayed, free).

- **Index**: `nordic_options/vix_index.py` — CBOE VIX methodology: per-expiration
  forwards via put-call parity, model-free variance from OTM mid-quotes with the
  zero-quote wing-truncation rule, total-variance interpolation to a constant
  30-day horizon. A 30-day ATM IV is computed alongside as a robustness check.
- **Daily job**: `update_index.py` — aggregates the last pre-close minute-files
  (quotes are pulled from the feed after 17:25 Stockholm), computes the index,
  appends to `data/history.csv` (idempotent per trade date) and regenerates the
  static website `docs/index.html` from `docs/template.html`.
- **Automation**: `.github/workflows/update.yml` runs the job on GitHub Actions
  every weekday at 16:45 UTC and commits the updated history and site.
- **Website**: <https://wilandh-prog.github.io/svix30/> (GitHub Pages, served
  from `docs/`) — fully self-contained (history chart, term structure, tables,
  light/dark).

Also included: `main.py`, an interactive CLI for browsing Nasdaq Nordic option
chains and futures with Black-76 implied volatilities.

```
pip install -r requirements.txt
python update_index.py          # compute today's value and rebuild the site
python main.py -u OMXS30        # browse the option chain
```

Unofficial research index, not investment advice. The risk-free rate is
hardcoded in `update_index.py` (`RISK_FREE`); update it when the Riksbank moves.
