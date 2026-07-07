#!/usr/bin/env python3
"""
Nordic Options — interactive CLI for Nasdaq Nordic pre-trade derivatives data.

Usage examples
--------------
  python main.py                        # Show OMXS30 chain, latest file
  python main.py --underlying OMXS30    # Same
  python main.py --underlying AAK       # Show AAK option chain
  python main.py --futures ERICB        # Show Ericsson B futures
  python main.py --summary              # Coverage overview (all underlyings)
  python main.py --expiry Jun-26        # Filter to one expiry
  python main.py --file NordicDerivatives-pretrade-2026-05-20T1000
  python main.py --list                 # List available files
"""

import argparse
import sys

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nasdaq Nordic pre-trade options viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--underlying", "-u", default="OMXS30",
        help="Underlying ticker to display (default: OMXS30)",
    )
    parser.add_argument(
        "--expiry", "-e", default=None,
        help="Filter to a single expiry label, e.g. 'Jun-26'",
    )
    parser.add_argument(
        "--file", "-f", default=None,
        help="Specific file name to load (default: latest available)",
    )
    parser.add_argument(
        "--summary", "-s", action="store_true",
        help="Print coverage summary instead of chain",
    )
    parser.add_argument(
        "--futures", default=None, metavar="UNDERLYING",
        help="Show futures table for an underlying",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available files and exit",
    )
    parser.add_argument(
        "--no-weekly", action="store_true",
        help="Exclude weekly expirations from chain",
    )
    parser.add_argument(
        "--max-strikes", type=int, default=40,
        help="Max strike rows to display (default: 40)",
    )
    parser.add_argument(
        "--swedish-only", action="store_true",
        help="Filter to instruments with Swedish (SE…) underlyings only",
    )
    parser.add_argument(
        "--rate", type=float, default=2.25,
        help="Risk-free rate in %% for IV calculation (default: 2.25)",
    )
    parser.add_argument(
        "--forward", type=float, default=None,
        help="Override forward price (applied to all expirations)",
    )
    args = parser.parse_args()

    from nordic_options.fetcher import NordicFetcher
    from nordic_options.chain import enrich, latest_snapshot, build_chain
    from nordic_options.display import display_chain, display_summary, display_futures

    fetcher = NordicFetcher()

    # --list
    if args.list:
        console.print("[bold cyan]Available files (newest first):[/bold cyan]")
        for name in fetcher.list_available()[:20]:
            console.print(f"  {name}")
        return

    # Download + parse
    console.print(f"[dim]Fetching {'latest' if not args.file else args.file} …[/dim]")
    try:
        df = fetcher.fetch(args.file)
    except Exception as exc:
        console.print(f"[red]Download failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[dim]Loaded {len(df):,} rows — reducing to latest snapshot …[/dim]")
    df = latest_snapshot(df)
    console.print(f"[dim]Enriching {len(df):,} instruments …[/dim]")
    df = enrich(df)

    if args.swedish_only:
        df = df[df["underlying_isin"].str.startswith("SE", na=False)]

    # --summary
    if args.summary:
        display_summary(df)
        return

    # --futures
    if args.futures:
        display_futures(df, args.futures.upper())
        return

    # Default: option chain
    from nordic_options.iv import estimate_forward, time_to_expiry
    underlying = args.underlying.upper()
    r = args.rate / 100.0

    chain = build_chain(
        df,
        underlying=underlying,
        expiry_label=args.expiry,
        weekly=not args.no_weekly,
        r=r,
        forward_override=(
            {exp: args.forward for exp in df["expiry_label"].unique()}
            if args.forward else None
        ),
    )

    if chain.empty:
        console.print(f"[yellow]No options found for '{underlying}'.[/yellow]")
        opts = df[df["instrument_type"].isin(["call", "put"])]["underlying"].value_counts().head(20)
        console.print("[dim]Available option underlyings (top 20):[/dim]")
        for name, count in opts.items():
            console.print(f"  {name}  ({count} instruments)")
        return

    # If no specific expiry given, show each expiry separately
    if args.expiry is None:
        expiries = sorted(
            chain["expiry"].unique(),
            key=lambda e: (e[-2:], e[:3]),
        )
        for exp in expiries[:6]:
            sub = chain[chain["expiry"] == exp]
            fwd = args.forward or estimate_forward(chain, exp, r)
            display_chain(sub, underlying, expiry=exp,
                          max_strikes=args.max_strikes, forward=fwd)
    else:
        fwd = args.forward or estimate_forward(chain, args.expiry, r)
        display_chain(chain, underlying, expiry=args.expiry,
                      max_strikes=args.max_strikes, forward=fwd)


if __name__ == "__main__":
    main()
