"""Parse Nasdaq Nordic derivative instrument names into structured fields.

Naming conventions
------------------
Index futures   : ``OMXS306F``        → underlying=OMXS30, year=2026, month=Jun, type=future
Index monthly   : ``OMXS306F3000``    → call, strike=3000
                  ``OMXS306R3120``    → put, strike=3120
Index weekly    : ``OMXS306E22Y3095`` → call, expiry=May-22, strike=3095
Equity options  : ``AAK6F172.41X``    → call, strike=172.41, series=X
Equity futures  : ``3ERICB6S``        → underlying=ERICB, future, series_no=3

Month codes (OCC convention)
-----------------------------
Calls  A-L  →  Jan–Dec
Puts   M-X  →  Jan–Dec
Futures also use A-L for the expiry month when no strike is present.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# OCC month-code maps
_CALL_MONTHS: dict[str, int] = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKL")}
_PUT_MONTHS: dict[str, int] = {c: i + 1 for i, c in enumerate("MNOPQRSTUVWX")}
_ALL_OCC: dict[str, int] = {**_CALL_MONTHS, **_PUT_MONTHS}
_MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Instruments that need series-number prefix stripped before further parsing
_SERIES_PREFIX = re.compile(r"^(\d+)([A-Z].+)$")

# Weekly-option day suffix, e.g. "22Y" or "05Y"
_WEEKLY = re.compile(r"(\d{2})Y")

# Trailing alphabetic series code on equity options, e.g. "X" in "172.41X"
_SERIES_SUFFIX = re.compile(r"([A-Z]+)$")

# Numeric strike (possibly decimal), optionally followed by series letters
_STRIKE_RE = re.compile(r"(\d+(?:\.\d+)?)([A-Z]*)$")


@dataclass
class Instrument:
    """Parsed representation of a Nasdaq Nordic derivative instrument."""

    raw_name: str
    isin: str
    underlying_isin: str

    # Decoded fields
    underlying: str = ""
    instrument_type: str = "unknown"   # "call" | "put" | "future" | "unknown"
    expiry_year: Optional[int] = None
    expiry_month: Optional[int] = None
    expiry_day: Optional[int] = None   # set for weekly options
    strike: Optional[float] = None
    series_no: str = ""                # numeric prefix ("3" in "3ERICB…")
    series_code: str = ""              # alpha suffix ("X" in "…172.41X")
    ccy: str = ""

    @property
    def expiry_label(self) -> str:
        """Human-readable expiry, e.g. ``Jun-26`` or ``22-May-26``."""
        if self.expiry_month is None or self.expiry_year is None:
            return "?"
        mo = _MONTH_NAMES[self.expiry_month]
        yr = str(self.expiry_year)[-2:]
        if self.expiry_day:
            return f"{self.expiry_day:02d}-{mo}-{yr}"
        return f"{mo}-{yr}"

    @property
    def option_type_label(self) -> str:
        return {"call": "C", "put": "P", "future": "F", "unknown": "?"}.get(
            self.instrument_type, "?"
        )

    def __repr__(self) -> str:
        if self.instrument_type == "future":
            return f"Future({self.underlying} {self.expiry_label})"
        if self.instrument_type in ("call", "put"):
            return (
                f"Option({self.underlying} {self.expiry_label} "
                f"{self.option_type_label} K={self.strike})"
            )
        return f"Instrument({self.raw_name})"


def parse_instrument(
    name: str,
    isin: str = "",
    underlying_isin: str = "",
    ccy: str = "",
) -> Instrument:
    """Return a parsed :class:`Instrument` for the given raw instrument name."""
    inst = Instrument(
        raw_name=name,
        isin=isin,
        underlying_isin=underlying_isin,
        ccy=ccy,
    )
    _parse_into(name, inst)
    return inst


# ---------------------------------------------------------------------------
# Internal parsing logic
# ---------------------------------------------------------------------------

def _parse_into(name: str, inst: Instrument) -> None:
    # 1. Strip numeric series prefix (e.g. "3" in "3ERICB6S")
    m = _SERIES_PREFIX.match(name)
    if m:
        inst.series_no = m.group(1)
        name = m.group(2)

    # 2. Find the year digit – it separates the underlying from the expiry code.
    #    The year digit is the first single digit that is followed by an OCC
    #    month letter (or two-letter prefix for some weeklies).
    year_m = re.search(r"(\d)([A-X])(.*)$", name)
    if not year_m:
        inst.underlying = name
        return

    inst.underlying = name[: year_m.start()]
    year_digit = year_m.group(1)
    rest = year_m.group(2) + year_m.group(3)  # e.g. "F3000", "R3120", "E22Y3095"

    # Year: 2020-based for single digit 0-9
    inst.expiry_year = 2020 + int(year_digit)

    # 3. Extract month code (first letter)
    month_code = rest[0]
    after_month = rest[1:]

    # 4. Check for weekly expiry: e.g. "22Y3095" → day=22, strike=3095
    wm = _WEEKLY.match(after_month)
    if wm:
        inst.expiry_day = int(wm.group(1))
        after_month = after_month[wm.end():]

    # 5. Determine instrument type from what remains
    if not after_month:
        # Nothing after month code → future
        inst.instrument_type = "future"
        inst.expiry_month = _decode_month_code(month_code, as_future=True)
    else:
        # Try to parse a numeric strike (possibly with trailing series letters)
        sm = _STRIKE_RE.search(after_month)
        if sm:
            inst.strike = float(sm.group(1))
            inst.series_code = sm.group(2)
            inst.expiry_month = _decode_month_code(month_code, as_future=False)
            inst.instrument_type = (
                "call" if month_code in _CALL_MONTHS else "put"
            )
        else:
            # Letters only after month code → treat as future with complex code
            inst.instrument_type = "future"
            inst.expiry_month = _decode_month_code(month_code, as_future=True)
            inst.series_code = after_month


def _decode_month_code(code: str, *, as_future: bool) -> int:
    if code in _ALL_OCC:
        return _ALL_OCC[code]
    return 0
