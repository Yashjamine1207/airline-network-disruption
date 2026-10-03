"""Small text helpers for the generated reports.

The helpers have one job: turn numbers into the same plain text every time, and refuse text that breaks the
project's writing rules (no long dashes, because the reports are read as plain markdown and copied around).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pandas as pd

LONG_DASHES = tuple(chr(code) for code in (0x2014, 0x2013, 0x2012, 0x2015))  # em dash, en dash, figure dash, horizontal bar


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def num(value, digits: int = 4) -> str:
    """0.059137 gives '0.0591'. Missing values give 'n/a'."""
    return "n/a" if is_missing(value) else f"{float(value):.{digits}f}"


def pct(value, digits: int = 1) -> str:
    """0.1047 gives '10.5%'. The input is a share between 0 and 1."""
    return "n/a" if is_missing(value) else f"{float(value) * 100:.{digits}f}%"


def whole(value) -> str:
    """1234567 gives '1,234,567'."""
    return "n/a" if is_missing(value) else f"{int(round(float(value))):,}"


def signed(value, digits: int = 4) -> str:
    """0.0157 gives '+0.0157'; -0.0032 gives '-0.0032'."""
    return "n/a" if is_missing(value) else f"{float(value):+.{digits}f}"


def interval(low, high, digits: int = 4, as_percent: bool = False) -> str:
    """'(0.0540 to 0.0652)'. Words, not a dash, between the two ends."""
    if is_missing(low) or is_missing(high):
        return "(interval n/a)"
    render = (lambda v: pct(v, digits)) if as_percent else (lambda v: num(v, digits))
    return f"({render(low)} to {render(high)})"


def ratio_text(value, digits: int = 1) -> str:
    """12.695 gives '12.7'."""
    return "n/a" if is_missing(value) else f"{float(value):.{digits}f}"


Column = tuple[str, str | Callable[[pd.Series], str]]


def md_table(frame: pd.DataFrame, columns: Sequence[tuple[str, str, Callable[[object], str] | None]]) -> str:
    """A markdown table. ``columns`` is a list of (header, source column, formatter). A missing column is an error, not a blank."""
    if frame is None or len(frame) == 0:
        return "_(no rows)_"
    missing = [source for _, source, _ in columns if source not in frame.columns]
    if missing:
        raise KeyError(f"table is missing columns {missing}; it has {list(frame.columns)}")
    header = "| " + " | ".join(h for h, _, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = []
        for _, source, formatter in columns:
            value = row[source]
            cells.append(str(formatter(value)) if formatter is not None else ("n/a" if is_missing(value) else str(value)))
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines)


def bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def check_clean(text: str, name: str = "report") -> str:
    """Return ``text`` unchanged, or raise if it holds a long dash or an unfilled placeholder."""
    for dash in LONG_DASHES:
        if dash in text:
            line = next(l for l in text.splitlines() if dash in l)
            raise ValueError(f"{name} contains a long dash: {line[:120]!r}")
    for marker in ("nan%", "None", "{{", "}}", "TODO", "TBD"):
        if marker in text:
            line = next(l for l in text.splitlines() if marker in l)
            raise ValueError(f"{name} contains {marker!r}: {line[:120]!r}")
    return text


def join_sections(sections: Sequence[str]) -> str:
    """Blank line between sections, single newline at the end."""
    return "\n\n".join(s.strip("\n") for s in sections if s and s.strip()) + "\n"


def is_close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
