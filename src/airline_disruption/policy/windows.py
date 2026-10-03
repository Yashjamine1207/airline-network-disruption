"""Expanding-window roles for the year-by-year backtests (Phase 7B).

For an evaluation year Y (UTC prediction time) the roles are:

    early_stop   the ``stop_months`` months before Y      (chooses the number of trees)
    fit          everything before that
    eval         the calendar year Y
    unused       later rows (never touched)

For Y = 2022 and six months this is exactly the locked protocol (fit before 2021-07-01, early stopping
2021-07-01 to 2021-12-31). The final-test year cannot be an evaluation year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FINAL_TEST_START_UTC = pd.Timestamp("2023-01-01", tz="UTC")
ROLE_FIT, ROLE_STOP, ROLE_EVAL, ROLE_UNUSED = "fit", "early_stop", "eval", "unused"


def expanding_window_roles(stamps: pd.Series, eval_year: int, stop_months: int = 6) -> np.ndarray:
    """Role of every row, by UTC prediction time. Raises ValueError for a locked year or empty roles."""
    if stop_months < 1:
        raise ValueError("stop_months must be at least 1")
    eval_start = pd.Timestamp(year=eval_year, month=1, day=1, tz="UTC")
    eval_end = pd.Timestamp(year=eval_year + 1, month=1, day=1, tz="UTC")
    if eval_end > FINAL_TEST_START_UTC:
        raise ValueError(f"Year {eval_year} reaches the locked final-test period. Evaluation years must end by 2022-12-31 UTC.")
    stop_start = eval_start - pd.DateOffset(months=stop_months)
    stamps = pd.Series(stamps)
    if stamps.dt.tz is None:
        raise ValueError("prediction timestamps must be timezone-aware UTC")
    if (stamps >= FINAL_TEST_START_UTC).any():
        raise ValueError("The rows include the locked final-test period. Drop them before assigning roles.")
    roles = np.select(
        [stamps < stop_start, stamps < eval_start, stamps < eval_end],
        [ROLE_FIT, ROLE_STOP, ROLE_EVAL],
        default=ROLE_UNUSED,
    ).astype(object)
    for name in (ROLE_FIT, ROLE_STOP, ROLE_EVAL):
        if not (roles == name).any():
            raise ValueError(f"No rows with role {name!r} for evaluation year {eval_year}")
    return roles
