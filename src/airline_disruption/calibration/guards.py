"""Split-safety checks for calibration (no pandas needed beyond a timestamp constant).

Phase 7A, Step 9. Three rules from the scope and DEC-015, each enforced by a function that raises
ValueError so a mistake stops the run instead of quietly producing a leaky number:

1. No row from the final-test period is ever used. (``reject_final_test_rows``)
2. Calibration rows come strictly BEFORE the rows they are evaluated on, in UTC prediction time.
   (``require_calibration_before_evaluation``)
3. The rows a calibrator is fitted on are never the rows it is scored on. (``require_disjoint``)
"""

from __future__ import annotations

import numpy as np

# 2023-01-01 00:00:00 UTC in seconds since 1970-01-01. The audited final-test period starts here (DEC-015).
FINAL_TEST_START_SECONDS = 1_672_531_200


def reject_final_test_rows(prediction_seconds, where: str) -> None:
    """Raise ValueError if any prediction time is on or after 2023-01-01 UTC."""
    seconds = np.asarray(prediction_seconds, dtype="int64")
    late = int((seconds >= FINAL_TEST_START_SECONDS).sum())
    if late:
        raise ValueError(f"{late:,} {where} rows are from the final-test period (2023). Calibration never touches it.")


def require_calibration_before_evaluation(calibration_seconds, evaluation_seconds, name: str) -> None:
    """The latest calibration row must be strictly earlier than the earliest evaluation row."""
    calibration_seconds = np.asarray(calibration_seconds, dtype="int64")
    evaluation_seconds = np.asarray(evaluation_seconds, dtype="int64")
    if len(calibration_seconds) == 0 or len(evaluation_seconds) == 0:
        raise ValueError(f"{name}: there are no calibration rows or no evaluation rows")
    if not calibration_seconds.max() < evaluation_seconds.min():
        raise ValueError(
            f"{name}: calibration rows end at {int(calibration_seconds.max())} but evaluation rows start at "
            f"{int(evaluation_seconds.min())} (seconds since 1970 UTC). Calibration must come first."
        )


def require_disjoint(calibration_ids, evaluation_ids, name: str) -> None:
    """No row id may be in both sets."""
    overlap = np.intersect1d(np.asarray(calibration_ids), np.asarray(evaluation_ids))
    if len(overlap):
        raise ValueError(f"{name}: {len(overlap):,} rows are used both to fit the calibrator and to evaluate it")
