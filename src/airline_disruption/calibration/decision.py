"""The pre-registered rules that choose the calibration method (DEC-023).

Phase 7A, Step 9. The rules are functions of numbers only, so they are fixed before any result
exists and are covered by tests.

Rule 1, which method. Fit on the calibration rows. On the FIRST half of 2022 (selection) a method is
a candidate only if its Brier score is below the uncalibrated Brier score. On the SECOND half of 2022
(confirmation) a candidate is adopted only if the paired day-level bootstrap of
(Brier(method) - Brier(uncalibrated)) has its whole 95% interval below zero. If both Platt and
isotonic pass, isotonic is preferred only if it also beats Platt with the whole interval below zero;
otherwise Platt (two numbers, cannot change the ranking) is chosen. If nothing passes, the scores
stay uncalibrated. Simple beats flexible unless the evidence is clear.

Rule 2, how the final-test calibrator will be fitted. If the chosen method fitted on the first half
of 2022 beats the same method fitted on the older calibration rows, on the second half of 2022, with
the whole 95% interval below zero, recency matters and the final-test calibrator is fitted on the
most recent labelled year before the test. Otherwise it is fitted on the same kind of rows as here.
"""

from __future__ import annotations

import numpy as np

METHOD_ORDER = ("uncalibrated", "platt", "isotonic")


def _below_zero(upper_end: float) -> bool:
    """True only for a finite upper interval end below zero. NaN counts as no."""
    return bool(np.isfinite(upper_end) and upper_end < 0.0)


def choose_method(
    brier_selection: dict[str, float],
    ci_high_vs_uncalibrated: dict[str, float],
    ci_high_isotonic_vs_platt: float,
) -> tuple[str, str]:
    """Return ``(method, reason)`` under Rule 1.

    ``brier_selection`` maps every method (including ``uncalibrated``) to its Brier score on the
    selection half. ``ci_high_vs_uncalibrated`` maps ``platt`` and ``isotonic`` to the UPPER end of the
    95% interval of (Brier(method) - Brier(uncalibrated)) on the confirmation half.
    ``ci_high_isotonic_vs_platt`` is the same for (isotonic - platt).
    """
    for method in METHOD_ORDER:
        if method not in brier_selection:
            raise ValueError(f"brier_selection is missing {method!r}")
    reference = brier_selection["uncalibrated"]
    candidates = [m for m in ("platt", "isotonic") if brier_selection[m] < reference]
    if not candidates:
        return "uncalibrated", "No calibrator had a lower Brier score than the uncalibrated scores on the selection half."
    confirmed = [m for m in candidates if _below_zero(ci_high_vs_uncalibrated.get(m, np.nan))]
    if not confirmed:
        return "uncalibrated", (
            "A calibrator improved the selection half, but no improvement over the uncalibrated scores was "
            "confirmed on the second half (whole 95% interval below zero)."
        )
    if confirmed == ["platt"]:
        return "platt", "Platt improved the selection half and the improvement was confirmed on the second half."
    if confirmed == ["isotonic"]:
        return "isotonic", "Isotonic improved the selection half and the improvement was confirmed on the second half."
    if _below_zero(ci_high_isotonic_vs_platt):
        return "isotonic", "Both were confirmed; isotonic also beat Platt with the whole 95% interval below zero."
    return "platt", "Both were confirmed, but isotonic did not clearly beat Platt, so the simpler Platt is chosen."


def choose_final_fit_source(method: str, ci_high_recent_vs_old: float) -> tuple[str, str]:
    """Return ``(source, reason)`` under Rule 2.

    ``source`` is ``"none"`` (nothing to fit), ``"recent_year"`` or ``"same_as_here"``.
    ``ci_high_recent_vs_old`` is the upper end of the 95% interval of
    (Brier(fitted on 2022 first half) - Brier(fitted on the older rows)) on the second half of 2022.
    """
    if method == "uncalibrated":
        return "none", "The uncalibrated scores were kept, so there is no calibrator to fit."
    if _below_zero(ci_high_recent_vs_old):
        return "recent_year", "A calibrator fitted on more recent rows was clearly better, so recency matters."
    return "same_as_here", "Fitting on more recent rows was not clearly better, so the same kind of rows is used."
