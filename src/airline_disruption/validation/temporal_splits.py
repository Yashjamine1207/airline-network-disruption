from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Split labels exactly as they appear in temporal_split_manifest.csv.
SPLIT_DEVELOPMENT = "development"
SPLIT_VALIDATION = "validation"
SPLIT_FINAL_TEST = "final_test_locked"

# Label given to rows removed by the UTC embargo.
SPLIT_EMBARGOED = "embargoed_boundary"

# Chronological order of the three retained splits.
SPLIT_ORDER = (SPLIT_DEVELOPMENT, SPLIT_VALIDATION, SPLIT_FINAL_TEST)

# Operating regimes from the Phase 5-7B scope update.
REGIME_2019 = "2019_pre_covid"
REGIME_2020 = "2020_covid_shock"
REGIME_2021_2022 = "2021_2022_transition"
REGIME_2023 = "2023_partial_final"
REGIME_OTHER = "other"


@dataclass(frozen=True)
class UtcSplitBoundaries:
    """UTC instants that separate the three splits.

    A development row must have prediction time < ``validation_start_utc``.
    A validation row must have ``validation_start_utc`` <= time < ``final_test_start_utc``.
    A final-test row must have time >= ``final_test_start_utc``.
    """

    validation_start_utc: pd.Timestamp
    final_test_start_utc: pd.Timestamp

    def __post_init__(self) -> None:
        for name in ("validation_start_utc", "final_test_start_utc"):
            value = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware (UTC)")
        if str(self.validation_start_utc.tz) != "UTC" or str(self.final_test_start_utc.tz) != "UTC":
            raise ValueError("Split boundaries must be expressed in UTC")
        if not self.validation_start_utc < self.final_test_start_utc:
            raise ValueError("validation_start_utc must be earlier than final_test_start_utc")


def boundaries_from_config(config_path: str | Path) -> UtcSplitBoundaries:
    """Read the split dates from ``configs/validation.yaml``.

    Uses ``primary_temporal_split.validation_start`` and ``validation_end``.
    ``validation_end`` is an inclusive calendar date, so the final test period
    starts on the following day at 00:00 UTC.
    """
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    block = config["primary_temporal_split"]

    # str() protects against YAML turning an unquoted date into a date object.
    validation_start = pd.Timestamp(str(block["validation_start"])).tz_localize("UTC")
    validation_end = pd.Timestamp(str(block["validation_end"])).tz_localize("UTC")
    final_test_start = validation_end + pd.Timedelta(days=1)
    return UtcSplitBoundaries(validation_start, final_test_start)


def apply_utc_embargo(
    split_labels: pd.Series,
    prediction_timestamp_utc: pd.Series,
    boundaries: UtcSplitBoundaries,
) -> pd.Series:
    """Return the Phase 6 split label for every row.

    Rows whose UTC prediction timestamp is missing, or lies on the wrong side
    of the boundaries for their original split, get ``SPLIT_EMBARGOED``.

    Raises ValueError for naive timestamps, mismatched lengths, or an original
    label that is not one of the three expected splits.
    """
    if len(split_labels) != len(prediction_timestamp_utc):
        raise ValueError("split_labels and prediction_timestamp_utc must have the same length")
    if not isinstance(prediction_timestamp_utc.dtype, pd.DatetimeTZDtype):
        raise ValueError("prediction_timestamp_utc must be a timezone-aware datetime column")

    labels = split_labels.astype(str).to_numpy()
    unknown = set(np.unique(labels)) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"Unexpected split labels: {sorted(unknown)}")

    stamps = prediction_timestamp_utc.dt.tz_convert("UTC")
    known_time = stamps.notna().to_numpy()
    # Comparing against NaT gives False, so missing timestamps fall through to the embargo.
    before_validation = (stamps < boundaries.validation_start_utc).to_numpy()
    before_final = (stamps < boundaries.final_test_start_utc).to_numpy()

    keep_development = (labels == SPLIT_DEVELOPMENT) & known_time & before_validation
    keep_validation = (labels == SPLIT_VALIDATION) & known_time & ~before_validation & before_final
    keep_final_test = (labels == SPLIT_FINAL_TEST) & known_time & ~before_final

    result = np.select(
        [keep_development, keep_validation, keep_final_test],
        [SPLIT_DEVELOPMENT, SPLIT_VALIDATION, SPLIT_FINAL_TEST],
        default=SPLIT_EMBARGOED,
    )
    return pd.Series(
        pd.Categorical(result, categories=[*SPLIT_ORDER, SPLIT_EMBARGOED]),
        index=split_labels.index,
        name="phase6_split",
    )


def assign_regime(flight_dates: pd.Series) -> pd.Series:
    """Map the local flight date to an operating regime.

    2019 pre-COVID reference, 2020 shock, 2021-2022 recovery/transition,
    2023 partial-year final period. Anything else is labelled ``other``.
    """
    years = pd.to_datetime(flight_dates).dt.year.to_numpy()
    result = np.select(
        [years == 2019, years == 2020, (years == 2021) | (years == 2022), years == 2023],
        [REGIME_2019, REGIME_2020, REGIME_2021_2022, REGIME_2023],
        default=REGIME_OTHER,
    )
    categories = [REGIME_2019, REGIME_2020, REGIME_2021_2022, REGIME_2023, REGIME_OTHER]
    return pd.Series(pd.Categorical(result, categories=categories), index=flight_dates.index, name="regime")


def to_boolean(series: pd.Series) -> pd.Series:
    """Convert a 0/1, True/False or float 0.0/1.0 column to plain bool.

    Missing values become False. Used for flags such as ``completed_flight``.
    """
    return series.astype("float64").fillna(0).astype("int64").astype(bool)


def chronology_report(split_labels: pd.Series, prediction_timestamp_utc: pd.Series) -> pd.DataFrame:
    """Per-split min/max prediction time and whether each split ends before the next starts.

    Only the three retained splits are considered, in development, validation,
    final-test order. The column ``ends_before_next_starts`` is True for the
    last split by convention. Use ``chronology_ok`` for a single verdict.
    """
    frame = pd.DataFrame({"split": split_labels.astype(str), "ts": prediction_timestamp_utc})
    rows = []
    for name in SPLIT_ORDER:
        subset = frame.loc[frame["split"] == name, "ts"]
        rows.append({"split": name, "rows": len(subset), "min_utc": subset.min(), "max_utc": subset.max()})
    report = pd.DataFrame(rows).set_index("split")
    next_min = report["min_utc"].shift(-1)
    report["ends_before_next_starts"] = (report["max_utc"] < next_min) | next_min.isna()
    return report


def chronology_ok(split_labels: pd.Series, prediction_timestamp_utc: pd.Series) -> bool:
    """True when development < validation < final test in UTC prediction time."""
    report = chronology_report(split_labels, prediction_timestamp_utc)
    return bool(report["ends_before_next_starts"].all())