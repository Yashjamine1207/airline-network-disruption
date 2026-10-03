"""Model inputs for the Phase 6 neural networks.

Phase 6, Step 7. Input version ``phase6_deep_inputs_v1``.

Two kinds of input
------------------
STATIC   the ``base_no_year`` schedule features of the predicted flight (DEC-018):
         four categorical columns turned into integer codes for embeddings, and the
         eight numeric columns turned into ten numbers (see ``NUMERIC_NAMES``).
SEQUENCE the 48-hour airport schedule lookback (DEC-017), read per batch from the
         airport table and scaled with ``SequenceScaler``.

Everything that is LEARNED from data (category vocabularies, medians, means and
standard deviations) is learned from the FIT rows only and stored in a spec. The
early-stopping rows, the 2022 validation rows and, later, the final-test rows are
only ever transformed with that spec.

Categories
----------
A level seen in fewer than ``MIN_LEVEL_COUNT`` fit rows, or never seen in the fit
rows, gets code 0 ("unknown"). A model therefore never learns an embedding for a
level it has hardly seen, and an unseen airport or route is scored like any other
rare level. LightGBM gets the same treatment through ``apply_fit_categories``.

Numbers
-------
* Month, day of week, and the scheduled departure and arrival clock times are
  written as sine and cosine of their position in the cycle, so 23:59 sits next to
  00:00 and December next to January.
* Distance and scheduled elapsed time are ``log1p`` transformed, then standardised.
* A missing number is replaced by its fit-row median before any transform.
* The calendar year is not an input (DEC-018).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from airline_disruption.features.feature_sets import BASE_CATEGORICAL_COLUMNS, BASE_NO_YEAR_COLUMNS

INPUT_VERSION = "phase6_deep_inputs_v1"

CATEGORICAL_COLUMNS = BASE_CATEGORICAL_COLUMNS  # origin_airport, destination_airport, route, carrier_identifier
MIN_LEVEL_COUNT = 20
UNKNOWN_CODE = 0

MONTH = "scheduled_departure_month"
DAY_OF_WEEK = "scheduled_departure_day_of_week"
DEP_HOUR, DEP_MINUTE = "scheduled_departure_hour_local", "scheduled_departure_minute_local"
ARR_HOUR, ARR_MINUTE = "scheduled_arrival_hour_local", "scheduled_arrival_minute_local"
ELAPSED = "scheduled_elapsed_time_minutes"
DISTANCE = "distance_miles"

NUMERIC_SOURCE_COLUMNS = (MONTH, DAY_OF_WEEK, DEP_HOUR, DEP_MINUTE, ARR_HOUR, ARR_MINUTE, ELAPSED, DISTANCE)
NUMERIC_NAMES = (
    "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
    "departure_time_sin", "departure_time_cos",
    "arrival_time_sin", "arrival_time_cos",
    "log_distance_z", "log_elapsed_z",
)

ROLE_CODES = {"fit": 0, "early_stop": 1, "validation": 2}
CHANNEL_COUNT = 3  # the airport table has three schedule channels


# ---------------------------------------------------------------------------
# Static features
# ---------------------------------------------------------------------------
@dataclass
class StaticSpec:
    """Everything learned from the fit rows for the static inputs."""

    vocabularies: dict[str, list[str]]  # column -> levels; the code of levels[i] is i + 1
    medians: dict[str, float]  # numeric source column -> fit median
    log_mean: dict[str, float]  # "distance_miles" / "scheduled_elapsed_time_minutes" -> mean of log1p
    log_std: dict[str, float]
    min_level_count: int = MIN_LEVEL_COUNT

    def to_json(self) -> dict:
        return {
            "input_version": INPUT_VERSION,
            "vocabularies": self.vocabularies,
            "medians": self.medians,
            "log_mean": self.log_mean,
            "log_std": self.log_std,
            "min_level_count": self.min_level_count,
        }

    @classmethod
    def from_json(cls, content: dict) -> StaticSpec:
        return cls(content["vocabularies"], content["medians"], content["log_mean"], content["log_std"], content["min_level_count"])

    def vocabulary_sizes(self) -> dict[str, int]:
        """Number of embedding rows per column: the known levels plus one for 'unknown'."""
        return {column: len(levels) + 1 for column, levels in self.vocabularies.items()}


def embedding_dim(n_levels: int) -> int:
    """Embedding width: ``1.6 * n^0.56`` (a common rule of thumb), between 4 and 32."""
    return int(min(32, max(4, round(1.6 * n_levels**0.56))))


def fit_static_spec(X_fit: pd.DataFrame, min_level_count: int = MIN_LEVEL_COUNT) -> StaticSpec:
    """Learn vocabularies, medians and scaling from the FIT rows. Never call this on other rows."""
    missing = [c for c in (*CATEGORICAL_COLUMNS, *NUMERIC_SOURCE_COLUMNS) if c not in X_fit.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    vocabularies = {}
    for column in CATEGORICAL_COLUMNS:
        counts = X_fit[column].astype(str).value_counts()
        counts = counts[counts >= min_level_count]
        # Most frequent first, ties by name, so the coding is the same on every run.
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        vocabularies[column] = [level for level, _ in ordered]

    medians = {c: float(X_fit[c].median()) for c in NUMERIC_SOURCE_COLUMNS}
    log_mean, log_std = {}, {}
    for column in (DISTANCE, ELAPSED):
        logged = np.log1p(X_fit[column].fillna(medians[column]).to_numpy(dtype="float64"))
        log_mean[column] = float(logged.mean())
        log_std[column] = float(max(logged.std(), 1e-6))
    return StaticSpec(vocabularies, medians, log_mean, log_std, min_level_count)


def encode_categories(column: pd.Series, levels: list[str]) -> np.ndarray:
    """Integer code of each row: 1..len(levels) for a known level, 0 for anything else."""
    lookup = {level: i + 1 for i, level in enumerate(levels)}
    if isinstance(column.dtype, pd.CategoricalDtype):
        # Map the (few) categories once, then index by the integer codes.
        table = np.array([lookup.get(str(c), UNKNOWN_CODE) for c in column.cat.categories] + [UNKNOWN_CODE], dtype="int32")
        return table[column.cat.codes.to_numpy()]  # code -1 (missing) indexes the last entry: unknown
    return column.astype(str).map(lookup).fillna(UNKNOWN_CODE).to_numpy(dtype="int32")


def transform_static(X: pd.DataFrame, spec: StaticSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(codes, numeric)``: int32 ``(n, 4)`` and float32 ``(n, 10)``.

    Uses only the values stored in ``spec``. Missing numbers are replaced by the fit median.
    """
    codes = np.stack([encode_categories(X[column], spec.vocabularies[column]) for column in CATEGORICAL_COLUMNS], axis=1)

    values = {c: X[c].fillna(spec.medians[c]).to_numpy(dtype="float64") for c in NUMERIC_SOURCE_COLUMNS}

    def cycle(position: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
        angle = 2.0 * math.pi * position / period
        return np.sin(angle), np.cos(angle)

    month_sin, month_cos = cycle(values[MONTH] - 1.0, 12.0)
    dow_sin, dow_cos = cycle(values[DAY_OF_WEEK], 7.0)
    dep_sin, dep_cos = cycle(values[DEP_HOUR] + values[DEP_MINUTE] / 60.0, 24.0)
    arr_sin, arr_cos = cycle(values[ARR_HOUR] + values[ARR_MINUTE] / 60.0, 24.0)
    distance_z = (np.log1p(values[DISTANCE]) - spec.log_mean[DISTANCE]) / spec.log_std[DISTANCE]
    elapsed_z = (np.log1p(values[ELAPSED]) - spec.log_mean[ELAPSED]) / spec.log_std[ELAPSED]

    numeric = np.stack(
        [month_sin, month_cos, dow_sin, dow_cos, dep_sin, dep_cos, arr_sin, arr_cos, distance_z, elapsed_z], axis=1
    ).astype("float32")
    if not np.isfinite(numeric).all():
        raise ValueError("Static numeric features contain NaN or infinite values after transformation")
    return codes, numeric


# ---------------------------------------------------------------------------
# Sequence scaling
# ---------------------------------------------------------------------------
@dataclass
class SequenceScaler:
    """``log1p`` then standardise each channel. Mean and std come from REAL positions of fit rows."""

    mean: list[float]
    std: list[float]

    def to_json(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_json(cls, content: dict) -> SequenceScaler:
        return cls(content["mean"], content["std"])

    def apply(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Scale ``(n, L, C)`` values. Padded positions (mask False) are set to exactly 0."""
        scaled = (np.log1p(values) - np.asarray(self.mean, dtype="float32")) / np.asarray(self.std, dtype="float32")
        return np.where(mask[:, :, None], scaled, 0.0).astype("float32")


def fit_sequence_scaler(values: np.ndarray, mask: np.ndarray) -> SequenceScaler:
    """Mean and std of ``log1p(count)`` per channel over the real positions only."""
    if not mask.any():
        raise ValueError("No real positions to fit the scaler on")
    logged = np.log1p(values[mask].astype("float64"))  # (real positions, channels)
    return SequenceScaler(
        mean=[float(m) for m in logged.mean(axis=0)],
        std=[float(max(s, 1e-6)) for s in logged.std(axis=0)],
    )


# ---------------------------------------------------------------------------
# Saved inputs
# ---------------------------------------------------------------------------
ARRAY_NAMES = ("codes", "numeric", "label", "role", "airport_index", "end_bin", "source_row_number", "prediction_seconds")


@dataclass
class DeepInputs:
    """Per-row arrays for the fit, early-stop and validation rows, in time order."""

    codes: np.ndarray  # int32 (n, 4)
    numeric: np.ndarray  # float32 (n, 10)
    label: np.ndarray  # int8 (n,)
    role: np.ndarray  # int8 (n,) values of ROLE_CODES
    airport_index: np.ndarray  # int16 (n,) row of the airport table for the sequence entity
    end_bin: np.ndarray  # int32 (n,) newest closed bin + 1
    source_row_number: np.ndarray  # int64 (n,)
    prediction_seconds: np.ndarray  # int64 (n,)
    spec: StaticSpec
    scaler: SequenceScaler
    manifest: dict = field(default_factory=dict)

    def rows(self, role: str) -> np.ndarray:
        """Positions of the rows with this role, in time order."""
        return np.flatnonzero(self.role == ROLE_CODES[role])


def save_deep_inputs(directory: str | Path, inputs: DeepInputs) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_NAMES:
        np.save(directory / f"{name}.npy", getattr(inputs, name))
    (directory / "spec.json").write_text(json.dumps(inputs.spec.to_json(), indent=1), encoding="utf-8")
    (directory / "sequence_scaler.json").write_text(json.dumps(inputs.scaler.to_json(), indent=1), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(inputs.manifest, indent=1, default=str), encoding="utf-8")


def load_deep_inputs(directory: str | Path) -> DeepInputs:
    directory = Path(directory)
    arrays = {name: np.load(directory / f"{name}.npy") for name in ARRAY_NAMES}
    n_rows = {len(a) for a in arrays.values()}
    if len(n_rows) != 1:
        raise ValueError("The saved input arrays have different lengths")
    spec = StaticSpec.from_json(json.loads((directory / "spec.json").read_text(encoding="utf-8")))
    scaler = SequenceScaler.from_json(json.loads((directory / "sequence_scaler.json").read_text(encoding="utf-8")))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return DeepInputs(spec=spec, scaler=scaler, manifest=manifest, **arrays)


def numeric_columns_used() -> tuple[str, ...]:
    """The raw columns the static features are built from (checked against the feature set in tests)."""
    return NUMERIC_SOURCE_COLUMNS


assert set(NUMERIC_SOURCE_COLUMNS) | set(CATEGORICAL_COLUMNS) == set(BASE_NO_YEAR_COLUMNS), "static inputs must equal base_no_year"
