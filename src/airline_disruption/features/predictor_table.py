"""Memory-friendly reader for the Phase 3 predictor CSV.

Phase 6, Step 3.

The Phase 3 feature file has 3 million rows. Read in one go, the timestamp and
date columns become millions of Python strings and the whole build needs about
3 GB of RAM. This reader converts each chunk to compact types as it goes:

* ``categorical_columns``  -> pandas category, with categories unified across chunks
* ``float32_columns``      -> float32 (exact for the integers and small decimals used here)
* ``utc_columns``          -> timezone-aware UTC datetimes
* ``date_columns``         -> datetime64 (local calendar dates)

All other selected columns keep the dtype pandas infers.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandas.api.types import union_categoricals

from airline_disruption.features.predictor_checks import to_utc


def read_predictor_csv(
    path: str | Path,
    columns: Iterable[str],
    categorical_columns: Iterable[str] = (),
    float32_columns: Iterable[str] = (),
    utc_columns: Iterable[str] = ("prediction_timestamp_utc", "scheduled_departure_utc"),
    date_columns: Iterable[str] = ("FL_DATE",),
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read ``columns`` from a CSV in chunks and return one compact DataFrame."""
    columns = list(columns)
    categorical = [c for c in categorical_columns if c in columns]
    float32 = [c for c in float32_columns if c in columns]
    utc = [c for c in utc_columns if c in columns]
    dates = [c for c in date_columns if c in columns]

    pieces: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=columns,
        dtype={c: "category" for c in categorical},
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        for column in utc:
            chunk[column] = to_utc(chunk[column])
        for column in dates:
            chunk[column] = pd.to_datetime(chunk[column])
        if float32:
            chunk[float32] = chunk[float32].astype("float32")
        pieces.append(chunk)

    if not pieces:
        raise ValueError(f"No rows read from {path}")

    # Categories differ between chunks, and a plain concat would turn them into
    # strings. Unify them first, take the category columns out, concat the rest,
    # then put the unified columns back.
    unified = {c: union_categoricals([piece[c].array for piece in pieces]) for c in categorical}
    for piece in pieces:
        piece.drop(columns=categorical, inplace=True)
    frame = pd.concat(pieces, ignore_index=True)
    del pieces  # release the chunks before the caller does more work
    for column in categorical:
        frame[column] = unified[column]

    return frame[columns]
