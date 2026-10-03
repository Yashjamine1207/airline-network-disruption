"""Assemble the neural-network inputs from the loaded rows (no file access, no TensorFlow).

Phase 6, Step 7. ``scripts/build_phase6_deep_inputs.py`` loads the rows and calls
``build_deep_inputs``; keeping the logic here lets the tests run it on small tables.

What this function guarantees
-----------------------------
* Only fit, early-stop and validation rows are accepted. A final-test row raises.
* The vocabularies, medians, scaling and the sequence scaler are learned from the FIT rows only.
* Every row's sequence index (airport row and newest closed bin) is checked three ways:
    - the airport row equals the row of the flight's own origin code,
    - the stored ``end_bin`` equals ``end_bins(prediction time)`` recomputed here,
    - no sequence includes a bin that closes after the prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airline_disruption.deep import inputs as di
from airline_disruption.features.feature_sets import ID_COLUMN
from airline_disruption.sequences import airport_bins as ab
from airline_disruption.sequences import lookback as lb

ALLOWED_ROLES = tuple(di.ROLE_CODES)  # fit, early_stop, validation


def _positions(index_ids: pd.Series, row_ids: pd.Series) -> np.ndarray:
    """Position in the sequence index of every row id. A missing id or a duplicated index id raises."""
    if index_ids.duplicated().any():
        raise ValueError("The sequence index has duplicate source_row_number values")
    positions = pd.Index(index_ids).get_indexer(row_ids)
    if (positions < 0).any():
        raise ValueError(f"{int((positions < 0).sum()):,} rows have no entry in the sequence index")
    return positions


def build_deep_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    roles: pd.Series,
    index: pd.DataFrame,
    table: np.ndarray,
    airports: np.ndarray,
    grid: ab.BinGrid,
    scaler_sample: int = 200_000,
    seed: int = 42,
    chunk: int = 500_000,
) -> tuple[di.DeepInputs, pd.DataFrame]:
    """Return ``(inputs, unseen_report)``.

    ``X`` holds the ``base_no_year`` columns, ``meta`` holds ``source_row_number`` and
    ``prediction_timestamp_utc``, ``roles`` holds the training role of each row, and ``index``
    is the Step 5 sequence index (``source_row_number, origin_index, end_bin``). All of ``X``,
    ``y``, ``meta`` and ``roles`` must be in the same row order (time order).
    """
    if not (len(X) == len(y) == len(meta) == len(roles)):
        raise ValueError("X, y, meta and roles must have the same number of rows")
    roles = roles.reset_index(drop=True)
    unexpected = sorted(set(roles.astype(str)) - set(ALLOWED_ROLES))
    if unexpected:
        raise ValueError(f"Only {ALLOWED_ROLES} rows may be built here; found {unexpected} (the final test stays locked)")
    fit_mask = (roles == "fit").to_numpy()
    if not fit_mask.any():
        raise ValueError("There are no fit rows to learn the vocabularies and scalers from")

    # Static inputs -------------------------------------------------------
    spec = di.fit_static_spec(X.loc[fit_mask])
    codes = np.empty((len(X), len(di.CATEGORICAL_COLUMNS)), dtype="int32")
    numeric = np.empty((len(X), len(di.NUMERIC_NAMES)), dtype="float32")
    for lo in range(0, len(X), chunk):
        hi = min(lo + chunk, len(X))
        codes[lo:hi], numeric[lo:hi] = di.transform_static(X.iloc[lo:hi], spec)

    # Sequence index, checked three ways ------------------------------------
    positions = _positions(index[ID_COLUMN], meta[ID_COLUMN])
    airport_index = index["origin_index"].to_numpy()[positions]
    end_bin = index["end_bin"].to_numpy()[positions]

    own_origin = ab.codes_to_index(X["origin_airport"], airports)
    if not np.array_equal(own_origin.astype("int64"), airport_index.astype("int64")):
        raise ValueError("The sequence index points at a different origin airport than the flight's own origin")

    prediction_seconds = ab.to_epoch_seconds(meta["prediction_timestamp_utc"])
    if not np.array_equal(lb.end_bins(prediction_seconds, grid).astype("int64"), end_bin.astype("int64")):
        raise ValueError("The stored end_bin does not match the prediction time recomputed from the grid")
    lb.assert_no_bin_after_prediction(end_bin, prediction_seconds, grid)

    # Sequence scaler from a sample of FIT rows -------------------------------
    rng = np.random.default_rng(seed)
    fit_positions = np.flatnonzero(fit_mask)
    sample = np.sort(rng.choice(fit_positions, size=min(scaler_sample, len(fit_positions)), replace=False))
    values, mask = lb.gather_sequences(table, airport_index[sample], end_bin[sample], grid)
    scaler = di.fit_sequence_scaler(values, mask)

    role_code = roles.map(di.ROLE_CODES).to_numpy(dtype="int8")
    inputs = di.DeepInputs(
        codes=codes,
        numeric=numeric,
        label=np.asarray(y, dtype="int8"),
        role=role_code,
        airport_index=airport_index.astype("int16"),
        end_bin=end_bin.astype("int32"),
        source_row_number=meta[ID_COLUMN].to_numpy(dtype="int64"),
        prediction_seconds=prediction_seconds.astype("int64"),
        spec=spec,
        scaler=scaler,
    )
    _check_ranges(inputs)
    return inputs, unseen_report(inputs)


def _check_ranges(inputs: di.DeepInputs) -> None:
    sizes = inputs.spec.vocabulary_sizes()
    for i, column in enumerate(di.CATEGORICAL_COLUMNS):
        if inputs.codes[:, i].min() < 0 or inputs.codes[:, i].max() >= sizes[column]:
            raise ValueError(f"Codes for {column} fall outside the vocabulary")
    if not np.isfinite(inputs.numeric).all():
        raise ValueError("Static numeric inputs contain NaN or infinite values")
    if not set(np.unique(inputs.label)) <= {0, 1}:
        raise ValueError("Labels must be 0 or 1")


def unseen_report(inputs: di.DeepInputs) -> pd.DataFrame:
    """Share of rows per role whose level maps to 'unknown' (code 0), for each categorical column."""
    rows = []
    for role, code in di.ROLE_CODES.items():
        selected = inputs.role == code
        if not selected.any():
            continue
        row = {"role": role, "rows": int(selected.sum())}
        for i, column in enumerate(di.CATEGORICAL_COLUMNS):
            row[f"unknown_share_{column}"] = float((inputs.codes[selected, i] == di.UNKNOWN_CODE).mean())
        rows.append(row)
    return pd.DataFrame(rows)
