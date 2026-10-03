"""Pre-registered tuning protocol for the LightGBM benchmark.

Phase 6, Step 6.

Why this exists
---------------
The Step 4 benchmark used Phase 4's settings and stopped after 24 to 78 trees.
If the LSTM, GRU and Transformer are later tuned and LightGBM is not, a win for
the deep model proves nothing. This module fixes, before any deep model is
trained, how LightGBM is tuned.

Protocol (written down before the results exist)
------------------------------------------------
* Search space: ``SEARCH_SPACE`` below. Nothing else is tuned.
* Budget: the default settings plus ``n_configs`` random draws (seeded).
* Fit rows and early-stopping rows are the same as in Step 4 (development only).
* SELECTION uses the first half of 2022 (prediction time before 2022-07-01 UTC).
  The configuration with the highest PR-AUC on that half is the challenger.
* CONFIRMATION uses the second half of 2022. The challenger replaces the default
  only if a paired day-level bootstrap says it beats the default there
  (95% interval for the PR-AUC difference above zero). Otherwise the default stays.
* The second half is never used to choose a configuration, so the confirmed gain
  is not inflated by picking the best of many draws.
* 2023 is never loaded.

Deep models follow the same select-on-H1, confirm-on-H2 rule, with a smaller
budget that is recorded next to their results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from airline_disruption.models.lightgbm_benchmark import LIGHTGBM_PARAMS

# Values chosen in advance. Every list contains the current default, so the
# default is always reachable.
SEARCH_SPACE: dict[str, list] = {
    "learning_rate": [0.02, 0.05],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [50, 100, 300, 1000],
    "reg_lambda": [0.0, 1.0, 10.0, 50.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.7, 1.0],  # subsample_freq is set to 1 whenever this is below 1
    # Settings that control how LightGBM splits on categories with many levels
    # (route has thousands). LightGBM's own defaults are cat_smooth=10,
    # min_data_per_group=100, cat_l2=10, max_cat_threshold=32.
    "cat_smooth": [10, 50, 200],
    "min_data_per_group": [100, 500, 2000],
    "cat_l2": [10, 50],
    "max_cat_threshold": [16, 32, 64],
}

# Settings a params dict may contain. ``random_state`` and ``n_jobs`` are deliberately absent:
# the fitting code sets them, so a settings file cannot change the seed or the thread count.
ALLOWED_KEYS = frozenset(LIGHTGBM_PARAMS) | frozenset(SEARCH_SPACE) | {"subsample_freq"}

SELECTION_END_UTC = pd.Timestamp("2022-07-01", tz="UTC")  # first instant of the confirmation half


def sample_configs(n_configs: int, seed: int = 42) -> list[dict]:
    """Override dictionaries to try. Entry 0 is ``{}``: the default settings.

    The other ``n_configs`` entries are distinct random draws from ``SEARCH_SPACE``.
    """
    if n_configs < 0:
        raise ValueError("n_configs must not be negative")
    rng = np.random.default_rng(seed)
    space_size = int(np.prod([len(values) for values in SEARCH_SPACE.values()]))
    if n_configs > space_size:
        raise ValueError(f"n_configs={n_configs} is larger than the search space ({space_size})")
    keys = list(SEARCH_SPACE)
    seen: set[tuple] = set()
    configs: list[dict] = [{}]
    while len(configs) < n_configs + 1:
        draw = {key: SEARCH_SPACE[key][int(rng.integers(len(SEARCH_SPACE[key])))] for key in keys}
        signature = tuple(draw[k] for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(draw)
    return configs


def params_for(overrides: dict) -> dict:
    """Complete LightGBM settings: the Phase 4 defaults with ``overrides`` on top.

    Raises ValueError for a key that is not allowed, so a typo cannot pass
    silently. ``LIGHTGBM_PARAMS`` itself is never modified.
    """
    bad = sorted(set(overrides) - ALLOWED_KEYS)
    if bad:
        raise ValueError(f"Unknown or reserved LightGBM settings: {bad}")
    params = {**LIGHTGBM_PARAMS, **overrides}
    if params.get("subsample", 1.0) < 1.0:
        params["subsample_freq"] = 1  # LightGBM ignores subsample unless a frequency is set
    return params


def validation_halves(prediction_time_utc: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks ``(selection, confirmation)`` for 2022 rows, split at 2022-07-01 UTC.

    Raises ValueError if any row is outside 2022, because the halves are defined
    for the validation year only.
    """
    stamps = prediction_time_utc
    outside = (stamps < pd.Timestamp("2022-01-01", tz="UTC")) | (stamps >= pd.Timestamp("2023-01-01", tz="UTC"))
    if outside.any():
        raise ValueError(f"{int(outside.sum()):,} rows are outside 2022")
    selection = (stamps < SELECTION_END_UTC).to_numpy()
    return selection, ~selection


def best_config_index(pr_auc_selection: list[float] | np.ndarray, exclude_default: bool = True) -> int:
    """Index of the highest selection-half PR-AUC. Ties go to the lower index; NaN is ignored.

    With ``exclude_default`` the default (index 0) is not a candidate: the search
    is for a CHALLENGER to the default.
    """
    scores = np.asarray(pr_auc_selection, dtype="float64").copy()
    if exclude_default:
        scores[0] = np.nan
    if np.isnan(scores).all():
        raise ValueError("No candidate configuration has a PR-AUC")
    return int(np.nanargmax(scores))


def decide_adoption(ci_low: float) -> bool:
    """Adopt the challenger only if the confirmation-half interval lies wholly above zero."""
    return bool(np.isfinite(ci_low) and ci_low > 0.0)


def load_params_file(path: str | Path) -> dict:
    """Read the ``params`` block of a selection file written by the tuning script.

    Returns a complete settings dict (without ``random_state`` and ``n_jobs``).
    Raises ValueError if the file has no ``params`` block or has keys that are not allowed.
    """
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if "params" not in content or not isinstance(content["params"], dict):
        raise ValueError(f"{path} has no 'params' block")
    bad = sorted(set(content["params"]) - ALLOWED_KEYS)
    if bad:
        raise ValueError(f"{path} has settings that are not allowed: {bad}")
    return dict(content["params"])
