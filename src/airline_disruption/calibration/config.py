"""Loading and checking ``configs/calibration.yaml``.

Phase 7A, Step 9. The file is read with PyYAML. Loading fails, with a message that says what is
wrong, if a required key is missing, if a method is unknown, or if the file tries to allow the
final test. The final test stays locked no matter what the file says.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from airline_disruption.calibration.methods import METHODS

REQUIRED = {
    "model": ("feature_set", "params_file", "seed", "reference_predictions", "reference_column"),
    "calibration": ("primary_fit_rows", "diagnostic_fit_rows"),
    "evaluation": ("selection", "confirmation"),
    "metrics": ("reliability_bins",),
    "bootstrap": ("resamples", "seed", "confidence"),
}


def load_calibration_config(path: str | Path) -> dict:
    """Read and validate the calibration settings. Raises ValueError for anything unexpected."""
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as problem:
        raise ValueError(f"Cannot read {path}: {problem}") from problem
    except yaml.YAMLError as problem:
        raise ValueError(f"{path} is not valid YAML: {problem}") from problem
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")

    for section, keys in REQUIRED.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{path}: missing section {section!r}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"{path}: missing {section}.{key}")

    methods = config.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError(f"{path}: 'methods' must be a non-empty list")
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"{path}: unknown methods {unknown}; choose from {METHODS}")
    if "uncalibrated" not in methods:
        raise ValueError(f"{path}: 'uncalibrated' must be listed, it is the reference")
    if sorted(methods) != sorted(METHODS):
        raise ValueError(f"{path}: 'methods' must list exactly {list(METHODS)} (the decision rules compare all three)")

    if config.get("final_test", {}).get("allowed", False) is not False:
        raise ValueError(f"{path}: final_test.allowed must be false. The final test is locked (DEC-015).")

    bootstrap = config["bootstrap"]
    if int(bootstrap["resamples"]) < 0 or not 0.5 < float(bootstrap["confidence"]) < 1.0:
        raise ValueError(f"{path}: bootstrap.resamples must be >= 0 and bootstrap.confidence between 0.5 and 1")
    if int(config["metrics"]["reliability_bins"]) < 2:
        raise ValueError(f"{path}: metrics.reliability_bins must be at least 2")
    return config
