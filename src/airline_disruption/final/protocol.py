"""Constants shared by the lock script and the final-test script (Phase 7B, Step 12)."""

from __future__ import annotations

TARGETS = {
    "severe_delay_120": {
        "eligibility": "eligible_severe_delay",
        "label": "Severe delay (arrival delay of 120 minutes or more)",
        "role": "selected model (settings tuned for this target, DEC-019)",
        "calibrate": True,
    },
    "cancelled": {
        "eligibility": "eligible_cancellation",
        "label": "Cancellation",
        "role": "untuned baseline (the severe-delay settings reused; ranking only, no calibration, no thresholds)",
        "calibrate": False,
    },
}

PARAMS_FILE = "configs/lightgbm_benchmark_selected.json"
CALIBRATION_CONFIG = "configs/calibration.yaml"
CALIBRATION_SELECTED = "configs/calibration_selected.json"
REFERENCE_PATH = "data/processed/phase6/predictions/lightgbm_validation_phase6_tuned_ladder.parquet"
REFERENCE_COLUMN = "lightgbm_base_no_year"
REPRODUCTION_TOLERANCE = 0.0005  # against the stored Phase 6 scores (as in Steps 9 to 11)

CAPACITIES = (0.005, 0.01, 0.05, 0.10)
BOOTSTRAP = {"unit": "whole UTC prediction days", "resamples": 1000, "seed": 42, "confidence": 0.95}
HISTORY_BASELINES = ("baseline_carrier_rate", "baseline_origin_rate", "baseline_destination_rate", "baseline_route_rate")

REJECTED_CHALLENGERS = {
    "lstm": "did not beat the MLP control that never sees the lookback (DEC-020)",
    "gru": "no clear gain over the LSTM or the MLP control (DEC-020)",
    "transformer": "whole interval below zero against both the MLP control and tuned LightGBM, overfits from epoch 1 (DEC-022)",
    "mlp_control": "not better than tuned LightGBM on validation",
}
NOT_SCORED_ON_FINAL_TEST = "Rejected challengers are not scored on 2023. They were rejected on 2022, and scoring them on the final test would turn it into a second selection round."

CLAIMS = {
    "C1_skill_over_chance": "The lower end of the 95% day-bootstrap interval of PR-AUC lift (PR-AUC divided by the 2023 event rate) is above 1.0.",
    "C2_beats_history": ("The whole 95% paired day-bootstrap interval of PR-AUC(LightGBM) minus PR-AUC(best_history_baseline) is above 0. "
                         "The baseline is the one of carrier, origin, destination and route rate with the highest 2022 PR-AUC, fixed in this lock."),
    "C3_ranking_not_only_seasonal": ("The lower end of the 95% day-bootstrap interval of Precision@10% with a fixed review capacity inside every day "
                                     "is above the 2023 event rate."),
}
CLAIM_NOTE = "These are the only pass/fail statements. Everything else in the final report is descriptive. Claims are judged the same way for both targets."
