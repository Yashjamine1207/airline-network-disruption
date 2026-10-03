"""Plot a compact Phase 4 validation model-comparison figure.

Classification panel: PR-AUC by model.
Regression panel: MAE (minutes) by model.
Saves a high-resolution PNG; does not retrain models or use 2023 data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports/tables/phase4_validation_summary.csv"
OUTPUT_PATH = ROOT / "reports/figures/phase4_model_comparison.png"

# Human-readable labels for the legend and axis ticks.
MODEL_LABELS = {
    "cancellation_logistic_unweighted": "Cancellation (Logistic)",
    "severe_delay_lightgbm_base": "Severe Delay (LightGBM)",
    "severe_delay_lightgbm_calibrated": "Severe Delay (LightGBM, calibrated)",
    "arrival_delay_ridge": "Arrival Delay (Ridge)",
}


def main() -> None:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Summary table not found: {SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)

    if summary.empty:
        raise ValueError("Summary table is empty.")

    # Classification: PR-AUC
    classification = summary.loc[
        summary["target"].isin(
            ["cancelled_target", "severe_delay_120"]
        )
        & summary["average_precision"].notna()
    ].copy()

    # Regression: MAE
    regression = summary.loc[
        summary["target"].eq("arrival_delay_minutes")
        & summary["mae_minutes"].notna()
    ].copy()

    if classification.empty or regression.empty:
        raise ValueError(
            "Both classification and regression rows are required."
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True
    )

    # Classification panel
    ax = axes[0]
    ax.barh(
        classification["experiment_key"].map(MODEL_LABELS),
        classification["average_precision"],
        color="#1f77b4",
    )
    ax.set_xlabel("PR-AUC (validation 2022)")
    ax.set_title("Classification: PR-AUC")
    ax.set_xlim(0, max(classification["average_precision"]) * 1.15)

    # Regression panel
    ax = axes[1]
    ax.barh(
        regression["experiment_key"].map(MODEL_LABELS),
        regression["mae_minutes"],
        color="#d62728",
    )
    ax.set_xlabel("MAE (minutes, validation 2022)")
    ax.set_title("Regression: MAE")
    ax.invert_yaxis()  # Keep the same order as the left panel.

    fig.suptitle(
        "Phase 4 Validation Model Comparison (2019–2021 train, 2022 validate)"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(OUTPUT_PATH),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved model-comparison figure: {OUTPUT_PATH}")
    print("No 2023 final-test results are shown.")


if __name__ == "__main__":
    main()