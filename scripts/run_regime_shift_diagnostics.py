"""Phase 2: reproducible descriptive diagnostics for operating-regime shift."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("reports/tables/monthly_operational_disruption_summary.csv")
TABLE = Path("reports/tables/regime_shift_diagnostics.csv")
FIGURE = Path("reports/figures/regime_shift_standardised_metrics.png")
REPORT = Path("docs/data/regime_shift_analysis.md")
RANDOM_SEED = 2026
N_PERMUTATIONS = 20_000
BASELINE = "2019 pre-COVID reference"
COMPARISONS = ["2020 COVID operational shock", "2021-2023 recovery and transition"]
METRICS = ["scheduled_flights", "cancellation_rate_pct", "severe_delay_rate_pct", "mean_arrival_delay_minutes"]


def permutation_p_value(reference: np.ndarray, comparison: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided randomisation test for difference in monthly means."""
    observed = abs(comparison.mean() - reference.mean())
    pooled = np.concatenate([reference, comparison])
    n_reference = len(reference)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        difference = abs(shuffled[n_reference:].mean() - shuffled[:n_reference].mean())
        extreme += difference >= observed
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def standardised_mean_difference(reference: np.ndarray, comparison: np.ndarray) -> float:
    """Cohen's d using the pooled sample standard deviation."""
    pooled_variance = ((len(reference) - 1) * reference.var(ddof=1) + (len(comparison) - 1) * comparison.var(ddof=1)) / (len(reference) + len(comparison) - 2)
    if pooled_variance == 0:
        return np.nan
    return (comparison.mean() - reference.mean()) / np.sqrt(pooled_variance)


def main() -> None:
    for path in (TABLE, FIGURE, REPORT):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT}. Run analyse_temporal_regimes.py first.")

    monthly = pd.read_csv(INPUT, parse_dates=["month"])
    required = {"regime", *METRICS}
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    reference_data = monthly[monthly["regime"] == BASELINE]
    if len(reference_data) != 12:
        raise ValueError("2019 reference period must contain 12 monthly observations.")

    for comparison_name in COMPARISONS:
        comparison_data = monthly[monthly["regime"] == comparison_name]
        for metric in METRICS:
            reference = reference_data[metric].dropna().to_numpy(dtype=float)
            comparison = comparison_data[metric].dropna().to_numpy(dtype=float)
            relative_change = np.nan if reference.mean() == 0 else 100 * (comparison.mean() - reference.mean()) / abs(reference.mean())
            rows.append({
                "reference_regime": BASELINE,
                "comparison_regime": comparison_name,
                "metric": metric,
                "reference_months": len(reference),
                "comparison_months": len(comparison),
                "reference_monthly_mean": reference.mean(),
                "comparison_monthly_mean": comparison.mean(),
                "absolute_difference": comparison.mean() - reference.mean(),
                "relative_change_pct": relative_change,
                "cohens_d": standardised_mean_difference(reference, comparison),
                "permutation_p_value": permutation_p_value(reference, comparison, rng),
            })

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(TABLE, index=False)

    metric_labels = {
        "scheduled_flights": "Scheduled flights",
        "cancellation_rate_pct": "Cancellation rate (%)",
        "severe_delay_rate_pct": "Severe-delay rate (%)",
        "mean_arrival_delay_minutes": "Mean arrival delay (min)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, metric in zip(axes.ravel(), METRICS):
        subset = diagnostics[diagnostics["metric"] == metric]
        axis.bar(subset["comparison_regime"], subset["cohens_d"], color=["#d95f02", "#1b9e77"])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(metric_labels[metric])
        axis.set_ylabel("Standardised mean difference (Cohen's d)")
        axis.tick_params(axis="x", rotation=18)
    fig.suptitle("Observed Monthly Regime Differences Relative to 2019", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Regime Shift Analysis", "", "## Method", "",
        "This report quantifies observed differences in monthly operational metrics relative to the 2019 pre-COVID reference period.",
        "",
        "- Unit of analysis: calendar month, not individual flight.",
        "- Comparison periods: 2020 COVID operational shock and 2021-2023 recovery/transition.",
        f"- Permutation tests: {N_PERMUTATIONS:,} two-sided random permutations with random seed {RANDOM_SEED}.",
        "- Effect size: Cohen's d based on monthly values.",
        "- These diagnostics identify distributional differences; they do not establish causes.",
        "- The recovery/transition period includes more months than 2019 and ends in August 2023, so partial-year coverage remains a limitation.",
        "", "## Diagnostics", "",
        "| Comparison | Metric | 2019 monthly mean | Comparison monthly mean | Difference | Relative change | Cohen's d | Permutation p-value |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['comparison_regime']} | {row['metric']} | {row['reference_monthly_mean']:.3f} | {row['comparison_monthly_mean']:.3f} | {row['absolute_difference']:.3f} | {row['relative_change_pct']:.2f}% | {row['cohens_d']:.3f} | {row['permutation_p_value']:.5f} |"
        )
    lines.extend([
        "", "## Interpretation rules", "",
        "- A small permutation p-value means the observed difference in monthly means is unusual under the test's exchangeability assumption; it is not proof of a causal mechanism.",
        "- Cohen's d sign shows direction relative to 2019; magnitude shows separation in units of pooled monthly standard deviation.",
        "- The results justify retaining 2020 as a distribution-shift regime and reporting later model performance by regime.",
        "", "## Outputs", "", f"- Diagnostics table: `{TABLE.as_posix()}`", f"- Figure: `{FIGURE.as_posix()}`", "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Regime-shift diagnostics complete.")
    print(f"- Table: {TABLE}")
    print(f"- Report: {REPORT}")
    print(f"- Figure: {FIGURE}")


if __name__ == "__main__":
    main()