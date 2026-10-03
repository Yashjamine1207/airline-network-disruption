"""Figures for the Phase 7A reports (matplotlib, Agg backend). Visual checks only; the numbers come from tables.

Every function returns True when a file was written and False when matplotlib is missing, so a run does not
fail just because a plotting library is absent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("  matplotlib is not installed, so figures were skipped.")
        return None


def shap_beeswarm(path: Path, contrib: pd.DataFrame, values: pd.DataFrame, numeric_columns: set[str], importance: pd.DataFrame,
                  max_features: int = 14, sample: int = 4000, seed: int = 42, subtitle: str = "") -> bool:
    """One row per feature (largest mean |SHAP| at the top), one dot per flight, coloured by the feature's value.

    Categorical features (airports, route, carrier) have no low or high, so their dots are grey.
    """
    plt = _plt()
    if plt is None:
        return False
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(contrib), size=min(sample, len(contrib)), replace=False))
    features = list(importance["feature"].iloc[:max_features])
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(features) + 2.2))
    for row, feature in enumerate(reversed(features)):
        x = contrib[feature].to_numpy()[take]
        y = row + rng.uniform(-0.32, 0.32, len(x))
        if feature in numeric_columns:
            v = values[feature].iloc[take].astype("float64")
            color = v.rank(pct=True).fillna(0.5).to_numpy()
            ax.scatter(x, y, c=color, cmap="coolwarm", s=5, alpha=0.6, vmin=0, vmax=1, linewidths=0)
        else:
            ax.scatter(x, y, color="#8a8a8a", s=5, alpha=0.5, linewidths=0)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(list(reversed(features)))
    ax.set_xlabel("SHAP value: change in the model's log-odds of severe delay (right = higher predicted risk)")
    ax.set_title("What the tuned LightGBM leans on (association with the prediction, not a cause)\n" + subtitle, fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low value", "high value"])
    cbar.set_label("numeric features only; grey dots are categorical", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def local_explanations(path: Path, panels: list[dict], top: int = 8) -> bool:
    """A grid of horizontal bar charts, one per case. Each panel dict: title, contributions [(feature, value, shap)]."""
    plt = _plt()
    if plt is None or not panels:
        return False
    columns = 2
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.6 * rows), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        items = panel["contributions"][:top][::-1]
        labels = [f"{feature} = {value}" for feature, value, _ in items]
        heights = [shap for _, _, shap in items]
        ax.barh(range(len(items)), heights, color=["#c0392b" if h > 0 else "#2471a3" for h in heights])
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_title("\n".join(textwrap.fill(line, 62) for line in panel["title"].split("\n")), fontsize=8.8)
        ax.set_xlabel("SHAP value (log-odds)", fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="x", alpha=0.25)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Local explanations: the typical (median-score) case of each category. Red raises predicted risk, blue lowers it.", fontsize=10.5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def pr_curves(path: Path, y: np.ndarray, scores: dict[str, np.ndarray], subtitle: str = "",
              title: str = "Precision-recall, severe delay (>= 120 min)") -> bool:
    """Precision-recall curves on one set of rows, with the no-skill line at the event rate."""
    plt = _plt()
    if plt is None:
        return False
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(8, 5.6))
    body_precision = 0.0
    for name, s in scores.items():
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, lw=1.5, label=f"{name} (PR-AUC {average_precision_score(y, s):.4f})")
        if (recall >= 0.02).any():
            body_precision = max(body_precision, float(precision[recall >= 0.02].max()))
    ax.axhline(float(np.mean(y)), color="#999999", ls="--", lw=1, label=f"No skill (event rate {np.mean(y):.4f})")
    # The first handful of top-ranked alerts can sit at 100% precision and squash everything else, so the
    # axis is cropped to twice the best precision beyond 2% recall. The label says so.
    top = min(1.0, max(2.0 * body_precision, 0.1))
    cropped = top < 1.0
    ax.set(xlim=(0, 1), ylim=(0, top), xlabel="Recall",
           ylabel="Precision" + (" (axis cropped; the very first alerts can reach 100%)" if cropped else ""),
           title=title + "\n" + subtitle)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def training_curves(path: Path, histories: dict[str, pd.DataFrame]) -> bool:
    """PR-AUC by epoch on the fit rows and on the early-stopping rows, one panel per network."""
    plt = _plt()
    if plt is None or not histories:
        return False
    columns = 2
    rows = int(np.ceil(len(histories) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.6 * rows), squeeze=False)
    for ax, (name, history) in zip(axes.ravel(), histories.items()):
        ax.plot(history["epoch"], history["pr_auc"], marker="o", color="#2471a3", label="fit rows")
        ax.plot(history["epoch"], history["val_pr_auc"], marker="s", color="#c0392b", label="early-stopping rows")
        best = int(history["val_pr_auc"].idxmax())
        ax.axvline(history.loc[best, "epoch"], color="#999999", ls=":", lw=1)
        ax.set(title=f"{name}: best epoch {int(history.loc[best, 'epoch'])}", xlabel="epoch", ylabel="PR-AUC")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(histories):]:
        ax.axis("off")
    fig.suptitle("Training curves (seed 42, default configuration). Rising on the fit rows while falling on later rows means overfitting.", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def review_figure(path: Path, capacity: pd.DataFrame, utility: pd.DataFrame, target_label: str, focus_year: int = 2022) -> bool:
    """Two panels for the review scenarios: recall against review capacity, and the illustrative penalty table.

    ``capacity`` has one row per year, scheme and capacity (columns eval_year, scheme, fraction, recall, break_even_ratio).
    ``utility`` has the penalty table for the focus year and the whole-year scheme (miss_to_false_alert_ratio,
    net_reduction_units, fraction, capacity, break_even_ratio). The right panel is an assumption-driven sensitivity
    picture, not a measurement.
    """
    plt = _plt()
    if plt is None or capacity.empty:
        return False
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))
    fractions = sorted(capacity["fraction"].unique())
    left.plot([f * 100 for f in fractions], [f for f in fractions], color="#999999", ls=":", lw=1.2, label="random review")
    styles = {"whole_year": "-", "by_month": "-.", "by_day": "--"}
    for year in sorted(capacity["eval_year"].unique()):
        for scheme in ("whole_year", "by_day"):
            part = capacity[(capacity["eval_year"] == year) & (capacity["scheme"] == scheme)].sort_values("fraction")
            if part.empty:
                continue
            strong = year == focus_year
            left.plot(part["fraction"] * 100, part["recall"], styles[scheme], marker="o" if strong else None, lw=2 if strong else 1,
                      alpha=1 if strong else 0.55, label=f"{year}, {scheme.replace('_', ' ')}")
    left.set(xscale="log", xlabel="Share of flights reviewed (%)", ylabel="Share of events captured (recall)",
             title=textwrap.fill(f"{target_label}: events captured by review capacity", 52))
    left.title.set_fontsize(10)
    left.set_xticks([f * 100 for f in fractions])
    left.set_xticklabels([f"{f * 100:g}%" for f in fractions])
    left.grid(alpha=0.25)
    left.legend(fontsize=8)
    for fraction, group in utility.groupby("fraction"):
        group = group.sort_values("miss_to_false_alert_ratio")
        breakeven = float(group["break_even_ratio"].iloc[0])
        right.plot(group["miss_to_false_alert_ratio"], group["net_reduction_units"], marker="o", lw=1.6,
                   label=f"top {fraction * 100:g}% (break-even about {breakeven:.0f} to 1)")
    right.axhline(0, color="#444444", lw=0.9)
    right.set(xscale="log", xlabel="Assumed penalty of a missed event, in false alerts (illustrative)",
              ylabel="Net penalty reduction vs no review (units of one false alert)",
              title=textwrap.fill(f"Illustrative only ({focus_year}, whole-year ranking). Above zero means review pays off", 52))
    right.title.set_fontsize(10)
    right.grid(alpha=0.25)
    right.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def reliability_figure(path: Path, tables: dict[str, pd.DataFrame], title: str, subtitle: str = "") -> bool:
    """Reliability curves: mean predicted probability against observed event rate in equal-count bins.

    ``tables`` maps a label to a table with ``mean_predicted`` and ``observed_rate`` (``calibration.metrics.reliability_table``).
    The dotted diagonal is perfect calibration. Both axes are linear and share one range, so a curve below the
    diagonal reads directly as "the model predicts more events than happened".
    """
    plt = _plt()
    if plt is None or not tables:
        return False
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    top = max(float(max(t["mean_predicted"].max(), t["observed_rate"].max())) for t in tables.values()) * 1.08
    ax.plot([0, top], [0, top], color="#999999", ls=":", lw=1.2, label="perfect calibration")
    for label, table in tables.items():
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", ms=4, lw=1.4, label=label)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted probability (equal-count bins)", ylabel="Observed event rate",
           title=textwrap.fill(title, 60) + ("\n" + subtitle if subtitle else ""))
    ax.title.set_fontsize(10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
