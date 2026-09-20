# Metric Definitions

## Purpose

This document defines the metrics used to evaluate classification, regression, recovery, propagation, network, calibration, and prioritisation analyses.

Metrics are selected to match the project’s rare disruption targets, time-dependent validation design, and retrospective analytical scope.

## Classification metrics

Classification is used for:

- Severe arrival-delay prediction.
- Cancellation prediction.
- Potential propagation-related severe-delay and cancellation analyses.

Because severe delays and cancellations are expected to be relatively rare, accuracy is not a headline metric.

A model can obtain high accuracy by predicting the majority non-disruption class while failing to identify the events that matter.

## Primary classification metric: PR-AUC

The primary classification metric is **Precision-Recall Area Under the Curve (PR-AUC)**.

PR-AUC measures the trade-off between:

- Precision: among flights predicted as disrupted, the proportion that were actually disrupted.
- Recall: among actual disrupted flights, the proportion correctly identified.

PR-AUC is appropriate when the positive event is uncommon.

The project will compare PR-AUC against the positive-event prevalence baseline for each target and evaluation period.

## Supporting classification metrics

### Precision

\[
\text{Precision} =
\frac{TP}{TP + FP}
\]

Precision answers:

```text
Of the flights flagged as high risk, how many were actually disruption events?
```

### Recall

\[
\text{Recall} =
\frac{TP}{TP + FN}
\]

Recall answers:

```text
Of all actual disruption events, how many did the model identify?
```

### F1 score

\[
F1 =
2 \times
\frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
\]

F1 is a balanced summary of precision and recall at a selected classification threshold.

### ROC-AUC

Receiver Operating Characteristic Area Under the Curve is reported as secondary context.

ROC-AUC must not be used as the headline metric for rare severe-delay or cancellation targets because it can appear favourable even when precision is low.

### Confusion matrix

At each documented probability threshold, report:

| Actual / predicted | Predicted non-event | Predicted event |
|---|---:|---:|
| Actual non-event | True negatives | False positives |
| Actual event | False negatives | True positives |

The threshold must be selected using validation data only, not the final temporal test set.

## Ranked-alert metrics

The project treats high-risk predictions as a ranking and prioritisation exercise, not an autonomous airline decision system.

For analyst-review capacities of the top 0.5%, 1%, 5%, and 10% of ranked flights, report:

### Precision at K

```text
Precision@K = disruption events within the top K ranked flights / K
```

### Recall at K

```text
Recall@K = disruption events within the top K ranked flights / all disruption events
```

### Events captured

```text
EventsCaptured@K = number of actual disruption events within the top K ranked flights
```

### False alerts

```text
FalseAlerts@K = number of non-events within the top K ranked flights
```

All capacity values and utility assumptions are illustrative analytical scenarios, not airline staffing plans or operating policies.

## Probability calibration metrics

Predicted probabilities must be evaluated for calibration.

A model is well calibrated when flights predicted at approximately 20% risk experience the event at approximately 20% frequency over comparable groups.

### Brier score

\[
\text{Brier Score} =
\frac{1}{N}
\sum_{i=1}^{N}
(p_i - y_i)^2
\]

Where:

- \(p_i\) is the predicted event probability.
- \(y_i\) is the observed binary outcome.

Lower Brier scores are better.

### Reliability diagram

A reliability diagram compares predicted probability groups with observed event frequencies.

The project will report calibration by meaningful operating regime where adequate data are available.

## Regression metrics

Regression is used for final arrival delay in minutes among the documented completed-flight cohort.

## Primary regression metric: MAE

The primary regression metric is **Mean Absolute Error (MAE)**.

\[
\text{MAE} =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i - \hat{y}_i|
\]

MAE is reported in minutes.

It answers:

```text
On average, how many minutes away from the final arrival delay was the prediction?
```

### Supporting regression metrics

#### Root Mean Squared Error

\[
\text{RMSE} =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i - \hat{y}_i)^2
}
\]

RMSE gives greater weight to very large errors.

#### Median Absolute Error

The median of the absolute prediction errors.

This provides a robust view of typical error when arrival-delay distributions have heavy tails.

#### Bias

\[
\text{Bias} =
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i - y_i)
\]

Positive bias indicates systematic overprediction.

Negative bias indicates systematic underprediction.

#### Upper-tail error

The project will report regression error for high-delay subsets, including flights with severe delays, where sufficient data are available.

This checks whether a model that performs well on typical flights still fails on the disruption cases of interest.

## Recovery and survival metrics

Recovery analysis will model time from a documented disruption-episode start to recovery, subject to documented censoring rules.

### Kaplan-Meier curve

A Kaplan-Meier curve estimates the probability that an episode has not yet recovered over time.

### Median recovery time

The time at which the estimated probability of remaining unrecovered reaches 50%, where estimable.

### Concordance index

Where an appropriate survival model is fitted, the concordance index measures how well predicted risk ordering agrees with observed recovery ordering.

### Survival calibration

Where supported by the final survival-model design, the project may report survival calibration and integrated Brier score.

All survival results must state:

- Episode definition.
- Recovery definition.
- Censoring rule.
- Eligible cohort.
- Assumptions and limitations.

## Propagation metrics

Propagation analysis is descriptive and restricted to validated aircraft-rotation sequences.

The project will report:

- Prior-leg delay.
- Current-leg delay.
- Delay amplification.
- Severe-delay propagation probability.
- Cancellation propagation probability.
- Cascade depth.
- Cascade duration.
- Cumulative rotation delay.
- Eligible-rotation coverage.

No single propagation or network “accuracy” metric will be invented.

## Network metrics

Airport-route network analysis is descriptive.

Candidate measures include:

- Degree.
- Weighted degree.
- Betweenness centrality.
- Closeness centrality.
- PageRank.
- Route diversity.
- Route concentration.
- Network density.
- Airport and route exposure.
- Downstream disruption indicators.
- Network recovery or resilience measures.

Network measures must be constructed from documented past-only windows when used as predictive features.

## Reporting rules

For every headline experiment, report:

- Target prevalence or cohort size.
- Chronological train, validation, and test periods.
- Primary metric.
- Supporting metrics.
- Confidence intervals or uncertainty estimates where feasible.
- Performance by operating regime.
- Calibration results for classification models.
- Error and subgroup analysis where sample size supports it.
- Important data-quality and coverage limitations.

Metrics must not be used to imply production readiness, causal effects, airline financial outcomes, or real-time operational value.