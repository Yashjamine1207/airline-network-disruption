# Historical outcome availability decision

## Prediction context

The pre-flight prediction timestamp is scheduled departure UTC minus
two hours. A historical outcome feature may be used only if the earlier
flight's outcome was available by that timestamp.

## Evidence from the audited source

The flight file provides final arrival-delay and cancellation outcomes,
but no field recording when each outcome became available to a user of
this retrospective dataset.

The project reconstructed and destination-clock-validated gate-arrival
event times for 2,459,301 development/validation flights. These are
times when flights arrived, not verified times when their records became
available. No reliable cancellation-availability timestamp exists.

## Phase 3 decision

Do not include historical severe-delay rates, historical cancellation
rates, historical mean/median arrival delay, or rolling outcome-based
disruption rates in pre-flight predictors under the current evidence.

Keep `validated_completed_arrivals.csv` for retrospective timing and
data-quality analysis only. Do not join it into
`phase3_schedule_network_features.csv`.

The approved historical predictors are previous-completed-UTC-month
scheduled-flight counts and schedule-only airport-route network
features. The target table remains separate from predictor tables.

This is a documented limitation, not evidence that historical outcome
rates would have no predictive value. Revisit them only if a defensible
outcome-availability rule and point-in-time tests are established.