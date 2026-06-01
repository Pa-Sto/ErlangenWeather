# Temperature Transition Brief

This note focuses on the days where Erlangen's daily mean temperature changed the most inside the scored prediction history.

![Temperature transition cases](visuals/temperature_transition_cases.svg)

## Reading guide

- Orange line: actual hourly temperature from `historical_data.csv`.
- Blue line: model prediction from `site/data/history/history_predictions.json`.
- Strong underreaction means the model is acting too much like persistence.

## Strongest transition days

| Date | Actual delta (C) | Pred delta (C) | Gap (C) | Comment |
|---|---:|---:|---:|---|
| 2026-05-12 | -6.13 | -0.54 | +5.59 | underreacted |
| 2026-02-21 | +5.89 | +0.17 | -5.72 | underreacted |
| 2025-11-24 | +5.44 | +1.61 | -3.83 | underreacted |
| 2026-01-14 | +5.30 | +1.48 | -3.82 | underreacted |
| 2025-12-08 | +5.04 | +1.84 | -3.20 | underreacted |
| 2026-04-04 | +4.55 | +0.10 | -4.46 | underreacted |

## Main takeaway

The current model usually gets the direction of major moves, but it compresses the amplitude. That is consistent with the conservative behavior already seen in the aggregate metrics.

## What to test next

1. Train on a more recent history window and compare these same transition days.
2. Add temperature transition weighting so large day-to-day changes matter more during training.
3. Score a dedicated jump-day benchmark alongside normal RMSE for every new experiment.