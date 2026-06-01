# Experiments — ErlangenWeather

This file tracks *planned* and *executed* model experiments in a consistent format.
Each experiment must include a baseline comparison and a clear next step.

---

## Baseline (2026-02-03)
**Evaluation command**
```bash
.venv/bin/python evaluation.py \
  --cache historical_data.csv \
  --history history_predictions.json \
  --history-multi history_predictions_multi.json \
  --min-lag-days 5
```

**Metrics**
- Temp RMSE: 3.171 °C  
- Rain RMSE: 0.055 mm/h  
- Cloud RMSE: 56.848 %

**Lead bins**
- Temp: 0–24h 2.538 | 24–48h 4.413 | 48–72h 5.117  
- Rain: 0–24h 0.053 | 24–48h 0.050 | 48–72h 0.061  
- Cloud: 0–24h 66.542 | 24–48h 57.068 | 48–72h 44.839

---

## EXP-001 — Add spatial context (3×3 grid) [Executed]
**Hypothesis**  
Adding nearby grid points (3×3, 75 km radius) will improve rain/cloud accuracy by giving spatial context of moving systems.

**Change**  
Enable multi-location grid inputs in training.

**Run**
```bash
.venv/bin/python WeatherData.py \
  --source archive \
  --download-data --days 365 \
  --extend-to-present \
  --grid --grid-n 3 --grid-radius-km 75 \
  --grid-cache-dir grid_cache \
  --loss huber \
  --tag EXP-001 --note "3x3 spatial grid"
```

**Evaluate**
```bash
.venv/bin/python evaluation.py \
  --cache historical_data.csv \
  --history history_predictions.json \
  --history-multi history_predictions_multi.json \
  --min-lag-days 5
```

**Success criteria**
- Rain RMSE improves vs baseline without worsening Temp RMSE > +0.2 °C.  
- Cloud RMSE improves or stays stable.

**Result (2026-05-12)**  
- Temp RMSE: `3.198` °C vs baseline `3.171` (`+0.027`)  
- Rain RMSE: `0.052` mm/h vs baseline `0.055` (`-0.003`)  
- Cloud RMSE: `50.725` % vs baseline `56.848` (`-6.123`)  

**Interpretation**  
The spatial grid helps Rain and Cloud, but does not fix the conservative temperature behavior yet. The next comparison should isolate the current model on a fixed benchmark rather than relying only on mixed history predictions.

**Status**  
Executed — partial win, not yet a full model upgrade.
