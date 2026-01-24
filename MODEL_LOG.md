# ErlangenWeather — Model Evolution & Design Notes

> This document is **curated manually** to explain the big-picture evolution of our local deep-learning weather model for Erlangen (49.59, 11.00).  
> We only record **milestone versions** here (V0, V0.1, …). Routine training runs remain in `model_log.txt` (JSONL) for internal reproducibility.

---

## TL;DR (current public model)
- **Version:** V0.1  
- **Horizon:** 72 h (3 days, hourly)  
- **Targets:** Temperature (°C), Rain (mm/h), Cloud cover (%)  
- **Input window:** 10 days (240 h), stride 24 h  
- **Feature set (27 features total):**  
  - *Meteorology (9):* `temperature_2m`, `relativehumidity_2m`, `pressure_msl`, `windspeed_10m`, `winddirection_10m`, `cloudcover`, `shortwave_radiation`, `precipitation`, `rain`  
  - *Derived (6):* `wind_u_10m`, `wind_v_10m`, `dewpoint_2m`, `dp_dt_3h`, `temp_prev_day`, `temp_delta_prev_day`  
  - *Time (12):* hour/dow/doy/month + their sin/cos, `timestamp`
- **Architecture:** Transformer encoder with 1-D **patching** (Conv1D k=4, s=2), `d_model=128`, `heads=4`, `layers=4`, `ffn=256`, `dropout=0.1`  
- **Loss:** time-decayed, channel-weighted MSE (temp×1, rain×3, cloud×0.5)  
- **Data:** ERA5 archive for training/metrics; Open-Meteo forecast (near-real-time) for daily updates  
- **Deployment:** GitHub Actions daily @ **00:01 Europe/Berlin** → GitHub Pages

---

## Why milestones?
We iterate frequently. To keep this page readable, we summarize only **meaningful releases** (Vx.y). Detailed per-run logs (hyper-params, timing, val-loss curves) are still captured to `model_log.txt` (JSONL).

---

## Version history

### V0 — First public prototype (temperature only)
**Period:** 2025-08 → 2025-12  
**Goal:** End-to-end pipeline (data → model → CI → site) for 24 h temperature (°C) forecast.

**Highlights**
- **Targets:** Temperature only (24 steps)  
- **Inputs:** ~26 features (ERA5 + time features)  
- **Architecture:** Transformer v1 (CLS + sinusoidal PE), `d_model=64`, `heads=4`, `layers=2`, `ffn=128`  
- **Loss:** time-decayed MSE (near-term > long-term)  
- **Training:** 10 d → 1 d; batch=64; max 100 epochs; EarlyStopping(p=5)  
- **Deployment:** Daily CI; `prediction.json` + point summary; single skill vs. persistence metric

**Limitations / Lessons**
- Captured diurnal cycle; struggled with fronts/precip events (no precip target).  
- Single target limited the website’s usefulness.  
- “% vs persistence” is OK for experts, harder to explain to general users.

---

### V0.1 — Multi-target, 3-day horizon, patching transformer *(current)*
**Released:** 2026-01  
**Goal:** 72 h horizon, **3 targets** (temp/rain/cloud), better temporal context with patching.

**What changed**
- **Targets:** `temperature_2m (°C)`, `rain (mm/h)`, `cloudcover (%)` (72 hourly steps)  
- **Inputs (27 features):** 9 meteo + 6 derived + 12 time features (see TL;DR)  
- **Windowing:** 10 d (240 h) input, stride 24 h → 3 d (72 h) output  
- **Architecture:** Encoder-only Transformer + **Conv1D patching** (k=4, s=2)  
  — `d_model=128`, `heads=4`, `layers=4`, `ffn=256`, `dropout=0.1`  
- **Loss:** Weighted multi-output MSE with time decay (1.0 → 0.5 across horizon)  
  — channel weights: temp×1, rain×3, cloud×0.5  
- **Optim:** Adam(3e-4), ReduceLROnPlateau(p=3,×0.5), EarlyStopping(p=5), batch=64  
- **Metrics (site):** RMSE for T/Rain/Cloud with 5-day lag (archive availability) + sample counts; legacy skill vs persistence retained

**Impact**
- More stable multi-day evolution (frontal passages, cloud variability) via patching + longer horizon.  
- New rain/cloud forecasts + mini-charts; daily totals and min/max temps shown.

**Known limitations**
- Rain is intermittent & skewed → MSE under-penalizes false alarms on dry hours and underweights rare heavy bursts.  
- Single-grid-point input (no spatial context).  
- Deterministic outputs (no uncertainty quantiles yet).

**Planned (V0.2)**
- **Probabilistic rain** (calibrated PoP or quantile regression / pinball loss)  
- **Multi-resolution time modeling** (larger patch size + cross-scale fusion)  
- **Spatial context** (neighbor tiles or learned local embeddings)  
- **Exogenous nowcasts/radar** when available  
- **Stronger baselines** (MOS-like diurnal/persistence blends for clearer skill)

---

## What we record per milestone (for transparency)
1. **Version & release date**  
2. **Horizon & targets** (e.g., 72 h; temp/rain/cloud)  
3. **Data & span** (ERA5 range, forecast window, TZ, coordinates)  
4. **Input windowing** (days × hours, stride, #features, feature list, scaling)  
5. **Architecture** (type, d_model, heads, layers, FFN, dropout, patch k/s)  
6. **Loss & training setup** (weights/shape, optimizer, LR schedule, batch, epochs, early-stop)  
7. **Metrics** (RMSE per target; skill vs persistence; sample counts; lag window)  
8. **Release notes** (what & why)  
9. **Limitations** (known failure modes)  
10. **Repro snippet** (commands + prerequisites)

---

## Reproduce (current V0.1)

```bash
# Train locally (once), then commit the trained `model/` directory
python3 WeatherData.py \
  --source archive \
  --download-data --days 365 \
  --extend-to-present \
  --lat 49.59 --lon 11.00

# Daily prediction (CI uses forecast source for near-real-time features)
python3 WeatherData.py \
  --predict-only \
  --source forecast --past-days 20 --forecast-days 3 \
  --extend-to-present \
  --lat 49.59 --lon 11.00
### 2026-01-21T22:35:11Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 30.302786189369023
- **commit**: 3a7bb486790814b3a5467529afc1eb23bc51cf69
- **branch**: main

### 2026-01-21T23:24:24Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 30.215171234243844
- **commit**: c29d36c6c4f222431f883307964deccdcc651834
- **branch**: main

### 2026-01-22T22:33:00Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 30.215171234243844
- **commit**: 0b9c110d7524ae410125fa9a2f56d1a235491e1b
- **branch**: main

### 2026-01-22T23:20:25Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.92182005721235
- **commit**: beb696fb89830cc9bde97f31f60cd7b94ac2421a
- **branch**: main

### 2026-01-23T22:28:34Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.92182005721235
- **commit**: 72e1eeeb2453f9d675b3c5e06d38da6d2daa00c9
- **branch**: main

### 2026-01-23T23:20:54Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.634110248969925
- **commit**: 9562db3ad4e564bcb66fe00473bba1cd6ed022ae
- **branch**: main

### 2026-01-24T22:28:46Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.634110248969925
- **commit**: fe6bd736d928205032d299ed34d7e18cf3958ca0
- **branch**: main

### 2026-01-24T23:19:01Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.35188062755116
- **commit**: a5cb3f0e46e4aa18646f535c303d6ec78085ef53
- **branch**: main
