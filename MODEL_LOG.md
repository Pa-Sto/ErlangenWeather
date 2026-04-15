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

### 2026-01-25T22:29:39Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.35188062755116
- **commit**: 603c112eea6f1181938dddede9d959db1b86ede5
- **branch**: main

### 2026-01-25T23:19:40Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.07497609332898
- **commit**: c19cfac3f75a9a2965c4f57666caeee66a69db0d
- **branch**: main

### 2026-01-26T22:32:32Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.07497609332898
- **commit**: bb40e5fd7fd904a723799f6cd4f8f0d5d477b813
- **branch**: main

### 2026-01-26T23:21:31Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.803247344793196
- **commit**: 3283af8e6d5301eddebbc2bc7b52ffe613eb02b9
- **branch**: main

### 2026-01-27T22:31:27Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.803247344793196
- **commit**: 43b6925b27866ad199e2bae00be1151d7063ce57
- **branch**: main

### 2026-01-27T23:21:12Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.679466524616245
- **commit**: 45295d41b2f71f33e7fd0b4f31a3e972f85fbdf5
- **branch**: main

### 2026-01-28T22:37:22Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.679466524616245
- **commit**: d13c85ab5f37772218f337c30a9c08a73a7c201a
- **branch**: main

### 2026-01-28T23:25:49Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.650003636833745
- **commit**: d11f8167e66568e01de99a1469082d15e8b7e25e
- **branch**: main

### 2026-01-29T22:37:30Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.650003636833745
- **commit**: 1f429578d3aaebb0b217e20b90dfb87dfe67e094
- **branch**: main

### 2026-01-29T23:25:53Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.042964660525467
- **commit**: 0e7b17d8a952d42b3f52a50e114b7e2fda5f17c4
- **branch**: main

### 2026-01-30T22:36:18Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 29.042964660525467
- **commit**: d6cb8191c369da8d686195226f5b72e55e7697d4
- **branch**: main

### 2026-01-30T23:24:32Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.781316330250462
- **commit**: 8a3338156ed81c21ae2175d03c9e0621c71aa5d7
- **branch**: main

### 2026-01-31T22:33:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.781316330250462
- **commit**: 2b8ed0c155cc5a39193d29e23d2efc0dff46fa01
- **branch**: main

### 2026-01-31T23:21:35Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.524340291587517
- **commit**: 3696f6d024727b928831bbc26961ed09c34abc27
- **branch**: main

### 2026-02-01T22:35:22Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.524340291587517
- **commit**: ac9903be33b2f9994be128c9cef7ad39ed1710bd
- **branch**: main

### 2026-02-01T23:23:37Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.27191250139648
- **commit**: d59b92cffe7940703ca7756a9d770f5181359075
- **branch**: main

### 2026-02-02T23:27:51Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.08833100009451
- **commit**: 40a6ef19f5d0d43d2da4269a4f251715cf49e3de
- **branch**: main

### 2026-02-03T22:39:59Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 28.08833100009451
- **commit**: 9c6e76876e614d8a8fcfa4260789a55ddf1aee8b
- **branch**: main

### 2026-02-03T23:27:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.844084643571946
- **commit**: 8bd67f7149ee17741175849fd531dd6f0862e3c5
- **branch**: main

### 2026-02-04T22:39:00Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.844084643571946
- **commit**: c935f98c30e698b5effebc255ee94ea71101c75d
- **branch**: main

### 2026-02-04T23:26:40Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.6992853453985
- **commit**: 324457d9d08daa4ef86396a805a63184eeb9f313
- **branch**: main

### 2026-02-05T22:38:26Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.6992853453985
- **commit**: 3a9c328d5ecc735cfa4fc0b2b9d810a8ea98d087
- **branch**: main

### 2026-02-05T23:22:39Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.46253931680535
- **commit**: 53e79cd5c157e940bda52a5f18e16aa1589f6854
- **branch**: main

### 2026-02-06T22:35:52Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.46253931680535
- **commit**: 25928ab20c265a6d37255d606331dd51b4ebfac7
- **branch**: main

### 2026-02-06T23:24:48Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.229805932764627
- **commit**: 457a9c4e6a209578922b9ca5009148d5b8151580
- **branch**: main

### 2026-02-07T22:36:20Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.229805932764627
- **commit**: 54e1555d337cf3a988672c34d04d305ea038b6ad
- **branch**: main

### 2026-02-07T23:28:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.372518792259687
- **commit**: 98051e29e7080f042374dee959b7019a97ded3f0
- **branch**: main

### 2026-02-08T22:38:16Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.372518792259687
- **commit**: c56cb4a8b2f9b295753c91813d0460168faaad1a
- **branch**: main

### 2026-02-08T23:30:20Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.144414468990856
- **commit**: baaebae59db9b0023e577ba414ed93fad48d1a75
- **branch**: main

### 2026-02-09T22:50:01Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.144414468990856
- **commit**: 498b3441df32a36d57c7508d71b0d04bd1b2f32d
- **branch**: main

### 2026-02-09T23:36:29Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.3096930543128
- **commit**: b9a4982f8d66d879569d6708fb1b2e137665c306
- **branch**: main

### 2026-02-10T22:53:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.3096930543128
- **commit**: 8009668803b73e45794ea70d136c80fd53275d21
- **branch**: main

### 2026-02-10T23:37:28Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.085843111244664
- **commit**: 7d5c18c7b9a1df828c04d54298ababde775c570c
- **branch**: main

### 2026-02-11T22:40:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.085843111244664
- **commit**: 2806f3aea2a9d7a4156ff2fa43694eb7236066c3
- **branch**: main

### 2026-02-11T23:30:43Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.053964520324175
- **commit**: 47c2eae69de08a45d962c10ad03c92ce3131d1b4
- **branch**: main

### 2026-02-12T22:40:51Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 27.053964520324175
- **commit**: 9e65de9333026b4e9c65084ad3c59a9fca5bcf42
- **branch**: main

### 2026-02-12T23:28:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.835787387095753
- **commit**: c03d3f5fd69d1f0db4d0a8c21f6436d3b8f46a54
- **branch**: main

### 2026-02-13T22:42:51Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.835787387095753
- **commit**: 1e9700ca3717cbcd485b615ca4c339d59dc2bb0f
- **branch**: main

### 2026-02-13T23:29:59Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.72329349833626
- **commit**: 65b44c131ad7b5a28f084c25c6fb0547e1a01c51
- **branch**: main

### 2026-02-14T22:34:36Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.72329349833626
- **commit**: 55de2a7460301a27b56ed78ae52bfbc7194e7a54
- **branch**: main

### 2026-02-14T23:23:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.511203867397086
- **commit**: 3a9abef1c81e97c171b6d2b900d0e8add526c885
- **branch**: main

### 2026-02-15T22:35:12Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.511203867397086
- **commit**: e1c70b6c12072baa418dc8c9e9cca2be80cab9e5
- **branch**: main

### 2026-02-15T23:25:37Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.302454230645925
- **commit**: a1e50d0b6723b9f7efe734ed9ba9ace514f941b2
- **branch**: main

### 2026-02-16T22:39:09Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.302454230645925
- **commit**: adabde75b3f735490a9ac6211ab78f59e8cd70a7
- **branch**: main

### 2026-02-16T23:28:07Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.096966306969
- **commit**: 8a805736748c945595b71a3ae0ae203b31db4917
- **branch**: main

### 2026-02-17T22:42:40Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 26.096966306969
- **commit**: b292d4fae3122c9800d0817cbf4862e01e379b83
- **branch**: main

### 2026-02-17T23:27:56Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.89922262939325
- **commit**: fd756ea59d59de536857fecacce122faa8e7a449
- **branch**: main

### 2026-02-18T22:42:37Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.89922262939325
- **commit**: 209207aa29cab61b4bab80bbb388f613ede9d84e
- **branch**: main

### 2026-02-18T23:28:59Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.699997839936376
- **commit**: 038837c45f512102846c7aefd1b76799d3828925
- **branch**: main

### 2026-02-19T22:41:50Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.699997839936376
- **commit**: 36e1797a865007b7ffed920614fa9da326c43aa8
- **branch**: main

### 2026-02-19T23:29:14Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.503814650318542
- **commit**: e4af799cbc3296af41df9b23327704a5c8a7eaa3
- **branch**: main

### 2026-02-20T22:37:25Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.503814650318542
- **commit**: 2f9b6da0db4e505d8aea7d968deaf269cdd296cc
- **branch**: main

### 2026-02-20T23:28:13Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.310603933270674
- **commit**: 0daca9b50018322bad0c09d397fc0600d1d33114
- **branch**: main

### 2026-02-21T22:34:54Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.310603933270674
- **commit**: a221c0b31e2914885e2af55fc16d5082699e38fa
- **branch**: main

### 2026-02-21T23:23:18Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.120298640539318
- **commit**: 697bc001258769ad6f0a93ae3637d06eec6ca8e6
- **branch**: main

### 2026-02-22T22:35:05Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.120298640539318
- **commit**: f41104c83824e80360a172f660385b33374c32c2
- **branch**: main

### 2026-02-22T23:24:49Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.93283372531141
- **commit**: a0162ee6d9b2599901dd88fb0fee67efe63ca49d
- **branch**: main

### 2026-02-23T22:53:26Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.93283372531141
- **commit**: 7a6c1a9a2bf58bce55f8fbc86ba65ca2a11d9680
- **branch**: main

### 2026-02-23T23:34:03Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.035431294913554
- **commit**: 4f0ae65d4e277ca5c79bb9740923a69363701029
- **branch**: main

### 2026-02-24T22:42:55Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 25.035431294913554
- **commit**: 004da331a195ace024be4a7d9765cb7ee1e9452f
- **branch**: main

### 2026-02-24T23:30:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.85134724127449
- **commit**: 5c63dac1f25cdad435b9e227a8b6a37c38d367b6
- **branch**: main

### 2026-02-25T22:43:40Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.85134724127449
- **commit**: 7fe518e93d2c36a3da790a58971d1325e04a8989
- **branch**: main

### 2026-02-25T23:27:26Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.669950546082703
- **commit**: 4efe1c16097a88e303981af886e84b0800b04f33
- **branch**: main

### 2026-02-26T22:42:57Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.669950546082703
- **commit**: 1470d7e1c7d9c04f8b77083f8f26559790cd3ac6
- **branch**: main

### 2026-02-26T23:28:31Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.491182788502396
- **commit**: f1fc322db22c4b463a2d273883992b30459670db
- **branch**: main

### 2026-02-27T22:34:04Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.491182788502396
- **commit**: 2686bea69ca81c4b4cbbf814e3d246cbfc5ef704
- **branch**: main

### 2026-02-27T23:23:51Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.31498722887288
- **commit**: d6eec86bebe068495ded538c90cd98e87f3bfdce
- **branch**: main

### 2026-02-28T22:32:57Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.31498722887288
- **commit**: 6732d0f9f2bd7373e06834442fb99cb9ea6c04a2
- **branch**: main

### 2026-02-28T23:21:24Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.141308748666646
- **commit**: 09d11f1e0b88489da6d012d944c89d105d48f4c6
- **branch**: main

### 2026-03-01T22:34:22Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 24.141308748666646
- **commit**: b9e355e9d9569ba20664226f288a94b85f85b058
- **branch**: main

### 2026-03-01T23:22:32Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.970093793002345
- **commit**: 1627a99a2a812c9c2f4d83a56a52123e669d07ab
- **branch**: main

### 2026-03-02T22:37:50Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.970093793002345
- **commit**: 2c2d6a43cb7d636ad7ff235227109679f8bc6336
- **branch**: main

### 2026-03-02T23:25:44Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.801290315586833
- **commit**: b5aaf29205e8f4cbbcca6001bcdc46f98ec37788
- **branch**: main

### 2026-03-03T22:39:50Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.801290315586833
- **commit**: 9713cabfe41eb485bb5c90220c331179a6b80917
- **branch**: main

### 2026-03-03T23:26:30Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.634847725967347
- **commit**: e67e0c920f9f041fc108483ea182563fdd1d136f
- **branch**: main

### 2026-03-04T22:41:18Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.634847725967347
- **commit**: 6012b6008c5ca3d532ca230101575feb3a521327
- **branch**: main

### 2026-03-04T23:29:16Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.47071683898146
- **commit**: 59e5c86675193f7a6b5cb092eff19bdca4779d95
- **branch**: main

### 2026-03-05T23:11:48Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.30884982629883
- **commit**: 4187431ae3048eebdbf1ace0ec54b4d7767b95db
- **branch**: main

### 2026-03-05T23:51:01Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.30884982629883
- **commit**: 99bd46738897914f4ca232db9ff02b0ce5ed4465
- **branch**: main

### 2026-03-06T22:38:57Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.30884982629883
- **commit**: 8af6f298bb5402b54e21203dda4b52cb3e966332
- **branch**: main

### 2026-03-06T23:26:48Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.14920016995432
- **commit**: e041289af4cd36016c393446d1a65124460502d1
- **branch**: main

### 2026-03-07T22:33:36Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 23.14920016995432
- **commit**: 38b25dcfea746f1810d6a25c432f7934f96ef05d
- **branch**: main

### 2026-03-07T23:21:35Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.991722617777757
- **commit**: 55a8aa56a0f90f9990a3721f4a66a7da88acdebf
- **branch**: main

### 2026-03-08T22:34:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.991722617777757
- **commit**: 3196c35673e2afed26e9c948c57587d3e47869c6
- **branch**: main

### 2026-03-08T23:22:50Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.83637314063061
- **commit**: cc7842ffbfe70eadfe9cd3e9c06d3fb86899b7b3
- **branch**: main

### 2026-03-09T22:39:14Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.83637314063061
- **commit**: 56747bab26ec44ae9b012271cd01b0fb960b4c24
- **branch**: main

### 2026-03-09T23:27:44Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.683108891364633
- **commit**: 2f77fcc40a89f8f77bca393cbee87a359a19b99e
- **branch**: main

### 2026-03-10T22:38:30Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.683108891364633
- **commit**: 5a585dc03407ed0e0381bb5be3fb8a9aeb207b02
- **branch**: main

### 2026-03-10T23:27:19Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.531888165422203
- **commit**: 27b1a7a9f25db91f1b6baa9cb30145335f188ee0
- **branch**: main

### 2026-03-11T22:36:14Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.531888165422203
- **commit**: bbfb266b9ac1424aa44a824ed5743118848624e1
- **branch**: main

### 2026-03-11T23:25:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.382670363002187
- **commit**: d124a147ed4c34ba81cfe70cc4462f5432569a43
- **branch**: main

### 2026-03-12T22:36:17Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.382670363002187
- **commit**: 1e2797e6fe898e4ce450ba934b745e827a5e4919
- **branch**: main

### 2026-03-12T23:27:54Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.23541595271928
- **commit**: 067feaf03214b18e0cf0c18f759c3c901f297398
- **branch**: main

### 2026-03-13T22:37:16Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.23541595271928
- **commit**: 3f1e201519a6cedda40ac06f5673e8729573f921
- **branch**: main

### 2026-03-13T23:28:15Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.090086436688434
- **commit**: a3cd228493a1e19e89992534d83c1d5332495358
- **branch**: main

### 2026-03-14T22:36:54Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.090086436688434
- **commit**: c9ff3c176fc30db443debc3c906d16d74767c658
- **branch**: main

### 2026-03-14T23:26:48Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.946644316969678
- **commit**: 49e3f122b006c5d5a65c5b7806afc4b2cd0ddd02
- **branch**: main

### 2026-03-15T22:37:58Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.946644316969678
- **commit**: 11916fef525f692ab576d00be95338d54bfd5663
- **branch**: main

### 2026-03-15T23:28:18Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.971491339688892
- **commit**: dcef3ebae50a001d80b046cc8d67b5a6612fa014
- **branch**: main

### 2026-03-16T22:44:32Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.971491339688892
- **commit**: 36a80f2f068862babf04fab4a6fa5588999628fa
- **branch**: main

### 2026-03-16T23:31:14Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.104160982595506
- **commit**: 1caddf4653a5d987914c7429953e7adb345f509c
- **branch**: main

### 2026-03-17T22:43:43Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.104160982595506
- **commit**: a36692fb8bf5c43edb27b463b6659d604a77db35
- **branch**: main

### 2026-03-17T23:31:43Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.095859908251654
- **commit**: e9481425feed93651d02afe460cd0ba4a1b48171
- **branch**: main

### 2026-03-18T22:41:13Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 22.095859908251654
- **commit**: fade0d193b33d2e394e223f936de93041aca3040
- **branch**: main

### 2026-03-18T23:30:04Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.956012693642467
- **commit**: c7583aa16d82aeb4a13985b9a0222acf12329f3f
- **branch**: main

### 2026-03-19T22:38:54Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.956012693642467
- **commit**: 4713b2de0f582cd8d23aed239ef971d7155007d0
- **branch**: main

### 2026-03-19T23:29:04Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.817924563493772
- **commit**: 29e79c01da28abc468d8dfb949516a380476904f
- **branch**: main

### 2026-03-20T22:39:04Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.817924563493772
- **commit**: d7ead38d8800b65da4feaad58ef1a6cab2df9861
- **branch**: main

### 2026-03-20T23:27:58Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.68156253497194
- **commit**: c1efb8bbe238717cc0e555b8c5990da8f3c23f68
- **branch**: main

### 2026-03-21T22:35:31Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.68156253497194
- **commit**: 9bd04fbfcceb9a960db6e58b6b6220dac9b49c98
- **branch**: main

### 2026-03-21T23:24:08Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.546894444692608
- **commit**: b9a22a2c4f9c6eff9e176bb496bca4b1de54cf92
- **branch**: main

### 2026-03-22T22:35:59Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.546894444692608
- **commit**: 74cd99231d557fcf783496156af0f226d0006655
- **branch**: main

### 2026-03-22T23:27:30Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.413888923429074
- **commit**: 47e95cd2886aefdb7ae17149ada348d21805fb49
- **branch**: main

### 2026-03-23T22:42:18Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.413888923429074
- **commit**: ee3a8c82cef369b5b79b60c62fd5e40eb61913a9
- **branch**: main

### 2026-03-23T23:31:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.282515371751593
- **commit**: ec3daa71830572d475f75922756d65f8cd8041c6
- **branch**: main

### 2026-03-24T22:41:49Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.282515371751593
- **commit**: 0033cbc25545446e5c3ba306a63643ccd629b53f
- **branch**: main

### 2026-03-24T23:31:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.152743936557986
- **commit**: 83fdae035b2bae730e7935be23aec2661bfa7aab
- **branch**: main

### 2026-03-25T22:49:27Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.152743936557986
- **commit**: 8eed43b954d1a66d5d5b177ddc0eabfa533e68ad
- **branch**: main

### 2026-03-25T23:33:47Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.024545488457637
- **commit**: 708350c29b3fc94456673a3a732faf058997f2cf
- **branch**: main

### 2026-03-26T22:40:55Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 21.024545488457637
- **commit**: 1204e52aa0e7c801e5fbf2cf509bd402de36d1b6
- **branch**: main

### 2026-03-26T23:31:32Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.89789159997295
- **commit**: 7916dd0df781a6b377d15c56e127d951508ee7eb
- **branch**: main

### 2026-03-27T22:44:09Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.89789159997295
- **commit**: b9b50f571dbe3e3f223502f594396fe83dd463b1
- **branch**: main

### 2026-03-27T23:31:54Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.77275452452401
- **commit**: f1d51a81df7d657617c333374f8947079bcb579a
- **branch**: main

### 2026-03-28T22:39:24Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.77275452452401
- **commit**: 9bb0c29c7c8fa321f8e576b216b7704edfc4a08d
- **branch**: main

### 2026-03-28T23:30:32Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.64910717616375
- **commit**: fe95019a69e2db75c17114d6395882359991bba8
- **branch**: main

### 2026-03-29T22:41:30Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.5269231100326
- **commit**: b816fb2e63b9753ff873e299f3f76b30748d333a
- **branch**: main

### 2026-03-29T23:31:58Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.5269231100326
- **commit**: 6318d409c1e2127ccf8fe28b80c66fb31005ca47
- **branch**: main

### 2026-03-30T22:49:20Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.406176503502998
- **commit**: 3ab4df44c2f2f243ddc38f83ba27ad15d899b9a2
- **branch**: main

### 2026-03-30T23:33:56Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.406176503502998
- **commit**: 4fb3344d523765641bc2ede12a3a651afa05e9ad
- **branch**: main

### 2026-03-31T22:43:56Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.28684213798544
- **commit**: ef74103f8007c5acd274ef36d04324a8ca2c37ff
- **branch**: main

### 2026-03-31T23:33:23Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.28684213798544
- **commit**: 5f09b29e2fb94155e6bf1a6894082b49cfb6bf2b
- **branch**: main

### 2026-04-01T22:50:52Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.168895381369243
- **commit**: 40659e0717fc65f1600ae748ce51200957112516
- **branch**: main

### 2026-04-01T23:35:13Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.168895381369243
- **commit**: d165d9835988b04ef96c3e19568892f6e817b40c
- **branch**: main

### 2026-04-02T22:41:31Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.052312171072312
- **commit**: 7010ef97527b572b6dacaa84218a0d7b733667e7
- **branch**: main

### 2026-04-02T23:35:03Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 20.052312171072312
- **commit**: 5fa816eab997912839264d19de2e607eff54c3b1
- **branch**: main

### 2026-04-03T22:44:27Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.937068997675343
- **commit**: d94b1ef52306399d549aa3eb8f46e18c71f740f3
- **branch**: main

### 2026-04-03T23:33:01Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.937068997675343
- **commit**: cda879b525fda32fe378bfe773c8b2b1ad5dbf46
- **branch**: main

### 2026-04-04T22:40:00Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.8231428891172
- **commit**: f5851410af07cb092db4acb87b5ae066c4d08729
- **branch**: main

### 2026-04-04T23:29:51Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.8231428891172
- **commit**: 75395a9416b13eb9e9d46b3cf27b69a559426459
- **branch**: main

### 2026-04-05T22:41:38Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.710511395429037
- **commit**: b02a3c16a8946bc7d8baf7f66b4fb487402ff631
- **branch**: main

### 2026-04-05T23:33:02Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.710511395429037
- **commit**: 5ee1b2b245811b95bd2feccb9393677185d732f2
- **branch**: main

### 2026-04-06T22:49:12Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.599152573985933
- **commit**: 16feb2c9334fa19e22e69748d029012e43e7b3a8
- **branch**: main

### 2026-04-06T23:34:53Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.599152573985933
- **commit**: b22bf619f6f2480c6d660f80477a41cf8d6bd86f
- **branch**: main

### 2026-04-07T22:51:24Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.489044975255677
- **commit**: c0ff8d4980d1ab320fa125baff314cae1a16259b
- **branch**: main

### 2026-04-07T23:36:20Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.489044975255677
- **commit**: 64567645d4003925abcfddb34f5115f0ccf6df09
- **branch**: main

### 2026-04-08T22:53:51Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.3801676290252
- **commit**: c347e5b77922b950d77ad869251dc3c9ed31c5a8
- **branch**: main

### 2026-04-08T23:37:33Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.3801676290252
- **commit**: 294155f43a74fc938532f72581d97c6f3c24f379
- **branch**: main

### 2026-04-09T22:54:16Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.41470137769318
- **commit**: af5a66222298ae1e9bbc47678ac129b68c952e31
- **branch**: main

### 2026-04-09T23:36:53Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.41470137769318
- **commit**: 235109fcf776d4f74e50c512003779348ebe7d9b
- **branch**: main

### 2026-04-10T22:49:52Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.44248902113744
- **commit**: ffc32878e0a4803570ccac79501e668ddd91f337
- **branch**: main

### 2026-04-10T23:34:21Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.44248902113744
- **commit**: 8f5e616d560a700b4a62206573cc62d063d2cea7
- **branch**: main

### 2026-04-11T22:42:20Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.33566215838394
- **commit**: 0dd24903b712ed3f5b38888197062bb527e2c1c3
- **branch**: main

### 2026-04-11T23:32:37Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.33566215838394
- **commit**: 663a8d2b4cc40ae8b07610600e988aa6e9d8f014
- **branch**: main

### 2026-04-12T22:46:39Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.230002802327196
- **commit**: 84a9e77d49cd385d2be7c14652973079006ab05b
- **branch**: main

### 2026-04-12T23:33:56Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.230002802327196
- **commit**: 1259ac03b21f5d83aa87e82b20fd1c4c6b41dafe
- **branch**: main

### 2026-04-13T22:54:49Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.162020206487117
- **commit**: 5e4d7389f16fefc8da7752f1f1a0d1d7d7442c79
- **branch**: main

### 2026-04-13T23:41:31Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.162020206487117
- **commit**: 302c836870669f844cdc6252be9e158693fba18e
- **branch**: main

### 2026-04-14T22:54:25Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.340605541797093
- **commit**: 9b0b28820dbbd82cf3bf5f9d70ff87d570240231
- **branch**: main

### 2026-04-14T23:41:15Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.340605541797093
- **commit**: d9a8741f71a2198650d69abe48d55b9285885ea7
- **branch**: main

### 2026-04-15T22:51:53Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.589190431345315
- **commit**: 873e078863049bbf6c95b9cffe6dc7815954004a
- **branch**: main

### 2026-04-15T23:41:00Z — predict

- **source**: forecast
- **targets**: ["temperature_2m", "rain", "cloudcover"]
- **seq_days**: 10
- **label_days**: 3
- **horizon_hours**: 72
- **overall_accuracy**: 19.589190431345315
- **commit**: 03f031c396c33b6d7c333509ed04a41acb32b10f
- **branch**: main
