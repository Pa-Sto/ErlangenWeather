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
