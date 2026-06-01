# ErlangenWeather Model Documentation

This page is the public model record for the ErlangenWeather repository. It describes the model and data that are currently present in the repo, without internal planning notes.

## Current Public Model

| Item | Value |
|---|---|
| Location | Erlangen, Germany (`49.59`, `11.00`) |
| Deployed model file | `model/` SavedModel |
| Forecast horizon | `72` hours (`3` days), hourly |
| Targets | `temperature_2m`, `rain`, `cloudcover` |
| Input window | `240` hours (`10` days) |
| Model input shape | `240 x 28` |
| Model output shape | `72 x 3` |
| Daily automation | GitHub Actions writes to `site/data/current/` and `site/data/history/` |

## Data

![Training data footprint](visuals/training_grid.svg)

The production forecast uses the Erlangen base point. A local `3x3` grid cache exists for spatial experiments, but it is not the model currently deployed by GitHub Actions.

| Dataset | Current repo state |
|---|---|
| Archive cache | `historical_data.csv` |
| Archive span | `1941-03-04 00:00` to `2026-06-01 23:00` |
| Archive rows | `747,264` hourly rows |
| Published current forecast | `site/data/current/` |
| Published history | `site/data/history/` |
| Temperature history entries | `285`, from `2025-08-15` to `2026-06-01` |
| Multi-target history entries | `133`, from `2026-01-20` to `2026-06-01` |

## Feature Set

The deployed model consumes `28` features per hour.

| Group | Count | Features |
|---|---:|---|
| Weather variables | `9` | `temperature_2m`, `relativehumidity_2m`, `pressure_msl`, `windspeed_10m`, `winddirection_10m`, `cloudcover`, `shortwave_radiation`, `precipitation`, `rain` |
| Time features | `13` | `hour`, `dow`, `doy`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos`, `month`, `month_sin`, `month_cos`, `timestamp` |
| Derived features | `6` | `wind_u_10m`, `wind_v_10m`, `dewpoint_2m`, `dp_dt_3h`, `temp_prev_day`, `temp_delta_prev_day` |

The preprocessing pipeline sorts the hourly time index, fills gaps with time interpolation plus forward/backward fill, scales features, and creates day-aligned windows.

## Architecture

![Model architecture](visuals/model_architecture.svg)

| Component | Value |
|---|---|
| Model family | Encoder-only Transformer with temporal patching |
| Input | `240 x 28` |
| Patching | `Conv1D`, kernel `4`, stride `2` |
| Token sequence | `119 x 128` |
| Transformer blocks | `4` |
| Attention heads | `4` |
| Feed-forward width | `256` |
| Dropout | `0.1` |
| Pooling | `GlobalAveragePooling1D` |
| Output head | Dense `216`, reshaped to `72 x 3` |
| Training objective | Weighted multi-target MSE with horizon decay |

## Published Metrics

The current published-history evaluation uses `site/data/history/` and ignores predictions newer than five days, so archive actuals have time to become available.

```bash
.venv/bin/python evaluation.py --cache historical_data.csv --min-lag-days 5
```

| Target | RMSE | Baseline | Delta | Points |
|---|---:|---:|---:|---:|
| Temperature | `3.532 C` | `3.171 C` | `+0.361` | `12,027` |
| Rain | `0.220 mm/h` | `0.055 mm/h` | `+0.165` | `9,123` |
| Cloud cover | `39.868 %` | `56.848 %` | `-16.980` | `9,123` |

Lead-time view:

| Target | 0-24h | 24-48h | 48-72h |
|---|---:|---:|---:|
| Temperature | `3.069 C` | `3.804 C` | `3.974 C` |
| Rain | `0.216 mm/h` | `0.220 mm/h` | `0.223 mm/h` |
| Cloud cover | `37.791 %` | `40.068 %` | `41.682 %` |

## Temperature Transitions

![Temperature transition cases](visuals/temperature_transition_cases.svg)

The strongest temperature-transition cases show that the model often compresses large day-to-day temperature moves. This is useful context for interpreting the aggregate temperature RMSE.

## Version History

| Version | Period | Summary |
|---|---|---|
| `V0` | 2025-08 to 2025-12 | First public prototype: temperature-only daily forecast with an end-to-end data, model, CI, and website pipeline. |
| `V0.1` | 2026-01 onward | Current public model: 72-hour multi-target forecast for temperature, rain, and cloud cover using a patching Transformer. |

## Reproduce

Run the daily prediction path locally without writing into the published site files:

```bash
.venv/bin/python WeatherData.py \
  --predict-only \
  --source forecast \
  --past-days 20 \
  --forecast-days 3 \
  --extend-to-present \
  --cache-file historical_data.csv \
  --lat 49.59 \
  --lon 11.00 \
  --model-path model
```

Run the same path in published mode:

```bash
.venv/bin/python WeatherData.py \
  --predict-only \
  --source forecast \
  --past-days 20 \
  --forecast-days 3 \
  --extend-to-present \
  --cache-file historical_data.csv \
  --lat 49.59 \
  --lon 11.00 \
  --site-current-dir site/data/current \
  --site-history-dir site/data/history \
  --log-dir site/data/history/logs \
  --model-path model
```
