# ErlangenWeather

Hourly 3-day weather forecasts for Erlangen, Germany (`49.59`, `11.00`) using a local Transformer model.

- **Website:** [pa-sto.github.io/ErlangenWeather](https://pa-sto.github.io/ErlangenWeather/)
- **Model, data, features, architecture, and metrics:** [MODEL_LOG.md](MODEL_LOG.md)

## Repository Layout

| Path | Purpose |
|---|---|
| `WeatherData.py` | Data fetching, preprocessing, training, prediction, metrics, and output writing |
| `evaluation.py` | Evaluates published prediction history against archive actuals |
| `index.html` | GitHub Pages UI |
| `model/` | Deployed SavedModel used by GitHub Actions |
| `historical_data.csv` | Archive cache for actual weather data |
| `site/data/current/` | Current published forecast, metrics, and model metadata |
| `site/data/history/` | Published prediction history from GitHub Actions |
| `visuals/` | Public diagrams used by the model documentation |

Local experiments write to `runs/local/<timestamp>/` by default, so they do not overwrite the published GitHub Actions record.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a local prediction without touching published site data:

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

Evaluate the published history:

```bash
.venv/bin/python evaluation.py --cache historical_data.csv --min-lag-days 5
```

## Daily Automation

The GitHub Action in `.github/workflows/daily-prediction.yml` runs the prediction path once per day and commits changes under `site/data/current/` and `site/data/history/`.
