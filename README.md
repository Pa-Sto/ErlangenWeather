# ErlangenWeather — Local Deep-Learning Weather Forecasts

Hourly 3-day forecasts for Erlangen (49.59, 11.00) with a custom Transformer model.
Deployed daily via GitHub Actions and visualized on GitHub Pages.

- **Live site:** (GitHub Pages for this repo)
- **Model evolution & design notes:** [MODEL_LOG.md](MODEL_LOG.md)

---

## What’s in here?

- **`WeatherData.py`** — data fetch/cache, feature engineering, training, prediction, metrics.
- **`evaluation.py`** — evaluates the published prediction history against archive actuals.
- **`index.html`** — lightweight site that reads the JSON outputs.
- **`model/`** — saved model used by the action (commit your trained weights here).
- **Published site data** — `site/data/current/` for current forecast files and `site/data/history/` for GitHub Actions history.
- **Local runs** — `runs/local/<timestamp>/` keeps experiments separate from published predictions.
- **CI** — `.github/workflows/daily-prediction.yml` runs daily to refresh predictions.

---

## Current public model (V0.1)

- **Horizon:** 72 h (3 days, hourly)
- **Targets:** Temperature (°C), Rain (mm/h), Cloud cover (%)
- **Input window:** 10 days (240 h) context
- **Architecture:** Transformer encoder with 1-D patching (Conv1D k=4, s=2), d_model=128, heads=4, layers=4, FFN=256, dropout=0.1  
- **Metrics shown on site:** RMSE (Temp/Rain/Cloud) with a 5-day lag (archive availability)

More details: [MODEL_LOG.md](MODEL_LOG.md)

---

## Evaluate Published History

```bash
.venv/bin/python evaluation.py --cache historical_data.csv --min-lag-days 5
```

By default this reads `site/data/history/history_predictions.json` and
`site/data/history/history_predictions_multi.json`, so local experiments do not
overwrite the GitHub Actions record.

---

## Quickstart (local)

```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
