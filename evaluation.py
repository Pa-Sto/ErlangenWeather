#!/usr/bin/env python3
"""Evaluate published ErlangenWeather forecast history against archive actuals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASELINES = {
    "temp": 3.171,
    "rain": 0.055,
    "cloud": 56.848,
}


def parse_time(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert("UTC").tz_localize(None)


def pick_column(columns: pd.Index, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def load_actuals(cache_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Archive cache not found: {cache_path}")

    df = pd.read_csv(cache_path, index_col="time", parse_dates=["time"]).sort_index()
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[~idx.isna()].copy()
    df.index = idx[~idx.isna()].tz_convert("UTC").tz_localize(None)

    columns = {
        "temp": pick_column(df.columns, ["temperature_2m", "temp_c", "temperature_c", "temperature"]),
        "rain": pick_column(df.columns, ["rain", "rain_mm", "precip_mm", "precipitation"]),
        "cloud": pick_column(df.columns, ["cloudcover", "cloudcover_pct"]),
    }
    missing = [target for target, column in columns.items() if column is None]
    if missing:
        raise ValueError(f"Missing archive columns for: {', '.join(missing)}")
    return df, columns  # type: ignore[return-value]


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_entries(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("entries"), list):
            return [entry for entry in payload["entries"] if isinstance(entry, dict)]
        if isinstance(payload.get("history"), list):
            return [entry for entry in payload["history"] if isinstance(entry, dict)]
        return [{"series": payload}]
    return []


def value_from_mapping(value: Any, target: str) -> float | None:
    if isinstance(value, dict):
        candidates = {
            "temp": ["temp_c", "temperature_c", "temperature_2m", "temp"],
            "rain": ["rain_mm", "rain", "precip_mm", "precipitation"],
            "cloud": ["cloudcover_pct", "cloudcover"],
        }[target]
        for key in candidates:
            if value.get(key) is not None:
                return float(value[key])
        return None
    if target == "temp" and isinstance(value, (int, float)):
        return float(value)
    return None


def collect_predictions(history_path: Path, history_multi_path: Path) -> pd.DataFrame:
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def add_series(entry: dict[str, Any], source: str, targets: list[str]) -> None:
        series = entry.get("series") or entry.get("pred") or entry.get("prediction") or entry.get("data")
        if not isinstance(series, dict):
            return

        parsed_times = [(ts_raw, parse_time(ts_raw)) for ts_raw in series]
        parsed_times = [(ts_raw, ts) for ts_raw, ts in parsed_times if ts is not None]
        if not parsed_times:
            return

        forecast_start = min(ts for _, ts in parsed_times)
        entry_id = str(entry.get("generated_at") or entry.get("date") or f"{source}:{forecast_start.isoformat()}")

        for ts_raw, ts in parsed_times:
            key = (entry_id, ts.isoformat())
            row = records.setdefault(
                key,
                {
                    "entry_id": entry_id,
                    "source": source,
                    "ts": ts,
                    "lead_hours": (ts - forecast_start).total_seconds() / 3600.0,
                },
            )
            value = series.get(ts_raw)
            for target in targets:
                pred = value_from_mapping(value, target)
                if pred is not None:
                    row[f"pred_{target}"] = pred

    for entry in iter_entries(read_json(history_multi_path)):
        add_series(entry, "multi", ["temp", "rain", "cloud"])
    for entry in iter_entries(read_json(history_path)):
        add_series(entry, "temp_history", ["temp"])

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records.values()).sort_values(["entry_id", "ts"])


def rmse(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return float(math.sqrt(np.mean(np.square(values.astype(float)))))


def score_predictions(
    actuals: pd.DataFrame,
    columns: dict[str, str],
    predictions: pd.DataFrame,
    min_lag_days: int,
) -> dict[str, Any]:
    cutoff = pd.Timestamp.utcnow().tz_convert("UTC").tz_localize(None) - pd.Timedelta(days=min_lag_days)
    predictions = predictions[predictions["ts"] <= cutoff].copy()
    if predictions.empty:
        raise ValueError("No predictions are old enough to score yet.")

    actual_view = actuals.copy()
    actual_view["ts"] = actual_view.index
    merged = pd.merge(predictions, actual_view, on="ts", how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps between predictions and archive actuals.")

    bins = {
        "0-24h": (0, 24),
        "24-48h": (24, 48),
        "48-72h": (48, 72),
    }
    result: dict[str, Any] = {"min_lag_days": min_lag_days, "targets": {}}
    for target in ["temp", "rain", "cloud"]:
        pred_col = f"pred_{target}"
        actual_col = columns[target]
        if pred_col not in merged:
            continue
        scored = merged[[pred_col, actual_col, "lead_hours"]].dropna()
        error = scored[pred_col].astype(float) - scored[actual_col].astype(float)
        target_result: dict[str, Any] = {
            "n": int(len(scored)),
            "rmse": rmse(error),
            "lead_bins": {},
        }
        for label, (start, end) in bins.items():
            subset = scored[(scored["lead_hours"] >= start) & (scored["lead_hours"] < end)]
            bin_error = subset[pred_col].astype(float) - subset[actual_col].astype(float)
            target_result["lead_bins"][label] = {"n": int(len(subset)), "rmse": rmse(bin_error)}
        result["targets"][target] = target_result
    return result


def format_delta(value: float | None, baseline: float) -> str:
    if value is None:
        return "n/a"
    delta = value - baseline
    direction = "improved" if delta < 0 else "worse" if delta > 0 else "unchanged"
    return f"{delta:+.3f} ({direction})"


def print_report(result: dict[str, Any], baselines: dict[str, float]) -> None:
    labels = {
        "temp": ("Temp", "C"),
        "rain": ("Rain", "mm/h"),
        "cloud": ("Cloud", "%"),
    }
    print("=== Published History Evaluation ===")
    print(f"Minimum archive lag: {result['min_lag_days']} day(s)")
    for target, (label, unit) in labels.items():
        target_result = result["targets"].get(target, {})
        value = target_result.get("rmse")
        if value is None:
            print(f"{label} RMSE: n/a")
            continue
        print(f"{label} RMSE: {value:.3f} {unit} | baseline delta: {format_delta(value, baselines[target])}")
        for bin_label, bin_result in target_result.get("lead_bins", {}).items():
            bin_value = bin_result.get("rmse")
            if bin_value is not None:
                print(f"  {bin_label}: {bin_value:.3f} {unit} (n={bin_result['n']})")

    print("\n=== Experiment Summary (Markdown) ===")
    print("| Target | RMSE | Baseline | Delta | n |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for target, (label, unit) in labels.items():
        target_result = result["targets"].get(target, {})
        value = target_result.get("rmse")
        if value is None:
            continue
        print(
            f"| {label} | {value:.3f} {unit} | {baselines[target]:.3f} {unit} | "
            f"{format_delta(value, baselines[target])} | {target_result['n']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ErlangenWeather published prediction history.")
    parser.add_argument("--cache", default="historical_data.csv", help="Archive cache CSV with actual weather data.")
    parser.add_argument(
        "--history",
        default="site/data/history/history_predictions.json",
        help="Temperature prediction history JSON.",
    )
    parser.add_argument(
        "--history-multi",
        default="site/data/history/history_predictions_multi.json",
        help="Multi-target prediction history JSON.",
    )
    parser.add_argument("--min-lag-days", type=int, default=5, help="Only score predictions older than this lag.")
    parser.add_argument("--baseline-temp", type=float, default=DEFAULT_BASELINES["temp"])
    parser.add_argument("--baseline-rain", type=float, default=DEFAULT_BASELINES["rain"])
    parser.add_argument("--baseline-cloud", type=float, default=DEFAULT_BASELINES["cloud"])
    args = parser.parse_args()

    actuals, columns = load_actuals(Path(args.cache))
    predictions = collect_predictions(Path(args.history), Path(args.history_multi))
    if predictions.empty:
        raise SystemExit("No prediction history found.")

    result = score_predictions(actuals, columns, predictions, args.min_lag_days)
    baselines = {
        "temp": args.baseline_temp,
        "rain": args.baseline_rain,
        "cloud": args.baseline_cloud,
    }
    print_report(result, baselines)


if __name__ == "__main__":
    main()
