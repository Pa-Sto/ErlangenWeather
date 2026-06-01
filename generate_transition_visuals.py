#!/usr/bin/env python3
"""
Generate a visual briefing for large day-to-day temperature transitions.

Outputs:
- visuals/temperature_transition_cases.svg
- TEMPERATURE_TRANSITIONS.md
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


ROOT = Path(__file__).resolve().parent
VIS_DIR = ROOT / "visuals"
SVG_PATH = VIS_DIR / "temperature_transition_cases.svg"
MD_PATH = ROOT / "TEMPERATURE_TRANSITIONS.md"
PREDICTION_HISTORY_PATH = ROOT / "site" / "data" / "history" / "history_predictions.json"


def load_predictions(path: Path) -> pd.DataFrame:
    data = json.load(path.open("r", encoding="utf-8"))
    rows = []
    for entry in data:
        forecast_day = entry.get("date")
        for ts, value in (entry.get("series") or {}).items():
            rows.append(
                {
                    "ts": pd.to_datetime(ts),
                    "pred_temp": float(value),
                    "forecast_day": forecast_day,
                }
            )
    return pd.DataFrame(rows)


def build_transition_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    actuals = pd.read_csv(
        ROOT / "historical_data.csv", index_col="time", parse_dates=["time"]
    ).sort_index()
    actuals = actuals[["temperature_2m"]].copy()
    actuals["ts"] = actuals.index

    preds = load_predictions(PREDICTION_HISTORY_PATH)
    merged = preds.merge(actuals, on="ts", how="inner")
    merged["date"] = merged["ts"].dt.floor("D")

    daily = (
        merged.groupby("date")
        .agg(
            pred_mean=("pred_temp", "mean"),
            actual_mean=("temperature_2m", "mean"),
        )
        .reset_index()
    )
    daily["pred_delta"] = daily["pred_mean"].diff()
    daily["actual_delta"] = daily["actual_mean"].diff()
    daily["delta_gap"] = daily["pred_delta"] - daily["actual_delta"]
    daily["abs_actual_delta"] = daily["actual_delta"].abs()
    daily = daily.dropna().sort_values("abs_actual_delta", ascending=False)
    return merged, daily


def compute_cases(merged: pd.DataFrame, daily: pd.DataFrame, top_n: int = 4):
    cases = []
    for _, row in daily.head(top_n).iterrows():
        event_day = pd.Timestamp(row["date"])
        window_start = event_day - pd.Timedelta(days=1)
        window_end = event_day + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        sub = merged[(merged["ts"] >= window_start) & (merged["ts"] <= window_end)].copy()
        sub = sub.sort_values("ts")
        if len(sub) < 24:
            continue
        cases.append(
            {
                "date": event_day,
                "window_start": window_start,
                "window_end": window_end,
                "actual_delta": float(row["actual_delta"]),
                "pred_delta": float(row["pred_delta"]),
                "delta_gap": float(row["delta_gap"]),
                "actual_mean": float(row["actual_mean"]),
                "pred_mean": float(row["pred_mean"]),
                "series": sub,
            }
        )
    return cases


def _line_path(points):
    if not points:
        return ""
    out = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    out.extend(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
    return " ".join(out)


def render_svg(cases, daily_top):
    VIS_DIR.mkdir(exist_ok=True)
    width = 1420
    height = 1235
    card_w = 640
    card_h = 360
    left_margin = 60
    top_margin = 160
    x_gap = 20
    y_gap = 40

    all_values = []
    for case in cases:
        all_values.extend(case["series"]["temperature_2m"].tolist())
        all_values.extend(case["series"]["pred_temp"].tolist())
    y_min = min(all_values) - 1.5
    y_max = max(all_values) + 1.5
    y_span = max(1.0, y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f6f1"/>',
        '<text x="60" y="62" font-family="Georgia, serif" font-size="34" fill="#1a2630">Temperature Transition Days</text>',
        '<text x="60" y="96" font-family="Arial, sans-serif" font-size="17" fill="#5f6871">Actual vs predicted hourly temperature curves for the strongest day-to-day jumps in the scored history.</text>',
        '<rect x="1010" y="28" width="350" height="108" rx="18" fill="#13212f"/>',
        '<text x="1185" y="60" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#d9e7f2">Current reading</text>',
        '<text x="1185" y="90" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#ffffff">The model catches the direction of jumps,</text>',
        '<text x="1185" y="116" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#ffffff">but still underestimates their size.</text>',
    ]

    for idx, case in enumerate(cases):
        col = idx % 2
        row = idx // 2
        x0 = left_margin + col * (card_w + x_gap)
        y0 = top_margin + row * (card_h + y_gap)
        plot_x = x0 + 28
        plot_y = y0 + 98
        plot_w = card_w - 56
        plot_h = 190

        parts.append(f'<rect x="{x0}" y="{y0}" width="{card_w}" height="{card_h}" rx="22" fill="#fffdf8" stroke="#d9d2c7" stroke-width="2"/>')
        parts.append(f'<text x="{x0 + card_w/2:.1f}" y="{y0+38}" text-anchor="middle" font-family="Georgia, serif" font-size="24" fill="#1d2832">{case["date"].strftime("%Y-%m-%d")}</text>')
        parts.append(
            f'<text x="{x0 + card_w/2:.1f}" y="{y0+68}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#5f6871">'
            f'Actual daily mean jump {case["actual_delta"]:+.2f} C, predicted {case["pred_delta"]:+.2f} C'
            '</text>'
        )

        parts.append(f'<line x1="{plot_x}" y1="{plot_y+plot_h}" x2="{plot_x+plot_w}" y2="{plot_y+plot_h}" stroke="#c8c1b5" stroke-width="1.5"/>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y+plot_h}" stroke="#c8c1b5" stroke-width="1.5"/>')

        for frac, label in [(0.0, "D-1 00"), (0.5, "D 00"), (1.0, "D+1 00")]:
            xx = plot_x + plot_w * frac
            parts.append(f'<line x1="{xx:.1f}" y1="{plot_y}" x2="{xx:.1f}" y2="{plot_y+plot_h}" stroke="#ebe4d8" stroke-width="1" stroke-dasharray="4 4"/>')
            anchor = "middle"
            lx = xx
            if frac == 0.0:
                anchor = "start"
                lx = xx + 2
            elif frac == 1.0:
                anchor = "end"
                lx = xx - 2
            parts.append(f'<text x="{lx:.1f}" y="{plot_y+plot_h+24}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="12" fill="#6d665d">{label}</text>')

        def map_x(i, n):
            return plot_x + (plot_w * i / max(1, n - 1))

        def map_y(v):
            return plot_y + plot_h - ((v - y_min) / y_span) * plot_h

        actual_points = []
        pred_points = []
        series = case["series"].reset_index(drop=True)
        for i, rec in series.iterrows():
            actual_points.append((map_x(i, len(series)), map_y(rec["temperature_2m"])))
            pred_points.append((map_x(i, len(series)), map_y(rec["pred_temp"])))

        parts.append(f'<path d="{_line_path(actual_points)}" fill="none" stroke="#d66a37" stroke-width="3"/>')
        parts.append(f'<path d="{_line_path(pred_points)}" fill="none" stroke="#34699a" stroke-width="3"/>')

        # Keep y-axis value labels inside the plot frame so they do not collide with the legend.
        parts.append(f'<text x="{plot_x}" y="{plot_y+12}" font-family="Arial, sans-serif" font-size="12" fill="#6d665d">{y_max:.1f} C</text>')
        parts.append(f'<text x="{plot_x}" y="{plot_y+plot_h-6}" font-family="Arial, sans-serif" font-size="12" fill="#6d665d">{y_min:.1f} C</text>')

        legend_y = y0 + card_h - 18
        parts.append(f'<circle cx="{x0+34}" cy="{legend_y}" r="6" fill="#d66a37"/>')
        parts.append(f'<text x="{x0+48}" y="{legend_y+5}" font-family="Arial, sans-serif" font-size="13" fill="#4c555d">actual hourly temperature</text>')
        parts.append(f'<circle cx="{x0+316}" cy="{legend_y}" r="6" fill="#34699a"/>')
        parts.append(f'<text x="{x0+330}" y="{legend_y+5}" font-family="Arial, sans-serif" font-size="13" fill="#4c555d">predicted hourly temperature</text>')

    table_x = 60
    table_y = 955
    parts.append(f'<rect x="{table_x}" y="{table_y}" width="1300" height="220" rx="18" fill="#ffffff" stroke="#ddd6ca" stroke-width="2"/>')
    parts.append(f'<text x="{table_x+24}" y="{table_y+34}" font-family="Georgia, serif" font-size="22" fill="#1d2832">Top transition days by absolute daily mean change</text>')
    headers = ["Date", "Actual delta", "Pred delta", "Gap", "Actual mean", "Pred mean"]
    col_x = [84, 260, 430, 570, 700, 870]
    for hx, header in zip(col_x, headers):
        parts.append(f'<text x="{hx}" y="{table_y+66}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#55606a">{header}</text>')

    for ridx, (_, row) in enumerate(daily_top.iterrows()):
        yy = table_y + 92 + ridx * 22
        values = [
            pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            f'{row["actual_delta"]:+.2f} C',
            f'{row["pred_delta"]:+.2f} C',
            f'{row["delta_gap"]:+.2f} C',
            f'{row["actual_mean"]:.2f} C',
            f'{row["pred_mean"]:.2f} C',
        ]
        for vx, value in zip(col_x, values):
            parts.append(f'<text x="{vx}" y="{yy}" font-family="Arial, sans-serif" font-size="13" fill="#24303a">{escape(value)}</text>')

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def render_markdown(cases, daily_top):
    lines = [
        "# Temperature Transition Brief",
        "",
        "This note focuses on the days where Erlangen's daily mean temperature changed the most inside the scored prediction history.",
        "",
        f"![Temperature transition cases](visuals/{SVG_PATH.name})",
        "",
        "## Reading guide",
        "",
        "- Orange line: actual hourly temperature from `historical_data.csv`.",
        "- Blue line: model prediction from `site/data/history/history_predictions.json`.",
        "- Strong underreaction means the model is acting too much like persistence.",
        "",
        "## Strongest transition days",
        "",
        "| Date | Actual delta (C) | Pred delta (C) | Gap (C) | Comment |",
        "|---|---:|---:|---:|---|",
    ]

    for _, row in daily_top.iterrows():
        comment = "underreacted" if abs(row["pred_delta"]) < abs(row["actual_delta"]) else "matched or overshot"
        lines.append(
            f"| {pd.Timestamp(row['date']).strftime('%Y-%m-%d')} | {row['actual_delta']:+.2f} | {row['pred_delta']:+.2f} | {row['delta_gap']:+.2f} | {comment} |"
        )

    lines.extend(
        [
            "",
            "## Main takeaway",
            "",
            "The current model usually gets the direction of major moves, but it compresses the amplitude. That is consistent with the conservative behavior already seen in the aggregate metrics.",
            "",
            "## What to test next",
            "",
            "1. Train on a more recent history window and compare these same transition days.",
            "2. Add temperature transition weighting so large day-to-day changes matter more during training.",
            "3. Score a dedicated jump-day benchmark alongside normal RMSE for every new experiment.",
        ]
    )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    merged, daily = build_transition_dataset()
    cases = compute_cases(merged, daily, top_n=4)
    if not cases:
        raise SystemExit("No usable transition cases found.")
    daily_top = daily.head(6).copy()
    render_svg(cases, daily_top)
    render_markdown(cases, daily_top)
    print(f"Wrote {SVG_PATH.relative_to(ROOT)}")
    print(f"Wrote {MD_PATH.name}")


if __name__ == "__main__":
    main()
