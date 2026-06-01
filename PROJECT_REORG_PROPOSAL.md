# Project Reorganization Proposal

This proposal separates three concerns that are currently mixed together:

1. Published forecast data used by the website and GitHub Actions
2. Local experiment outputs and metrics
3. Reusable raw caches and model artifacts

## Current Problems

## Non-Negotiable Preservation Rule

The existing GitHub Actions outputs on the repository are historical records and must be preserved.

That means:

- do not delete or rewrite existing committed prediction history
- do not replace old history with locally regenerated versions
- do not collapse old files into a smaller derived summary if that loses forecast detail
- treat the current committed history as the authoritative public record

The reorganization should therefore be a **migration of paths and ownership**, not a reset of history.

### 1. Published and local outputs share the same files

`WeatherData.py` currently writes local runs into the same root-level files that GitHub Actions updates:

- `prediction.json`
- `prediction_multi.json`
- `prediction_point.json`
- `history_predictions.json`
- `history_predictions_multi.json`
- `metrics.json`
- `accuracy_overall.json`
- `historical_data.csv`

That means:

- local testing overwrites the same files that the website reads
- `git pull` brings remote prediction updates into files you may already have changed locally
- evaluation history mixes model versions and local test runs

### 2. Root directory is overloaded

The repo root currently contains source code, published data, local caches, model files, logs, and experiment assets together. That makes it hard to answer:

- what is production/published?
- what is local-only?
- what should be committed?
- what should be ignored?

### 3. Model artifacts are duplicated

There are two model storage styles in the repo:

- `model/` (SavedModel directory)
- `model.keras` (single-file Keras format)

This should be reduced to one production model location plus optional experiment checkpoints.

### 4. History evaluation is version-mixed

`history_predictions.json` and `history_predictions_multi.json` accumulate predictions over time, but they do not cleanly separate:

- forecasts made by GitHub Actions for the public model
- forecasts made locally during experiments

That makes long-horizon RMSE harder to trust.

## Proposed Target Layout

```text
ErlangenWeather/
├── src/
│   ├── WeatherData.py
│   ├── evaluation.py
│   └── tools/
├── site/
│   ├── index.html
│   └── data/
│       ├── current/
│       │   ├── prediction.json
│       │   ├── prediction_multi.json
│       │   ├── prediction_point.json
│       │   ├── metrics.json
│       │   ├── accuracy_overall.json
│       │   └── model_info.json
│       └── history/
│           ├── daily/
│           │   ├── pred_2026-02-09.json
│           │   └── ...
│           ├── history_predictions.json
│           └── history_predictions_multi.json
├── data/
│   ├── cache/
│   │   ├── historical_data.csv
│   │   └── grid/
│   │       ├── historical_L0.csv
│   │       └── ...
│   └── snapshots/
│       └── optional frozen evaluation sets
├── models/
│   ├── production/
│   │   ├── model.keras
│   │   └── model_info.json
│   └── experiments/
│       ├── exp-001/
│       │   ├── model.keras
│       │   ├── metrics.json
│       │   └── notes.md
│       └── ...
├── runs/
│   └── local/
│       ├── 2026-06-01_exp-002/
│       │   ├── outputs/
│       │   ├── metrics/
│       │   └── logs/
│       └── ...
├── docs/
│   ├── MODEL_LOG.md
│   ├── EXPERIMENTS.md
│   ├── PROJECT_VISUAL_OVERVIEW.md
│   └── visuals/
└── .github/
    └── workflows/
```

## Ownership Rules

### `site/data/current/`

Purpose:
- only the currently published forecast and metrics

Writer:
- GitHub Actions only

Reader:
- website only

Local runs:
- should never write here by default

### `site/data/history/`

Purpose:
- long-lived public forecast history for evaluation

Writer:
- GitHub Actions only

Reader:
- evaluation scripts and website

Local runs:
- read-only by default

### `data/cache/`

Purpose:
- raw archive and grid caches used for training and scoring

Writer:
- local training runs and optionally GitHub Actions

Commit policy:
- decide explicitly
- if repo size is a concern, this should move to ignored local storage

### `models/production/`

Purpose:
- the model currently used by the website/GitHub Actions

Writer:
- explicit promotion step only

### `models/experiments/`

Purpose:
- saved outputs for experimental runs

Writer:
- local experiments

### `runs/local/`

Purpose:
- scratch outputs from local tests

Writer:
- local runs only

Commit policy:
- gitignored

## Recommended Command-Level Changes

The code should stop assuming root-level filenames and instead accept explicit directories.

### New runtime arguments

Recommended additions:

- `--site-current-dir`
- `--site-history-dir`
- `--cache-dir`
- `--model-dir`
- `--run-dir`
- `--mode published|local`

### Default behavior

#### Local mode

Default local run writes to:

- `runs/local/<timestamp>/outputs/`
- `runs/local/<timestamp>/metrics/`
- `runs/local/<timestamp>/logs/`

It reads from:

- `data/cache/`
- optionally `models/production/model.keras`

It does **not** modify:

- `site/data/current/`
- `site/data/history/`

#### Published mode

GitHub Actions writes to:

- `site/data/current/`
- `site/data/history/`

It reads from:

- `models/production/model.keras`
- `data/cache/` or a smaller dedicated published cache

## Recommended GitHub Actions Change

Current workflow commits root-level forecast files and the main cache.

Target workflow should only commit:

- `site/data/current/*`
- `site/data/history/*`
- optionally `models/production/model_info.json`
- optionally selected docs/log summaries

It should not mix published data with local experiment artifacts.

## Minimal Migration Plan

### Phase 1: Safe separation without changing the website structure too much

1. Keep `index.html` where it is for now
2. Copy existing published history into `site/data/history/` without deleting the old files yet
3. Copy current published forecast files into `site/data/current/`
4. Update `index.html` to read from `site/data/current/`
5. Add local output directories under `runs/local/`
6. Update `WeatherData.py` to write local runs there by default

### Phase 2: Clean up model and cache locations

1. Standardize on `models/production/model.keras`
2. Move grid cache into `data/cache/grid/`
3. Move base cache into `data/cache/historical_data.csv`
4. Only after verification, stop writing new published outputs to the legacy root-level files

### Phase 3: Better experiment control

1. Add run manifests per experiment
2. Save evaluation summaries per run
3. Add a clean benchmark dataset under `data/snapshots/`

## History Migration Strategy

To preserve the public record safely:

1. Existing committed files remain untouched during the first migration step.
2. We introduce the new `site/data/...` structure alongside the current root files.
3. We copy historical files into the new structure once.
4. GitHub Actions starts appending only to the new published-history location.
5. After the website and evaluation scripts are confirmed to work from the new paths, we can decide whether the old root-level published files should remain as compatibility mirrors or be frozen.

This avoids losing git-tracked history and prevents local experiment runs from colliding with the published archive.

## Recommended First Implementation

If we want the smallest change with the biggest payoff, do this first:

1. Create `site/data/current/` and `site/data/history/`
2. Copy the existing committed published records into those directories
3. Change GitHub Actions to write there going forward
4. Change `index.html` to read from there
5. Add `--run-dir` to `WeatherData.py`
6. Make local runs default to `runs/local/<timestamp>/`

That will solve the immediate conflict between:

- pulling published prediction history from GitHub Actions
- testing local model variants without overwriting published JSON files

## Current Repo-Specific Notes

Observed in the repo today:

- `predictions/` already acts like a partial history store, but it is mixed with root-level history files
- `model_info.json` is stale relative to the current code
- `model/` and `model.keras` coexist and should be normalized
- `grid_cache/` is clearly local/training-oriented and should not stay mixed with published site data

## Recommendation

Use this proposal as the basis for the actual refactor. The first code change should be output path parameterization, because that unlocks everything else cleanly.
