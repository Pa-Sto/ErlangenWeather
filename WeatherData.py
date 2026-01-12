import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import List
import json
import os
from datetime import datetime, timedelta, date
import time
import sys
import argparse
import subprocess
from requests.adapters import HTTPAdapter

from urllib3.util.retry import Retry
from dateutil.tz import gettz

# Earliest supported start date for Open-Meteo Archive API (reanalysis)

MIN_ARCHIVE_DATE = date(1940, 1, 1)

# Normalize Open-Meteo hourly variable names (avoid API 4xx/5xx due to mismatches)
OPEN_METEO_HOURLY_ALIASES = {
    "wind_speed_10m": "windspeed_10m",  # common alias → official name
    "windspeed_10m": "windspeed_10m",
    "winddirection_10m": "winddirection_10m",
}

# ---- Forecast targets & window config (global) ----
TARGETS = ["temperature_2m", "rain", "cloudcover"]  # 3-target training/prediction
NUM_TARGETS = len(TARGETS)
SEQ_DAYS = 10
LABEL_DAYS = 3

def normalize_hourly_variables(variables: List[str]) -> List[str]:
    normalized = [OPEN_METEO_HOURLY_ALIASES.get(v, v) for v in variables]
    # de-duplicate while preserving order
    seen, out = set(), []
    for v in normalized:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def get_historical_data(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    variables: List[str],
    timezone: str = "UTC",
    retry: int = 3,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Fetches historical measured data via Open-Meteo Archive API.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    # Clamp start/end to supported archive window and validate
    start_date_obj = date.fromisoformat(start[:10])
    end_date_obj = date.fromisoformat(end[:10])
    if start_date_obj < MIN_ARCHIVE_DATE:
        print(f"[Archive] start_date {start_date_obj} < {MIN_ARCHIVE_DATE}; clamping.")
        start_date_obj = MIN_ARCHIVE_DATE
    if end_date_obj < start_date_obj:
        raise ValueError(f"end_date ({end_date_obj}) is earlier than start_date ({start_date_obj}).")

    variables = normalize_hourly_variables(variables)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "timezone": timezone,
        "start_date": start_date_obj.isoformat(),
        "end_date": end_date_obj.isoformat()
    }
    # Robust session with retry/backoff
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    headers = {"User-Agent": "ErlangenWeather/1.0 (+https://github.com/Pa-Sto/ErlangenWeather)"}

    try:
        # (connect timeout, read timeout)
        r = session.get(base_url, params=params, headers=headers, timeout=(5, timeout))
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Archive API error {r.status_code}: {r.text}") from e

    data = r.json().get("hourly", {})
    times = data.get("time", [])
    cols = {}
    for v in variables:
        vals = data.get(v)
        if vals is None:
            # some variables may not exist for very old periods; fill with NaNs
            vals = [np.nan] * len(times)
        cols[v] = vals
    df = pd.DataFrame(cols, index=pd.to_datetime(times))
    df.index.name = "time"
    return df

def get_and_cache_past_data(
    latitude: float,
    longitude: float,
    days: int,
    variables: List[str],
    timezone: str = "UTC",
    cache_file: str = "historical_data.csv",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Fetches the past `days` days of weather data via Open-Meteo and caches to CSV.
    If `cache_file` exists and `force_download` is False, loads from CSV instead.
    """
    if os.path.exists(cache_file) and not force_download:
        df = pd.read_csv(cache_file, index_col='time', parse_dates=['time'])
    else:
        end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=days)
        start = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df = get_historical_data(
            latitude, longitude, start, end, variables, timezone
        )
        df.to_csv(cache_file)
    return df

def _iter_date_chunks(start_d: date, end_d: date, chunk_days: int = 10):
    """Yield (chunk_start, chunk_end) pairs inclusive, split by chunk_days."""
    cur = start_d
    delta = timedelta(days=chunk_days)
    while cur <= end_d:
        nxt = min(cur + delta - timedelta(days=1), end_d)
        yield (cur, nxt)
        cur = nxt + timedelta(days=1)

def _print_progress(iteration: int, total: int, prefix: str = "[Archive]", length: int = 30):
    if total <= 0:
        return
    filled = int(length * iteration / total)
    bar = "█" * filled + "-" * (length - filled)
    pct = 100.0 * iteration / total
    sys.stdout.write(f"\r{prefix} |{bar}| {pct:5.1f}% ({iteration}/{total})")
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write("\n")

def _get_git_info():
    commit = os.environ.get("GITHUB_SHA")
    branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF")
    if commit:
        return {"commit": commit, "branch": branch}
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        try:
            b = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            b = None
        return {"commit": c, "branch": b}
    except Exception:
        return {"commit": None, "branch": branch}

def _append_model_log(entry: dict, jsonl_path: str = "model_log.txt", md_path: str = "MODEL_LOG.md"):
    ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    entry = dict(entry)
    entry.setdefault("timestamp", ts)
    # JSONL append
    try:
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Log] Could not append {jsonl_path}: {e}")
    # Markdown append
    try:
        with open(md_path, "a") as f:
            f.write(f"\n### {entry.get('timestamp')} — {entry.get('event','event')}\n\n")
            order = [
                "tag","note","source","targets","seq_days","label_days","horizon_hours","n_features",
                "n_train_windows","n_val_windows","best_val_loss","final_val_loss","epochs_run","train_seconds",
                "overall_accuracy","d_model","num_heads","d_ff","num_layers","commit","branch"
            ]
            for k in order:
                if k in entry and entry[k] is not None:
                    v = entry[k]
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v, ensure_ascii=False)
                    f.write(f"- **{k}**: {v}\n")
    except Exception as e:
        print(f"[Log] Could not append {md_path}: {e}")

def update_cache_with_historical(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    variables: List[str],
    timezone: str = "UTC",
    cache_file: str = "historical_data.csv",
    retry: int = 3,
    timeout: int = 30,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical data for [start, end], merge with existing cache (if any),
    de-duplicate on index, sort, and write back to CSV.
    Returns the merged DataFrame.
    """
    # Build chunked requests over the desired date span
    start_d = date.fromisoformat(start[:10])
    end_d = date.fromisoformat(end[:10])
    if start_d < MIN_ARCHIVE_DATE:
        print(f"[Archive] start_date {start_d} < {MIN_ARCHIVE_DATE}; clamping.")
        start_d = MIN_ARCHIVE_DATE
    if end_d < start_d:
        raise ValueError(f"end_date ({end_d}) is earlier than start_date ({start_d}).")

    def _fetch_range(a: date, b: date, depth: int = 0, max_depth: int = 8) -> pd.DataFrame:
        """Fetch range [a,b] with recursive halving on server errors/timeouts."""
        try:
            return get_historical_data(
                latitude, longitude, a.isoformat(), b.isoformat(), variables, timezone, retry=retry, timeout=timeout
            )
        except Exception as e:
            days_span = (b - a).days
            if days_span >= 1 and depth < max_depth:
                mid = a + timedelta(days=days_span // 2)
                left = _fetch_range(a, mid, depth + 1, max_depth)
                right = _fetch_range(mid + timedelta(days=1), b, depth + 1, max_depth)
                return pd.concat([left, right], axis=0)
            else:
                print(f"[Archive] Skipping {a}→{b} after errors: {e}")
                return pd.DataFrame()

    frames = []
    chunks = list(_iter_date_chunks(start_d, end_d, chunk_days=10))
    total = len(chunks)
    if show_progress:
        print(f"[Archive] Downloading {total} chunk(s) from {start_d} to {end_d}…")
        _print_progress(0, total, prefix=f"[Archive] {start_d}→{end_d}")
    for i, (a, b) in enumerate(chunks, start=1):
        df_chunk = _fetch_range(a, b)
        if not df_chunk.empty:
            frames.append(df_chunk)
        if show_progress:
            _print_progress(i, total, prefix=f"[Archive] {start_d}→{end_d}")
        time.sleep(0.1)
    df_new = pd.concat(frames, axis=0) if frames else pd.DataFrame()
    # Merge with existing cache and write to disk
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        df_old = pd.read_csv(cache_file, index_col='time', parse_dates=['time']).sort_index()
        df_merged = pd.concat([df_old, df_new], axis=0)
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')].sort_index()
    else:
        df_merged = df_new
    df_merged.to_csv(cache_file)
    return df_merged
def extend_cache_to_present(
    latitude: float,
    longitude: float,
    variables: List[str],
    timezone: str = "UTC",
    cache_file: str = "historical_data.csv",
    show_progress: bool = True,
):
    """
    Extend the local cache forward to 'today' (local time) by downloading the missing
    date range from the last cached timestamp + 1 day up to today.
    If the cache does not exist or is empty, start from MIN_ARCHIVE_DATE.
    """
    today = datetime.now(gettz(timezone)).date()
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        dfc = pd.read_csv(cache_file, index_col="time", parse_dates=["time"]).sort_index()
        if not dfc.empty:
            last_day = dfc.index.max().date()
            start_d = last_day + timedelta(days=1)
        else:
            start_d = MIN_ARCHIVE_DATE
    else:
        start_d = MIN_ARCHIVE_DATE

    end_d = today
    if start_d > end_d:
        print("[Archive] Cache is already up-to-date through today.")
        return pd.read_csv(cache_file, index_col="time", parse_dates=["time"]) if os.path.exists(cache_file) else pd.DataFrame()

    start = start_d.isoformat()
    end = end_d.isoformat()
    print(f"[Extend] Filling cache forward: {start} → {end}")
    return update_cache_with_historical(
        latitude, longitude, start, end, variables,
        timezone=timezone, cache_file=cache_file, show_progress=show_progress
    )
def get_recent_forecast(
    latitude: float,
    longitude: float,
    variables: List[str],
    past_days: int = 10,
    forecast_days: int = 1,
    timezone: str = "Europe/Berlin",
    retry: int = 3,
    timeout: int = 20,
) -> pd.DataFrame:
    """
    Fetches the last `past_days` and next `forecast_days` from Open-Meteo Forecast API.
    This is near-real-time model analysis/forecast (more up-to-date than ERA5 archive).
    Returns a DataFrame indexed by local time with the requested hourly variables.
    """
    base_url = "https://api.open-meteo.com/v1/forecast"
    variables = normalize_hourly_variables(variables)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
        "timezone": timezone,
    }
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    headers = {"User-Agent": "ErlangenWeather/1.0 (+https://github.com/Pa-Sto/ErlangenWeather)"}
    for attempt in range(retry):
        r = session.get(base_url, params=params, headers=headers, timeout=(5, timeout))
        try:
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == retry - 1:
                raise
    data = r.json().get("hourly", {})
    times = data.get("time", [])
    cols = {}
    for v in variables:
        vals = data.get(v)
        cols[v] = vals if vals is not None else [np.nan] * len(times)
    df = pd.DataFrame(cols, index=pd.to_datetime(times))
    df.index.name = "time"
    return df


from tensorflow.keras.models import load_model


# -- 1. Data wrapper (as before) ------------------------------------------

def get_open_meteo_data(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    variables: List[str],
    timezone: str = "UTC",
    retry: int = 3,
    timeout: int = 10
) -> pd.DataFrame:
    base_url = "https://api.open-meteo.com/v1/forecast"
    variables = normalize_hourly_variables(variables)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "timezone": timezone,
        "start": start,
        "end": end
    }
    for attempt in range(retry):
        r = requests.get(base_url, params=params, timeout=timeout)
        try:
            r.raise_for_status()
            break
        except:
            if attempt == retry - 1:
                raise
    data = r.json().get("hourly", {})
    times = data.get("time", [])
    cols = {}
    for v in variables:
        vals = data.get(v)
        if vals is None:
            vals = [np.nan] * len(times)
        cols[v] = vals
    df = pd.DataFrame(cols, index=pd.to_datetime(times))
    df.index.name = "time"
    return df

# -- 2. Time features -----------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek
    df["doy"]  = df.index.dayofyear - 1
    # cyclic
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
    df["doy_sin"]  = np.sin(2*np.pi*df["doy"]/365)
    df["doy_cos"]  = np.cos(2*np.pi*df["doy"]/365)
    # additional date-based features
    df["month"] = df.index.month - 1
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # raw timestamp (seconds since epoch)
    df["timestamp"] = df.index.view("int64") / 1e9  # seconds since epoch (avoids FutureWarning)
    return df

# -- Derived physical features -------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind components, dew point, pressure tendency, and previous-day helpers."""
    df = df.copy()
    # Backwards-compat: unify wind speed column name
    if "wind_speed_10m" in df.columns and "windspeed_10m" not in df.columns:
        df["windspeed_10m"] = df["wind_speed_10m"]
    # Wind components from speed + direction (meteorological: from-which → minus signs)
    if {"windspeed_10m", "winddirection_10m"}.issubset(df.columns):
        rad = np.deg2rad(df["winddirection_10m"])  # degrees → radians
        df["wind_u_10m"] = -df["windspeed_10m"] * np.sin(rad)
        df["wind_v_10m"] = -df["windspeed_10m"] * np.cos(rad)
    # Dew point (Magnus formula approximation)
    if {"temperature_2m", "relativehumidity_2m"}.issubset(df.columns):
        T = df["temperature_2m"].astype(float)
        RH = df["relativehumidity_2m"].clip(1, 100).astype(float)
        a, b = 17.27, 237.7
        alpha = (a * T) / (b + T) + np.log(RH / 100.0)
        df["dewpoint_2m"] = (b * alpha) / (a - alpha)
    # Pressure tendency (3-hour difference)
    if "pressure_msl" in df.columns:
        df["dp_dt_3h"] = df["pressure_msl"].diff(3)
    # Previous-day persistence helpers
    if "temperature_2m" in df.columns:
        df["temp_prev_day"] = df["temperature_2m"].shift(24)
        df["temp_delta_prev_day"] = df["temperature_2m"] - df["temp_prev_day"]
    return df

# -- 3. Windowing ---------------------------------------------------------


def make_windows(
    data: np.ndarray,
    window_in: int,
    window_out: int
):
    X, y = [], []
    for i in range(len(data) - window_in - window_out + 1):
        X.append(data[i:i+window_in])
        y.append(data[i+window_in:i+window_in+window_out, 0])  # temp only
    return np.array(X), np.array(y)

def _full_day_start_indices(idx: pd.DatetimeIndex) -> np.ndarray:
    """Return positions (int) where a full day (00:00..23:00 hourly) starts in idx."""
    # Ensure hourly, contiguous; rely on earlier interpolation/fill to create continuity
    # Find all midnights
    is_midnight = (idx.hour == 0)
    midnight_positions = np.where(is_midnight)[0]
    # Keep only those whose next 23 hours exist contiguously
    good_starts = []
    for p in midnight_positions:
        end_p = p + 23
        if end_p < len(idx) and (idx[end_p] - idx[p]).components.days == 0 and (idx[end_p] - idx[p]).components.hours == 23:
            # additionally ensure consecutive hourly steps
            if (idx[p:end_p+1].to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
                good_starts.append(p)
    return np.array(good_starts, dtype=int)

def prepare_training_data_days(
    df: pd.DataFrame,
    seq_days: int,
    label_days: int,
    train_ratio: float = 0.8,
    require_train: bool = True,
    targets: List[str] = None,
):
    """
    Day-aligned preparation:
    - Sort & impute gaps
    - Scale using TRAIN portion only when require_train is True and enough train days exist
      otherwise fall back to scaling on the full data (predict-only robustness)
    - Build windows that use `seq_days` consecutive full days as input and the
      next `label_days` full day(s) (24*label_days hours) as labels.
    Returns: X_train, X_val, y_train, y_val, split_windows, mean, std
    """
    df = df.sort_index()
    df = df.interpolate(method="time").ffill().bfill()

    idx = df.index
    values = df.values.astype("float32")

    # Find day start positions where we have a full 24-hour day
    starts = _full_day_start_indices(idx)
    if len(starts) == 0:
        raise ValueError("No full 24h days found in index; ensure hourly continuity.")

    num_days = len(starts)
    total_needed_days = seq_days + label_days

    # Determine training days (for scaling) and whether we have enough for strict train split
    train_days = int(num_days * train_ratio)
    have_strict_train = train_days >= total_needed_days

    # Select scaling rows
    if require_train and have_strict_train:
        # scale on rows up to the end of the last training day
        last_train_day_start = starts[train_days - 1]
        last_train_row = last_train_day_start + 24 - 1
        train_vals_rows = values[: last_train_row + 1]
    else:
        # robust fallback: scale on full available data (predict-only or small datasets)
        train_vals_rows = values

    mean = np.nanmean(train_vals_rows, axis=0)
    std = np.nanstd(train_vals_rows, axis=0)
    std[std == 0] = 1e-6
    values = (values - mean) / std

    # Build day-aligned windows with stride 24
    X, y = [], []
    # determine target indices in the current column order
    if targets is None:
        targets = ["temperature_2m"]
    target_idx = []
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"Target '{t}' not found in dataframe columns")
        target_idx.append(int(df.columns.get_loc(t)))
    for d in range(0, num_days - total_needed_days + 1):
        start_row = starts[d]
        in_end_row = start_row + seq_days * 24
        lbl_end_row = in_end_row + label_days * 24
        X.append(values[start_row:in_end_row])
        # next label_days full days for all targets → shape (H, C)
        Yh = values[in_end_row:lbl_end_row][:, target_idx]
        y.append(Yh)
    X = np.array(X)
    y = np.array(y)

    # Determine split_windows
    if require_train and have_strict_train:
        # Number of training windows so that labels are strictly inside training span
        train_windows = train_days - total_needed_days + 1
        if train_windows < 0:
            train_windows = 0
        split_windows = train_windows
    else:
        # Predict-only or small dataset: ensure at least one validation window when possible
        split_windows = max(len(X) - 1, 0)

    return (
        X[:split_windows], X[split_windows:],
        y[:split_windows], y[split_windows:],
        split_windows,
        mean, std,
        target_idx,
    )

def prepare_training_data(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    train_ratio: float = 0.8
):
    """
    - Sort & impute gaps (time interpolation + ffill/bfill)
    - Scale using TRAIN split only (no leakage)
    - Window into (X, y)
    Returns: X_train, X_val, y_train, y_val, split_windows
    """
    # 1) sort & impute
    df = df.sort_index()
    df = df.interpolate(method="time").ffill().bfill()

    # 2) compute split on raw rows (pre-window)
    values = df.values.astype("float32")
    split_rows = int(len(values) * train_ratio)

    # 3) scale with train-only stats
    train_vals = values[:split_rows]
    mean = np.nanmean(train_vals, axis=0)
    std = np.nanstd(train_vals, axis=0)
    std[std == 0] = 1e-6
    values = (values - mean) / std

    # 4) window after scaling
    X, y = make_windows(values, seq_len, horizon)
    split_windows = int(len(X) * train_ratio)

    return (
        X[:split_windows], X[split_windows:],
        y[:split_windows], y[split_windows:],
        split_windows
    )

# -- 4. Positional Encoding Layer -----------------------------------------

class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = np.arange(seq_len)[:, None]
        i   = np.arange(d_model)[None, :]
        angle = pos / np.power(10000, (2 * (i//2)) / d_model)
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pos_encoding = tf.cast(pe[None, :, :], tf.float32)

    def call(self, x):
        # x shape = (batch, seq_len, d_model)
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]

# -- 5. Transformer Block ------------------------------------------------

def transformer_block(d_model, num_heads, d_ff, dropout=0.1, name=None):
    inputs = layers.Input(shape=(None, d_model))
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(inputs, inputs)
    attn_out = layers.Dropout(dropout)(attn_out)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_out)

    ff = layers.Dense(d_ff, activation="relu")(out1)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff)

    return tf.keras.Model(inputs=inputs, outputs=out2, name=name or "transformer_block")

class CLSToken(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.cls = self.add_weight(
            name="cls_token", shape=(1, 1, d_model),
            initializer="glorot_uniform", trainable=True
        )
    def call(self, x):
        b = tf.shape(x)[0]
        cls = tf.repeat(self.cls, repeats=b, axis=0)
        return tf.concat([cls, x], axis=1)  # (batch, 1+seq_len, d_model)

def weighted_mse(y_true, y_pred):
    # Heavier weight on near-term horizons, linearly decaying to 0.5
    # Create weights of length H (forecast horizon)
    H = tf.shape(y_pred)[-1]              # int32 tensor
    num = tf.cast(H, tf.int32)            # ensure integer type for tf.linspace
    w = tf.linspace(1.0, 0.5, num)        # shape: (H,)
    w = tf.reshape(w, (1, -1))            # shape: (1, H) for broadcasting over batch

    sq_err = tf.square(y_true - y_pred)   # shape: (batch, H)
    return tf.reduce_mean(w * sq_err)     # mean over batch and horizon

# --- Robust model loader for custom loss ---
def load_model_safe(path: str):
    """Load a Keras model handling custom loss 'weighted_mse'.
    Tries with custom_objects first, then falls back to compile=False for inference-only.
    """
    try:
        return load_model(path, custom_objects={"weighted_mse": weighted_mse, "weighted_mse_multi": weighted_mse_multi})
    except Exception as e1:
        print(f"[Model] load_model with custom_objects failed: {e1}\n[Model] Retrying with compile=False …")
        return load_model(path, compile=False)

def weighted_mse_multi(y_true, y_pred):
    """Time-decay + per-channel weighting that adapts to TARGETS length.
    y_* shape: (batch, H, C)
    """
    H = tf.shape(y_pred)[1]
    C = tf.shape(y_pred)[2]
    # time weights from 1.0 → 0.5 across horizon
    w_t = tf.linspace(1.0, 0.5, H)      # (H,)
    w_t = tf.reshape(w_t, (1, H, 1))    # (1,H,1)
    # channel weights based on declared TARGETS
    name2w = {"temperature_2m": 1.0, "precipitation": 3.0, "rain": 3.0, "cloudcover": 0.5}
    w_list = [name2w.get(n, 1.0) for n in TARGETS]
    w_c = tf.constant(w_list, dtype=tf.float32)
    w_c = tf.reshape(w_c, (1, 1, -1))   # (1,1,Tlen)

    # Align to actual prediction channel count C (slice or pad with ones)
    tlen_static = w_c.shape[-1]
    if tlen_static is not None and y_pred.shape[-1] is not None:
        c_static = int(y_pred.shape[-1])
        tlen = int(tlen_static)
        if tlen >= c_static:
            w_c_sel = w_c[..., :c_static]
        else:
            pad = c_static - tlen
            w_pad = tf.ones((1, 1, pad), dtype=tf.float32)
            w_c_sel = tf.concat([w_c, w_pad], axis=-1)
    else:
        # fully dynamic fallback
        pad = tf.math.maximum(C - tf.shape(w_c)[-1], 0)
        w_pad = tf.ones((1, 1, pad), dtype=tf.float32)
        w_c_sel = tf.concat([w_c[..., :C], w_pad], axis=-1)

    # final weight = time * channel
    w = w_t * w_c_sel
    se = tf.square(y_true - y_pred)
    return tf.reduce_mean(w * se)
# -- 6. Build the time-series Transformer model --------------------------

def build_transformer_model(
    seq_len: int,
    feature_dim: int,
    window_out: int,
    num_targets: int = 1,
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 128,
    num_layers: int = 2,
):
    inputs = layers.Input(shape=(seq_len, feature_dim))
    # project features into d_model dims
    x = layers.Dense(d_model)(inputs)
    # prepend CLS token and add positional encodings for (seq_len + 1)
    x = CLSToken(d_model)(x)
    x = PositionalEncoding(seq_len + 1, d_model)(x)

    # stack N transformer blocks
    for i in range(num_layers):
        block = transformer_block(d_model, num_heads, d_ff, name=f"transformer_block_{i}")
        x = block(x)

    # CLS pooling
    x = layers.Lambda(lambda z: z[:, 0, :])(x)

    # final MLP → (window_out * num_targets) then reshape to (H, C)
    x = layers.Dense(window_out * num_targets)(x)
    outputs = layers.Reshape((window_out, num_targets))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ts_transformer_multi")
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=weighted_mse_multi)
    return model

# -- 7. Example end-to-end -----------------------------------------------


def save_prediction(
    model,
    X_val,
    df,
    seq_len: int,
    split: int,
    stride: int,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    target_idx: List[int] = None,
    lat: float = None,
    lon: float = None,
    output_file: str = 'prediction.json',
    point_file: str = 'prediction_point.json',
    timezone: str = 'Europe/Berlin',
):
    """
    Predicts next horizon with the trained model. Supports multi-output (H, C) where
    channels correspond to TARGETS. Writes:
      - prediction_multi.json : {ts: {temp_c, precip_mm, rain_mm, cloudcover_pct}}
      - prediction.json       : {ts: temp_c}  (for backward compatibility)
      - prediction_point.json : summary for temperature channel
      - updates history_predictions.json
    """
    last_window = X_val[-1:]
    # --- Compatibility shim: align features with model's expected input width ---
    try:
        expected_feat = int(model.input_shape[-1])
    except Exception:
        expected_feat = last_window.shape[-1]
    current_feat = int(last_window.shape[-1])
    if current_feat != expected_feat:
        if current_feat > expected_feat:
            print(f"[Compat] Truncating features from {current_feat} to {expected_feat} to match model input.")
            last_window = last_window[..., :expected_feat]
        else:
            pad = expected_feat - current_feat
            print(f"[Compat] Padding features from {current_feat} to {expected_feat} with zeros to match model input.")
            last_window = np.pad(last_window, ((0,0),(0,0),(0,pad)), mode='constant')
    # (summary block removed here; will create summary after all variables are defined)
    print(f"[Predict] Using input window shape: {last_window.shape}, model expects: (None, None, {expected_feat})")
    # --------------------------------------------------------------------------
    pred = model.predict(last_window)[0]
    # pred shape (H,) old models → upgrade to (H,1)
    if pred.ndim == 1:
        pred = pred[:, None]
    H, C = pred.shape
    # --- Denormalize & align output channels to TARGETS order ---
    T = len(target_idx) if target_idx is not None else 1
    # Initialize output matrix in TARGETS channel order
    den = np.zeros((H, T), dtype=np.float32)

    if mean is None or std is None or not target_idx:
        # No stats: best effort mapping
        if C >= 1:
            # Map first channel to temperature if available
            try:
                temp_ci = TARGETS.index("temperature_2m")
            except ValueError:
                temp_ci = 0
            den[:, temp_ci] = pred[:, 0]
        # others remain zero
    else:
        if C == T:
            # 1:1 mapping by channel index
            for ci, col_i in enumerate(target_idx):
                den[:, ci] = pred[:, ci] * std[col_i] + mean[col_i]
        elif C == 1:
            # Legacy model: single temperature channel
            try:
                temp_ci = TARGETS.index("temperature_2m")
            except ValueError:
                temp_ci = 0
            temp_col = target_idx[temp_ci]
            den[:, temp_ci] = pred[:, 0] * std[temp_col] + mean[temp_col]
            # other targets remain zero (precip/rain/cloud)
        else:
            # Fallback: map min(C, T) channels in order
            m = min(C, T)
            for ci in range(m):
                col_i = target_idx[ci]
                den[:, ci] = pred[:, ci] * std[col_i] + mean[col_i]
            # remaining channels stay zero

    # Replace any NaNs with zeros to avoid downstream JSON issues
    den = np.nan_to_num(den, nan=0.0)
    # ------------------------------------------------------------------

    # Align base time to end of last input window
    last_val_i = X_val.shape[0] - 1
    end_idx = (seq_len - 1) + (split + last_val_i) * stride
    base_time = df.index[end_idx]

    generated_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    now_local = datetime.now(gettz(timezone))
    forecast_day_dt = now_local.date()
    forecast_day = forecast_day_dt.isoformat()
    forecast_day_name = now_local.strftime('%A')

    multi = {}
    temp_only = {}
    for i in range(H):
        ts = (base_time + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%dT%H:%M:%S')
        # Map by TARGETS order
        vals = { TARGETS[ci]: float(den[i, ci]) for ci in range(min(C, len(TARGETS))) }
        entry = {
            "temp_c": float(vals.get("temperature_2m", np.nan)),
            "precip_mm": max(0.0, float(vals.get("precipitation", 0.0))),
            "rain_mm": max(0.0, float(vals.get("rain", 0.0))),
            "cloudcover_pct": float(vals.get("cloudcover", np.nan)),
        }
        multi[ts] = entry
        temp_only[ts] = entry["temp_c"]

    with open('prediction_multi.json', 'w') as f:
        json.dump(multi, f, indent=2)
    with open(output_file, 'w') as f:
        json.dump(temp_only, f, indent=2)

    # Temperature summary point
    temps = np.array([v["temp_c"] for v in multi.values()], dtype=float)
    argmax = int(np.nanargmax(temps))
    argmin = int(np.nanargmin(temps))
    t_keys = list(multi.keys())
    t_max = t_keys[argmax]
    t_min = t_keys[argmin]
    point_summary = {
        'lat': lat,
        'lon': lon,
        'units': 'C',
        'horizon_hours': H,
        'start_time': (base_time).strftime('%Y-%m-%dT%H:%M:%S'),
        'forecast_day': forecast_day,
        'forecast_day_name': forecast_day_name,
        'generated_at': generated_at,
        'max_temp_c': float(temps[argmax]),
        'max_time': t_max,
        'min_temp_c': float(temps[argmin]),
        'min_time': t_min,
        'mean_temp_c': float(np.nanmean(temps)),
    }
    with open(point_file, 'w') as f:
        json.dump(point_summary, f, indent=2)

    # Persist temp-only series for accuracy
    try:
        os.makedirs("predictions", exist_ok=True)
        daily_payload = {
            "meta": {
                "forecast_day": forecast_day,
                "forecast_day_name": forecast_day_name,
                "generated_at": generated_at,
                "lat": lat, "lon": lon,
                "units": "C"
            },
            "series": temp_only
        }
        daily_path = os.path.join("predictions", f"pred_{forecast_day}.json")
        with open(daily_path, "w") as f:
            json.dump(daily_payload, f, indent=2)
        _append_history_prediction("history_predictions.json", forecast_day, temp_only, generated_at)
    except Exception as e:
        print(f"[Warn] Could not write daily/history prediction files: {e}")

    # Build a summary for logging/diagnostics
    summary = {
        "horizon_hours": H,
        "n_channels": C,
        "start_time": (base_time).strftime('%Y-%m-%dT%H:%M:%S'),
        "generated_at": generated_at,
    }

    print(f"[Predict] Wrote {H} hourly values × {C} channel(s) spanning {(H//24)} day(s).")
    print('Prediction (temp only) written to', output_file)
    print('Multi-output prediction written to prediction_multi.json')
    print('Summary point written to', point_file)

    return summary

# -- Persist history + compute single-number accuracy ----------------------

def _append_history_prediction(history_file: str, forecast_day: str, pred_dict: dict, generated_at: str):
    """Append/replace the entry for forecast_day in a consolidated history JSON file."""
    history = []
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    # de-duplicate by date
    history = [h for h in history if h.get("date") != forecast_day]
    history.append({
        "date": forecast_day,
        "generated_at": generated_at,
        "series": pred_dict,
    })
    history.sort(key=lambda x: x.get("date", ""))
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def _ensure_cache_has_days(cache_file: str, latitude: float, longitude: float, days: list, timezone: str):
    """Ensure historical_data.csv has all hours for the given days; fetch missing days from Open-Meteo."""
    if not days:
        return
    # Load existing cache if present
    have = set()
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        dfc = pd.read_csv(cache_file, index_col="time", parse_dates=["time"]).sort_index()
        # mark days with full 24 hours present
        if not dfc.empty:
            # count hours per UTC date string
            counts = dfc.groupby(dfc.index.strftime('%Y-%m-%d')).size()
            have = set(counts[counts >= 24].index.tolist())
    # Identify which days are missing or incomplete
    missing = [d for d in sorted(set(days)) if d not in have]
    for d in missing:
        try:
            update_cache_with_historical(
                latitude, longitude,
                start=d, end=d,
                variables=["temperature_2m", "relativehumidity_2m", "pressure_msl",
                           "windspeed_10m", "winddirection_10m", "cloudcover", "shortwave_radiation",
                           "precipitation", "rain"],
                timezone=timezone,
                cache_file=cache_file,
                show_progress=False
            )
        except Exception as e:
            print(f"[Accuracy] Could not fetch actuals for {d}: {e}")


def update_accuracy_from_history(cache_file: str,
                                 history_file: str = "history_predictions.json",
                                 overall_file: str = "accuracy_overall.json",
                                 timezone: str = "Europe/Berlin",
                                 min_age_days: int = 5):
    """Compute a single accuracy % using skill vs persistence baseline.
    accuracy_percent = mean over days of max(0, 100 * (1 - MSE_model / MSE_persistence)).
    Persistence forecast uses actual(T-24h) for each hour.
    Only evaluates days at least `min_age_days` before today (to allow archive latency).
    """
    if not (os.path.exists(history_file) and os.path.getsize(history_file) > 0):
        print("[Accuracy] No history_predictions.json yet; skipping overall accuracy.")
        return
    with open(history_file, "r") as f:
        history = json.load(f)
    if not history:
        return
    # Only evaluate days at least min_age_days before today (local time)
    today_local = datetime.now(gettz(timezone)).date()
    cutoff = (today_local - timedelta(days=min_age_days)).isoformat()
    eval_entries = [h for h in history if h.get("date", "") <= cutoff]
    if not eval_entries:
        print("[Accuracy] No past days to evaluate yet.")
        return
    # Ensure cache has those days and the preceding day for persistence
    days_needed = set(h["date"] for h in eval_entries)
    # Also need previous day for persistence baseline
    prev_days = set((date.fromisoformat(d) - timedelta(days=1)).isoformat() for d in days_needed)
    _ensure_cache_has_days(cache_file, LAT, LON, list(days_needed | prev_days), timezone)

    # Load cache after ensuring
    dfc = pd.read_csv(cache_file, index_col="time", parse_dates=["time"]).sort_index()

    daily_percents = []
    for h in eval_entries:
        fday = h.get("date")
        series = h.get("series", {})
        # Align to hours present in both prediction and actuals
        times = sorted(series.keys())
        if not times:
            continue
        # Build arrays
        y_pred = []
        y_true = []
        y_pers = []
        for ts in times:
            try:
                t = pd.Timestamp(ts)
                # actual
                if t in dfc.index:
                    actual = dfc.loc[t, "temperature_2m"]
                else:
                    # try without seconds
                    actual = dfc.loc.get(t, np.nan)
                if pd.isna(actual):
                    continue
                # persistence = actual at t - 24h
                t_prev = t - pd.Timedelta(hours=24)
                if t_prev in dfc.index:
                    pers = dfc.loc[t_prev, "temperature_2m"]
                else:
                    pers = np.nan
                if pd.isna(pers):
                    continue
                y_true.append(float(actual))
                y_pred.append(float(series[ts]))
                y_pers.append(float(pers))
            except Exception:
                continue
        if len(y_true) < 6:  # require at least 6 hours to compute stable metric
            continue
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        y_pers = np.array(y_pers, dtype=float)
        mse_model = float(np.mean((y_pred - y_true) ** 2))
        mse_pers = float(np.mean((y_pers - y_true) ** 2)) if np.any(~np.isnan(y_pers)) else None
        if mse_pers is None or mse_pers < 1e-6:
            # Fallback: if persistence has ~zero error (rare) use small epsilon
            mse_pers = 1e-6
        acc = 100.0 * (1.0 - (mse_model / mse_pers))
        # clip to [0, 100] for display
        acc = float(max(0.0, min(100.0, acc)))
        daily_percents.append({"date": fday, "accuracy_percent": acc})

    if not daily_percents:
        print("[Accuracy] Could not compute any daily accuracy values.")
        return

    overall = float(np.mean([d["accuracy_percent"] for d in daily_percents]))
    payload = {
        "accuracy_percent": overall,
        "n_days": len(daily_percents),
        "updated_at": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "definition": "100*(1 - MSE_model/MSE_persistence) clipped to [0,100]",
    }
    with open(overall_file, "w") as f:
        json.dump(payload, f, indent=2)
    # Also write per-day breakdown (optional)
    with open("accuracy_daily.json", "w") as f:
        json.dump(daily_percents, f, indent=2)
    print(f"[Accuracy] Overall accuracy: {overall:.1f}% over {len(daily_percents)} day(s)")

def save_accuracy(*args, **kwargs):
    """Deprecated: legacy accuracy.json removed. Update single-number accuracy for the site.
    This writes only accuracy_overall.json based on history_predictions.json and actuals in
    historical_data.csv. Left in place so older calls don't break.
    """
    try:
        update_accuracy_from_history(
            cache_file=kwargs.get("cache_file", "historical_data.csv"),
            history_file="history_predictions.json",
            overall_file="accuracy_overall.json",
            timezone="Europe/Berlin",
        )
        print("[Accuracy] accuracy_overall.json updated.")
    except Exception as e:
        print(f"[Accuracy] Skipped updating overall accuracy: {e}")

def summarize_cache(cache_file: str = "historical_data.csv"):
    """
    Print coverage of the cached CSV: time span, expected hours, missing hours,
    and per-variable non-null counts.
    """
    if not os.path.exists(cache_file):
        print(f"[Cache] {cache_file} not found.")
        return
    dfc = pd.read_csv(cache_file, index_col="time", parse_dates=["time"])  # local var to avoid confusion
    dfc = dfc.sort_index()
    if dfc.empty:
        print("[Cache] Cache is empty.")
        return
    start, end = dfc.index.min(), dfc.index.max()
    expected = pd.date_range(start, end, freq="H")
    missing_index = expected.difference(dfc.index)
    coverage_pct = 100.0 * (len(expected) - len(missing_index)) / len(expected)
    print("=== Cache summary ===")
    print(f"Time span: {start} → {end}  ({len(expected)} expected hours)")
    print(f"Present rows: {len(dfc):,}  | Missing hours: {len(missing_index):,}  | Coverage: {coverage_pct:.2f}%")
    nn = dfc.notna().sum().sort_values(ascending=False)
    total = len(dfc)
    print("\nPer-variable non-null counts:")
    for col, cnt in nn.items():
        print(f"  {col:>24}: {cnt:>8} ({100.0*cnt/total:5.1f}%)")
    if len(missing_index) > 0:
        print("\nExamples of missing hours:")
        for ts in list(missing_index[:10]):
            print(f"  {ts}")

def _has_data_window(lat, lon, start_d: date, end_d: date, variables, timezone="UTC") -> bool:
    dfw = get_historical_data(
        lat, lon,
        start=start_d.isoformat(),
        end=end_d.isoformat(),
        variables=variables,
        timezone=timezone
    )
    return not dfw.dropna(how="all").empty

def probe_archive_range(lat, lon, variables, timezone="UTC", window_days=30):
    """
    Find earliest and latest available dates with data for the given location/variables
    by probing the Archive API with a small window (default 30 days).
    """
    today = datetime.utcnow().date()
    # — Earliest —
    lo, hi = MIN_ARCHIVE_DATE, today
    w = timedelta(days=window_days)
    earliest = None
    while lo <= hi:
        mid_ord = (lo.toordinal() + hi.toordinal()) // 2
        mid = date.fromordinal(mid_ord)
        has = False
        try:
            has = _has_data_window(lat, lon, mid, min(mid + w, today), variables, timezone)
        except Exception:
            has = False
        if has:
            earliest = mid
            hi = mid - timedelta(days=1)
        else:
            lo = mid + timedelta(days=1)
    # — Latest —
    lo, hi = MIN_ARCHIVE_DATE, today
    latest = None
    while lo <= hi:
        mid_ord = (lo.toordinal() + hi.toordinal()) // 2
        mid = date.fromordinal(mid_ord)
        start_win = max(MIN_ARCHIVE_DATE, mid - w)
        has = False
        try:
            has = _has_data_window(lat, lon, start_win, mid, variables, timezone)
        except Exception:
            has = False
        if has:
            latest = mid
            lo = mid + timedelta(days=1)
        else:
            hi = mid - timedelta(days=1)
    return earliest, latest

# --- Metrics: per-target MSE (temp, rain, cloud) -------------------------------

def compute_and_save_mse(
    cache_csv: str = "historical_data.csv",
    history_path: str = "history_predictions.json",
    multi_path: str = "prediction_multi.json",
    out_path: str = "metrics.json",
    min_lag_days: int = 5,
) -> None:
    """Compute MSE for temperature (°C), rain (mm/h), and cloudcover (%) by
    aligning past predictions to archive actuals. Safe no-op if inputs are missing.

    Sources:
      - Actuals:        cache_csv (historical_data.csv) with columns like
                        temperature_2m, rain/precipitation, cloudcover
      - Predictions:    Prefer consolidated history (history_predictions.json)
                        for temperature; fall back to latest multi-output
                        (prediction_multi.json) to also score rain/cloud when available.
    """
    import os, json
    import numpy as np
    import pandas as pd
    from datetime import datetime

    def _pick(cols, cands):
        for c in cands:
            if c in cols:
                return c
        return None

    # 1) Load actuals (archive cache)
    if not os.path.exists(cache_csv) or os.path.getsize(cache_csv) == 0:
        print(f"[Metrics] No cache CSV at {cache_csv}; skipping MSE")
        return
    try:
        df = pd.read_csv(cache_csv, index_col="time", parse_dates=["time"]).sort_index()
    except Exception as e:
        print(f"[Metrics] Failed reading {cache_csv}: {e}")
        return

    # Actual columns (names can vary)
    col_temp  = _pick(df.columns, ["temperature_2m", "temp_c", "temperature_c", "temperature"])  # °C
    col_rain  = _pick(df.columns, ["rain", "precipitation", "rain_mm", "precip_mm"])             # mm/h
    col_cloud = _pick(df.columns, ["cloudcover", "cloudcover_pct"])                               # %

    # 2) Load predictions
    preds = []  # rows: {ts, pred_temp, pred_rain, pred_cloud}

    # (a) Preferred: history_predictions.json (temp only)
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        try:
            hist = json.load(open(history_path, "r", encoding="utf-8"))
            for entry in (hist if isinstance(hist, list) else []):
                series = entry.get("series", {})
                for ts, val in series.items():
                    try:
                        t = pd.to_datetime(ts)
                        preds.append({"ts": t, "pred_temp": float(val), "pred_rain": None, "pred_cloud": None})
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Metrics] Failed reading {history_path}: {e}")

    # (b) Fallback/augment: prediction_multi.json (latest run with rain/cloud)
    if os.path.exists(multi_path) and os.path.getsize(multi_path) > 0:
        try:
            multi = json.load(open(multi_path, "r", encoding="utf-8"))
            # multi is mapping ts -> {temp_c, precip_mm, rain_mm, cloudcover_pct}
            for ts, obj in multi.items():
                try:
                    t = pd.to_datetime(ts)
                    row = {
                        "ts": t,
                        "pred_temp": float(obj.get("temp_c")) if obj.get("temp_c") is not None else None,
                        "pred_rain": (float(obj.get("rain_mm")) if obj.get("rain_mm") is not None
                                      else (float(obj.get("precip_mm")) if obj.get("precip_mm") is not None else None)),
                        "pred_cloud": float(obj.get("cloudcover_pct")) if obj.get("cloudcover_pct") is not None else None,
                    }
                    preds.append(row)
                except Exception:
                    continue
        except Exception as e:
            print(f"[Metrics] Failed reading {multi_path}: {e}")

    if not preds:
        print("[Metrics] No predictions available; skipping MSE")
        return

    P = pd.DataFrame(preds).dropna(subset=["ts"]).sort_values("ts")

    # Only evaluate predictions old enough that the archive has ground truth
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=min_lag_days)
    P = P[P["ts"] <= cutoff]
    if P.empty:
        print("[Metrics] No sufficiently old predictions to score yet; skipping MSE")
        return

    # 3) Join with actuals by timestamp
    A = df.copy()
    A["ts"] = A.index
    M = pd.merge(P, A, on="ts", how="inner")
    if M.empty:
        print("[Metrics] No overlapping timestamps between predictions and archive; skipping MSE")
        return

    def _mse(pred_col, act_col):
        if pred_col not in M or act_col not in M:
            return None, 0
        sub = M[[pred_col, act_col]].dropna()
        if sub.empty:
            return None, 0
        err = (sub[pred_col].astype(float) - sub[act_col].astype(float)) ** 2
        return float(np.mean(err)), int(len(sub))

    mse_temp, n_t = _mse("pred_temp",  col_temp)  if col_temp  else (None, 0)
    rain_truth_col = col_rain
    mse_rain, n_r = _mse("pred_rain",  rain_truth_col) if rain_truth_col else (None, 0)
    mse_cloud, n_c = _mse("pred_cloud", col_cloud) if col_cloud else (None, 0)

    out = {
        "updated_at": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "window_min_lag_days": int(min_lag_days),
        "n_points": {"temp": n_t, "rain": n_r, "cloud": n_c},
        "mse": {
            "temperature_c": mse_temp,
            "rain_mm_per_h": mse_rain,
            "cloudcover_pct": mse_cloud,
        },
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[Metrics] Saved MSE metrics to {out_path}")
    except Exception as e:
        print(f"[Metrics] Failed writing {out_path}: {e}")

if __name__ == "__main__":
    # parameters via CLI
    parser = argparse.ArgumentParser(description="Train or run prediction for ErlangenWeather model")
    parser.add_argument("--predict-only", action="store_true", help="Skip training; load saved model and only write predictions")
    parser.add_argument("--download-data", action="store_true", help="Download an absolute/relative range before running")
    parser.add_argument("--extend-to-present", action="store_true", default=False, help="Extend cache forward to today from last cached day")
    parser.add_argument("--days", type=int, default=365, help="Relative range (days) if --download-data is set")
    parser.add_argument("--absolute-start", type=str, default=None, help="YYYY-MM-DD absolute start (optional)")
    parser.add_argument("--absolute-end", type=str, default=None, help="YYYY-MM-DD absolute end (optional)")
    parser.add_argument("--cache-file", type=str, default="historical_data.csv", help="Path to CSV cache file")
    parser.add_argument("--lat", type=float, default=49.59)
    parser.add_argument("--lon", type=float, default=11.00)
    parser.add_argument("--source", type=str, choices=["archive", "forecast"], default="archive",
                        help="Data source for feature dataframe: 'archive' (ERA5, delayed) or 'forecast' (past_days, near real-time)")
    parser.add_argument("--past-days", type=int, default=10, help="When --source=forecast, include this many past days")
    parser.add_argument("--forecast-days", type=int, default=1, help="When --source=forecast, include this many future days")
    parser.add_argument("--note", type=str, default=None, help="Freeform note to include in model logs")
    parser.add_argument("--tag", type=str, default=None, help="Short tag or version label for this run")
    args = parser.parse_args()

    training = not args.predict_only
    download_data = args.download_data
    extend_to_present = args.extend_to_present
    days = args.days
    cache_file = args.cache_file
    LAT, LON = args.lat, args.lon

    # fetch & prep (with caching)
    vars = [
        "temperature_2m", "relativehumidity_2m", "pressure_msl",
        "windspeed_10m", "winddirection_10m", "cloudcover", "shortwave_radiation",
        "precipitation", "rain"
    ]
    if download_data and days > 30846 and not (args.absolute_start and args.absolute_end):
        print(f"[Warning] days={days} implies a very large download. Skipping auto-download. Set days smaller or provide absolute range.")
    elif download_data:
        if args.absolute_start and args.absolute_end:
            start = args.absolute_start
            end = args.absolute_end
        else:
            end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=days)
            start = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            end = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        print(f"[Download] Start: {start[:10]} End: {end[:10]}")
        update_cache_with_historical(
            LAT, LON, start, end, vars,
            timezone="Europe/Berlin",
            cache_file=cache_file,
            show_progress=True
        )
    if extend_to_present:
        extend_cache_to_present(
            LAT, LON, vars,
            timezone="Europe/Berlin",
            cache_file=cache_file,
            show_progress=True
        )
    if args.source == "archive":
        df = pd.read_csv(cache_file, index_col='time', parse_dates=['time']).sort_index()
        print("[Source] Using ARCHIVE (ERA5) cache for training/prediction.")
    else:
        print("[Source] Using FORECAST past_days window for near-real-time features.")
        df = get_recent_forecast(
            LAT, LON, vars,
            past_days=max(1, args.past_days),
            forecast_days=max(0, args.forecast_days),
            timezone="Europe/Berlin",
        )

    # --- Ensure all TARGETS columns exist in df; backfill if missing ---
    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        if args.source == "archive":
            # Backfill missing target columns from the archive for the df span
            span_start = df.index.min().date().isoformat()
            span_end = df.index.max().date().isoformat()
            print(f"[Archive] Backfilling missing targets {missing_targets} for {span_start} → {span_end}")
            try:
                df_fill = get_historical_data(
                    LAT, LON, span_start, span_end, missing_targets, timezone="Europe/Berlin"
                )
                # join new columns into df
                df = df.join(df_fill[missing_targets], how="left")
                # also persist in cache for future runs
                if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
                    dfc = pd.read_csv(cache_file, index_col="time", parse_dates=["time"]).sort_index()
                    dfc = dfc.join(df_fill[missing_targets], how="left")
                    dfc.to_csv(cache_file)
            except Exception as e:
                print(f"[Archive] Could not backfill missing targets {missing_targets}: {e}")
        else:
            # For forecast source: re-fetch including the missing targets explicitly
            print(f"[Forecast] Re-fetching to include missing targets {missing_targets}")
            try:
                df = get_recent_forecast(
                    LAT, LON, normalize_hourly_variables(vars + missing_targets),
                    past_days=max(1, args.past_days),
                    forecast_days=max(0, args.forecast_days),
                    timezone="Europe/Berlin",
                )
            except Exception as e:
                print(f"[Forecast] Could not re-fetch with missing targets {missing_targets}: {e}")
    # -------------------------------------------------------------------

    df = add_time_features(df)
    df = add_derived_features(df)

    # --- Availability checks ---
    # summarize_cache(cache_file)
    # earliest, latest = probe_archive_range(49.59, 11.00, vars, timezone="Europe/Berlin")
    # if earliest and latest:
    #     total_days = (latest - earliest).days + 1
    #     total_hours = total_days * 24
    #     print(f"[API availability @ 49.59,11.00] {earliest} → {latest}  (~{total_days} days, ~{total_hours} hours)")
    # else:
    #     print("[API availability] Could not determine earliest/latest via probe.")

    # windows: input last 10 days (10*24h), predict next 1 day (24h) temperatures
    SEQ_LEN = SEQ_DAYS * 24
    HORIZON = LABEL_DAYS * 24

    X_train, X_val, y_train, y_val, split, mean, std, target_idx = prepare_training_data_days(
        df, SEQ_DAYS, LABEL_DAYS, train_ratio=0.8, require_train=training, targets=TARGETS
    )

    if training:
        # Explicit hyperparameters (also logged)
        D_MODEL, N_HEADS, D_FF, N_LAYERS = 64, 4, 128, 2
        model = build_transformer_model(
            seq_len=SEQ_LEN,
            feature_dim=X_train.shape[-1],
            window_out=HORIZON,
            num_targets=NUM_TARGETS,
            d_model=D_MODEL,
            num_heads=N_HEADS,
            d_ff=D_FF,
            num_layers=N_LAYERS,
        )
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
        ]
        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        train_seconds = int(time.time() - t0)

        # Save without optimizer state; we only need inference in CI / predict-only
        model.save("model", include_optimizer=False)

        # ---- Log training event ----
        hist = history.history
        val_losses = hist.get("val_loss", [])
        best_val = float(min(val_losses)) if val_losses else None
        final_val = float(val_losses[-1]) if val_losses else None
        epochs_run = int(len(hist.get("loss", [])))
        git = _get_git_info()
        _append_model_log({
            "event": "train",
            "tag": args.tag,
            "note": args.note,
            "source": args.source,
            "targets": TARGETS,
            "seq_days": SEQ_DAYS,
            "label_days": LABEL_DAYS,
            "horizon_hours": HORIZON,
            "n_features": int(X_train.shape[-1]),
            "n_train_windows": int(X_train.shape[0]),
            "n_val_windows": int(X_val.shape[0]),
            "best_val_loss": best_val,
            "final_val_loss": final_val,
            "epochs_run": epochs_run,
            "train_seconds": train_seconds,
            "d_model": D_MODEL,
            "num_heads": N_HEADS,
            "d_ff": D_FF,
            "num_layers": N_LAYERS,
            **git,
        })
    else:
        if not os.path.exists("model"):
            raise SystemExit("[Model] No saved model found at 'model'. Train once locally (run without --predict-only) or commit the 'model/' directory.")
        model = load_model_safe("model")
    # Save outputs via helper functions and capture summary for logging
    pred_summary = save_prediction(
        model, X_val, df,
        seq_len=SEQ_LEN, split=split, stride=24,
        mean=mean, std=std, target_idx=target_idx,
        lat=LAT, lon=LON,
        output_file='prediction.json',
        point_file='prediction_point.json',
        timezone='Europe/Berlin',
    )
    # (Obsolete: legacy accuracy.json output removed)
    # Update single-number accuracy (vs persistence baseline) from saved history
    update_accuracy_from_history(
        cache_file=cache_file,
        history_file="history_predictions.json",
        overall_file="accuracy_overall.json",
        timezone="Europe/Berlin",
        min_age_days=5,
    )
    # Compute per-target MSE metrics and save to metrics.json (non-fatal)
    try:
        compute_and_save_mse(
            cache_csv=cache_file,
            history_path="history_predictions.json",
            multi_path="prediction_multi.json",
            out_path="metrics.json",
            min_lag_days=5,
        )
    except Exception as _e:
        print(f"[Metrics] Skipped MSE computation: {_e}")    # ---- Log prediction event ----

    overall_acc = None
    try:
        with open("accuracy_overall.json", "r") as f:
            overall_acc = float(json.load(f).get("accuracy_percent"))
    except Exception:
        pass
    try:
        git = _get_git_info()
        _append_model_log({
            "event": "predict",
            "tag": args.tag,
            "note": args.note,
            "source": args.source,
            "targets": TARGETS,
            "seq_days": SEQ_DAYS,
            "label_days": LABEL_DAYS,
            "horizon_hours": pred_summary.get("horizon_hours") if isinstance(pred_summary, dict) else None,
            "n_channels": pred_summary.get("n_channels") if isinstance(pred_summary, dict) else None,
            "start_time": pred_summary.get("start_time") if isinstance(pred_summary, dict) else None,
            "generated_at": pred_summary.get("generated_at") if isinstance(pred_summary, dict) else None,
            "overall_accuracy": overall_acc,
            **git,
        })
    except Exception as e:
        print(f"[Log] Skipped writing prediction log: {e}")
