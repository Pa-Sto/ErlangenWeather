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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Earliest supported start date for Open-Meteo Archive API (reanalysis)

MIN_ARCHIVE_DATE = date(1940, 1, 1)

# Normalize Open-Meteo hourly variable names (avoid API 4xx/5xx due to mismatches)
OPEN_METEO_HOURLY_ALIASES = {
    "wind_speed_10m": "windspeed_10m",  # common alias → official name
    "windspeed_10m": "windspeed_10m",
    "winddirection_10m": "winddirection_10m",
}

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
    Extend the local cache forward to 'today' (UTC) by downloading the missing
    date range from the last cached timestamp + 1 day up to today.
    If the cache does not exist or is empty, start from MIN_ARCHIVE_DATE.
    """
    today = datetime.utcnow().date()
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
    train_ratio: float = 0.8
):
    """
    Day-aligned preparation:
    - Sort & impute gaps
    - Scale using TRAIN portion only (rows up to end of the train day split)
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
    # Determine the number of training days
    train_days = int(num_days * train_ratio)
    if train_days < (seq_days + label_days):
        raise ValueError("Not enough days in training split to form at least one window.")

    # Compute scaling stats using rows up to the end of the training days
    last_train_day_start = starts[train_days - 1]
    last_train_row = last_train_day_start + 24 - 1  # end of that day
    train_vals_rows = values[: last_train_row + 1]
    mean = np.nanmean(train_vals_rows, axis=0)
    std = np.nanstd(train_vals_rows, axis=0)
    std[std == 0] = 1e-6
    values = (values - mean) / std

    # Build day-aligned windows with stride 24
    X, y = [], []
    # valid window starts are such that we have seq_days inputs and label_days labels
    total_needed_days = seq_days + label_days
    for d in range(0, num_days - total_needed_days + 1):
        start_row = starts[d]
        in_end_row = start_row + seq_days * 24
        lbl_end_row = in_end_row + label_days * 24
        X.append(values[start_row:in_end_row])
        # predict temperature (column 0) for next day(s)
        y.append(values[in_end_row:lbl_end_row, 0])
    X = np.array(X)
    y = np.array(y)

    # Number of training windows so that labels are strictly inside training span
    train_windows = train_days - (seq_days + label_days) + 1
    if train_windows < 0:
        train_windows = 0
    split_windows = train_windows

    return (
        X[:split_windows], X[split_windows:],
        y[:split_windows], y[split_windows:],
        split_windows,
        mean, std
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

# -- 6. Build the time-series Transformer model --------------------------

def build_transformer_model(
    seq_len: int,
    feature_dim: int,
    window_out: int,
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

    # final MLP to forecast next window_out temps
    outputs = layers.Dense(window_out)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ts_transformer")
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=weighted_mse, metrics=["mae"])
    return model

# -- 7. Example end-to-end -----------------------------------------------


def save_prediction(
    model,
    X_val,
    df,
    seq_len: int,
    split: int,
    stride: int,
    temp_mean: float = None,
    temp_std: float = None,
    lat: float = None,
    lon: float = None,
    output_file: str = 'prediction.json',
    point_file: str = 'prediction_point.json'
):
    """
    Generates next-horizon prediction from the last window of X_val,
    then writes a timestamp->value mapping to output_file.
    """
    last_window = X_val[-1:]
    next_pred = model.predict(last_window)[0]
    # Denormalize if stats provided
    if temp_mean is not None and temp_std is not None:
        next_vals = next_pred * temp_std + temp_mean
    else:
        next_vals = next_pred
    horizon = len(next_vals)

    # Align timestamps to the end of the last validation input window (day stride)
    last_val_i = X_val.shape[0] - 1
    end_idx = (seq_len - 1) + (split + last_val_i) * stride
    base_time = df.index[end_idx]

    # Metadata for display
    generated_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    forecast_day_dt = (base_time + pd.Timedelta(days=1)).date()
    forecast_day = forecast_day_dt.isoformat()
    forecast_day_name = (base_time + pd.Timedelta(days=1)).strftime('%A')

    # Build mapping
    pred_dict = {}
    for i, val in enumerate(next_vals):
        t = base_time + pd.Timedelta(hours=i+1)
        pred_dict[t.strftime('%Y-%m-%dT%H:%M:%S')] = float(val)

    # Write series JSON (Celsius values)
    with open(output_file, 'w') as f:
        json.dump(pred_dict, f, indent=2)

    # Also write a summary point file: max/min/mean over the next day
    arr = np.array(next_vals)
    argmax = int(np.argmax(arr))
    argmin = int(np.argmin(arr))
    t_max = (base_time + pd.Timedelta(hours=argmax+1)).strftime('%Y-%m-%dT%H:%M:%S')
    t_min = (base_time + pd.Timedelta(hours=argmin+1)).strftime('%Y-%m-%dT%H:%M:%S')
    point_summary = {
        'lat': lat,
        'lon': lon,
        'units': 'C',
        'horizon_hours': horizon,
        'start_time': base_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'forecast_day': forecast_day,
        'forecast_day_name': forecast_day_name,
        'generated_at': generated_at,
        'max_temp_c': float(arr[argmax]),
        'max_time': t_max,
        'min_temp_c': float(arr[argmin]),
        'min_time': t_min,
        'mean_temp_c': float(arr.mean())
    }
    with open(point_file, 'w') as f:
        json.dump(point_summary, f, indent=2)

    print('Next day Prediction (°C) written to', output_file)
    print('Summary point written to', point_file)

def save_accuracy(model, X_val, y_val, df, split, seq_len, output_file='accuracy.json', stride: int = 1, temp_mean: float = None, temp_std: float = None):
    """
    Computes MAE and MSE for the first forecast hour over the validation set,
    maps each to its timestamp, and writes a list of dicts to output_file.
    """
    # Predict full horizon for all validation windows
    val_preds = model.predict(X_val)
    # Determine horizon length
    horizon = val_preds.shape[1] if val_preds.ndim > 1 else 1
    # Compute per-step errors and squared errors
    if temp_mean is not None and temp_std is not None:
        # Denormalize to Celsius
        val_preds_c = val_preds * temp_std + temp_mean
        y_val_c = y_val * temp_std + temp_mean
        errors = np.abs(val_preds_c - y_val_c)
        mses = errors ** 2
    else:
        errors = np.abs(val_preds - y_val)
        mses = errors ** 2

    # Compute corresponding timestamps
    # End-of-input index for each window i in validation set (i counted from 0 within val)
    # Global window index = split + i; end position = seq_len - 1 + (split + i) * stride
    end_positions = [seq_len - 1 + (split + i) * stride for i in range(X_val.shape[0])]
    val_times = df.index[end_positions]

    # Build history list for each validation sample and each forecast hour
    history_list = []
    for i, t0 in enumerate(val_times):
        for h in range(horizon):
            t = t0 + pd.Timedelta(hours=h+1)
            history_list.append({
                'date': t.strftime('%Y-%m-%dT%H:%M:%S'),
                'mae': float(errors[i, h] if horizon > 1 else errors[i]),
                'mse': float(mses[i, h] if horizon > 1 else mses[i])
            })

    # Write out JSON
    with open(output_file, 'w') as f:
        json.dump(history_list, f, indent=2)

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
        "windspeed_10m", "winddirection_10m", "cloudcover", "shortwave_radiation"
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
    df = pd.read_csv(cache_file, index_col='time', parse_dates=['time']).sort_index()
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
    SEQ_DAYS = 10
    LABEL_DAYS = 1
    SEQ_LEN = SEQ_DAYS * 24
    HORIZON = LABEL_DAYS * 24

    # Prepare day-aligned training data (clean, scale-on-train, day windows)
    X_train, X_val, y_train, y_val, split, mean, std = prepare_training_data_days(
        df, SEQ_DAYS, LABEL_DAYS, train_ratio=0.8
    )
    temp_mean = float(mean[0])
    temp_std = float(std[0])

    if training:
        # build & fit
        model = build_transformer_model(
            seq_len=SEQ_LEN,
            feature_dim=X_train.shape[-1],
            window_out=HORIZON
        )
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
        ]
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        model.save("model")
    else:
        model = load_model("model")
    # Save outputs via helper functions
    save_prediction(
        model, X_val, df,
        seq_len=SEQ_LEN, split=split, stride=24,
        temp_mean=temp_mean, temp_std=temp_std,
        lat=LAT, lon=LON,
        output_file='prediction.json',
        point_file='prediction_point.json'
    )
    save_accuracy(
        model, X_val, y_val, df,
        split, SEQ_LEN,
        output_file='accuracy.json', stride=24,
        temp_mean=temp_mean, temp_std=temp_std
    )
    # Optionally verify JSON files
