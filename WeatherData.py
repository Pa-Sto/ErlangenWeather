import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import List

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
    data = r.json()["hourly"]
    df = pd.DataFrame({v: data[v] for v in variables},
                      index=pd.to_datetime(data["time"]))
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
    attn_out = layers.MultiHeadAttention(num_heads, key_dim=d_model)(inputs, inputs)
    attn_out = layers.Dropout(dropout)(attn_out)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_out)

    ff = layers.Dense(d_ff, activation="relu")(out1)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff)

    return tf.keras.Model(inputs=inputs, outputs=out2, name=name or "transformer_block")

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
    x = PositionalEncoding(seq_len, d_model)(x)

    # stack N transformer blocks
    for i in range(num_layers):
        block = transformer_block(d_model, num_heads, d_ff, name=f"transformer_block_{i}")
        x = block(x)

    # optionally pool: take final time-step
    x = layers.Lambda(lambda z: z[:, -1, :])(x)
    # or global average: z = layers.GlobalAveragePooling1D()(x)

    # final MLP to forecast next window_out temps
    outputs = layers.Dense(window_out)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ts_transformer")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# -- 7. Example end-to-end -----------------------------------------------

if __name__ == "__main__":
    # fetch & prep
    vars = ["temperature_2m","relativehumidity_2m","pressure_msl",
            "wind_speed_10m","cloudcover"]
    df = get_open_meteo_data(52.52, 13.41,
        start="2025-07-01T00:00:00Z",
        end="2025-07-15T23:00:00Z",
        variables=vars,
        timezone="Europe/Berlin"
    )
    df = add_time_features(df)
    data = (df.values - df.values.mean(axis=0)) / df.values.std(axis=0)

    # windows
    SEQ_LEN, HORIZON = 24, 3
    X, y = make_windows(data, SEQ_LEN, HORIZON)
    # train/val split
    split = int(len(X)*0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # build & fit
    model = build_transformer_model(
        seq_len=SEQ_LEN,
        feature_dim=X.shape[-1],
        window_out=HORIZON
    )
    model.summary()

    model.fit(
      X_train, y_train,
      validation_data=(X_val, y_val),
      epochs=10, batch_size=16
    )