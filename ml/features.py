import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12",
    "rv_6", "rv_12", "rv_24",
    "volume_z", "volume_change",
    "range_pct", "body_pct",
    "ema_gap_12_36", "ema_gap_24_72",
    "funding_rate", "funding_z",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def zscore(s, w=48):
    mean = s.rolling(w).mean()
    std = s.rolling(w).std().replace(0, np.nan)
    return (s - mean) / std


def _cyclical_time_features(x):
    ts = pd.to_datetime(x["t"].astype("int64"), unit="ms", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    return (
        np.sin(2*np.pi*hour/24.0), np.cos(2*np.pi*hour/24.0),
        np.sin(2*np.pi*dow/7.0), np.cos(2*np.pi*dow/7.0),
    )


def build_features(df, funding_rate=0.0, funding_series=None):
    """Only information known at the candle timestamp is used."""
    x = df.copy()
    close = x["c"].astype(float)
    volume = x["v"].astype(float)
    logp = np.log(close.replace(0, np.nan))

    x["log_ret_1"] = logp.diff(1)
    x["log_ret_3"] = logp.diff(3)
    x["log_ret_6"] = logp.diff(6)
    x["log_ret_12"] = logp.diff(12)
    x["rv_6"] = x["log_ret_1"].rolling(6).std()
    x["rv_12"] = x["log_ret_1"].rolling(12).std()
    x["rv_24"] = x["log_ret_1"].rolling(24).std()
    x["volume_change"] = volume.pct_change()
    x["volume_z"] = zscore(volume, 48)
    x["range_pct"] = (x["h"] - x["l"]) / close.replace(0, np.nan)
    x["body_pct"] = (x["c"] - x["o"]) / close.replace(0, np.nan)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema36 = close.ewm(span=36, adjust=False).mean()
    ema24 = close.ewm(span=24, adjust=False).mean()
    ema72 = close.ewm(span=72, adjust=False).mean()
    x["ema_gap_12_36"] = (ema12-ema36)/close.replace(0, np.nan)
    x["ema_gap_24_72"] = (ema24-ema72)/close.replace(0, np.nan)

    if funding_series is not None and len(funding_series) == len(x):
        x["funding_rate"] = pd.Series(funding_series, index=x.index).astype(float)
    else:
        x["funding_rate"] = float(funding_rate)
    x["funding_z"] = zscore(x["funding_rate"], 48).fillna(0.0)

    x["hour_sin"], x["hour_cos"], x["dow_sin"], x["dow_cos"] = _cyclical_time_features(x)
    return x[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def binary_target(df, horizon):
    future = df["c"].shift(-horizon)
    return (future > df["c"]).astype(float).where(future.notna())


def future_return(df, horizon):
    return df["c"].shift(-horizon) / df["c"] - 1.0


def live_overlay(trade_buy_ratio, ofi, microprice_edge, recent_ret):
    """Heuristic live overlay, never used as a historical trained feature.
    Returns [-1,1]. It is deliberately small at inference.
    """
    trade_imb = (float(trade_buy_ratio)-0.5)*2.0
    raw = (
        0.40*np.tanh(trade_imb*2.0) +
        0.25*np.tanh(float(ofi)*2.5) +
        0.20*np.tanh(float(microprice_edge)*2500.0) +
        0.15*np.tanh(float(recent_ret)*40.0)
    )
    return float(np.clip(raw, -1.0, 1.0))
