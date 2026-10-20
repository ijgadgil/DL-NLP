"""
Incremental, per-user feature engineering.

The SAME `FeatureEngine` is used for offline dataset building and for the
real-time streaming detector, so training and serving features can never
drift apart. Every feature is computed causally: it only uses the user's
history *before* the current transaction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datasets import SYNTHETIC, DatasetSpec

BASE_FEATURES = [
    "log_amount", "amount_zscore", "log_amount_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "log_hours_since_last", "txns_last_10min", "txns_last_1h",
    "log_km_from_home", "km_jump_from_last", "is_foreign", "new_device",
    "new_category", "is_night", "hour_deviation",
]


def feature_names(spec: DatasetSpec = SYNTHETIC) -> list[str]:
    return BASE_FEATURES + [f"cat_{c}" for c in spec.categories]


def n_features(spec: DatasetSpec = SYNTHETIC) -> int:
    return len(BASE_FEATURES) + len(spec.categories)


# backwards-compatible names for the synthetic dataset
FEATURE_NAMES = feature_names(SYNTHETIC)
N_FEATURES = len(FEATURE_NAMES)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


@dataclass
class UserState:
    home_lat: float
    home_lon: float
    home_country: str = "IN"
    n: int = 0
    mean_log_amt: float = 0.0
    m2_log_amt: float = 0.0
    mean_hour: float = 12.0
    last_ts: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    devices: set = field(default_factory=set)
    categories: set = field(default_factory=set)
    recent_ts: deque = field(default_factory=lambda: deque(maxlen=50))
    history: deque | None = None  # last T feature vectors (for streaming)


class FeatureEngine:
    def __init__(self, users: pd.DataFrame, seq_len: int = 10, spec: DatasetSpec = SYNTHETIC):
        self.seq_len = seq_len
        self.spec = spec
        self.categories = list(spec.categories)
        has_country = "home_country" in users.columns
        self.homes = {r.user_id: (r.home_lat, r.home_lon, r.home_country if has_country else spec.default_country)
                      for r in users.itertuples()}
        self.states: dict[str, UserState] = {}

    def _state(self, user_id: str) -> UserState:
        st = self.states.get(user_id)
        if st is None:
            lat, lon, country = self.homes.get(user_id, (np.nan, np.nan, self.spec.default_country))
            st = UserState(home_lat=lat, home_lon=lon, home_country=country, history=deque(maxlen=self.seq_len))
            self.states[user_id] = st
        return st

    def update(self, txn) -> tuple[np.ndarray, dict]:
        """Featurise one transaction, then fold it into the user's state.

        `txn` needs: user_id, timestamp, amount, category, lat, lon, country, device_id.
        Returns (feature_vector, raw_signals) where raw_signals feed the rule baseline.
        """
        st = self._state(txn["user_id"])
        ts: pd.Timestamp = pd.Timestamp(txn["timestamp"])
        t = ts.timestamp()
        amount = float(txn["amount"])
        log_amt = np.log1p(amount)
        hour = ts.hour + ts.minute / 60
        dow = ts.dayofweek

        std = np.sqrt(st.m2_log_amt / (st.n - 1)) if st.n > 2 else 1.0
        z = (log_amt - st.mean_log_amt) / max(std, 0.25) if st.n >= 3 else 0.0
        ratio = log_amt - st.mean_log_amt if st.n >= 3 else 0.0
        hrs_since = (t - st.last_ts) / 3600 if st.last_ts is not None else 72.0
        n10 = sum(1 for x in st.recent_ts if t - x <= 600)
        n60 = sum(1 for x in st.recent_ts if t - x <= 3600)
        home_lat = st.home_lat if not np.isnan(st.home_lat) else txn["lat"]
        home_lon = st.home_lon if not np.isnan(st.home_lon) else txn["lon"]
        km_home = haversine_km(home_lat, home_lon, txn["lat"], txn["lon"])
        km_jump = haversine_km(st.last_lat, st.last_lon, txn["lat"], txn["lon"]) if st.last_lat is not None else 0.0
        new_device = float(st.n > 0 and txn["device_id"] not in st.devices)
        new_cat = float(st.n >= 5 and txn["category"] not in st.categories)
        is_foreign = float(txn["country"] != st.home_country)
        is_night = float(hour < 5)
        hour_dev = min(abs(hour - st.mean_hour), 24 - abs(hour - st.mean_hour)) / 12 if st.n >= 3 else 0.0

        cat_onehot = [float(txn["category"] == c) for c in self.categories]
        vec = np.array([
            log_amt, np.clip(z, -6, 10), ratio,
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.log1p(min(hrs_since, 720)), n10, n60,
            np.log1p(km_home), np.log1p(km_jump), is_foreign, new_device,
            new_cat, is_night, hour_dev,
            *cat_onehot,
        ], dtype=np.float32)

        prior_mean_amt = float(np.expm1(st.mean_log_amt)) if st.n >= 3 else None
        raw = dict(
            amount=amount, prior_n=st.n, prior_mean_amount=prior_mean_amt,
            hour=hour, txns_last_10min=n10, km_from_home=km_home,
            is_foreign=bool(is_foreign), new_device=bool(new_device),
            high_risk_category=txn["category"] in self.spec.high_risk,
        )

        # ---- fold transaction into state (Welford update) ----
        st.n += 1
        delta = log_amt - st.mean_log_amt
        st.mean_log_amt += delta / st.n
        st.m2_log_amt += delta * (log_amt - st.mean_log_amt)
        # circular-ish running mean of hour (simple EMA is fine for profiling)
        diff = (hour - st.mean_hour + 12) % 24 - 12
        st.mean_hour = (st.mean_hour + diff * (1 / min(st.n, 20))) % 24
        st.last_ts, st.last_lat, st.last_lon = t, txn["lat"], txn["lon"]
        st.devices.add(txn["device_id"])
        st.categories.add(txn["category"])
        st.recent_ts.append(t)
        st.history.append(vec)
        return vec, raw


def build_features(df: pd.DataFrame, users: pd.DataFrame, seq_len: int = 10, spec: DatasetSpec = SYNTHETIC):
    """Run the engine over a time-sorted transaction log.

    Returns
    -------
    X        : (N, F) raw feature matrix
    seq_idx  : (N, seq_len) row indices of each transaction's history window,
               left-padded with -1; the last column is the transaction itself
    raw_df   : DataFrame of raw signals used by the rule-based baseline
    engine   : the warmed-up FeatureEngine (state after the last transaction)
    """
    engine = FeatureEngine(users, seq_len, spec)
    n = len(df)
    X = np.zeros((n, n_features(spec)), dtype=np.float32)
    seq_idx = np.full((n, seq_len), -1, dtype=np.int64)
    user_rows: dict[str, deque] = {}
    raws = []
    cols = ["user_id", "timestamp", "amount", "category", "lat", "lon", "country", "device_id"]
    for i, rec in enumerate(df[cols].to_dict("records")):
        vec, raw = engine.update(rec)
        X[i] = vec
        raws.append(raw)
        dq = user_rows.setdefault(rec["user_id"], deque(maxlen=seq_len))
        dq.append(i)
        seq_idx[i, seq_len - len(dq):] = list(dq)
    return X, seq_idx, pd.DataFrame(raws), engine


class Scaler:
    """Minimal standard scaler that is trivial to serialise into a checkpoint."""

    def __init__(self, mean=None, scale=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)

    def fit(self, X):
        self.mean = X.mean(0).astype(np.float32)
        self.scale = X.std(0).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale


def make_sequences(X_scaled: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray | None = None):
    """Gather (n, T, F+1) sequences. The extra channel is a 'present' mask (0 for padding)."""
    idx = seq_idx if rows is None else seq_idx[rows]
    present = (idx >= 0)
    seq = X_scaled[np.where(present, idx, 0)]
    seq[~present] = 0.0
    return np.concatenate([seq, present[..., None].astype(np.float32)], axis=-1)


def sequence_from_history(history, scaler: Scaler, seq_len: int) -> np.ndarray:
    """Build one (1, T, F+1) sequence from a user's live history deque (streaming)."""
    h = np.stack(list(history))[-seq_len:]
    out = np.zeros((seq_len, h.shape[1] + 1), dtype=np.float32)
    out[seq_len - len(h):, :-1] = scaler.transform(h)
    out[seq_len - len(h):, -1] = 1.0
    return out[None]
