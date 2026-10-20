"""Simulated real-time streaming inference.

Transactions arrive one at a time. For each event the detector:
  1. updates the user's behavioural state (incremental features),
  2. assembles the user's last-T-transactions sequence from in-memory state,
  3. runs a single forward pass of the LSTM/GRU,
  4. also evaluates the rule baseline for comparison,
and records end-to-end latency per event.
"""
from __future__ import annotations

import time
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch

from .baseline import RuleBasedDetector
from .features import FeatureEngine, sequence_from_history
from .pipeline import TrainedModel

_COLS = ["txn_id", "user_id", "timestamp", "amount", "category", "lat", "lon",
         "country", "city", "device_id", "is_fraud", "fraud_type"]


class StreamingDetector:
    def __init__(self, trained: TrainedModel, users: pd.DataFrame, rules: RuleBasedDetector | None = None):
        self.trained = trained
        self.seq_len = trained.config.seq_len
        self.engine = FeatureEngine(users, self.seq_len, trained.spec)
        self.rules = rules or RuleBasedDetector(trained.spec)
        self.model = trained.model.eval()

    def warm_up(self, history_df: pd.DataFrame):
        """Replay historical transactions so every user has a behavioural profile."""
        for rec in history_df[_COLS].to_dict("records"):
            self.engine.update(rec)

    def process(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        _, raw = self.engine.update(txn)
        hist = self.engine.states[txn["user_id"]].history
        seq = sequence_from_history(hist, self.trained.scaler, self.seq_len)
        with torch.inference_mode():
            prob = float(torch.sigmoid(self.model(torch.from_numpy(seq)))[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        fired = self.rules.explain(raw)
        return dict(
            **{k: txn.get(k) for k in _COLS},
            risk=prob, dl_alert=prob >= self.trained.threshold,
            rule_alert=len(fired) >= self.rules.min_rules, rules_fired=", ".join(fired),
            latency_ms=latency_ms,
        )

    def stream(self, events: pd.DataFrame) -> Iterator[dict]:
        for rec in events[_COLS].to_dict("records"):
            yield self.process(rec)


def summarise_latency(latencies: Iterable[float]) -> dict:
    a = np.asarray(list(latencies))
    if a.size == 0:
        return dict(p50=0, p95=0, p99=0, mean=0, throughput=0)
    return dict(p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), mean=float(a.mean()),
                throughput=float(1000 / a.mean()))
