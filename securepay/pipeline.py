"""End-to-end pipeline: dataset preparation, training (SMOTE / focal loss), inference, checkpoints."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE

from .datasets import SPECS, SYNTHETIC, DatasetSpec
from .features import Scaler, build_features, feature_names, make_sequences, n_features
from .metrics import best_f1_threshold
from .models import FocalLoss, SequenceFraudNet

IMBALANCE_STRATEGIES = ["smote+focal", "focal", "smote", "none"]


@dataclass
class TrainConfig:
    cell: str = "lstm"            # "lstm" | "gru"
    seq_len: int = 10
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 8
    batch_size: int = 256
    lr: float = 2e-3
    imbalance: str = "smote+focal"
    smote_ratio: float = 0.15     # minority : majority after SMOTE
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    seed: int = 7


@dataclass
class PreparedData:
    df: pd.DataFrame
    users: pd.DataFrame
    X: np.ndarray
    seq_idx: np.ndarray
    raw: pd.DataFrame
    y: np.ndarray
    train_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray
    seq_len: int
    spec: DatasetSpec = SYNTHETIC


def prepare(df: pd.DataFrame, users: pd.DataFrame, seq_len: int = 10, spec: DatasetSpec = SYNTHETIC,
            train_frac: float = 0.6, val_frac: float = 0.15, cuts=None) -> PreparedData:
    """Featurise and split *chronologically* (train -> validation -> test).

    `cuts` = (validation_start, test_start) timestamps overrides the fractions,
    e.g. to reuse a dataset's official train/test boundary.
    """
    X, seq_idx, raw, _ = build_features(df, users, seq_len, spec)
    t = df["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    if cuts is not None:
        c1, c2 = (int(pd.Timestamp(c).timestamp()) for c in cuts)
    else:
        t0, t1 = t.min(), t.max()
        c1 = t0 + (t1 - t0) * train_frac
        c2 = t0 + (t1 - t0) * (train_frac + val_frac)
    rows = np.arange(len(df))
    return PreparedData(
        df=df, users=users, X=X, seq_idx=seq_idx, raw=raw, y=df["is_fraud"].values.astype(np.float32),
        train_rows=rows[t < c1], val_rows=rows[(t >= c1) & (t < c2)], test_rows=rows[t >= c2],
        seq_len=seq_len, spec=spec,
    )


def _apply_smote(seqs: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    n, T, F = seqs.shape
    n_pos = int(y.sum())
    if n_pos < 2 or ratio <= n_pos / max(n - n_pos, 1):
        return seqs, y
    sm = SMOTE(sampling_strategy=ratio, k_neighbors=min(5, n_pos - 1), random_state=seed)
    Xr, yr = sm.fit_resample(seqs.reshape(n, T * F), y.astype(int))
    Xr = Xr.reshape(-1, T, F).astype(np.float32)
    # synthetic samples interpolate the padding mask -> snap it back to {0,1}
    mask = (Xr[..., -1] > 0.5).astype(np.float32)
    Xr[..., -1] = mask
    Xr[..., :-1] *= mask[..., None]
    return Xr, yr.astype(np.float32)


def predict_proba(model: nn.Module, seqs: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(seqs[i:i + batch]))).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@dataclass
class TrainedModel:
    model: SequenceFraudNet
    scaler: Scaler
    threshold: float
    config: TrainConfig
    history: list
    train_seconds: float
    spec: DatasetSpec = SYNTHETIC

    @property
    def name(self) -> str:
        return f"{self.config.cell.upper()} ({self.config.imbalance})"

    def score_rows(self, data: PreparedData, rows: np.ndarray) -> np.ndarray:
        seqs = make_sequences(self.scaler.transform(data.X), data.seq_idx, rows)
        return predict_proba(self.model, seqs)

    # ---------- persistence ----------
    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            state_dict=self.model.state_dict(), config=asdict(self.config),
            scaler_mean=self.scaler.mean, scaler_scale=self.scaler.scale,
            threshold=self.threshold, history=self.history, train_seconds=self.train_seconds,
            feature_names=feature_names(self.spec), dataset=self.spec.name,
        ), path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**ck["config"])
        spec = SPECS[ck.get("dataset", "synthetic")]
        model = SequenceFraudNet(n_features(spec) + 1, cfg.cell, cfg.hidden, cfg.layers, cfg.dropout)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return cls(model, Scaler(ck["scaler_mean"], ck["scaler_scale"]), ck["threshold"],
                   cfg, ck["history"], ck["train_seconds"], spec)


def train(data: PreparedData, cfg: TrainConfig,
          on_epoch: Callable[[int, dict], None] | None = None) -> TrainedModel:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    start = time.time()

    scaler = Scaler().fit(data.X[data.train_rows])
    Xs = scaler.transform(data.X).astype(np.float32)
    tr_seq = make_sequences(Xs, data.seq_idx, data.train_rows)
    tr_y = data.y[data.train_rows]
    va_seq = make_sequences(Xs, data.seq_idx, data.val_rows)
    va_y = data.y[data.val_rows]

    if "smote" in cfg.imbalance:
        tr_seq, tr_y = _apply_smote(tr_seq, tr_y, cfg.smote_ratio, cfg.seed)
    loss_fn = FocalLoss(cfg.focal_alpha, cfg.focal_gamma) if "focal" in cfg.imbalance else nn.BCEWithLogitsLoss()

    model = SequenceFraudNet(n_features(data.spec) + 1, cfg.cell, cfg.hidden, cfg.layers, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    Xt, yt = torch.from_numpy(tr_seq), torch.from_numpy(tr_y)

    history, best_state, best_ap = [], None, -1.0
    from sklearn.metrics import average_precision_score
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        va_p = predict_proba(model, va_seq)
        va_ap = float(average_precision_score(va_y, va_p)) if va_y.sum() > 0 else 0.0
        logs = dict(epoch=epoch, train_loss=total / len(Xt), val_pr_auc=va_ap)
        history.append(logs)
        if va_ap > best_ap:
            best_ap = va_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if on_epoch:
            on_epoch(epoch, logs)

    model.load_state_dict(best_state)
    threshold = best_f1_threshold(va_y, predict_proba(model, va_seq)) if va_y.sum() > 0 else 0.5
    return TrainedModel(model, scaler, threshold, cfg, history, time.time() - start, data.spec)
