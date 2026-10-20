"""Evaluation helpers: precision / recall / F1 and friends."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def best_f1_threshold(y_true, scores) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def evaluate(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan"),
        pr_auc=average_precision_score(y_true, scores) if len(set(y_true)) > 1 else float("nan"),
        false_positive_rate=fp / max(fp + tn, 1),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        alerts_per_1k=1000 * (tp + fp) / max(len(y_true), 1),
        threshold=float(threshold),
    )
    return out


def recall_by_type(fraud_type: pd.Series, y_pred) -> pd.Series:
    df = pd.DataFrame({"type": fraud_type.values, "pred": np.asarray(y_pred)})
    df = df[df["type"] != "legit"]
    return df.groupby("type")["pred"].mean()
