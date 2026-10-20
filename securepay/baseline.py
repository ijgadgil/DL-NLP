"""Rule-based baseline, modelled on typical static bank fraud rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import SYNTHETIC, DatasetSpec

RULE_KEYS = [
    "R1_large_amount", "R2_spend_spike", "R3_velocity", "R4_foreign",
    "R5_night_spike", "R6_new_device_spike", "R7_far_high_risk",
]


def rule_descriptions(spec: DatasetSpec = SYNTHETIC) -> dict[str, str]:
    return {
        "R1_large_amount":     f"Amount above {spec.money(spec.large_amount)}",
        "R2_spend_spike":      "Amount > 5x the user's average (>= 5 prior txns)",
        "R3_velocity":         ">= 3 transactions in the last 10 minutes",
        "R4_foreign":          "Transaction outside the home country",
        "R5_night_spike":      "00:00-05:00 and amount > 3x average",
        "R6_new_device_spike": "New device and amount > 2x average",
        "R7_far_high_risk":    "> 300 km from home in a high-risk category "
                               f"({', '.join(sorted(spec.high_risk))})",
    }


# kept for backwards compatibility (synthetic wording)
RULES = rule_descriptions(SYNTHETIC)


class RuleBasedDetector:
    def __init__(self, spec: DatasetSpec = SYNTHETIC, min_rules: int = 1):
        self.spec = spec
        self.min_rules = min_rules
        self.descriptions = rule_descriptions(spec)

    def _fired(self, r) -> dict:
        avg = r["prior_mean_amount"]
        has_hist = avg is not None and not (isinstance(avg, float) and np.isnan(avg))
        ratio = r["amount"] / avg if has_hist and avg > 0 else 0.0
        return {
            "R1_large_amount": r["amount"] > self.spec.large_amount,
            "R2_spend_spike": has_hist and r["prior_n"] >= 5 and ratio > 5,
            "R3_velocity": r["txns_last_10min"] >= 2,  # 2 earlier + current = 3
            "R4_foreign": bool(r["is_foreign"]),
            "R5_night_spike": r["hour"] < 5 and ratio > 3,
            "R6_new_device_spike": bool(r["new_device"]) and ratio > 2,
            "R7_far_high_risk": r["km_from_home"] > 300 and bool(r["high_risk_category"]),
        }

    @property
    def n_rules(self) -> int:
        return len(RULE_KEYS)

    def explain(self, raw: dict) -> list[str]:
        return [k for k, v in self._fired(raw).items() if v]

    def score_raw(self, raw: dict) -> float:
        return sum(self._fired(raw).values()) / self.n_rules

    def score(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Continuous score = fraction of rules fired (used for PR/ROC curves)."""
        return np.array([self.score_raw(r) for r in raw_df.to_dict("records")], dtype=np.float32)

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        return (self.score(raw_df) * self.n_rules >= self.min_rules).astype(int)

    @property
    def threshold(self) -> float:
        """Score threshold equivalent to `min_rules` fired."""
        return self.min_rules / self.n_rules - 1e-9
