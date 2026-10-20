"""
Dataset specifications and loaders.

A `DatasetSpec` tells the feature engine, the rule baseline and the app what a
dataset contains (merchant categories, currency, whether device IDs and
foreign transactions exist). Two datasets are supported:

* synthetic : SecurePay's own behaviour-rich generator (`securepay.data`)
* sparkov   : Kaggle "Credit Card Transactions Fraud Detection Dataset"
              by Kartik Shenoy (kartik2112), generated with the Sparkov tool.
              https://www.kaggle.com/datasets/kartik2112/fraud-detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CATEGORIES as SYN_CATEGORIES
from .data import HIGH_RISK_CATEGORIES as SYN_HIGH_RISK


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    categories: tuple
    high_risk: frozenset
    currency: str
    large_amount: float           # threshold for rule R1
    has_device: bool
    has_foreign: bool
    default_country: str = "IN"
    notes: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def money(self, x: float) -> str:
        return f"{self.currency}{x:,.0f}"


SYNTHETIC = DatasetSpec(
    name="synthetic", title="SecurePay synthetic",
    categories=tuple(SYN_CATEGORIES), high_risk=frozenset(SYN_HIGH_RISK),
    currency="₹", large_amount=50_000, has_device=True, has_foreign=True, default_country="IN",
    notes="Behaviour-rich synthetic data with three labelled fraud patterns, devices and foreign travel.",
)

SPARKOV_CATEGORIES = (
    "gas_transport", "grocery_pos", "home", "shopping_pos", "kids_pets", "shopping_net", "entertainment",
    "food_dining", "personal_care", "health_fitness", "misc_pos", "misc_net", "grocery_net", "travel",
)
SPARKOV = DatasetSpec(
    name="sparkov", title="Kaggle Sparkov (kartik2112)",
    categories=SPARKOV_CATEGORIES, high_risk=frozenset({"shopping_net", "misc_net"}),
    # R1 threshold = 99.5th percentile of amounts in fraudTrain.csv ($844), rounded
    currency="$", large_amount=850, has_device=False, has_foreign=False, default_country="US",
    notes="Public Kaggle dataset: 1.85M US card transactions (2019-2020), 999 cards, 0.52% fraud. "
          "No device IDs or foreign transactions; fraud is labelled yes/no only.",
)

SPECS = {s.name: s for s in (SYNTHETIC, SPARKOV)}
# Kaggle's own split: fraudTest.csv starts here. Used as the test boundary.
SPARKOV_TEST_START = pd.Timestamp("2020-06-21 12:14:00")


def load_sparkov(csv_dir: str | Path, n_cards: int | None = None, seed: int = 42):
    """Read fraudTrain.csv + fraudTest.csv and map them onto SecurePay's schema.

    Returns (transactions_df, users_df). `n_cards` keeps a random subset of cards
    (all of each kept card's history), which preserves the fraud rate while
    reducing memory and training time. None or 0 = all 999 cards.
    """
    csv_dir = Path(csv_dir)
    cols = ["trans_date_trans_time", "cc_num", "category", "amt", "city", "state", "lat", "long",
            "trans_num", "merch_lat", "merch_long", "is_fraud"]
    parts = [pd.read_csv(csv_dir / f, usecols=cols) for f in ("fraudTrain.csv", "fraudTest.csv")]
    raw = pd.concat(parts, ignore_index=True)

    if n_cards:
        cards = raw["cc_num"].drop_duplicates().sample(n=min(n_cards, raw["cc_num"].nunique()), random_state=seed)
        raw = raw[raw["cc_num"].isin(set(cards))]

    uid = "C" + raw["cc_num"].astype(str)
    df = pd.DataFrame({
        "txn_id": raw["trans_num"].values,
        "user_id": uid.values,
        "timestamp": pd.to_datetime(raw["trans_date_trans_time"]).values,
        "amount": raw["amt"].astype(float).values,
        "category": raw["category"].values,
        "city": (raw["city"] + ", " + raw["state"]).values,
        "country": "US",
        "lat": raw["merch_lat"].astype(float).values,
        "lon": raw["merch_long"].astype(float).values,
        "device_id": (uid + "-D0").values,     # dataset has no device information
        "is_fraud": raw["is_fraud"].astype(int).values,
    })
    df["fraud_type"] = np.where(df["is_fraud"] == 1, "fraud", "legit")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    first = raw.assign(user_id=uid).drop_duplicates("user_id")
    users = pd.DataFrame({
        "user_id": first["user_id"].values,
        "home_city": (first["city"] + ", " + first["state"]).values,
        "home_lat": first["lat"].astype(float).values,
        "home_lon": first["long"].astype(float).values,
        "home_country": "US",
    })
    return df, users
