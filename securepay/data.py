"""
Synthetic transaction generator.

Each user gets a behavioural profile (typical spend, active hours, favourite
merchant categories, home city, devices). Legitimate transactions are sampled
from that profile, including realistic "noise" such as occasional travel or a
new phone. Fraud episodes are then injected for a subset of victim users:

* account_takeover : burst of high-value purchases from a new device/location
* card_testing     : several tiny probing charges followed by a large one
* behavioral_drift : subtle fraud - moderately inflated amounts at unusual
                     hours/categories from a known device (hard for rules)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery", "dining", "fuel", "utilities", "ecommerce",
    "travel", "entertainment", "electronics", "gift_cards", "crypto",
]
HIGH_RISK_CATEGORIES = {"electronics", "gift_cards", "crypto"}

CITIES = {
    "Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
    "Chennai": (13.08, 80.27), "Hyderabad": (17.39, 78.49), "Pune": (18.52, 73.86),
    "Kolkata": (22.57, 88.36), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.79),
}
FOREIGN_CITIES = {
    "Singapore": (1.35, 103.82), "Dubai": (25.20, 55.27), "London": (51.51, -0.13),
    "Lagos": (6.52, 3.38), "Bucharest": (44.43, 26.10),
}
START_DATE = pd.Timestamp("2026-07-01")


@dataclass
class GeneratorConfig:
    n_users: int = 800
    n_days: int = 75
    victim_fraction: float = 0.12
    seed: int = 42


def _jitter(lat: float, lon: float, km: float, rng: np.random.Generator):
    """Move a point by roughly `km` kilometres in a random direction."""
    angle = rng.uniform(0, 2 * np.pi)
    dlat = (km / 111.0) * np.cos(angle)
    dlon = (km / (111.0 * max(np.cos(np.radians(lat)), 0.2))) * np.sin(angle)
    return lat + dlat, lon + dlon


def _make_users(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    city_names = list(CITIES)
    rows = []
    for uid in range(cfg.n_users):
        city = rng.choice(city_names)
        lat, lon = CITIES[city]
        prefs = rng.dirichlet([3, 2.5, 1.5, 1, 2.5, 0.4, 1.2, 0.3, 0.08, 0.04])
        rows.append(dict(
            user_id=f"U{uid:05d}",
            home_city=city,
            home_lat=lat, home_lon=lon,
            amt_mu=rng.normal(6.4, 0.7),            # log-rupees; exp(6.4) ~ Rs 600
            amt_sigma=rng.uniform(0.35, 0.85),
            rate_per_day=rng.uniform(0.6, 3.0),
            peak_hour=float(np.clip(rng.normal(15, 3), 8, 22)),
            hour_spread=rng.uniform(2.0, 4.0),
            n_devices=int(rng.integers(1, 3)),
            prefs=prefs,
        ))
    return pd.DataFrame(rows)


def _legit_transactions(user: pd.Series, cfg: GeneratorConfig, rng: np.random.Generator):
    rows = []
    devices = [f"{user.user_id}-D{i}" for i in range(user.n_devices)]
    travel_until = -1
    travel_city = None
    for day in range(cfg.n_days):
        # occasional legitimate travel lasting a few days
        if day > travel_until and rng.random() < 0.012:
            travel_until = day + int(rng.integers(2, 6))
            pool = FOREIGN_CITIES if rng.random() < 0.15 else CITIES
            travel_city = rng.choice([c for c in pool if c != user.home_city])
        travelling = day <= travel_until
        n = rng.poisson(user.rate_per_day)
        for _ in range(n):
            hour = float(np.clip(rng.normal(user.peak_hour, user.hour_spread), 0, 23.99))
            ts = START_DATE + pd.Timedelta(days=day, hours=hour) + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            prefs = user.prefs.copy()
            if travelling:
                prefs[CATEGORIES.index("travel")] += 0.4
                prefs[CATEGORIES.index("dining")] += 0.2
                prefs = prefs / prefs.sum()
            cat = rng.choice(CATEGORIES, p=prefs)
            amt = float(np.exp(rng.normal(user.amt_mu, user.amt_sigma)))
            if cat in ("travel", "electronics"):
                amt *= rng.uniform(2, 6)
            if cat == "utilities":
                amt *= rng.uniform(1, 2.5)
            if rng.random() < 0.005:  # occasional legitimate big-ticket purchase
                amt *= rng.uniform(5, 20)
            if travelling:
                base_city = travel_city
                lat0, lon0 = {**CITIES, **FOREIGN_CITIES}[base_city]
                country = "IN" if base_city in CITIES else "FOREIGN"
            else:
                base_city, lat0, lon0, country = user.home_city, user.home_lat, user.home_lon, "IN"
            lat, lon = _jitter(lat0, lon0, rng.exponential(4), rng)
            device = rng.choice(devices)
            if rng.random() < 0.004:  # legit new phone / browser
                device = f"{user.user_id}-D{len(devices)}"
                devices.append(device)
            rows.append(dict(
                user_id=user.user_id, timestamp=ts, amount=round(amt, 2), category=cat,
                city=base_city, country=country, lat=lat, lon=lon, device_id=device,
                is_fraud=0, fraud_type="legit",
            ))
    return rows


def _fraud_episode(user: pd.Series, kind: str, start: pd.Timestamp, rng: np.random.Generator):
    rows = []
    typical = float(np.exp(user.amt_mu))
    attacker_device = f"ATK-{rng.integers(1e9)}"
    if rng.random() < 0.3:
        city = rng.choice(list(FOREIGN_CITIES)); lat0, lon0 = FOREIGN_CITIES[city]; country = "FOREIGN"
    else:
        city = rng.choice(list(CITIES)); lat0, lon0 = CITIES[city]; country = "IN"
    t = start

    def add(amount, category, device, c=city, la=lat0, lo=lon0, co=country):
        lat, lon = _jitter(la, lo, rng.exponential(3), rng)
        rows.append(dict(
            user_id=user.user_id, timestamp=t, amount=round(float(amount), 2), category=category,
            city=c, country=co, lat=lat, lon=lon, device_id=device, is_fraud=1, fraud_type=kind,
        ))

    if kind == "account_takeover":
        device = attacker_device if rng.random() < 0.85 else f"{user.user_id}-D0"
        for _ in range(int(rng.integers(2, 6))):
            if rng.random() < 0.7:
                cat = rng.choice(["electronics", "gift_cards", "crypto", "ecommerce"], p=[.35, .3, .15, .2])
            else:  # some attackers blend in with everyday categories
                cat = rng.choice(CATEGORIES)
            add(typical * float(np.exp(rng.uniform(np.log(3), np.log(60)))), cat, device)
            t = t + pd.Timedelta(minutes=float(rng.uniform(2, 25)))
    elif kind == "card_testing":
        device = attacker_device if rng.random() < 0.5 else f"{user.user_id}-D0"
        if rng.random() < 0.5:  # tester sits in the victim's home city (proxy / local mule)
            city, lat0, lon0, country = user.home_city, user.home_lat, user.home_lon, "IN"
        for _ in range(int(rng.integers(2, 7))):
            add(rng.uniform(1, 49), rng.choice(["ecommerce", "gift_cards", "entertainment", "dining"]), device)
            t = t + pd.Timedelta(seconds=float(rng.uniform(20, 300)))
        for _ in range(int(rng.integers(1, 3))):
            add(typical * rng.uniform(3, 15), rng.choice(["electronics", "gift_cards", "ecommerce", "travel"]), device)
            t = t + pd.Timedelta(minutes=float(rng.uniform(1, 10)))
    else:  # behavioral_drift: subtle, mostly from home city and a known device
        device = f"{user.user_id}-D0" if rng.random() < 0.7 else attacker_device
        # happens at hours this user is normally inactive
        off_hour = (user.peak_hour + 12 + rng.uniform(-3, 3)) % 24
        t = start.normalize() + pd.Timedelta(hours=off_hour)
        for _ in range(int(rng.integers(2, 5))):
            # pick a category this user rarely uses
            cat = rng.choice(CATEGORIES, p=(1 - user.prefs) ** 4 / ((1 - user.prefs) ** 4).sum())
            add(typical * rng.uniform(2.0, 6.0), cat, device,
                c=user.home_city, la=user.home_lat, lo=user.home_lon, co="IN")
            t = t + pd.Timedelta(hours=float(rng.uniform(0.2, 3)))
    return rows


def generate_transactions(cfg: GeneratorConfig | None = None):
    """Return (transactions_df, users_df)."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    users = _make_users(cfg, rng)
    rows = []
    for _, u in users.iterrows():
        rows.extend(_legit_transactions(u, cfg, rng))

    victims = users.sample(frac=cfg.victim_fraction, random_state=cfg.seed)
    for _, u in victims.iterrows():
        for _ in range(int(rng.integers(1, 3))):
            kind = rng.choice(["account_takeover", "card_testing", "behavioral_drift"], p=[.4, .25, .35])
            day = rng.uniform(7, cfg.n_days - 0.5)  # leave a week of history first
            # fraudsters favour night hours, but not always
            hour = rng.uniform(0, 5) if rng.random() < 0.55 else rng.uniform(5, 24)
            start = START_DATE + pd.Timedelta(days=int(day), hours=hour)
            rows.extend(_fraud_episode(u, kind, start, rng))

    df = pd.DataFrame(rows).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    users_out = users.drop(columns=["prefs"])
    return df, users_out
