# SecurePay — Real-Time Fraud Detection using Deep Neural Networks

SecurePay models each customer's transaction history as a sequence and uses an
**LSTM / GRU** to detect behavioural deviations that indicate fraud. Class
imbalance is handled with **SMOTE** and **focal loss**, real-time viability is
tested with **simulated streaming inference**, and everything is evaluated
against a **rule-based baseline** on precision, recall and F1.

## Quick start

Trained models for both datasets are included, so the app works straight away.

**Easiest:** double-click `run_app.bat` (Windows) or run `./run_app.sh` (Linux / macOS).
The first run creates a virtual environment and installs the libraries; later runs start at once.

**Manual:**

```bash
python3 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only: small CPU build
pip install -r requirements.txt
streamlit run app.py                           # opens http://localhost:8501
```

Optional retraining: `python train.py --ablation` (synthetic, about 3 minutes) or
`python train.py --dataset sparkov` (see below).

## The app

A compact console where every page fits on one screen; secondary views sit in tabs.

| Section | Page | What it shows |
|---|---|---|
| Monitor | **Overview** | Headline KPIs, alerts an analyst would review per detector (fraud caught vs false alarms), precision / recall / F1, the scoring pipeline |
| Monitor | **Live stream** | Replays the test period one transaction at a time: live risk chart, alert feed, latency and throughput |
| Monitor | **Score a transaction** | Pick a customer, craft a payment, get a risk gauge, a verdict, the rules that fired and the behavioural signals |
| Model | **Evaluation** | Tabs: Summary, Curves (PR curve, score separation), Errors (confusion matrices), Threshold explorer, Ablation |
| Model | **Train models** | Train LSTM / GRU with any imbalance strategy; learning curves of the loaded models |
| Model | **Data explorer** | Distributions, fraud patterns, one customer's timeline, raw records; generate a new synthetic dataset |

Switch between **SecurePay synthetic** and **Kaggle Sparkov** in the sidebar.

## Second dataset: Kaggle Sparkov (public benchmark)

SecurePay is also evaluated on the public **"Credit Card Transactions Fraud
Detection Dataset"** by Kartik Shenoy (kartik2112),
<https://www.kaggle.com/datasets/kartik2112/fraud-detection>. It holds 1.85M US
card transactions (Jan 2019 – Dec 2020) from 999 cards, 0.52 % fraud, and was
generated with Brandon Harris's Sparkov tool, so it contains no real customers.

Pre-trained Sparkov models are already in `artifacts/sparkov`; pick
**Kaggle Sparkov** in the app's sidebar. To retrain:

```bash
# put fraudTrain.csv and fraudTest.csv in ./data
python train.py --dataset sparkov            # 250 random cards (~460k txns), ~15-30 min on a CPU
python train.py --dataset sparkov --cards 0  # all 999 cards, needs ~8 GB RAM, much slower
```

How it is mapped and split:

* Card number → customer; merchant latitude/longitude → transaction location;
  customer latitude/longitude → home. All 14 Sparkov categories are one-hot features.
* The dataset has **no device IDs and no foreign transactions**, so those features
  stay constant and rules R4 / R6 never fire. Fraud is labelled yes/no only.
* **Kaggle's own split is kept:** `fraudTest.csv` (from 21 Jun 2020) is the test set;
  the last 3 months of `fraudTrain.csv` are validation, the rest is training.
* Rule R1's threshold is $850, the 99.5th percentile of amounts in `fraudTrain.csv`.

Results (250 cards, seed 42; test = 138,481 transactions, 535 fraud):

| Model | Precision | Recall | F1 | PR-AUC | False positives |
|---|---|---|---|---|---|
| Rule-based baseline | 0.031 | 0.755 | 0.060 | 0.098 | 12,576 |
| **LSTM (SMOTE + focal)** | **0.963** | **0.880** | **0.920** | 0.946 | **18** |
| GRU (SMOTE + focal) | 0.959 | 0.869 | 0.912 | 0.937 | 20 |

Streaming latency on Sparkov: p50 0.6 ms, p95 0.9 ms per event (~1,500 txn/s).
The rules were written for the synthetic data and not tuned for Sparkov, which
partly explains their very high false-positive count; even so, the gap shows
the value of learning each customer's own behaviour.

## Project layout

```
securepay/
├── app.py                    Streamlit UI (dashboard with sidebar navigation)
├── run_app.sh / run_app.bat  One-step setup and launch
├── .streamlit/config.toml    Theme
├── assets/                   Logo and icon
├── artifacts/                Trained models + datasets (synthetic; sparkov/ for Kaggle)
├── train.py                  Headless training / evaluation / streaming benchmark (--dataset synthetic|sparkov)
├── data/                     Put the Kaggle Sparkov CSVs here (see data/README.txt)
├── requirements.txt
└── securepay/
    ├── data.py               Synthetic per-user transaction generator + fraud injection
    ├── datasets.py           Dataset specs (categories, currency, ...) + Kaggle Sparkov loader
    ├── features.py           Incremental, causal per-user feature engine (shared by training & streaming)
    ├── models.py             SequenceFraudNet (LSTM/GRU + attention pooling) and FocalLoss
    ├── pipeline.py           Chronological split, SMOTE on sequences, training loop, checkpoints
    ├── baseline.py           7 static bank-style rules
    ├── metrics.py            Precision/recall/F1, PR-AUC, best-F1 threshold
    └── streaming.py          StreamingDetector with per-user in-memory state + latency stats
```

## Method

**Data.** No public dataset combines per-user histories, device, location and
timing, so SecurePay generates one: each of N users has a spending profile
(typical amount, active hours, favourite categories, home city, devices).
Legit noise is included — travel, foreign trips, new phones, occasional
big-ticket purchases — which is exactly what makes static rules noisy. Three
fraud patterns are injected for ~12 % of users:

* **Account takeover** — burst of high-value buys from a new device/location
* **Card testing** — several tiny probes within minutes, then a large charge
* **Behavioural drift** — moderately inflated buys at the user's off-hours, in
  categories they rarely use, often from their own device (hard for rules)

**Features (27 per transaction + padding mask).** Computed causally from the user's
own past: log amount, amount z-score vs history, Δt since last transaction,
velocity (10 min / 1 h), hour & weekday (cyclic), deviation from usual hour,
distance from home and from previous transaction, foreign, new device, new
category, night flag, category one-hot.

**Model.** The last *T* (default 10) transactions → linear projection →
2-layer LSTM or GRU → [last hidden state ‖ attention-pooled context] → MLP →
fraud probability. Best epoch chosen by validation PR-AUC; decision threshold
chosen to maximise validation F1.

**Imbalance.** SMOTE runs on flattened training sequences (the padding mask is
snapped back to {0,1} afterwards); focal loss (α = 0.75, γ = 2) down-weights
easy legitimate examples.

**Evaluation.** Chronological split — first 60 % of the timeline train, next
15 % validation, last 25 % test — so the model never sees the future.

**Streaming.** `StreamingDetector` keeps per-user state in memory, featurises
each event incrementally, builds the sequence and runs one forward pass.
Streaming scores match offline batch scores to ~1e-7, proving there is no
train/serve skew.

## Reference results (`python train.py --ablation`, seed 42)

107k transactions, 0.5 % fraud; test window = 27k transactions, 193 fraud.

| Model | Precision | Recall | F1 | PR-AUC | False positives |
|---|---|---|---|---|---|
| Rule-based baseline | 0.103 | 0.658 | 0.178 | 0.350 | 1107 |
| **LSTM (SMOTE + focal)** | **0.975** | 0.803 | **0.881** | 0.931 | **4** |
| GRU (SMOTE + focal) | 0.951 | 0.798 | 0.868 | 0.940 | 8 |
| LSTM (no imbalance handling) | 0.899 | 0.829 | 0.863 | 0.898 | 18 |
| LSTM (SMOTE only) | 0.975 | 0.808 | 0.884 | 0.934 | 4 |
| LSTM (focal only) | 0.981 | 0.813 | 0.890 | 0.948 | 3 |

Recall by pattern — rules vs LSTM: account takeover 0.96 / 0.88, card testing
0.84 / 0.93, behavioural drift 0.27 / 0.66.

Streaming latency: p50 ≈ 0.7 ms, p95 ≈ 1 ms per event on one CPU thread
(~1,300+ transactions/s).

**Reading these honestly.** The deep models cut false positives by more than
two orders of magnitude at similar recall, and they catch far more of the
subtle drift fraud. The ablation shows that any imbalance handling improves
precision over plain BCE, but SMOTE, focal loss and their combination perform
similarly on this data — differences are within run-to-run noise. Because the
data is synthetic, absolute numbers will be lower on real bank data; the
comparative findings are the point.

## Extending to real data

Supply a transactions DataFrame with `user_id, timestamp, amount, category,
lat, lon, country, city, device_id, is_fraud` (plus `fraud_type`, or fill with
`"legit"`/`"fraud"`) and a users DataFrame with `user_id, home_lat, home_lon`,
then call `securepay.pipeline.prepare(...)` and `train(...)`.
