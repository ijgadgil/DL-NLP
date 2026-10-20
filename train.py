"""
Headless training + evaluation.

Synthetic data (default):
    python train.py                       # LSTM + GRU vs rule baseline
    python train.py --ablation            # also compare imbalance strategies
    python train.py --users 1500 --days 90 --epochs 10

Kaggle Sparkov dataset (put fraudTrain.csv + fraudTest.csv in ./data):
    python train.py --dataset sparkov                 # 250 random cards (~460k txns)
    python train.py --dataset sparkov --cards 0       # all 999 cards (1.85M txns, needs ~8 GB RAM)

Artifacts go to ./artifacts (synthetic) or ./artifacts/sparkov and are picked up by the Streamlit app.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from securepay.baseline import RuleBasedDetector
from securepay.data import GeneratorConfig, generate_transactions
from securepay.datasets import SPARKOV, SPARKOV_TEST_START, SYNTHETIC, load_sparkov
from securepay.metrics import evaluate, recall_by_type
from securepay.pipeline import TrainConfig, prepare, train
from securepay.streaming import StreamingDetector, summarise_latency

ROOT = Path(__file__).parent
ART_DIRS = {"synthetic": ROOT / "artifacts", "sparkov": ROOT / "artifacts" / "sparkov"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "sparkov"])
    # synthetic options
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--days", type=int, default=75)
    ap.add_argument("--victims", type=float, default=0.12)
    # sparkov options
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="folder with fraudTrain.csv and fraudTest.csv")
    ap.add_argument("--cards", type=int, default=250, help="random subset of cards; 0 = all 999")
    # model options
    ap.add_argument("--epochs", type=int, default=None, help="default: 8 synthetic, 5 sparkov")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--imbalance", default="smote+focal", choices=["smote+focal", "focal", "smote", "none"])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.epochs is None:
        args.epochs = 8 if args.dataset == "synthetic" else 5
    art = ART_DIRS[args.dataset]
    art.mkdir(parents=True, exist_ok=True)

    if args.dataset == "synthetic":
        spec, cuts = SYNTHETIC, None
        print("Generating transactions ...")
        df, users = generate_transactions(GeneratorConfig(args.users, args.days, args.victims, args.seed))
    else:
        spec = SPARKOV
        # Kaggle's own test file starts at SPARKOV_TEST_START; the last 3 months before it validate.
        cuts = (SPARKOV_TEST_START - pd.Timedelta(days=91), SPARKOV_TEST_START)
        print(f"Loading Sparkov CSVs from {args.data_dir} ...")
        df, users = load_sparkov(args.data_dir, n_cards=args.cards or None, seed=args.seed)
    df.to_parquet(art / "transactions.parquet")
    users.to_parquet(art / "users.parquet")
    print(f"  {len(df):,} transactions | {df.user_id.nunique():,} customers | "
          f"{df.is_fraud.sum():,} fraud ({df.is_fraud.mean():.2%})")

    print("Engineering behavioural features ...")
    data = prepare(df, users, args.seq_len, spec, cuts=cuts)
    test = data.test_rows
    y_test = data.y[test]
    print(f"  split  train={len(data.train_rows):,}  val={len(data.val_rows):,}  test={len(test):,}"
          f"  (test fraud={int(y_test.sum())})")

    results = {}
    rules = RuleBasedDetector(spec)
    rule_scores = rules.score(data.raw.iloc[test])
    results["Rule-based baseline"] = evaluate(y_test, rule_scores, rules.threshold)

    for cell in ["lstm", "gru"]:
        cfg = TrainConfig(cell=cell, seq_len=args.seq_len, epochs=args.epochs, imbalance=args.imbalance)
        print(f"Training {cell.upper()} [{cfg.imbalance}] ...")
        tm = train(data, cfg, on_epoch=lambda e, l: print(
            f"  epoch {e:2d}  loss={l['train_loss']:.4f}  val PR-AUC={l['val_pr_auc']:.3f}", flush=True))
        tm.save(art / f"model_{cell}.pt")
        results[f"{cell.upper()} ({cfg.imbalance})"] = evaluate(y_test, tm.score_rows(data, test), tm.threshold)
        if cell == "lstm":
            lstm = tm

    if args.ablation:
        for strat in ["none", "smote", "focal"]:
            if strat == args.imbalance:
                continue
            print(f"Ablation: LSTM [{strat}] ...")
            tm = train(data, TrainConfig(cell="lstm", seq_len=args.seq_len, epochs=args.epochs, imbalance=strat))
            results[f"LSTM ({strat})"] = evaluate(y_test, tm.score_rows(data, test), tm.threshold)

    table = pd.DataFrame(results).T[["precision", "recall", "f1", "roc_auc", "pr_auc",
                                     "false_positive_rate", "tp", "fp", "fn"]]
    print(f"\n=== Test-set results: {spec.title} (chronological hold-out) ===")
    print(table.round(3).to_string())

    ft = df["fraud_type"].iloc[test]
    print("\nRecall by fraud type:")
    print(pd.DataFrame({
        "Rules": recall_by_type(ft, rules.predict(data.raw.iloc[test])),
        "LSTM": recall_by_type(ft, (lstm.score_rows(data, test) >= lstm.threshold).astype(int)),
    }).round(3).to_string())

    print("\nStreaming simulation (2,000 events) ...")
    det = StreamingDetector(lstm, users)
    first_test = test.min()
    det.warm_up(df.iloc[:first_test])
    lat = [r["latency_ms"] for r in det.stream(df.iloc[first_test:first_test + 2000])]
    s = summarise_latency(lat)
    print(f"  latency p50={s['p50']:.2f} ms  p95={s['p95']:.2f} ms  p99={s['p99']:.2f} ms"
          f"  ~{s['throughput']:.0f} txn/s single-threaded")

    (art / "results.json").write_text(json.dumps(
        {"dataset": spec.name, "metrics": results, "latency": s}, indent=2, default=float))
    print(f"\nSaved artifacts to {art.resolve()}")


if __name__ == "__main__":
    main()
