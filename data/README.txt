Put the Kaggle Sparkov CSV files here:

    fraudTrain.csv
    fraudTest.csv

Dataset: "Credit Card Transactions Fraud Detection Dataset" by Kartik Shenoy (kartik2112)
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Then run from the project folder:

    python train.py --dataset sparkov            # 250 random cards, ~15-30 min on a laptop CPU
    python train.py --dataset sparkov --cards 0  # all 999 cards (1.85M transactions, needs ~8 GB RAM, slow)

The CSVs are only needed for (re)training. The app reads artifacts/sparkov/.
