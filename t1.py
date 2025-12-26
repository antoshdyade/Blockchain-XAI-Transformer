import pandas as pd
print("Features columns:", pd.read_csv("data/elliptic_txs_features.csv", nrows=2).columns.tolist())
print("Classes columns:", pd.read_csv("data/elliptic_txs_classes.csv", nrows=2).columns.tolist())