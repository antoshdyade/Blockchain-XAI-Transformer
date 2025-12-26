import pandas as pd, os

DATA_FOLDER = "data"

# Load raw files
df_feat = pd.read_csv("data/elliptic_txs_features.csv", header=None)
df_class = pd.read_csv("data/elliptic_txs_classes.csv")

# Fix: first column of features file is the transaction ID
df_feat.columns = ["txId"] + [f"f{i}" for i in range(1, df_feat.shape[1])]

# Merge features + labels
df = df_feat.merge(df_class, on="txId", how="inner")

# Remove unknown labels
df = df[df['class'] != 'unknown']

# Convert illicit=1, licit=0
df['label'] = df['class'].apply(lambda x: 1 if x == 1 else 0)

# Drop old label column
df = df.drop(columns=['class', 'class'])

# Save cleaned dataset
df.to_csv("data/elliptic_bitcoin.csv", index=False)
print("✔ Bitcoin dataset ready → data/elliptic_bitcoin.csv")



# ---- Convert Ethereum Fraud ----
eth_files = [
    "transaction_dataset.csv",
    "Ethereum_Fraud_Detection.csv",
    "Ethereum_Fraud_Detection_Data.csv",
    "ethereum_fraud.csv"
]

found = False
for file in eth_files:
    path = os.path.join(DATA_FOLDER, file)
    if os.path.exists(path):
        df = pd.read_csv(path)

        # identify fraud column
        fraud_col = None
        for col in df.columns:
            if col.lower() in ["flag", "isfraud", "fraud", "label"]:
                fraud_col = col
                break

        if fraud_col:
            df['label'] = df[fraud_col].apply(lambda x: 1 if x == 1 else 0)
            df = df.drop(columns=[fraud_col])

            out_eth = os.path.join(DATA_FOLDER, "ethereum_fraud.csv")
            df.to_csv(out_eth, index=False)
            print("Ethereum dataset converted →", out_eth)
            found = True
            break
        else:
            print("Fraud label column not found in", file)
            found = True
            break

if not found:
    print("Ethereum raw files not found, skipping Ethereum conversion.")
