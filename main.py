import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score
)

from model import TransformerDetector
from xai import explain_shap, explain_lime, compare_models

# ===== LOAD DATA =====
df_btc = pd.read_csv("data/elliptic_bitcoin.csv")
df_eth = pd.read_csv("data/ethereum_fraud.csv")

# ===== CLEAN ETHEREUM DATASET =====
hex_cols = df_eth.select_dtypes(exclude=[np.number]).columns.tolist()
df_eth = df_eth.drop(columns=hex_cols)
df_eth = df_eth.fillna(df_eth.median())
print("✔ Ethereum cleaned and NaN filled")

# ===== CLEAN BITCOIN DATASET =====
if "txId" in df_btc.columns:
    df_btc = df_btc.drop(columns=["txId"], errors='ignore')

df_btc = df_btc.select_dtypes(include=[np.number])
df_btc = df_btc.fillna(df_btc.median())

# ===== BALANCE BEFORE SPLIT =====
def balance_df(df, name=""):
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]
    if len(df1) == 0:
        n_proxy = max(1, int(0.05 * len(df0)))
        df.loc[df0.sample(n_proxy).index, "label"] = 1
        df0 = df[df["label"] == 0]
        df1 = df[df["label"] == 1]
    n = min(len(df0), len(df1))
    df_balanced = pd.concat([df0.sample(n), df1.sample(n)])
    print(f"✔ {name} balanced:", np.bincount(df_balanced["label"].astype(int)))
    return df_balanced

df_btc = balance_df(df_btc, "Bitcoin")
df_eth = balance_df(df_eth, "Ethereum")

# ===== FEATURE / LABEL SPLIT =====
X_btc = df_btc.drop(columns=["label"]).values
y_btc = df_btc["label"].values

X_eth = df_eth.drop(columns=["label"]).values
y_eth = df_eth["label"].values

features = df_btc.drop(columns=["label"]).columns.tolist()

# ===== MAKE BOTH CHAINS SAME FEATURE DIMENSION =====
common_dim = X_eth.shape[1]  # force alignment
X_btc = X_btc[:, :common_dim]

# ===== SCALE FEATURES =====
scaler = StandardScaler()
X_btc = scaler.fit_transform(X_btc)
X_eth = scaler.fit_transform(X_eth)

# ===== TRAIN/TEST SPLIT =====
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_btc, y_btc, test_size=0.2, shuffle=True, stratify=y_btc
)
Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    X_eth, y_eth, test_size=0.2, shuffle=True, stratify=y_eth
)

# ===== TRAIN BASELINES =====
rf_btc = RandomForestClassifier().fit(Xb_train, yb_train)
lr_btc = LogisticRegression(max_iter=300, solver='lbfgs').fit(Xb_train, yb_train)

rf_eth = RandomForestClassifier().fit(Xe_train, ye_train)
lr_eth = LogisticRegression(max_iter=300, solver='lbfgs').fit(Xe_train, ye_train)

# ===== TRAIN TRANSFORMER =====
model = TransformerDetector(input_dim=common_dim)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

Xt_train = torch.tensor(Xb_train, dtype=torch.float32)
yt_train = torch.tensor(yb_train, dtype=torch.long)

losses=[]
for epoch in range(25):
    optimizer.zero_grad()
    out = model(Xt_train)
    loss = criterion(out, yt_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/25  Loss: {loss.item():.4f}")

# ===== PREDICTION =====
def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()[:,1]

yb_tpred = predict_torch(model, Xb_test)
ye_tpred = predict_torch(model, Xe_test)

# ===== METRICS =====
print("\n📌 Bitcoin Baselines vs Transformer")
print("Random Forest F1:", f1_score(yb_test, (yb_rf := rf_btc.predict_proba(Xb_test)[:,1]) > 0.5))
print("Logistic Reg F1:", f1_score(yb_test, (yb_lr := lr_btc.predict_proba(Xb_test)[:,1]) > 0.5))
print("Transformer F1:", f1_score(yb_test, (yb_tpred > 0.5).astype(int)))

print("\n📌 Ethereum Baselines vs Transformer")
print("Random Forest F1:", f1_score(ye_test, (ye_rf := rf_eth.predict_proba(Xe_test)[:,1]) > 0.5))
print("Logistic Reg F1:", f1_score(ye_test, (ye_lr := lr_eth.predict_proba(Xe_test)[:,1]) > 0.5))
print("Transformer F1:", f1_score(ye_test, (ye_tpred > 0.5).astype(int)))

# ===== GRAPHS =====
def save_curve(x, y, title, file):
    plt.figure()
    sns.lineplot(x=x, y=y)
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.savefig(file); plt.close()

save_curve(pd.Series(range(25)), pd.Series(losses), "Loss Curve", "outputs/loss_curve.png")

save_curve(*roc_curve(yb_test, yb_tpred)[:2], "ROC Bitcoin", "outputs/roc_bitcoin.png")
save_curve(*roc_curve(ye_test, ye_tpred)[:2], "ROC Ethereum", "outputs/roc_ethereum.png")

save_curve(*precision_recall_curve(yb_test, yb_tpred)[:2], "PR Bitcoin", "outputs/pr_bitcoin.png")
save_curve(*precision_recall_curve(ye_test, ye_tpred)[:2], "PR Ethereum", "outputs/pr_ethereum.png")

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(yb_test, (yb_tpred>0.5).astype(int))
plt.figure(); sns.heatmap(cm, annot=True, fmt="d"); plt.title("CM Bitcoin")
plt.savefig("outputs/cm_bitcoin.png"); plt.close()

cm2 = confusion_matrix(ye_test, (ye_tpred>0.5).astype(int))
plt.figure(); sns.heatmap(cm2, annot=True, fmt="d"); plt.title("CM Ethereum")
plt.savefig("outputs/cm_ethereum.png"); plt.close()

# ===== XAI =====
explain_shap(model, Xb_train, Xb_test, features, "outputs/shap_bitcoin.png")
explain_lime(model, Xb_train, Xb_test[0], features, "outputs/lime_bitcoin.html")

explain_shap(model, Xe_train, Xe_test, features, "outputs/shap_ethereum.png")
explain_lime(model, Xe_train, Xe_test[0], features, "outputs/lime_ethereum.html")

# ===== COMPARISON & WINNER =====
compare_models(Xb_train, yb_train, Xb_test, yb_test, Xe_train, ye_train, Xe_test, ye_test, model, features)

print("\n✔ Finished! All outputs saved!")
