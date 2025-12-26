import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from sklearn.metrics import classification_report, f1_score

# ===== SHAP GLOBAL EXPLANATION =====
def explain_shap(model, X_train, X_test, features, out_file):
    def predict_fn(x):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    X_bg = X_train[:100]
    explainer = shap.KernelExplainer(predict_fn, X_bg)
    shap_values = explainer.shap_values(X_test[:50])

    plt.figure()
    shap.summary_plot(shap_values, X_test[:50], feature_names=features, show=False)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print("✔ SHAP explanation saved →", out_file)

# ===== LIME LOCAL EXPLANATION =====
def explain_lime(model, X_train, X_sample, features, html_file):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=features,
        class_names=["Normal", "Anomaly"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_sample,
        lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
        num_features=10
    )

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(exp.as_html())

    print("✔ LIME explanation saved →", html_file)

# ===== MODEL COMPARISON FUNCTION =====
def compare_models(Xb_train, yb_train, Xb_test, yb_test,
                   Xe_train, ye_train, Xe_test, ye_test,
                   transformer_model, feature_names):

    print("\n===== 🔍 MODEL PERFORMANCE COMPARISON =====\n")

    # Predictions
    def torch_pred(X):
        transformer_model.eval()
        with torch.no_grad():
            logits = transformer_model(torch.tensor(X, dtype=torch.float32))
            return torch.argmax(logits, dim=1).numpy()

    yb_tr = torch_pred(Xb_test)
    ye_tr = torch_pred(Xe_test)

    # F1 Scores
    f1_btc = f1_score(yb_test, yb_tr)
    f1_eth = f1_score(ye_test, ye_tr)

    print("📌 Bitcoin Transformer Report:")
    print(classification_report(yb_test, yb_tr))

    print("📌 Ethereum Transformer Report:")
    print(classification_report(ye_test, ye_tr))

    print("\nBitcoin Transformer F1 Score:", f1_btc)
    print("Ethereum Transformer F1 Score:", f1_eth)

    # Verdict
    if f1_eth > f1_btc:
        print("\n🏆 Transformer works better on Ethereum anomaly patterns")
    else:
        print("\n🏆 Transformer works better on Bitcoin anomaly patterns")

    print("\n===== ✔ Comparison Finished =====\n")
