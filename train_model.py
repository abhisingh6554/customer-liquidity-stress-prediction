from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mobile_money_customers.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Dataset not found. Run: python src/generate_data.py")

df = pd.read_csv(DATA)
features = [
    "avg_balance", "balance_volatility", "total_inflows",
    "total_withdrawals", "withdrawal_frequency",
    "transaction_frequency", "merchant_activity",
    "inflow_outflow_ratio", "days_low_balance",
    "avg_transaction_amount"
]
X = df[features]
y = df["liquidity_stress_30d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", min_samples_leaf=3
        ))
    ])
}

results = {}
best_name, best_model, best_auc = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(auc, 4)
    }

    if auc > best_auc:
        best_name, best_model, best_auc = name, model, auc

joblib.dump(best_model, MODEL_DIR / "liquidity_stress_model.pkl")
(OUTPUT_DIR / "model_metrics.json").write_text(
    json.dumps({"best_model": best_name, "results": results}, indent=2)
)

pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.close()

# Feature importance for Random Forest when selected.
if best_name == "Random Forest":
    importances = best_model.named_steps["model"].feature_importances_
    imp = pd.Series(importances, index=features).sort_values()
    plt.figure(figsize=(8, 5))
    imp.plot(kind="barh")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()

print("Model comparison:")
for name, metrics in results.items():
    print(name, metrics)
print(f"\nBest model: {best_name} | ROC-AUC: {best_auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
