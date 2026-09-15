from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "liquidity_stress_model.pkl"
DATA = ROOT / "data" / "mobile_money_customers.csv"

if not MODEL.exists():
    raise FileNotFoundError("Model not found. Run: python src/train_model.py")

model = joblib.load(MODEL)
df = pd.read_csv(DATA)

features = [
    "avg_balance", "balance_volatility", "total_inflows",
    "total_withdrawals", "withdrawal_frequency",
    "transaction_frequency", "merchant_activity",
    "inflow_outflow_ratio", "days_low_balance",
    "avg_transaction_amount"
]

sample = df.iloc[[0]][features]
probability = model.predict_proba(sample)[0, 1]
prediction = int(probability >= 0.5)

print(f"Customer: {df.iloc[0]['customer_id']}")
print(f"30-day liquidity stress probability: {probability:.2%}")
print(f"Prediction: {'Potential Stress' if prediction else 'No Expected Stress'}")
