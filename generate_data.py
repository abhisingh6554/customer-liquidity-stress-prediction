import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 2000

avg_balance = rng.lognormal(mean=8.2, sigma=0.9, size=n)
total_inflows = rng.lognormal(mean=9.0, sigma=0.8, size=n)
total_withdrawals = rng.lognormal(mean=8.8, sigma=0.85, size=n)
withdrawal_frequency = rng.poisson(8, n) + 1
transaction_frequency = rng.poisson(28, n) + 2
merchant_activity = rng.poisson(12, n)
balance_volatility = rng.lognormal(mean=5.2, sigma=0.8, size=n)
days_low_balance = rng.poisson(4, n)
avg_transaction_amount = rng.lognormal(mean=5.7, sigma=0.7, size=n)

inflow_outflow_ratio = total_inflows / (total_withdrawals + 1)

# Synthetic risk-generating process for portfolio demonstration.
risk_score = (
    -0.9
    - 0.000035 * avg_balance
    - 0.000018 * total_inflows
    + 0.000022 * total_withdrawals
    + 0.055 * withdrawal_frequency
    - 0.012 * transaction_frequency
    - 0.018 * merchant_activity
    + 0.012 * days_low_balance
    + 0.000025 * balance_volatility
    - 0.28 * np.log1p(inflow_outflow_ratio)
)
prob = 1 / (1 + np.exp(-risk_score))
liquidity_stress_30d = rng.binomial(1, prob)

df = pd.DataFrame({
    "customer_id": [f"CUST_{i:05d}" for i in range(1, n + 1)],
    "avg_balance": avg_balance.round(2),
    "balance_volatility": balance_volatility.round(2),
    "total_inflows": total_inflows.round(2),
    "total_withdrawals": total_withdrawals.round(2),
    "withdrawal_frequency": withdrawal_frequency,
    "transaction_frequency": transaction_frequency,
    "merchant_activity": merchant_activity,
    "inflow_outflow_ratio": inflow_outflow_ratio.round(3),
    "days_low_balance": days_low_balance,
    "avg_transaction_amount": avg_transaction_amount.round(2),
    "liquidity_stress_30d": liquidity_stress_30d
})

out = Path(__file__).resolve().parents[1] / "data" / "mobile_money_customers.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} synthetic customer records to {out}")
