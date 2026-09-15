# Customer Liquidity Stress Prediction

An end-to-end machine learning project that predicts whether a mobile-money customer may experience liquidity stress within the following 30 days using transaction behavior and account activity.

## Problem Statement
Financial liquidity stress can occur when a customer has insufficient available funds relative to their recent spending and withdrawal behavior. This project builds a binary classification model using mobile-money activity such as inflows, withdrawals, balances, transaction frequency, and merchant activity.

> **Data note:** This repository uses synthetic data for demonstration and portfolio purposes. No real customer financial data is included.

## Features
- Average balance
- Balance volatility
- Total inflows
- Total withdrawals
- Withdrawal frequency
- Transaction frequency
- Merchant activity
- Inflow/outflow ratio
- Days with low balance
- Average transaction amount

## Target
`liquidity_stress_30d`
- `0`: No expected liquidity stress
- `1`: Expected liquidity stress within 30 days

## Models
The notebook trains and compares:
- Logistic Regression
- Random Forest

The final model is selected using ROC-AUC on the validation/test data.

## Project Structure
```text
customer-liquidity-stress-prediction/
├── data/
│   └── mobile_money_customers.csv
├── models/
│   └── liquidity_stress_model.pkl
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_metrics.json
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run Locally
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
python src/predict.py
```

## Example Prediction
The prediction script demonstrates how the trained model can estimate a customer's 30-day liquidity-stress probability.

## Disclaimer
This is an educational machine-learning portfolio project. A production financial-risk system would require validated labels, representative data, fairness testing, privacy controls, monitoring, and domain/regulatory review.
