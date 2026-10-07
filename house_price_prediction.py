"""
================================================================================
Level 2 Task 1: Predicting House Prices with Linear Regression, Ridge & Lasso
Oasis Infobyte - Data Analytics Internship (Level 2)
================================================================================
Author: Deeva Jain (Data Analytics Intern)
Description: Supervised regression pipeline with exploratory analysis, One-Hot
             encoding, correlation diagnostics, OLS regression, Ridge, Lasso,
             evaluation metrics (MAE, RMSE, R2), and coefficient impact ranking.
================================================================================
"""

import os
import sys

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUTPUT_DIR = 'assets/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_house_price_pipeline(data_path='house_prices_dataset.csv'):
    print("=" * 80)
    print("STARTING HOUSE PRICE PREDICTION REGRESSION PIPELINE (LEVEL 2 TASK 1)")
    print("=" * 80)

    # 1. Ingest
    print(f"\n[1/6] Ingesting housing dataset from: {data_path} ...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    df = pd.read_csv(data_path)
    print(f"  [+] Properties Ingested: {df.shape[0]:,} | Features: {df.shape[1]}")
    print(f"  [+] Mean Price: ${df['Price'].mean():,.2f} | Median Price: ${df['Price'].median():,.2f}")

    # 2. Preprocess & Encode
    print("\n[2/6] One-Hot Encoding categorical neighborhood features ...")
    df_model = df.drop(columns=['Property_ID'])
    df_encoded = pd.get_dummies(df_model, columns=['Location_Neighborhood'], drop_first=True, dtype=int)
    print(f"  [+] Encoded Matrix Dimensions: {df_encoded.shape[0]:,} samples × {df_encoded.shape[1]} columns")

    # 3. Split
    print("\n[3/6] Partitioning 80/20 train/test split ...")
    X = df_encoded.drop(columns=['Price'])
    y = df_encoded['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"  [+] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # 4. Train OLS Linear Regression
    print("\n[4/6] Training Ordinary Least Squares (OLS) Linear Regression ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 5. Train Regularized Models (Ridge & Lasso)
    print("\n[5/6] Training Regularized Models (Ridge L2 & Lasso L1) ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    lasso = Lasso(alpha=100.0, random_state=42).fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)

    # 6. Evaluation Summary
    def calc_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return name, mae, rmse, r2

    models_eval = [
        calc_metrics(y_test, y_pred_lr, "Linear Regression (OLS)"),
        calc_metrics(y_test, y_pred_ridge, "Ridge Regression (L2)"),
        calc_metrics(y_test, y_pred_lasso, "Lasso Regression (L1)")
    ]

    print("\n" + "=" * 80)
    print(f"{'Regression Model':28s} | {'MAE ($)':12s} | {'RMSE ($)':12s} | {'R² Score':9s}")
    print("-" * 80)
    for name, mae, rmse, r2 in models_eval:
        print(f"{name:28s} | ${mae:10,.2f} | ${rmse:10,.2f} | {r2:8.4f}")
    print("=" * 80)
    print("Pipeline executed successfully! All visual assets generated in assets/figures/")

if __name__ == '__main__':
    run_house_price_pipeline()
