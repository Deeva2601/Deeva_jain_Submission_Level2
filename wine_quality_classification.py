"""
================================================================================
Level 2 Task 2: Physicochemical Wine Quality Classification
Oasis Infobyte - Data Analytics Internship (Level 2)
================================================================================
Author: Deeva Jain (Data Analytics Intern)
Description: Automated pipeline for class imbalance handling, feature scaling,
             multi-classifier training (Random Forest, SGD, SVC), Gini feature
             importance extraction, and comprehensive oenological evaluation.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = 'assets/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_wine_quality_pipeline(data_path='winequality-red.csv'):
    print("=" * 80)
    print("STARTING WINE QUALITY CLASSIFICATION PIPELINE (LEVEL 2 TASK 2)")
    print("=" * 80)

    # 1. Ingestion
    print(f"\n[1/5] Ingesting UCI Wine Quality dataset from: {data_path} ...")
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)
    df = pd.read_csv(data_path)
    print(f"  [+] Wine Samples Ingested: {df.shape[0]:,} | Features: {df.shape[1]}")

    # 2. Imbalance & Feature Engineering
    print("\n[2/5] Engineering binary quality target (Quality >= 7 as Good) ...")
    chem_features = [c for c in df.columns if c not in ['quality', 'Quality_Binary']]
    df['Quality_Binary'] = np.where(df['quality'] >= 7, 1, 0)
    good_cnt = sum(df['Quality_Binary'])
    std_cnt = len(df) - good_cnt
    print(f"  [+] Standard Table Wines (< 7): {std_cnt:,} ({std_cnt/len(df)*100:.1f}%)")
    print(f"  [+] Premium Good Wines   (>= 7): {good_cnt:,} ({good_cnt/len(df)*100:.1f}%)")

    # 3. Train/Test Split & Scaling
    print("\n[3/5] Performing stratified 80/20 train/test split & StandardScaler ...")
    X = df[chem_features]
    y = df['Quality_Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Multi-Classifier Training
    print("\n[4/5] Training 3 Classification Architectures ...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42),
        'SGD Classifier': SGDClassifier(loss='modified_huber', max_iter=1000, class_weight='balanced', random_state=42),
        'Support Vector (SVC)': SVC(kernel='rbf', C=1.5, class_weight='balanced', probability=True, random_state=42)
    }

    results = []
    for name, model in models.items():
        if name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        results.append((name, acc, prec, rec, f1, roc))

    # 5. Output Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Classifier Model':22s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s} | {'ROC-AUC':8s}")
    print("-" * 80)
    for name, acc, prec, rec, f1, roc in results:
        print(f"{name:22s} | {acc*100:7.2f}% | {prec*100:7.2f}% | {rec*100:6.2f}% | {f1:8.4f} | {roc:8.4f}")
    print("=" * 80)

    # Feature Importance
    rf = models['Random Forest']
    imp = pd.Series(rf.feature_importances_, index=chem_features).sort_values(ascending=False)
    print("\nTop 5 Chemical Quality Drivers (Random Forest Gini Importance):")
    for feat, val in imp.head(5).items():
        print(f"  • {feat:22s}: {val*100:.2f}%")
    print("\nPipeline execution completed successfully!")

if __name__ == '__main__':
    run_wine_quality_pipeline()
