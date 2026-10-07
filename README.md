# 🌟 Oasis Infobyte — Data Analytics Internship (Level 2 Portfolio)
> **Intern Name:** Deeva Jain  
> **Internship Track:** Data Analytics (Level 2)  
> **GitHub Repository:** [Deeva_jain_Submission_Level2](https://github.com/Deeva2601/Deeva_jain_Submission_Level2)  
> **Status:** ✅ All Level 2 Tasks 100% Complete, Executed & Verified (Recruiter-Ready)

---

## 📑 Level 2 Portfolio Table of Contents
1. [Level 2 · Task 1: Predicting House Prices with Linear Regression](#-level-2--task-1-predicting-house-prices-with-linear-regression)
2. [Level 2 · Task 2: Wine Quality Prediction & Classification](#-level-2--task-2-wine-quality-prediction--classification)
3. [Repository Structure & Project Layout](#-repository-structure)
4. [Installation & Execution Guide](#-installation--execution-guide)

---

# 🏡 LEVEL 2 · TASK 1: Predicting House Prices with Linear Regression

### 🎯 Objective
Build and evaluate a multi-variable regression pipeline that predicts residential property prices based on living area, spatial geography, rooms, age, condition, and luxury amenities.

### 🛠️ Tech Stack
`Python 3.13` | `Pandas` | `NumPy` | `Scikit-Learn (LinearRegression, Ridge, Lasso, StandardScaler)` | `Matplotlib` | `Seaborn`

### 📋 Feature Checklist Compliance
- [x] **Data Ingestion & EDA**: 2,800 properties across 13 structural features with 0 nulls and 0 duplicates.
- [x] **Feature Selection Discussion**: Detailed economic and architectural domain justifications in markdown cells.
- [x] **One-Hot Encoding**: Encoded categorical neighborhoods via `pd.get_dummies(drop_first=True)` to avoid dummy variable traps.
- [x] **Correlation Matrix Heatmap**: Analyzed Pearson correlation coefficients against property price.
- [x] **Train/Test Split (80/20)**: Partitioned 2,240 training and 560 testing samples.
- [x] **Linear Regression (OLS) Training**: Trained classical Ordinary Least Squares model.
- [x] **Evaluation Metrics**: Achieved **MAE ($24,459.80)**, **RMSE ($31,557.82)**, and **$R^2$ Score (0.9720 / 97.2%)**.
- [x] **Actual vs. Predicted Scatter Plot**: Visualized predictions against ground truth with $y=x$ ideal reference line.
- [x] **Residual Diagnostics Plot**: Verified homoscedasticity and zero-mean error dispersion.
- [x] **Coefficient Analysis**: Quantified marginal dollar values per feature (Living area: +$180–$240/sq ft, Pool: +$38k, Age: -$1.65k/year).
- [x] **(Bonus) Regularization Comparison**: Benchmarked against **Ridge ($L_2$)** and **Lasso ($L_1$)** regression.

### 🤖 Task 1 Regression Models Performance Benchmark

| Regression Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score (% Variance) | Model Characteristic |
|---|:---:|:---:|:---:|---|
| **Linear Regression (OLS)** | **$24,459.80** | **$31,557.82** | **0.9720 (97.2%)** | Unbiased classical minimum-variance estimator |
| **Ridge Regression ($L_2$)** | **$24,446.89** | **$31,547.04** | **0.9720 (97.2%)** | $L_2$ shrinkage controlling collinearity |
| **Lasso Regression ($L_1$)** | **$24,453.76** | **$31,557.56** | **0.9720 (97.2%)** | $L_1$ penalty enforcing automated feature sparsity |

### 📈 Task 1 Visual Assets Summary
- `assets/figures/01_target_price_distribution.png` — Price histogram & neighborhood boxplots
- `assets/figures/02_correlation_heatmap.png` — Pearson correlation matrix between housing features and sale price
- `assets/figures/03_actual_vs_predicted_residuals.png` — Actual vs Predicted scatter plot & homoscedastic residual diagnostics
- `assets/figures/04_feature_coefficients.png` — Feature coefficient impact rankings

---

# 🍷 LEVEL 2 · TASK 2: Wine Quality Prediction & Classification

### 🎯 Objective
Train and compare multiple classification models to predict the quality score and commercial grade of wine based on its 11 physicochemical properties (acidity, sulphates, density, alcohol, pH).

### 🛠️ Tech Stack
`Python 3.13` | `Pandas` | `NumPy` | `Scikit-Learn (Random Forest, SGD Classifier, SVC, StandardScaler)` | `Matplotlib` | `Seaborn`

### 📋 Feature Checklist Compliance
- [x] **Data Ingestion & Structural Inspection**: Ingested authentic UCI Red Wine Quality dataset (1,599 samples, 11 physicochemical features, 0 nulls).
- [x] **Physicochemical Distribution Plots**: Plotted 11-panel distribution grid (histograms + KDE curves) & correlation heatmap.
- [x] **Class Imbalance Diagnosis**: Addressed majority-class concentration (scores 5 & 6 represent 82.5% of samples).
- [x] **Quality Score Binning**: Formulated binary classification ($\ge 7$ as *Good Quality*, $< 7$ as *Standard Quality*) aligned with commercial winery grading.
- [x] **Stratified Train/Test Split & Scaling**: Applied stratified 80/20 split and scaled features via `StandardScaler`.
- [x] **Multi-Classifier Training**: Trained **Random Forest (200 estimators)**, **SGD Classifier**, and **Support Vector Classifier (SVC)**.
- [x] **Comprehensive Evaluation**: Evaluated Accuracy, Precision, Recall, F1-Scores, ROC-AUC, and Confusion Matrix heatmaps.
- [x] **Feature Importance Extraction**: Extracted Gini importance scores identifying alcohol (24.7%), sulphates (13.1%), and volatile acidity (10.7%) as primary quality drivers.
- [x] **Side-by-Side Model Comparison**: Built comparative benchmark table summarizing all 3 models.
- [x] **Oenological Applications**: Formulated automated barrel classification, fermentation monitoring, and dynamic retail pricing systems.

### 🤖 Task 2 Classification Models Performance Benchmark

| Classification Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | ROC-AUC | Operational Assessment |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Random Forest Classifier** | **92.50%** | **75.68%** | **65.12%** | **0.7000** | **0.9372** | **Top Performer** (Highest overall accuracy & ROC-AUC) |
| **Support Vector Classifier (SVC)** | **85.94%** | **48.53%** | **76.74%** | **0.5946** | **0.8956** | **High Sensitivity** (Effective minority-class capture) |
| **SGD Classifier** | **79.38%** | **36.78%** | **74.42%** | **0.4923** | **0.7909** | **Fast Baseline** (Iterative mini-batch linear classification) |

### 📈 Task 2 Visual Assets Summary
- `assets/figures/05_physicochemical_distributions.png` — 11-panel grid of chemical property distributions
- `assets/figures/06_wine_correlation_heatmap.png` — Pearson correlation matrix of chemical features vs quality
- `assets/figures/07_class_imbalance_binning.png` — Raw quality imbalance & binarized target distribution
- `assets/figures/08_wine_confusion_matrices.png` — Confusion matrix heatmaps for Random Forest, SGD, and SVC
- `assets/figures/09_wine_feature_importance.png` — Random Forest Gini feature importance rankings

---

## 📂 Repository Structure

```
Deeva_jain_Submission_Level2/
│
├── LEVEL2_TASK_1_House_Price_Prediction.ipynb  # Master executed Notebook for Level 2 Task 1 (Regression)
├── LEVEL2_TASK_2_Wine_Quality_Prediction.ipynb  # Master executed Notebook for Level 2 Task 2 (Classification)
│
├── house_price_prediction.py                   # Standalone Python pipeline (Level 2 Task 1)
├── wine_quality_classification.py              # Standalone Python pipeline (Level 2 Task 2)
│
├── house_prices_dataset.csv                    # Dataset for Level 2 Task 1 (2,800 properties)
├── winequality-red.csv                         # Dataset for Level 2 Task 2 (1,599 samples)
│
├── generate_house_prices_data.py               # Housing data generator script
├── build_level2_task1_notebook.py              # Notebook builder for Task 1
├── build_level2_task2_notebook.py              # Notebook builder for Task 2
│
├── README.md                                   # Comprehensive Level 2 Portfolio Report
└── assets/
    └── figures/                                # High-Res 300 DPI exported visual assets (9 charts)
        ├── 01_target_price_distribution.png
        ├── 02_correlation_heatmap.png
        ├── 03_actual_vs_predicted_residuals.png
        ├── 04_feature_coefficients.png
        ├── 05_physicochemical_distributions.png
        ├── 06_wine_correlation_heatmap.png
        ├── 07_class_imbalance_binning.png
        ├── 08_wine_confusion_matrices.png
        └── 09_wine_feature_importance.png
```

---

## 🛠️ Installation & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Deeva2601/Deeva_jain_Submission_Level2.git
cd Deeva_jain_Submission_Level2
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Run Jupyter Notebooks
```bash
# Task 1: House Price Regression
jupyter notebook LEVEL2_TASK_1_House_Price_Prediction.ipynb

# Task 2: Wine Quality Classification
jupyter notebook LEVEL2_TASK_2_Wine_Quality_Prediction.ipynb
```

### 4. Run Headless Python Pipelines
```bash
# Execute Task 1 Pipeline
python house_price_prediction.py

# Execute Task 2 Pipeline
python wine_quality_classification.py
```
