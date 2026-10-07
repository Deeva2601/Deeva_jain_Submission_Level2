# 🏡 Oasis Infobyte — Data Analytics Internship (Level 2)
## 📊 Level 2 · Task 1: Predicting House Prices with Linear Regression

> **Intern Name:** Deeva Jain  
> **Internship Track:** Data Analytics (Level 2)  
> **GitHub Repository:** [Deeva_jain_Submission_Level2](https://github.com/Deeva2601/Deeva_jain_Submission_Level2)  
> **Status:** ✅ 100% Complete, Fully Executed & Verified (Recruiter-Ready)

---

## 🎯 Executive Project Overview & Objectives
Accurate residential valuation models power modern FinTech lending, mortgage underwriting, real estate investment appraisals, and urban planning.

The primary objective of this project is to build an end-to-end **Supervised Regression Pipeline** to predict property sale prices based on living area, spatial geography, bedroom/bathroom configurations, property age, condition rating, and luxury amenities.

---

## 📋 Evaluation Checklist & Deliverables Matrix

| # | Feature Requirement | Status | Implementation Highlights |
|---|---|:---:|---|
| 1 | **Data Ingestion & Exploratory Data Analysis** | ✅ | Analyzed 2,800 properties across 13 structural features with 0 nulls and 0 duplicates. |
| 2 | **Feature Selection Discussion** | ✅ | Detailed architectural & economic rationales justifying each predictor in markdown cells. |
| 3 | **One-Hot Encoding & Preprocessing** | ✅ | Encoded categorical neighborhoods via `pd.get_dummies(drop_first=True)` to prevent dummy variable traps. |
| 4 | **Correlation Matrix Heatmap** | ✅ | Visualized Pearson correlation coefficients across all numerical attributes and target `Price`. |
| 5 | **Stratified Train/Test Split** | ✅ | Partitioned 80% training (2,240 properties) and 20% testing (560 properties). |
| 6 | **Linear Regression (OLS) Training** | ✅ | Fitted Ordinary Least Squares model with Scikit-Learn. |
| 7 | **Comprehensive Evaluation Metrics** | ✅ | Calculated **MAE ($24,459)**, **MSE**, **RMSE ($31,557)**, and **$R^2$ Score (0.9720 / 97.2%)**. |
| 8 | **Actual vs. Predicted Scatter Plot** | ✅ | Plotted predictions against ground truth with $y=x$ ideal diagonal fit line. |
| 9 | **Residual Diagnostics Plot** | ✅ | Verified homoscedasticity and random error dispersion around zero. |
| 10 | **Feature Coefficient Impact Analysis** | ✅ | Ranked marginal dollar contribution per unit feature using horizontal coefficient charts. |
| 11 | **(Bonus) Regularization Benchmark** | ✅ | Benchmarked OLS against **Ridge ($L_2$)** and **Lasso ($L_1$)** regularized regression. |

---

## 🤖 Regression Models Performance Benchmark

| Regression Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score (% Variance Explained) | Model Characteristic |
|---|:---:|:---:|:---:|---|
| **Linear Regression (OLS)** | **$24,459.80** | **$31,557.82** | **0.9720 (97.2%)** | Unbiased classical minimum-variance estimator |
| **Ridge Regression ($L_2$)** | **$24,446.89** | **$31,547.04** | **0.9720 (97.2%)** | $L_2$ regularization controlling collinearity |
| **Lasso Regression ($L_1$)** | **$24,453.76** | **$31,557.56** | **0.9720 (97.2%)** | $L_1$ penalty enforcing automated feature sparsity |

---

## 📈 Visual Assets Summary
All charts are exported at 300 DPI in [`assets/figures/`](assets/figures/):
- `01_target_price_distribution.png` — Target price distribution histogram & neighborhood boxplots
- `02_correlation_heatmap.png` — Pearson correlation matrix between housing features and sale price
- `03_actual_vs_predicted_residuals.png` — Actual vs Predicted scatter plot & homoscedastic residual diagnostics
- `04_feature_coefficients.png` — Linear regression coefficient rankings showing marginal dollar impacts

---

## 💡 Key Economic Findings & Coefficient Insights
1. **Living Area (`Area_SqFt`)**: The single strongest driver of valuation, contributing ~**$180–$240 per additional sq ft**.
2. **Neighborhood Premiums**: Locations like *Downtown Metro* and *Green Hills* command substantial location premiums over outer suburban zones.
3. **Amenities**: A dedicated swimming pool adds ~**+$38,000** and each garage bay adds ~**+$19,500** in appraised value.
4. **Structural Depreciation**: Property age exerts a steady depreciation of ~**-$1,650 per year of age**.

---

## 📂 Repository Structure

```
Deeva_jain_Submission_Level2/
│
├── LEVEL2_TASK_1_House_Price_Prediction.ipynb  # Master executed Jupyter Notebook with all outputs & charts
├── house_price_prediction.py                   # Standalone automated Python regression pipeline
├── house_prices_dataset.csv                    # Clean housing dataset (2,800 properties, 13 attributes)
├── generate_house_prices_data.py               # Dataset generator script
├── README.md                                   # Comprehensive Level 2 Project Report
└── assets/
    └── figures/                                # High-Res 300 DPI visualization assets
        ├── 01_target_price_distribution.png
        ├── 02_correlation_heatmap.png
        ├── 03_actual_vs_predicted_residuals.png
        └── 04_feature_coefficients.png
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

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook LEVEL2_TASK_1_House_Price_Prediction.ipynb
```

### 4. Run Headless Python Script
```bash
python house_price_prediction.py
```
