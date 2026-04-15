# Explainable & Cost-Aware Fraud Detection with Concept Drift Monitoring

> A full machine learning pipeline for credit card fraud detection — covering preprocessing, model comparison, hyperparameter tuning, SHAP explainability, and concept drift detection.

---

## Overview

This project investigates machine learning approaches for automated fraud detection on a large-scale consumer credit dataset (150,000 records, 18 features). The pipeline addresses the complete ML lifecycle — from raw data ingestion through to production-ready drift monitoring.

**Key results:**
- Best model: Tuned Random Forest — cross-validated **AUC-ROC of 0.906**
- Fraud recall improved from **26% → 76%** after hyperparameter tuning
- Overfitting gap reduced from **4.79% → 0.68%**
- Automated drift monitoring via sliding-window detection and Evidently AI

---

## Repository Structure

```
drift-detection-fraud/
│
├── Drift_detection_code.ipynb   # Main notebook — full pipeline end to end
├── data_drift_report.html       # Evidently AI drift report (reference vs current)
├── credit_final_model.pkl       # Saved final model (tuned Random Forest)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Pipeline Summary

### 1. Data Preprocessing
- Removed 2 spurious null rows → 150,000 clean records
- **Missing value imputation:** mode imputation for `MonthlyIncome` (19.8% missing), median for `NumberOfDependents` (2.9%) — fitted on training data only
- **Log transformation** (`log(x+1)`) applied to all numerical features to reduce skewness
- **Outlier handling:** 5th–95th percentile capping (thresholds from training data applied to test)
- **Categorical encoding:** One-Hot for Gender and Region; Ordinal for Housing, Occupation, Education
- **Feature scaling:** StandardScaler fitted on training, applied to both sets
- Train/test split: 90/10 (135,000 / 15,000) — split performed before any transformation

### 2. Model Comparison
Five classifiers benchmarked using AUC-ROC and a domain-specific financial cost function:

| Model | AUC-ROC | Financial Cost | Test Accuracy |
|---|---|---|---|
| Naive Bayes | 0.732 | £702,200 | ~85% |
| Decision Tree | 0.683 | £319,660 | ~91% |
| Random Forest | 0.662 | £123,830 | 95.2% |
| KNN | 0.627 | £73,570 | 94.4% |
| Logistic Regression | 0.537 | £34,370 | ~87% |

> **Cost function:** FN (missed fraud) = £500 penalty · FP (false alarm) = £10 penalty

### 3. Hyperparameter Tuning
Two-stage tuning on Random Forest:
- **Stage 1 — RandomizedSearchCV:** 15 iterations, 3-fold stratified CV → best AUC: 0.9066
- **Stage 2 — GridSearchCV:** narrow refined search → best AUC: 0.9019

Final parameters: `n_estimators=300, max_depth=12, min_samples_split=15, min_samples_leaf=2, class_weight='balanced'`

### 4. SHAP Explainability
SHAP TreeExplainer applied to 200 test samples. Top predictors by mean absolute SHAP value:

1. `RevolvingUtilizationOfUnsecuredLines` — 31.4% importance
2. `Central` (region) — 14.6%
3. `Education_re` — 12.9%

### 5. Concept Drift Detection
Two complementary approaches:

- **Sliding-window detector:** monitors mean predicted fraud probability across windows of 5,000 records; flags drift if shift exceeds threshold of 0.05
- **ADWIN (Adaptive Windowing):** online drift detector via the `river` library (`delta=0.002`); processes the prediction stream one sample at a time
- **Evidently AI report:** batch statistical testing — KS test for numerical features, chi-squared for categorical — comparing training (reference) vs test (current) distributions

No drift detected on the static test set, as expected. Both detectors are designed for live production streaming data.

---

## Getting Started

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies:
```
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
shap
evidently==0.5.0
river
feature-engine
imbalanced-learn
```

### Running the Notebook
1. Clone the repository
```bash
git clone https://github.com/srihari4420/drift-detection-fraud.git
cd drift-detection-fraud
```

2. Place `creditcard.csv` in the root directory (dataset not included due to size)

3. Open and run `Drift_detection_code.ipynb` top to bottom

> All transformers (scaler, encoders, imputation values) are fitted on training data and applied to the test set — no data leakage.

---

## Dataset

- **Source:** Europen credit card holders data set from kaggle
- **Records:** 150,000 (after cleaning)
- **Features:** 18 (numerical + categorical)
- **Target:** `Good_Bad` — Good (legitimate) / Bad (fraud/default)
- **Class split:** 93.96% Good / 6.04% Bad

The dataset is not included in this repository. Download and Place `creditcard.csv` in the root directory before running.

---

## Ethical Considerations

- Dataset contains anonymised demographic features (gender, region, education) — no PII included
- SHAP analysis provides transparency over which features drive individual predictions
- Demographic features carry potential for proxy discrimination; fairness auditing (e.g. equalised odds) recommended before any production deployment
- Financial cost function explicitly models the asymmetric impact of false positives vs false negatives

---

## Future Work
- [ ] Integrate with a streaming platform (Apache Kafka + MLflow) for end-to-end monitoring

---

## References

- Bhattacharyya et al. (2011) — Random Forests for credit card fraud
- Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
- Gama et al. (2014) — A Survey on Concept Drift Adaptation
- Evidently AI (2024) — [evidentlyai.com](https://www.evidentlyai.com)
- UK Finance (2024) — Annual Fraud Report

---

