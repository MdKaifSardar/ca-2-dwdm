<div align="center">

# 🔬 Integrated Data Cleansing & Hybrid Feature Selection

**A Reproducible Framework for Improving Predictive Performance in Large Datasets**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](#)
[![Imbalanced-Learn](https://img.shields.io/badge/imbalanced--learn-%23000000.svg?logo=python&logoColor=white)](#)

</div>

## 📌 Overview

This repository contains a reproducible, research-grade machine learning pipeline. It systematically evaluates how different data cleansing strategies and feature selection topologies (Filter, Wrapper, Embedded) impact classification performance on large datasets.

## 🏗️ Architecture

The framework is strictly modular, preventing data leakage and ensuring isolated experimental runs:

- ⚙️ **`src/config.py`**: Centralized hyperparameters and reproducible deterministic seeds.
- 🧹 **`src/data_cleaning.py`**: Imputation, deduplication, outlier handling (IQR), scaling, and SMOTE balancing.
- 🎯 **`src/feature_selection.py`**:
  - **Filter**: Mutual Information, Correlation
  - **Wrapper**: Recursive Feature Elimination (RFE)
  - **Embedded**: L1 Logistic, Random Forest Importance
- 🧠 **`src/model_training.py`**: Cross-validated Baseline Models (Logistic Regression, Random Forest).
- 📊 **`src/evaluation.py`**: Extraction of Precision, Recall, F1, ROC-AUC, and Confusion Matrices.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place the target dataset (e.g., Kaggle's Credit Card Fraud dataset) in the `data/` directory.

```text
your-project-root/
├── data/
│   └── creditcard.csv  <-- Place dataset here
├── experiments/
├── src/
...
```

### 3. Execution

Run the automated experimental pipeline:

```bash
python main.py
```

---

## 📈 Outputs & Metrics

The orchestrator dynamically processes the dataset through **5 distinct evaluation tracks**:

1. **Raw Data** (Baseline)
2. **Cleaned Data** (Imputed, Outlier Removed, Scaled, SMOTE Balanced)
3. **Cleaned + Filter Selection** (Mutual Information)
4. **Cleaned + Wrapper Selection** (RFE)
5. **Cleaned + Embedded Selection** (Random Forest)

All results are automatically generated and saved:

- 📄 `results/metrics.csv` _(Aggregated numerical telemetry: F1, ROC-AUC, Time)_
- 🖼️ `results/plots/` _(High-res Confusion Matrix visual plots)_
- 📝 `results/experiment.log` _(Detailed step instrumentation)_

---

## 🔬 Experimental Reproducibility

To ensure strict academic reproducibility and structural integrity:

- **Global Seeding**: All stochastic operations (Numpy arrays, SMOTE, Tree instantiation, KFold splits) are locked to `config.RANDOM_SEED`.
- **Data Leakage Prevention**: Resampling techniques (SMOTE) and target-aware feature selections are applied **strictly to the training subsets** during cross-validation boundaries.
