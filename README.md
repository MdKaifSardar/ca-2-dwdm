# Integrated Data Cleansing and Hybrid Feature Selection Framework

## Research Objective

This system provides an integrated preprocessing framework designed to evaluate the impacts of structural data cleansing and varying feature selection topologies (Filter, Wrapper, Embedded) on predictive classification performance for large datasets. It targets the persistent academic issue of comparative reproducibility in feature engineering pipelines.

## System Architecture

The framework follows strict separation of concerns to allow seamless swapping of dataset or internal techniques without cascading changes.

- **`src/config.py`**: Centralized parameters ensuring reproducible, consistent state across runs.
- **`src/utils.py`**: Shared logic for directories and standardized structured logging.
- **`src/data_cleaning.py`**: A modular toolkit encompassing imputation, deduplication, outlier identification, scaling, and balancing via SMOTE.
- **`src/feature_selection.py`**: Topologically diverse methods extending from simple Mutual Information filters to computationally intensive Wrapper (RFE) techniques and structurally embedded (L1 regularized/Tree-based) selections.
- **`src/model_training.py`**: Container for baseline models, encompassing strict training bounds and time tracing.
- **`src/evaluation.py`**: The analytical core, extracting multiple dimensionality metrics (F1, ROC-AUC) to rigorously compare disparate setups.

## Requirements

Python 3.8+ required. Dependencies are listed in `requirements.txt`.

## Installation

1. Ensure you have the target dataset available:
   Place the Kaggle `creditcard.csv` inside the `data/` directory relative to the repository root.

   ```bash
   mkdir -p data
   # copy creditcard.csv into data/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Steps

To execute the pipeline dynamically, run the entry point script from the repository root:

```bash
python main.py
```

This single command automates:

1. Loading the target dataset.
2. Generating Raw Baseline metrics.
3. Conducting rigorous cleansing, mitigating systemic issues like class imbalance securely within the training boundaries.
4. Diverging into three specialized feature selection tracks (Filter, Wrapper, Embedded).
5. Logging exhaustive step telemetry, culminating into `results/metrics.csv` and `results/plots/`.

## Experimental Methodology

All experimental sequences utilize **Stratified Train-Test Splitting** to ensure target label distribution parity. Synthetic Minority Over-sampling Technique (SMOTE) is strictly confined to the training subsets to prevent data leakage or superficial inflation to test-set precision/recall boundaries. Baseline models (Logistic Regression, Random Forest) are tracked computationally (training time) and statically (resultant feature count).

## Reproducibility Requirements

- The deterministic footprint (global seeding, specific scikit-learn estimator states) relies entirely upon `src/config.py = RANDOM_SEED`.
- Hyperparameters remain unmodified dynamically inside modules, assuring full auditability of evaluation outcomes mapped inside `results/metrics.csv`.
