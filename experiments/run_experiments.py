import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List

from src.utils import FrameworkLogger, DataLoadError, ensure_directories
from src.config import config
from src.data_cleaning import (
    handle_missing_values, remove_duplicates, detect_outliers_iqr,
    scale_features, handle_class_imbalance_smote
)
from src.feature_selection import (
    filter_by_correlation, filter_by_mutual_info,
    wrapper_rfe, embedded_l1_logistic, embedded_random_forest, apply_feature_selection
)
from src.model_training import train_logistic_regression, train_random_forest
from src.evaluation import evaluate_model, save_metrics, plot_confusion_matrix

logger = FrameworkLogger.get_logger(__name__)

def load_data() -> pd.DataFrame:
    """Load the dataset from the specified path."""
    try:
        logger.info(f"Loading data from {config.DATA_PATH}")
        df = pd.read_csv(config.DATA_PATH)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise DataLoadError(f"Could not read {config.DATA_PATH}")

def run_evaluation(X_train, y_train, X_test, y_test, config_name: str, metrics_list: List[Dict[str, Any]]):
    """Train baseline models and record their metrics."""
    logger.info(f"--- Running evaluation for: {config_name} ---")
    
    # Logistic Regression
    lr_model, lr_time = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", config_name, lr_time, X_train.shape[1])
    metrics_list.append(lr_metrics)
    plot_confusion_matrix(lr_model, X_test, y_test, f"{config_name} - Logistic Regression")
    
    # Random Forest
    rf_model, rf_time = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", config_name, rf_time, X_train.shape[1])
    metrics_list.append(rf_metrics)
    plot_confusion_matrix(rf_model, X_test, y_test, f"{config_name} - Random Forest")

def main():
    ensure_directories([config.RESULTS_DIR, config.PLOTS_DIR])
    logger.info("Starting Research Experiment Pipeline")
    
    # 1. Load Data
    df = load_data()
    
    # Exclude strict target and drop columns
    features = [c for c in df.columns if c != config.TARGET_COL and c not in config.DROP_COLS]
    
    X = df[features]
    y = df[config.TARGET_COL]
    
    all_metrics = []
    
    # ==========================================
    # 1) Raw Data Baseline
    # ==========================================
    logger.info(">>> Running Baseline: Raw Data")
    
    # Models cannot handle NaN natively. For the raw baseline, we simply drop NaNs
    # instead of imputing, to show the raw performance (or lack thereof)
    df_raw = df.copy().dropna(subset=features + [config.TARGET_COL])
    X_raw = df_raw[features]
    y_raw = df_raw[config.TARGET_COL]
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=config.TEST_SIZE, stratify=y_raw, random_state=config.RANDOM_SEED
    )
    run_evaluation(X_train_raw, y_train_raw, X_test_raw, y_test_raw, "1_Raw", all_metrics)
    
    # ==========================================
    # 2) Data Cleansing Phase
    # ==========================================
    logger.info(">>> Applying Data Cleansing Phase")
    df_clean = df.copy()
    df_clean = handle_missing_values(df_clean, strategy='mean')
    df_clean = remove_duplicates(df_clean)
    # Using IQR for generic outlier detection
    df_clean = detect_outliers_iqr(df_clean, features)
    df_clean = scale_features(df_clean, features, method='standard')
    
    X_clean = df_clean[features]
    y_clean = df_clean[config.TARGET_COL]
    
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=config.TEST_SIZE, stratify=y_clean, random_state=config.RANDOM_SEED
    )
    
    # Handle Imbalance on TRAINING SET ONLY to prevent data leakage
    X_train_res, y_train_res = handle_class_imbalance_smote(X_train_clean, y_train_clean, random_state=config.RANDOM_SEED)
    
    run_evaluation(X_train_res, y_train_res, X_test_clean, y_test_clean, "2_Cleaned", all_metrics)
    
    # ==========================================
    # 3) Cleaned + Filter Feature Selection
    # ==========================================
    logger.info(">>> Applying Feature Selection: Filter (Mutual Info)")
    # Apply to training data to avoid leakage
    sel_features_filter = filter_by_mutual_info(X_train_res, y_train_res, k=15, random_state=config.RANDOM_SEED)
    X_train_filter = apply_feature_selection(X_train_res, sel_features_filter)
    X_test_filter = apply_feature_selection(X_test_clean, sel_features_filter)
    
    run_evaluation(X_train_filter, y_train_res, X_test_filter, y_test_clean, "3_Cleaned_Filter_MI", all_metrics)
    
    # ==========================================
    # 4) Cleaned + Wrapper Feature Selection
    # ==========================================
    logger.info(">>> Applying Feature Selection: Wrapper (RFE)")
    sel_features_wrapper = wrapper_rfe(X_train_res, y_train_res, n_features_to_select=config.RFE_N_FEATURES_TO_SELECT)
    X_train_wrapper = apply_feature_selection(X_train_res, sel_features_wrapper)
    X_test_wrapper = apply_feature_selection(X_test_clean, sel_features_wrapper)
    
    run_evaluation(X_train_wrapper, y_train_res, X_test_wrapper, y_test_clean, "4_Cleaned_Wrapper_RFE", all_metrics)
    
    # ==========================================
    # 5) Cleaned + Embedded Feature Selection
    # ==========================================
    logger.info(">>> Applying Feature Selection: Embedded (Random Forest)")
    sel_features_embedded = embedded_random_forest(X_train_res, y_train_res, threshold=config.RF_IMPORTANCE_THRESHOLD)
    X_train_embedded = apply_feature_selection(X_train_res, sel_features_embedded)
    X_test_embedded = apply_feature_selection(X_test_clean, sel_features_embedded)
    
    run_evaluation(X_train_embedded, y_train_res, X_test_embedded, y_test_clean, "5_Cleaned_Embedded_RF", all_metrics)
    
    # Save Final Metrics
    logger.info(">>> Saving all evaluation metrics")
    save_metrics(all_metrics)
    
    # Identify Best Configuration based on ROC-AUC
    best_config = max(all_metrics, key=lambda x: x['ROC_AUC'])
    logger.info(f"*** Best Configuration Identified: {best_config['Configuration']} with {best_config['Model']} ***")
    logger.info(f"*** Best ROC-AUC: {best_config['ROC_AUC']:.4f} ***")

if __name__ == "__main__":
    main()
