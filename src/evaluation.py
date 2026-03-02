import os
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import FrameworkLogger
from src.config import config

logger = FrameworkLogger.get_logger(__name__)

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, config_name: str, training_time: float, num_features: int) -> Dict[str, Any]:
    """Compute varied metrics for the model on the test set."""
    logger.info(f"Evaluating {model_name} on configuration: {config_name}")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        'Configuration': config_name,
        'Model': model_name,
        'Num_Features': num_features,
        'Training_Time_sec': training_time,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Optional: Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix for {model_name} ({config_name}):\n{cm}")
    
    return metrics

def save_metrics(metrics_list: List[Dict[str, Any]]):
    """Save the aggregated metrics to a CSV file."""
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.csv")
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"Saved experiment metrics to {metrics_path}")

def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series, title_suffix: str):
    """Generate and save a confusion matrix plot."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    clean_suffix = title_suffix.replace(' ', '_').replace('+', 'and')
    plot_path = os.path.join(config.PLOTS_DIR, f"cm_{clean_suffix}.png")
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {plot_path}")
