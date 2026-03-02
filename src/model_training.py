import time
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.utils import FrameworkLogger
from src.config import config

logger = FrameworkLogger.get_logger(__name__)

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, float]:
    """Train a Logistic Regression model and return the model and training time."""
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED, solver='lbfgs')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"Logistic Regression training completed in {training_time:.4f} seconds.")
    return model, training_time

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, float]:
    """Train a Random Forest model and return the model and training time."""
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    logger.info(f"Random Forest training completed in {training_time:.4f} seconds.")
    return model, training_time

def evaluate_cv(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate a model using Stratified K-Fold Cross Validation."""
    logger.info(f"Evaluating model using {config.CV_FOLDS}-fold Stratified CV...")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    metrics = {
        'cv_accuracy': cv_results['test_accuracy'].mean(),
        'cv_precision': cv_results['test_precision'].mean(),
        'cv_recall': cv_results['test_recall'].mean(),
        'cv_f1': cv_results['test_f1'].mean(),
        'cv_roc_auc': cv_results['test_roc_auc'].mean(),
    }
    
    logger.info(f"CV Evaluation Completed: F1={metrics['cv_f1']:.4f}, ROC_AUC={metrics['cv_roc_auc']:.4f}")
    return metrics
