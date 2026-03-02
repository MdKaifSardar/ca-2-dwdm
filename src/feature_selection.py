import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List

from src.utils import FrameworkLogger

logger = FrameworkLogger.get_logger(__name__)

# --- FILTER METHODS ---

def filter_by_correlation(X: pd.DataFrame, threshold: float = 0.8) -> List[str]:
    """Filter features by correlation threshold. Removes one of the highly correlated pairs."""
    logger.info(f"Applying Correlation Filter with threshold: {threshold}")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = [col for col in X.columns if col not in to_drop]
    
    logger.info(f"Correlation Filter selected {len(selected_features)} out of {X.shape[1]} features.")
    return selected_features

def filter_by_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 15, random_state: int = 42) -> List[str]:
    """Filter features by Mutual Information. Selects top k features."""
    logger.info(f"Applying Mutual Information Filter (top {k} features)")
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    selected_features = mi_series.nlargest(k).index.tolist()
    logger.info(f"Mutual Info Filter selected {len(selected_features)} features.")
    return selected_features


# --- WRAPPER METHODS ---

def wrapper_rfe(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 15, step: int = 1, random_state: int = 42) -> List[str]:
    """Recursive Feature Elimination (RFE) using Logistic Regression."""
    logger.info(f"Applying Recursive Feature Elimination (RFE) targeting {n_features_to_select} features.")
    estimator = LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear')
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.support_].tolist()
    logger.info(f"RFE selected {len(selected_features)} features.")
    return selected_features


# --- EMBEDDED METHODS ---

def embedded_l1_logistic(X: pd.DataFrame, y: pd.Series, C: float = 0.1, random_state: int = 42) -> List[str]:
    """Embedded feature selection using L1 regularized Logistic Regression."""
    logger.info(f"Applying L1 Logistic Regression feature selection with C={C}")
    estimator = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=random_state, max_iter=1000)
    selector = SelectFromModel(estimator)
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    logger.info(f"L1 Logistic selected {len(selected_features)} features.")
    return selected_features

def embedded_random_forest(X: pd.DataFrame, y: pd.Series, threshold: str = 'mean', random_state: int = 42) -> List[str]:
    """Embedded feature selection using Random Forest feature importances."""
    logger.info(f"Applying Random Forest Embedded feature selection with threshold={threshold}")
    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    selector = SelectFromModel(estimator, threshold=threshold)
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    logger.info(f"Random Forest Embedded selected {len(selected_features)} features.")
    return selected_features

def apply_feature_selection(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Helper method to slice the dataframe based on selected features."""
    return X[selected_features].copy()
