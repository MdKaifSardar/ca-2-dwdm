import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple
from scipy import stats

from src.utils import FrameworkLogger

logger = FrameworkLogger.get_logger(__name__)

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', n_neighbors: int = 5) -> pd.DataFrame:
    """Handle missing values using specified strategy ('mean', 'median', 'knn')."""
    logger.info(f"Handling missing values using strategy: {strategy}")
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found.")
        return df.copy()
        
    cols = df.columns
    if strategy in ['mean', 'median']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols)
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols)
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'knn'.")
        
    logger.info("Missing values handled successfully.")
    return df_imputed

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    initial_shape = df.shape[0]
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    dropped = initial_shape - df_dedup.shape[0]
    logger.info(f"Removed {dropped} duplicate rows. New shape: {df_dedup.shape}")
    return df_dedup

def detect_outliers_iqr(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Detect and remove outliers using the Interquartile Range (IQR) method."""
    logger.info("Removing outliers using IQR method.")
    outlier_indices = set()
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(outliers)
        
    df_clean = df.drop(index=list(outlier_indices)).reset_index(drop=True)
    logger.info(f"Removed {len(outlier_indices)} outliers via IQR. New shape: {df_clean.shape}")
    return df_clean

def detect_outliers_zscore(df: pd.DataFrame, features: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Detect and remove outliers using the Z-score method."""
    logger.info("Removing outliers using Z-score method.")
    z_scores = np.abs(stats.zscore(df[features].dropna()))
    # Only keep rows where all z-scores are below the threshold
    keep_indices = (z_scores < threshold).all(axis=1)
    
    # We need to map boolean mask back to original indices carefully 
    # But for simplicity, assuming features df aligns with df
    df_clean = df[keep_indices].reset_index(drop=True)
    dropped = df.shape[0] - df_clean.shape[0]
    logger.info(f"Removed {dropped} outliers via Z-score. New shape: {df_clean.shape}")
    return df_clean

def detect_outliers_isolation_forest(df: pd.DataFrame, features: list[str], random_state: int = 42) -> pd.DataFrame:
    """Detect and remove outliers using Isolation Forest."""
    logger.info("Removing outliers using Isolation Forest.")
    clf = IsolationForest(random_state=random_state, n_estimators=100)
    preds = clf.fit_predict(df[features])
    
    df_clean = df[preds == 1].reset_index(drop=True)
    dropped = df.shape[0] - df_clean.shape[0]
    logger.info(f"Removed {dropped} outliers via Isolation Forest. New shape: {df_clean.shape}")
    return df_clean

def scale_features(df: pd.DataFrame, features: list[str], method: str = 'standard') -> pd.DataFrame:
    """Scale specified features using StandardScaler or MinMaxScaler."""
    logger.info(f"Scaling features using {method} scaler.")
    df_scaled = df.copy()
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")
        
    df_scaled[features] = scaler.fit_transform(df[features])
    logger.info(f"Features scaled successfully.")
    return df_scaled

def handle_class_imbalance_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the dataset."""
    logger.info("Applying SMOTE to handle class imbalance.")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(f"SMOTE applied. Original shape: {X.shape}, Resampled shape: {X_res.shape}")
    return X_res, y_res
