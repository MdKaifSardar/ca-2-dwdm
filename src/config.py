import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
    PLOTS_DIR: str = os.path.join(RESULTS_DIR, "plots")
    
    # Target and Features
    TARGET_COL: str = "Class"
    DROP_COLS: List[str] = field(default_factory=list)
    
    # Random Seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Train-test split
    TEST_SIZE: float = 0.2
    
    # Cross Validation
    CV_FOLDS: int = 5
    
    # Feature Selection Configs
    CORRELATION_THRESHOLD: float = 0.8
    RFE_STEP: int = 1
    RFE_N_FEATURES_TO_SELECT: int = 10
    L1_C: float = 0.1
    RF_IMPORTANCE_THRESHOLD: str = "mean"
    
config = Config()
