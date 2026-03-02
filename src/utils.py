import logging
import os
import sys

class FrameworkLogger:
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Console Handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            # File Handler
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            os.makedirs(results_dir, exist_ok=True)
            log_file = os.path.join(results_dir, "experiment.log")
            
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        return logger

logger = FrameworkLogger.get_logger(__name__)

def ensure_directories(dirs: list[str]):
    """Ensure that the given directories exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")

class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
