import traceback
import sys
from experiments.run_experiments import main as run_pipeline
from src.utils import FrameworkLogger

logger = FrameworkLogger.get_logger("main")

def main():
    try:
        logger.info("Starting Experimental Pipeline...")
        run_pipeline()
        logger.info("Experimental Pipeline Completed Successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
