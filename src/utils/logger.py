# src/utils/logger.py
import logging
import sys

def get_logger(name: str = "FraudDetection") -> logging.Logger:
    """Configures structured logger for production observability."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()
