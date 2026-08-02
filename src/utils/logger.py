import logging
import sys
import os
from typing import Optional

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Create or get a logger.

    If `level` is not provided, the function will read the `LOG_LEVEL`
    environment variable (default WARNING) to control verbosity. Set
    LOG_LEVEL=INFO to re-enable info logging.
    """
    logger = logging.getLogger(name)

    if level is None:
        lvl_name = os.getenv("LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, lvl_name, logging.WARNING)

    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
