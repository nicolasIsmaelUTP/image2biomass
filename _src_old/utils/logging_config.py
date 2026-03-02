"""Basic logging configuration for the src package.

Provides a single entry point to configure the root logger once and
retrieve child loggers across the codebase.
"""

import logging
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_LEVEL = logging.INFO


def configure_logging(level: int = DEFAULT_LEVEL) -> None:
    """Configure the root logger with a stream handler if not already set."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger, ensuring the root logger is configured first."""

    configure_logging()
    return logging.getLogger(name)
