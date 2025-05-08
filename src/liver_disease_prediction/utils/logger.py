"""Logging configuration for the package."""
import logging
import sys
from typing import Union


def setup_logging(level: Union[int, str] = logging.INFO) -> None:
    """Set up logging configuration.

    Args:
        level: The logging level to use. Defaults to INFO.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler) 