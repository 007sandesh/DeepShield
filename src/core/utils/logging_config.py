"""
Logging configuration for DeepShield.

Structured logging with loguru + optional JSON output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = "logs/deepshield.log",
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format (json, text)
        log_file: Optional log file path
        rotation: Log rotation size/time
        retention: How long to keep logs
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if log_format == "json":
        logger.add(
            sys.stderr,
            level=level,
            format="{message}",
            serialize=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "{message}"
            ),
            colorize=True,
        )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=True if log_format == "json" else False,
        )

    logger.info("Logging initialized", level=level, format=log_format)
