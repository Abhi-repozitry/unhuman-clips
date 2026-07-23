"""Structured logging configuration for the Unhuman Clips backend.

Provides a single setup_logging() call that configures Python's logging module
with consistent formatting, timestamps, and log levels across all modules.
"""
from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging"]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured formatting.

    Sets up a StreamHandler with timestamps, module names, and log levels.
    Called once at application startup.

    Args:
        level: Root logger level (default: INFO).
    """
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called multiple times
    if not root.handlers:
        root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
