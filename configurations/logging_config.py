"""
This module provides a utility function for configuring a centralized logger for a project.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
import colorlog

from configurations import config

# ===== Custom Title log level =====
TITLE_LEVEL_NUM = 25
logging.addLevelName(TITLE_LEVEL_NUM, "TITLE")


def title(self, message, *args, **kwargs):
    """
    Log a message at a custom TITLE level.
    Args:
        message: The message string to log.
        *args: Additional arguments to use for string formatting.
        **kwargs: Additional keyword arguments to be passed to the logging handler.
    """
    if self.isEnabledFor(TITLE_LEVEL_NUM):
        self._log(TITLE_LEVEL_NUM, message, args, **kwargs)


logging.Logger.title = title

# ===== Formats =====
LOG_FORMAT = "%(asctime)s %(filename)s:%(lineno)d - %(levelname)-8s - %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "red",
    "ERROR": "bold_red",
    "CRITICAL": "bold_red,bg_white",
    "TITLE": "blue",
}

IS_LOGGER_CONFIGURED = False


def configure_logger() -> logging.Logger:
    """
    Configure and returns a centralized project logger with both console and file handlers.

    Ensures the logger is only configured once per application run by using a global flag.
    It sets up a file handler that writes logs to 'project_log' with a log rotation policy to
    manage file size. It also sets up a console handler that outputs logs with color formatting
    for better readability in the terminal. The logger is set to the INFO level for both handlers,
    meaning all messages of severity INFO and higher will be captured.

    Note: The global ``IS_LOGGER_CONFIGURED`` flag is not thread-safe. This is acceptable for
    the current single-process pipeline and Uvicorn single-worker usage, but should be replaced
    with ``logging.config`` or a threading lock if multi-threaded configuration is needed.

    Returns:
        logging.Logger instance
    """
    global IS_LOGGER_CONFIGURED

    logger = logging.getLogger("project_logger")

    if IS_LOGGER_CONFIGURED:
        return logger
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    filepath = os.path.join(config.REPORTS_DIR, "project.log")

    file_handler = RotatingFileHandler(
        filepath, maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s" + LOG_FORMAT + "%(reset)s",
        datefmt=DATE_FORMAT,
        log_colors=LOG_COLORS,
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    IS_LOGGER_CONFIGURED = True
    return logger
