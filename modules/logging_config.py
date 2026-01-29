import os
import sys

from loguru import logger


def setup_logging() -> None:
    logger.remove()
    log_level = os.getenv('CHATBOT_LOG_LEVEL')
    if log_level:
        logger.add(sys.stderr, level=log_level)
