import logging
import sys
from pythonjsonlogger import jsonlogger


def setup_json_logger(level: int = logging.INFO):
    """Setup system log in JSON format to avoid log misinterpretation"""
    logger = logging.getLogger()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    # Rename to "level" and "timestamp" to meet standard logging format
    formatter = jsonlogger.JsonFormatter(
        "%(levelname)s %(name)s %(module)s %(message)s %(asctime)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
        json_ensure_ascii=False,
    )
    handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    # Set Uvicorn (FastAPI's server) to use this JSON format
    # So that HTTP request logs will also be in JSON
    UVICORN_LOGGERS = ["uvicorn", "uvicorn.access", "uvicorn.error"]
    for logger_name in UVICORN_LOGGERS:
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(handler)

    return logger
