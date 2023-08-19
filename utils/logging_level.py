import logging

def set_log_level(log_level: str) -> None:
    """sets logging level

    Args:
        log_level (str): is either FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL, and OFF
    """
    levels = {"FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG}
    if log_level not in levels.keys():
        raise ValueError(f"{log_level=} has to be in {levels=}")
    logging.basicConfig(level=levels[log_level])
