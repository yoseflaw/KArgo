import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING


def get_logger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_format = logging.Formatter("%(asctime)s|%(levelname)s|%(module)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(log_format)
    logger.addHandler(stream)
    return logger
