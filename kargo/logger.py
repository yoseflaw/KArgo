import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s|%(levelname)s|%(module)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(log_format)
    logger.addHandler(stream)
    return logger