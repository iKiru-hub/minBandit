import os, logging, coloredlogs, json, pprint



DEBUG = False


# logger
def setup_logger(name: str="MAIN", colored: bool=True) -> logging.Logger:

    """
    this function sets up a logger

    Parameters
    ----------
    name : str
        name of the logger. Default="MAIN"
    colored : bool
        use colored logs. Default=True

    Returns
    -------
    logger : object
        logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create a custom formatter
    if colored:
        formatter = coloredlogs.ColoredFormatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # create a colored stream handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # add the handler to the logger and disable propagation
        logger.addHandler(handler)

    logger.propagate = False

    # wrapper class 
    class LoggerWrapper:
        def __init__(self, logger):
            self.logger = logger

        def __repr__(self):

            return f"LoggerWrapper(name={self.logger.name})"

        def __call__(self, msg: str=""):
            self.logger.info(msg)

        def info(self, msg):
            self.logger.info(msg)

        def warning(self, msg):
            self.logger.warning(msg)

        def error(self, msg):
            self.logger.error(msg)

        def debug(self, msg, DEBUG: bool=True):
            if DEBUG:
                self.logger.debug(msg)

    return LoggerWrapper(logger)

logger = setup_logger(__name__)




def tqdm_enumerate(iter, **tqdm_kwargs):
    i = 0
    for y in tqdm(iter, **tqdm_kwargs):
        yield i, y
        i += 1


