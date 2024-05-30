import numpy as np
import matplotlib.pyplot as plt
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


""" visualization """


def plot_multiple(stats: dict):

    """
    plot all trials with for all models
    """

    rewards = stats["reward_list"]
    names = stats["names"]
    scores = stats["scores"]
    upper = stats["upper_bound_list"].mean()
    N, nb_trials, nb_rounds = rewards.shape 

    x = range(nb_trials * nb_rounds)
    chance_x = np.tile(stats["chance_list"], (nb_rounds, 1)).T.flatten()
    upper_bound_x = np.tile(stats["upper_bound_list"], (nb_rounds, 1)).T.flatten()

    # plt.plot(x, np.cumsum(chance_x), 'k--', label="chance", alpha=0.5)
    # plt.plot(x, np.cumsum(upper_bound_x), 'k-', label="upper", alpha=0.5)

    window = 500 
    # y = np.cumsum(upper_bound_x) - np.cumsum(chance_x)
    # y = np.convolve(y, np.ones(window), 'valid') / window
    # y = np.concatenate((np.array([0]*(len(x)-len(y))), y))
    # plt.plot(x, y, 'k--', label=f"chance", alpha=0.3)
    plt.axhline(0, color="black", alpha=0.5, label=f"upper bound [{upper:.2f}]")

    for i in range(N):
        # y = np.cumsum(upper_bound_x) - np.cumsum(rewards[i].flatten())
        y = upper_bound_x - rewards[i].flatten()
        y = np.convolve(y, np.ones(window), 'valid') / window
        y = np.concatenate((np.array([0]*(len(x)-len(y))), y))
        # y = np.cumsum(rewards[i].flatten())
        plt.plot(x, y, label=f"{names[i]} [{scores[i]:.2f}]",
                 alpha=0.4 if (i+1) != N else 0.9)

    plt.legend()
    plt.title(f"error wrt the upper bound")
    plt.grid()
    plt.xlabel("rounds for all trials")
    plt.ylabel("error")
    plt.show()


if __name__ == "__main__":

    x = np.array([2, 4])
    y = np.tile(x, (3, 1)).T.flatten()
    print(x)
    print(y)
