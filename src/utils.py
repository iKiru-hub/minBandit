import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import os, logging, coloredlogs, json, pprint

CACHE_PATH = r"/Users/daniekru/Research/lab/minBandit/src/cache"


DEBUG = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

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


def load_model(model_name: str=None):

    """
    load a model from the models folder

    Parameters
    ----------
    model_name : str
        name of the model

    Returns
    -------
    model : object
        model object
    """

    if model_name is None:
        files = {i: f for i, f in enumerate(os.listdir(CACHE_PATH))}
        pprint.pprint(files)

        model_name = files[int(input("Enter the model number: "))]

    if model_name not in os.listdir(CACHE_PATH):
        raise ValueError(f"Model {model_name} not found in {CACHE_PATH}")

    with open(os.path.join(CACHE_PATH, model_name), "r") as f:
        model_params = json.load(f)["genome"]

    return model_params



""" visualization """


def plot_multiple_regret(stats: dict, window: int=1, ax: plt.Axes=None):

    """
    plot all trials with for all models
    """

    rewards = stats["reward_list"]
    names = stats["names"]
    scores = stats["scores"]
    best_arms = stats["best_arm_list"]
    arms = stats["arm_list"]
    N, nb_reps, nb_trials, nb_rounds = rewards.shape

    x = range(nb_trials * nb_rounds)

    #
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))

    ax.axhline(0, color="black", alpha=0.3)
    for i in range(0, nb_trials):
        ax.axvline(nb_rounds * i, color="black", linestyle="--", alpha=0.3)
        ax.text(x=nb_rounds*i+5, y=-0.03, s=f"trial {i}")

    for i in range(N):

        y = ((arms - best_arms)**2).astype(int).sum(axis=3)/2
        y = y.mean(axis=0).flatten()
        yvar = y.var(axis=0).flatten()

        if window is not None:
            y = np.convolve(y, np.ones(window), 'valid') / window
            y = np.concatenate((np.array([0]*(len(x)-len(y))), y))

        # add variance
        ax.fill_between(x, y-yvar, y+yvar, alpha=0.1)
        ax.plot(x, y, label=f"{names[i]} [$R$={scores[i]:.3f}]",
                 alpha=0.4 if (i+1) != N else 0.9)

    ax.legend(loc="upper right")
    ax.set_title(f"Regret")
    # ax.grid()
    ax.set_xlabel("rounds for all trials")
    ax.set_ylim(-0.1, 1.3)

    plt.show()


def plot_multiple_reward(stats: dict, window: int=1, ax: plt.Axes=None):

    """
    plot all trials with for all models
    """

    rewards = stats["reward_list"]
    names = stats["names"]
    scores = stats["scores"]
    chance = stats["chance_list"]
    upper = stats["upper_bound_list"]
    N, nb_reps, nb_trials, nb_rounds = rewards.shape

    x = range(nb_trials * nb_rounds)

    #
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))

    # trial separator
    ax.axhline(0, color="black", alpha=0.3)
    if nb_trials < 10:
        for i in range(0, nb_trials):
            ax.axvline(nb_rounds * i, color="black", linestyle="--", alpha=0.3)
            ax.text(x=nb_rounds*i+5, y=-0.03, s=f"trial {i}")

    # bandit statistics
    ax.plot(x, upper.flatten(), 'k-', alpha=0.3)
    ax.plot(x, chance.flatten(), 'k-', alpha=0.3)

    ax.fill_between(x, upper.flatten(), chance.flatten(), alpha=0.15, color="grey", label="chance-upper")

    for i in range(N):

        y = rewards[i].mean(axis=0).flatten()
        yvar = y.var(axis=0).flatten()

        if window is not None:
            y = np.convolve(y, np.ones(window), 'valid') / window
            y = np.concatenate((np.array([0]*(len(x)-len(y))), y))

        # add variance
        ax.fill_between(x, y-yvar, y+yvar, alpha=0.1)
        ax.plot(x, y, label=f"{names[i]} [$R$={scores[i]:.3f}]",
                 alpha=0.4 if (i+1) != N else 0.9)

    ax.legend(loc="upper right")
    ax.set_title(f"Reward")
    # ax.grid()
    ax.set_xlabel("rounds for all trials")
    ax.set_ylim(-0.1, 1.3)
    ax.set_xlim(0, max(x))

    plt.show()







if __name__ == "__main__":

    x = np.array([2, 4])
    y = np.tile(x, (3, 1)).T.flatten()
    print(x)
    print(y)
