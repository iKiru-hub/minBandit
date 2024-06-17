import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import os, logging, coloredlogs, json, pprint
import argparse
from numba import jit

CACHE_PATH = r"/Users/daniekru/Research/lab/minBandit/src/cache"


DEBUG = False
try:
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'
except:
    import warnings
    warnings.warn("Could not set font properties")

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
        files = []
        for f in os.listdir(CACHE_PATH):
            if f.endswith(".json"):
                files.append(f)
        files = {i: f for i, f in enumerate(files)}
        pprint.pprint(files)

        model_name = files[int(input("Enter the model number: "))]

    if model_name not in os.listdir(CACHE_PATH):
        raise ValueError(f"Model {model_name} not found in {CACHE_PATH}")

    with open(os.path.join(CACHE_PATH, model_name), "r") as f:
        model_params = json.load(f)

    logger(f"Model info: {model_params['info']}")

    return model_params["genome"]


def make_probability_set(K: int, nb_trials: int,
                         fixed_p: bool=False,
                         normalize: bool=False) -> np.ndarray:

    """
    make a set of probabilities for the bandit

    Parameters
    ----------
    K : int
        number of arms
    nb_trials : int
        number of trials
    fixed_p : bool
        fix the probabilities. Default=False
    normalize : bool
        normalize the probabilities.
        Default=False

    Returns
    -------
    probabilities_set : np.ndarray
        set of probabilities
    """


    probabilities_set = []
    for i in range(nb_trials):

        if fixed_p:
            p = np.around(np.random.uniform(0.05, 0.3, K), 2)
            p[i%K] = fixed_p
        else:
            p = np.around(np.random.uniform(0.05, 0.95, K), 2)

        if normalize:
            p = p / p.sum()

        probabilities_set += [p.tolist()]

    return np.array(probabilities_set)



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


def plot_lr_policy(params: dict):

    """
    plot the learning rate policy
    """

    X = np.arange(0, 10, 0.05)
    Y = gaussian_sigmoid(x=X, alpha=params["alpha_lr"], beta=params["beta_lr"],
                         mu=params["mu_lr"], sigma=params["sigma_lr"],
                         r=params["r_lr"]).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(X, Y)
    ax.set_title("Learning rate policy $\\tilde{\eta}$")
    ax.set_xlabel("weight value")
    ax.set_ylabel("learning rate")
    ax.grid()
    plt.show()


def plot_value_function(params: dict):

    """
    plot the value function
    """

    X = np.arange(0, 10, 0.05)
    Y = gaussian_sigmoid(x=X, alpha=params["alpha"], beta=params["beta"],
                         mu=params["mu"], sigma=params["sigma"],
                         r=params["r"]).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(X, Y)
    ax.set_title("Value function $V$")
    ax.set_xlabel("weight value")
    ax.set_ylabel("value")
    ax.grid()
    plt.show()


def plot_activation_function(params: dict):

    """
    plot the activation function
    """

    X = np.arange(0, 10, 0.05)
    Y = generalized_sigmoid(x=X, gain=params["gain"], threshold=params["threshold"]).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(X, Y)
    ax.set_title("Activation function $\\sigma$")
    ax.set_xlabel("x")
    ax.set_ylabel("activation")
    ax.grid()
    plt.show()



""" jit functions """



@jit(nopython=True)
def softmax(x: np.ndarray, beta: float=1.) -> np.ndarray:

    """
    Compute the softmax of an array

    Parameters
    ----------
    x: np.ndarray
        the array to compute the softmax of
    beta: float
        the inverse temperature of the softmax

    Returns
    -------
    np.ndarray
        the softmax of the array
    """

    return np.exp(beta * x) / np.sum(np.exp(beta * x))


@jit(nopython=True)
def max_normalize(x: np.ndarray) -> np.ndarray:

    """
    Normalize an array to the maximum value

    Parameters
    ----------
    x: np.ndarray
        the array to normalize

    Returns
    -------
    np.ndarray
        the normalized array
    """

    return x / np.max(x)


@jit#(no_python=True)
def gaussian_sigmoid(x: np.ndarray, alpha: float, beta: float,
                     mu: float, sigma: float, r: float) -> np.ndarray:

    return r / (1 + np.exp(-beta*(x - alpha))) + \
        (1 - r) * np.exp(- ((x - mu)**2) / sigma)


@jit(nopython=True)
def sigmoid(x: np.ndarray, alpha: float=0., beta: float=1.) -> np.ndarray:

    return 1 / (1 + np.exp(-beta*(x - alpha)))


@jit(nopython=True)
def mlp(x: float, y: np.ndarray, param1: float, param2: float,
        param3: float, param4: float, param5: float, param6: float,
        param7: float, param8: float) -> float:

    return param7 * sigmoid(
        param1 * x + param2 * y + param3,
    ) + param8 * sigmoid(
        param4 * x + param5 * y + param6
    )

@jit
def generalized_sigmoid(x: np.ndarray, gain: float, threshold: float) -> np.ndarray:

    return 1 / (1 + np.exp(-gain*(x - threshold)))





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="plotting functions")
    parser.add_argument("--policy", type=str, default=None,
                        help="plot one of the policies: lr, value, activation")
    args = parser.parse_args()


    # load model
    model_params = load_model()

    if args.policy == "lr":
        plot_lr_policy(model_params)
    elif args.policy == "value":
        plot_value_function(model_params)
    elif args.policy == "activation":
        plot_activation_function(model_params)
    else:
        raise ValueError("Invalid policy")


