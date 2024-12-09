import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import os, logging, coloredlogs, json, pprint
from tqdm import tqdm
import argparse
from numba import jit

CACHE_PATH = r"/Users/daniekru/Research/lab/minBandit/src/_evo_cache"
CACHE_PATH_2 = r"/home/daniekru/lab/minBandit/src/_evo_cache"
MEDIA_PATH = r"/Users/daniekru/Research/lab/minBandit/media"
MEDIA_PATH_2 = r"/home/daniekru/lab/minBandit/media"


DEBUG = False
try:
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'
except:
    import warnings
    warnings.warn("Could not set font properties")


""" LOGGER """


def setup_logger(name: str="MAIN",
                 colored: bool=True,
                 level: int=0,
                 is_debugging: bool=True,
                 is_warning: bool=True) -> logging.Logger:

    """
    this function sets up a logger

    Parameters
    ----------
    name : str
        name of the logger. Default="MAIN"
    colored : bool
        use colored logs. Default=True
    level : int
        the level that is currently used.
        Default=0
    is_debugging : bool
        use debugging mode. Default=True
    is_warning : bool
        use warning mode. Default=True

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
        def __init__(self, logger,
                     level: int,
                     is_debugging: bool=False,
                     is_warning: bool=False):
            self.logger = logger
            self.level = level
            self.is_debugging = is_debugging
            self.is_warning = is_warning

            # self.logger.info(self)

        def __repr__(self):

            return f"LoggerWrapper(name={self.logger.name}," + \
                   f"level={self.level}, " + \
                   f"debugging={self.is_debugging}, " + \
                   f"warning={self.is_warning})"

        def __call__(self, msg: str="", level: int=1):
            if level <= self.level:
                self.logger.info(msg)

        def info(self, msg: str="", level: int=1):
            self(msg, level)

        def warning(self, msg: str=""):
            if self.is_warning:
                self.logger.warning(msg)

        def error(self, msg: str=""):
            if self.is_warning:
                self.logger.error(msg)

        def debug(self, msg):
            if self.is_debugging:
                self.logger.debug(msg)

        def set_debugging(self, is_debugging: bool):
            self.is_debugging = is_debugging

        def set_warning(self, is_warning: bool):
            self.is_warning = is_warning

        def set_level(self, level: int):
            self.level = level

    return LoggerWrapper(logger=logger, level=level,
                         is_debugging=is_debugging,
                         is_warning=is_warning)


logger = setup_logger(name="UTILS", colored=True,
                      level=0, is_debugging=False,
                      is_warning=False)


def edit_logger(level: int=-1,
                is_debugging: bool=True,
                is_warning: bool=False):
    global logger
    logger.set_level(level)
    logger.set_debugging(is_debugging)
    logger.set_warning(is_warning)


def tqdm_enumerate(iter, **tqdm_kwargs):
    i = 0
    for y in tqdm(iter, **tqdm_kwargs):
        yield i, y
        i += 1


def load_model(model_name: str=None, idx: int=None,
               verbose: bool=True, CACHE_PATH=CACHE_PATH):

    """
    load a model from the models folder

    Parameters
    ----------
    model_name : str
        name of the model
    idx : int
        index of the model. Default=None

    Returns
    -------
    model : object
        model object
    """

    # check path
    if not os.path.exists(CACHE_PATH):
        CACHE_PATH = CACHE_PATH_2

    if model_name is None:
        files = []
        for f in os.listdir(CACHE_PATH):
            if f.endswith(".json"):
                files.append(f)
        files = {i: f for i, f in enumerate(files)}

        if idx is None:
            pprint.pprint(files)
            idx = int(input("Enter the model number: "))

        model_name = files[idx]

    if model_name not in os.listdir(CACHE_PATH):
        raise ValueError(f"Model {model_name} not found" + \
            f" in {CACHE_PATH}")

    with open(os.path.join(CACHE_PATH, model_name),
              "r") as f:
        model_params = json.load(f)

    if verbose:
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
    idx_set = np.random.choice(list(range(K)), nb_trials,
                               replace=False)
    for i in range(nb_trials):

        if fixed_p:
            p = np.around(np.random.uniform(0.05, 0.3, K), 2)
            # p[i%K] = fixed_p
            p[idx_set[i%K]] = 0.9
            logger.warning(f"# Fixed probability: {p}")
        else:
            p = np.around(np.random.uniform(0.05, 0.95, K), 2)

        if normalize:
            p = p / p.sum()

        probabilities_set += [p.tolist()]

    return np.array(probabilities_set)


def calc_entropy(z: np.ndarray) -> float:

    """
    calculate the entropy of a distribution

    Parameters
    ----------
    z : np.ndarray
        the distribution

    Returns
    -------
    entropy : float
        the entropy of the distribution
    """

    # make probabilities
    z = z / z.sum()
    p = z[z > 0]
    return -np.sum(p * np.log(p))



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


def plot_multiple_reward(stats: dict, window: int=1,
                         ax: plt.Axes=None, title: str="",
                         show: bool=False):

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

    upper = np.tile(np.array(upper).reshape(2, 1), (1, nb_rounds))
    chance = np.tile(np.array(chance).reshape(2, 1), (1, nb_rounds))

    #
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    else:
        fig = ax.get_figure()

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
    ax.set_title(f"Reward{title}")
    # ax.grid()
    ax.set_xlabel("rounds for all trials")
    ax.set_ylim(-0.1, 1.3)
    ax.set_xlim(0, max(x))

    if show:
        plt.show()

    return fig


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


def plot_online_3d(ax: plt.Axes, t: int, U: np.ndarray,
                   V: np.ndarray, u: np.ndarray, v: np.ndarray,
                   title: str=""):

        ax.clear()
        # plot U as a black line in 3D
        ax.plot(U[:t, 0], U[:t, 1], U[:t, 2], 'k-',
                alpha=0.5)
        # plot V as a red line in 3D
        ax.plot(V[:t, 0], V[:t, 1], V[:t, 2], 'r-',
                alpha=0.5)
        # plot the current point in 3D
        ax.plot([u[0]], [u[1]], [u[2]],
                'ko', label="u")
        ax.plot([v[0]], [v[1]], [v[2]],
                'ro', label="v")
        ax.set_title(title)

        ax.set_xlim(0., 2.)
        ax.set_ylim(0., 2.)
        ax.set_zlim(0., 2.)

        ax.set_xticks([0., 2.])
        ax.set_yticks([0., 2.])
        ax.set_zticks([0., 2.])

        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_zlabel("C")

        ax.grid(False)

        ax.legend()


def plot_online_2d(ax: plt.Axes, t: int, U: np.ndarray,
                   V: np.ndarray, u: np.ndarray, v: np.ndarray,
                   title: str=""):

    ax.clear()
    ax.plot(U[:t, 0], U[:t, 1], 'k-', alpha=0.5)
    ax.plot(V[:t, 0], V[:t, 1], 'r-', alpha=0.5)
    ax.plot([u[0]], [u[1]], 'ko', label="u")
    ax.plot([v[0]], [v[1]], 'ro', label="v")
    ax.set_title(title)
    ax.set_xlim(0., 2.)
    ax.set_ylim(0., 2.)
    ax.set_xticks([0., 2.])
    ax.set_yticks([0., 2.])
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.grid(False)
    ax.legend()


def plot_online_tape(ax: plt.Axes, x: np.ndarray,
                     shape: tuple=(1, -1),
                     title: str="",
                     lsty: str="o-"):

    ax.clear()
    # ax.imshow(x.reshape(shape), cmap="Greys",
    #           vmin=0., vmax=2., aspect="auto")
    ax.plot(x.flatten(), lsty, color="black", lw=2,
            alpha=0.8 if lsty == "o-" else 0.3)
    ax.set_xlim(0, x.size-1)
    ax.set_ylim(0, 2.1)
    ax.set_aspect("auto")
    ax.set_title(title)
    ax.set_yticks([])
    if len(x) < 10:
        ax.set_xticks(range(x.size))
        ax.set_xticklabels(np.arange(1, x.size+1).astype(int))
    else:
        ax.set_xticks(np.linspace(0, x.size, 10).astype(int))
        ax.set_xticklabels(np.linspace(1, x.size, 10).astype(int))
    # ax.set_axis_off()


def plot_online_choices(ax: plt.Axes, K: int, choices: list, 
                        title: str=""):

    ax.clear()
    # ax.plot(choices, 'ko', markersize=5)

    length = min((100, len(choices)))
    z = np.zeros((length, K))
    z[np.arange(length), choices[-length:]] = 1

    ax.set_title(title)
    ax.set_ylim(-1, K)
    ax.set_ylabel(f"{K} arms")
    ax.grid(True)
    # if K < 10:
    ax.set_yticks(range(0, K))
    ax.set_yticklabels(np.arange(1, K+1).astype(str))
    # else:
    #     ax.set_yticks(np.linspace(0, K, 10).astype(int))
    #     ax.set_yticklabels(np.linspace(1, K+1, 10).astype(int).astype(str))

    ax.set_xticks([])
    ax.set_xlabel("rounds")

    ax.imshow(z.T, cmap="Greys", alpha=0.8,
              aspect="auto", interpolation="nearest")
    # ax.set_axis_off()





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


@jit#(no_python=True)
def neural_response_func(x: np.ndarray, gain: float, offset: float,
                         threshold: float) -> np.ndarray:

    z = 1 / (1 + np.exp(-gain*(x - offset)))
    return z * (z > threshold)



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




def render_func():

    x = np.arange(-10, 10, 0.01)
    y = gaussian_sigmoid(x, alpha=1, beta=10,
                         mu=1, sigma=1, r=0.5)
 

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.suptitle("Gaussian Sigmoid function",
                 fontsize=19)
    ax.plot(x, y)
    ax.set_xlabel("$x$", fontsize=15)
    ax.set_ylabel("$y$", fontsize=15)
    ax.grid()
    plt.show()

    # fig.savefig("../paper/figures/gaussian_sigmoid.png",
    #             dpi=500, bbox_inches="tight")
    print("saved")



if __name__ == "__main__":

    render_func()

    # parser = argparse.ArgumentParser(description="plotting functions")
    # parser.add_argument("--policy", type=str, default=None,
    #                     help="plot one of the policies: lr, value, activation")
    # args = parser.parse_args()


    # # load model
    # model_params = load_model()

    # if args.policy == "lr":
    #     plot_lr_policy(model_params)
    # elif args.policy == "value":
    #     plot_value_function(model_params)
    # elif args.policy == "activation":
    #     plot_activation_function(model_params)
    # else:
    #     raise ValueError("Invalid policy")


