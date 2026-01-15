"""
Test the sensitivity of the `epsilon` paramenter of the epsilon-greedy algorithm in different environments
"""

import numpy as np
import matplotlib.pyplot as plt
import time, argparse, json
from tqdm import tqdm

import sys, os
sys.path.append(os.getcwd().split("src")[0] + "src/core")

import envs
import models as mm
import utils

logger = utils.setup_logger(name=__name__, level=2)


def run(epsilon: float, K: int, nb_rounds: int,  nb_trials: int,
        env_type: str, fixed_p: float=0.6, verbose: bool=True) -> dict:

    # define proababilities set
    probabilities_set = utils.make_probability_set(K=K,
                                                   nb_trials=nb_trials,
                                                   fixed_p=fixed_p,
                                                   normalize=False)
    # define the environment
    if env_type == "driftv0":
        env = envs.KABdriftv0(K=K,
                              probabilities_set=probabilities_set,
                              verbose=verbose,
                              tau=10)
    elif env_type == "driftv1":
        env = envs.KABdriftv1(K=K,
                              verbose=verbose,
                              tau=100,
                              normalize=True,
                              fixed_p=0.9)
    elif env_type == "sinv0":
        frequencies = np.linspace(0.1, 0.4, K)
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            normalize=True,
                            verbose=verbose)
    else:
        env = envs.KABv0(K=K,
                         probabilities_set=probabilities_set,
                         verbose=verbose)

    # define the model
    model = mm.EpsilonGreedy(K=K, epsilon=epsilon)

    # run
    results = envs.trial(model=model,
                         environment=env,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                         verbose=verbose,
                         disable=True)

    return results


def redact_score(results):
    return (results['score'] - results['chance']) / (results['upper_bound'] - results['chance'])


def get_sgnf_d(x: float):
    return len(str(x).split('.')[1].split('0'))



plt.style.use('seaborn-v0_8-white')  # Requires matplotlib-style package

plt.rcParams.update({
    'figure.figsize': (10, 4),
    #'figure.dpi': 400,
    'figure.subplot.wspace': 0.1,
    'figure.subplot.hspace': 0.3,

    'text.usetex': True,

    'font.size': 23,
    'font.family': 'serif',
    'font.weight': 'normal',
    'font.serif': ['Computer Modern Roman'],

    'axes.labelsize': 23,
    'axes.titlesize': 23,
    'axes.labelweight': 'normal',
    'axes.spines.top': True,
    'axes.spines.right': True,

    'lines.linewidth': 1.0,
    'lines.markeredgewidth': 0.5,
    'lines.markeredgecolor': 'black',

    'xtick.labelsize': 23,
    'ytick.labelsize': 23,

})



if __name__ == "__main__":

    np.random.seed(3)

    # ---
    num_k = 5
    nb_rounds = 1500
    nb_trials = 1
    nb_reps = 100
    env_type = "default"
    fixed_p = 0.9

    # ---
    Ks = np.logspace(2, 10, num=num_k, base=2)
    Ks = np.flip(Ks, axis=0)
    logger(f"{Ks=}")
    Ks = Ks.astype(int)
    logger(f"(int) {Ks=}")

    #Eps = np.around(np.linspace(0.01, 0.9, 3), 2)
    Eps = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    num_epsilon = len(Eps)

    table = np.zeros((num_k, num_epsilon, 2))

    # ---
    for i in range(num_k):
        logger(f"-->> K={Ks[i]}")
        pbar = tqdm(range(num_epsilon))
        for j in pbar:
            pbar.set_description(f"epsilon={Eps[j]}")
            results = np.zeros(nb_reps)
            for l in range(nb_reps):
                result = run(epsilon=Eps[j], K=Ks[i],
                             nb_rounds=nb_rounds, nb_trials=nb_trials,
                              env_type=env_type, fixed_p=fixed_p, verbose=False)
                # table[i, j] += redact_score(results)
                results[l] = redact_score(result)

            # update tables
            table[i, j, 0] = results.mean()
            table[i, j, 1] = results.std()

    # logger(f"{results['upper_bound']=}")

    # ---
    fig, ax = plt.subplots()
    im = ax.imshow(table[:, :, 0], cmap="viridis")

    ax.set_xticks(range(num_epsilon))
    ax.set_xticklabels(Eps)
    ax.set_xlabel("$\\epsilon$")

    ax.set_yticks(range(num_k))
    ax.set_yticklabels(Ks)
    ax.set_ylabel("$K$")

    for i in range(len(table)):
        for j in range(len(table[i])):
            _text = f"{table[i, j, 0]:.2f}"
            _text += f"({get_sgnf_d(table[i, j, 1])})"
            _ = ax.text(j, i, _text, ha="center", va="center", color="black",
                        fontsize=23)

    plt.colorbar(im)

    plt.show()

    # -- save
    count = len([f for f in os.listdir(utils.FIG_PATH) if "eps_plot_" in f])
    logger(f"{count=}")
    fig.savefig(f'{utils.FIG_PATH}/eps_plot_{count}.svg', bbox_inches='tight')
    logger("[saved]")



