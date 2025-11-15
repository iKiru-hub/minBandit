import numpy as np
import matplotlib.pyplot as plt
import time, argparse, os, json
from tqdm import tqdm
from pprint import pprint

try:
    import envs
    import models as mm
    import utils
except ImportError:
    import src.envs as envs
    import src.models as mm
    import src.utils as utils

logger = utils.setup_logger(name=__name__, level=2)

# ---------------------------------
# this is meant to test the sensitivity of the various model paramenter
# ---------------------------------

IDX = 5


def run(param: str, value: float,
        K: int, nb_rounds: int,  nb_trials: int,
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
    params = utils.load_model(idx=IDX)
    params["K"] = K
    # pprint(params)
    assert param in tuple(params.keys()), f"{param} not recognized as a model parameters"
    params[param] = value

    # run
    results = envs.trial(model=mm.Model(**params),
                         environment=env,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                         verbose=verbose,
                         disable=True)

    return results


def redact_score(results):
    return (results['score'] - results['chance']) / (results['upper_bound'] - results['chance'])


if __name__ == "__main__":

    np.random.seed(1)

    # ---
    num_k = 3
    num_param = 3

    nb_rounds = 400
    nb_trials = 1
    nb_reps = 1
    env_type = "default"
    fixed_p = 0.9

    # ---
    Ks = np.logspace(2, 8, num=num_k, base=2)
    Ks = np.flip(Ks, axis=0)
    logger(f"{Ks=}")
    Ks = Ks.astype(int)
    logger(f"(int) {Ks=}")

    param = "dur_pre"
    values = np.around(np.linspace(100, 2000, num_param)).astype(int)

    table = np.zeros((num_k, num_param))

    # ---
    for i in tqdm(range(num_k)):
        logger(f"-->> K={Ks[i]}")
        for j in tqdm(range(num_param)):
            for _ in range(nb_reps):
                results = run(param=param, value=values[j], K=Ks[i], nb_rounds=nb_rounds,
                              nb_trials=nb_trials, env_type=env_type, fixed_p=fixed_p,
                              verbose=False)
                table[i, j] += redact_score(results)

            table[i, j] /= nb_reps

    logger(f"{results['upper_bound']=}")

    # ---
    # table = np.flip(table, axis=0)
    # values = np.flip(values, axis=0)
    fig, ax = plt.subplots(figsize=(3, 10))
    im = ax.imshow(table, cmap="viridis")

    ax.set_xticks(range(num_param))
    ax.set_xticklabels(values, fontsize=15)
    ax.set_xlabel(f"{param}", fontsize=19)

    ax.set_yticks(range(num_k))
    ax.set_yticklabels(Ks, fontsize=15)
    ax.set_ylabel("number of arms", fontsize=19)

    for i in range(len(table)):
        for j in range(len(table[i])):
            _ = ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", color="black",
                        fontsize=15)

    fig.suptitle(f"Model for different {param} and $K$", fontsize=21)

    plt.show()


