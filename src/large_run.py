import numpy as np
import json
import os, sys, json, time
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

# set the path to the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import setup_logger, make_probability_set
import src.models as mm
import src.envs as envs

logger = setup_logger(__name__,
                      level=2)

main_PATH = r"/Users/daniekru/Research/lab/minBandit/src/data"
pigeon_PATH = r"/Users/daniekru/Research/lab/pigeon/data"
main_PATH_cl = r"/home/daniekru/lab/minBandit/src/data"
pigeon_PATH_cl = r"/home/daniekru/lab/pigeon/data"
PATH = pigeon_PATH_cl

# from a sweep
model_params = {
    "K": None,
    "dur_pre": 2000,
    "dur_post": 2000,
    "lr": 2.62773,
    "gain": 0.83687,
    "threshold": 0.5,
    "alpha": 2.53669,
    "beta": 9.18994,
    "mu": -2.66142,
    "sigma": 4.57933,
    "r": 0.55691,
    "alpha_lr": -0.25916,
    "beta_lr": 2.16108,
    "mu_lr": -2.91196,
    "sigma_lr": 0.81556,
    "r_lr": 0.77974,
    "w_max": 4.38612,
    "value_function": "gaussian",
    "lr_function": "gaussian",
}


""" settings """


NB_ROUNDS = 200
NB_TRIALS = 2
NB_REPS = 1


""" some local functions """


def make_env(K: int,
             env_type: str,
             probabilities_set: list,
             tau: int,
             normalize=True):

    if env_type == "driftv0":
        env = envs.KABdriftv0(K=K,
                              probabilities_set=probabilities_set,
                              verbose=False,
                              tau=tau)
    elif env_type == "driftv1":
        env = envs.KABdriftv1(K=K,
                              verbose=False,
                              tau=tau,
                              normalize=normalize,
                              fixed_p=0.9)
    elif env_type == "sinv0":
        frequencies = np.linspace(0.1, 0.4, K)
        phases = np.random.uniform(0, 2*np.pi, K)
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            phases=phases,
                            normalize=normalize,
                            verbose=False)
    else:
        env = envs.KABv0(K=K,
                         probabilities_set=probabilities_set,
                         verbose=False)

    return env


def run_for_one_k(K: int):

    # update model parameters
    model_params["K"] = K

    # define proababilities set
    probabilities_set = make_probability_set(K=K,
                                             nb_trials=NB_TRIALS,
                                             fixed_p=0.9,
                                             normalize=False)

    # define the environment
    envs_list = [
            make_env(K=K,
                     env_type="v0",
                     probabilities_set=probabilities_set,
                     tau=None),
            make_env(K=K,
                     env_type="driftv0",
                     probabilities_set=probabilities_set,
                     tau=10),
            make_env(K=K,
                     env_type="driftv1",
                     probabilities_set=probabilities_set,
                     tau=100),
            make_env(K=K,
                     env_type="sinv0",
                     probabilities_set=probabilities_set,
                     tau=None),
    ]
 
    # run
    all_results = {}
    for env in envs_list:

        # define models
        model_list = [
            mm.ThompsonSampling(K=K),
            mm.EpsilonGreedy(K=K, epsilon=0.1),
            mm.UCB1(K=K),
            mm.Model(**model_params)
        ]

        # run
        results = envs.trial_multiple_models(
                             models=model_list,
                             environment=env,
                             nb_trials=NB_TRIALS,
                             nb_rounds=NB_ROUNDS,
                             nb_reps=NB_REPS,
                             bin_size=2,
                             verbose=False)

        all_results[str(env)] = results

    return all_results


def sanitize_for_json(data: dict):
    """
    Sanitize the data for json serialization.
    Convert any numpy array at any level of the dictionary to a list.
    """

    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = sanitize_for_json(v)
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()

    return data


def save(path: str, data: dict, name: str):
    with open(f"{path}/{name}.json", "w") as f:
        json.dump(data, f)
    logger(f"results saved as `{name}.json`")


def process_k(K):
    logger(f"running for K={K}...")
    return K, run_for_one_k(K=K)  # Return a tuple (K, result) for aggregation


def run_for_all_k(K_list: list):

    # Use a multiprocessing Pool
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_k, K_list), total=len(K_list)))

    # make folder with timestamp
    dirname = time.strftime("run_%d%m%Y_%H%M%S")
    path = f"{PATH}/{dirname}"
    os.makedirs(path, exist_ok=True)

    # save each result separately
    for K, result in results:
        data = sanitize_for_json(result)
        name = f"large_run_results_K{K}"
        save(path=path, data=data, name=name)


if __name__ == "__main__":


    """ settings """

    # K_list = [5, 10, 50, 100, 200, 500, 1000, 1500]
    K_list = [5]

    """ run """

    run_for_all_k(K_list=K_list)
    logger("done.")


