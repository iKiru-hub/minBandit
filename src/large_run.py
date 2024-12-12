import numpy as np
import json
import os, sys, json, time
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

# set the path to the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import setup_logger, make_probability_set, load_model
import src.models as mm
import src.envs as envs

logger = setup_logger(__name__,
                      level=2)

main_PATH = r"/Users/daniekru/Research/lab/minBandit/src/data"
tmp_PATH = r"/Users/daniekru/Research/lab/minBandit/src/tmp"
pigeon_PATH = r"/Users/daniekru/Research/lab/pigeon/data"
main_PATH_cl = r"/home/daniekru/lab/minBandit/src/data"
pigeon_PATH_cl = r"/home/daniekru/lab/pigeon/data"
PATH = pigeon_PATH_cl
# PATH = tmp_PATH

# from a sweep
# model_params = {
# "alpha":0.4450195718988339,
# "alpha_lr":-2.958221399416473,
# "beta":7.391580370882872,
# "beta_lr":0.9526338245859882,
# "gain":47.86910588825282,
# "lr":0.2773642826170334,
# "mu":2.6713225228884383,
# "mu_lr":-0.6084876849578418,
# "r":0.5161936304182647,
# "r_lr":0.044296190220556864,
# "sigma":0.5450255193262612,
# "sigma_lr":4.463404997440599,
# "threshold":0.3842711049268154,
# "w_max":2.461390370832377,
# "K": None,
# "dur_pre": 2000,
# "dur_post": 2000,
# "value_function": "gaussian",
# "lr_function": "gaussian"}

# from evolution
model_params = load_model(idx=16) # 1


""" settings """


NB_ROUNDS = 2000
NB_TRIALS = 2
# NB_REPS = 2

entropy_calc = False
K_list = [5, 10]
# K_list = [50, 100]
# K_list = [200, 1000]
# K_list = [5, 10, 50, 100, 200, 1000]


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
        frequencies = np.linspace(0.01, 0.4, K)
        phases = np.random.uniform(0, 2*np.pi, K)
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            phases=phases,
                            normalize=False,
                            verbose=False)
    elif env_type == "sinv1":
        frequencies = np.linspace(0.1, 0.4, K)
        phases = np.random.uniform(0, 2*np.pi, K)
        constants = np.random.uniform(0.1, 0.7, int(K/2))
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            phases=phases,
                            normalize=normalize,
                            constants=constants,
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
                                             fixed_p=False,
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
                     env_type="sinv0",
                     probabilities_set=probabilities_set,
                     tau=None),
            make_env(K=K,
                     env_type="sinv1",
                     probabilities_set=probabilities_set,
                     tau=None),
    ]
 
    # run
    all_results = {}
    for env in tqdm(envs_list):

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
                             nb_reps=1,
                             bin_size=20,
                             entropy_calc=entropy_calc,
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


def process_over_ks(empty):

    """Encapsulate the per-beta computation."""

    results = {}
    for K in tqdm(K_list):
        results_k = run_for_one_k(K=K)
        results[K] = {env_name: {"score": values["scores"],
                                 "max": np.mean(values["upper_bound_list"]),
                                 "chance": np.mean(values["chance_list"])} \
            for env_name, values in results_k.items()}

    return results


def parallel_run_over_ks(NUM_CORES: int, chunksize: int):

    """Run the computation over all Ks in parallel."""

    logger(f"running for Ks={K_list}...")

    with Pool(processes=NUM_CORES) as pool:
        results = list(
            tqdm(pool.imap(process_over_ks, [None] * NUM_REPS,
                           chunksize=chunksize),
                 total=NUM_REPS)
        )

    logger("run finished")

    # make folder with timestamp
    dirname = time.strftime("run_%d%m%Y_%H%M%S")
    path = f"{PATH}/{dirname}"
    os.makedirs(path, exist_ok=True)


    # save
    results = {i: res for i, res in enumerate(results)}
    # for result in results:
    data = sanitize_for_json(results)
    name = "large_run_" + "".join(["%" + str(l) for l in K_list])
    save(path=path, data=data, name=name)




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Run entropy experiment')
    parser.add_argument('--reps', type=int, default=4,
                        help='Number of repetitions')
    parser.add_argument('--cores', type=int, default=4,
                        help='Number of cores')

    args = parser.parse_args()

    """ settings """

    NUM_CORES = args.cores
    NUM_REPS = args.reps

    """ parallel computation """

    chunksize = NUM_REPS // NUM_CORES  # Divide the workload evenly

    logger(f"{NUM_CORES=}")
    logger(f"{NUM_REPS=}")
    logger(f"{chunksize=}")
    logger(f"running...")

    parallel_run_over_ks(NUM_CORES=NUM_CORES, chunksize=chunksize)

    # K_list = [5, 10, 50, 100, 200, 500, 1000, 1500]
    # K_list = [5, 10, 100]

    # run_for_all_k(K_list=K_list)
    # logger(f"{K_list} done.")

    # import time
    # time.sleep(180)
    # print("\n\n\n")

    # K_list = [200, 1000, 2000]
    # K_list = [50]

    # run_for_all_k(K_list=K_list)
    # logger(f"{K_list} done.")


