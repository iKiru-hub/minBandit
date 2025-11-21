import numpy as np
from multiprocessing import Pool, cpu_count
import os, json
from tqdm import tqdm
import time

import models as mm
import envs
import utils

# --

logger = utils.setup_logger(level=2)
logger(f"{logger}")


# -- general settings
NB_ROUNDS = 2000
NB_TRIALS = 2
ENV_TYPE = "v0"
VERBOSE = True
MODEL_IDX = 1
K_VALUE = 200
NUM_VALUES = 5


# -- model parameters
model_params = utils.load_model(idx=MODEL_IDX)
model_params["K"] = K_VALUE



""" local utils """

def make_probabililties_set(index: int) -> tuple:

    """ define a new set of distributions from the reference with a
    level of entropy dependant on the index """

    # -- reference probability distribution
    probability_max = 0.4
    distributions = []
    for _ in range(NB_TRIALS):
        _distr = np.random.uniform(0., probability_max, K_VALUE)
        _distr[K_VALUE//2] = probability_max
        distributions += [_distr]

    # -- beta values | 1, 0.57.., 0.32.., 0.19.., 0.1.., 0.0625, ...
    lambda_values = 1 / np.logspace(0, 4, num=NUM_VALUES, base=2)

    probabilities_set = []
    entropies = []
    for ref_distribution in distributions:
        distribution = ref_distribution.copy()
        # distribution[K_VALUE//2] = PROBABILITY_MAX + (1 - PROBABILITY_MAX) / (index + 1)
        distribution[K_VALUE//2] = probability_max + (1 - probability_max) * lambda_values[index]
        probabilities_set += [distribution]

        entropies += [utils.calc_entropy(distribution)]

    return probabilities_set, entropies


def make_probabililties_set_v2(index: int) -> tuple:

    """ define a new set of distributions from the reference with a
    level of entropy dependant on the index """

    # -- reference probability distribution
    probability_mean = 0.4
    distributions = []
    for _ in range(NB_TRIALS):
        _distr = np.clip(np.random.normal(probability_mean, 0.13, K_VALUE)
        _distr[K_VALUE//2] = 0.7
        distributions += [_distr]

    # -- beta values | 1, 0.57.., 0.32.., 0.19.., 0.1.., 0.0625, ...
    lambda_values = 5 / np.logspace(0, 5, num=NUM_VALUES, base=1.4)

    probabilities_set = []
    entropies = []
    for ref_distribution in distributions:
        distribution = utils.softmax(ref_distribution, lambda_values[i]) * ref_distribution.sum()
        distribution = np.clip(distribution, 0, 1)
        probabilities_set += [distribution]

        entropies += [utils.calc_entropy(distribution)]

    return probabilities_set, entropies


""" main functions """


def single_run(probabilities_set: list, params: dict):

    """ run all models on a given probability set """

    # define the environment
    env = envs.make_new_env(K=K_VALUE,
                            env_type=ENV_TYPE,
                            nb_trials=NB_TRIALS,
                            probabilities_set=probabilities_set)
    if VERBOSE:
        logger.info(f"%env: {env}")

    # define models
    params["K"] = K_VALUE

    model_list = [
        mm.ThompsonSampling(K=K_VALUE),
        mm.EpsilonGreedy(K=K_VALUE, epsilon=0.1),
        mm.UCB1(K=K_VALUE),
        mm.Model(**params)
    ]

    # run
    results = envs.trial_multiple_models(
                         models=model_list,
                         environment=env,
                         nb_trials=NB_TRIALS,
                         nb_rounds=NB_ROUNDS,
                         nb_reps=1,
                         entropy_calc=True,
                         verbose=False)
    return results


def run_multiple_indexes(empty):

    """ run a list of indexes """

    prob_entropy = []
    model_entropy = []
    model_reward = []
    model_entropy_std = []
    model_reward_std = []
    upper_list = []

    pbar = tqdm(range(NUM_VALUES))
    for index in pbar:

        pbar.set_description(f"{index=}")

        # define proababilities
        # probabilities_set, entropies = make_probabililties_set(index=index)
        probabilities_set, entropies = make_probabililties_set_v2(index=index)
        prob_entropy += [entropies]

        # run
        results = single_run(probabilities_set, model_params)
        model_reward += [results["score_list"][:, 0, :].tolist()]
        model_reward_std += [results["score_list"][:, 0, :].tolist()]
        upper_list += [results["upper_bound_list"].tolist()]
        for i in range(NB_TRIALS):
            model_entropy += [results["entropy_list"][:, :, i, :].mean(axis=2).mean(axis=1).tolist()]
            model_entropy_std += [results["entropy_list"][:, :, i, :].mean(axis=2).std(axis=1)]

    model_reward_std = np.stack(model_reward_std).tolist()
    model_entropy_std = np.stack(model_entropy_std).tolist()

    return prob_entropy, model_reward, model_reward_std, model_entropy, model_entropy_std, upper_list


# ================================================================================
# ================================================================================


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

    names = ['Thompson Sampling', 'Epsilon-Greedy', 'UCB1', 'Model']

    """ parallel computation """

    chunksize = NUM_REPS // NUM_CORES  # Divide the workload evenly

    logger(f"{NUM_CORES=}")
    logger(f"{NUM_REPS=}")
    logger(f"{chunksize=}")
    logger(f"NUM_VALUES={NUM_VALUES}")
    logger(f"{names=}")
    logger(f"running...")

    with Pool(processes=NUM_CORES) as pool:
        results = list(
            tqdm(pool.imap(run_multiple_indexes, [None] * NUM_REPS,
                           chunksize=chunksize),
                 total=NUM_REPS)
        )

    logger("run finished")

    """ collect results """

    data = {
        "prob_entropy": [],
        "model_reward": [],
        "model_reward_std": [],
        "model_entropy": [],
        "model_entropy_std": [],
        "upper_list": []
    }

    for i, res in enumerate(results):
        if i == 0:
            data["prob_entropy"] += [res[0]]
        data["model_reward"] += [res[1]]
        data["model_reward_std"] += [res[2]]
        data["model_entropy"] += [res[3]]
        data["model_entropy_std"] += [res[4]]
        data["upper_list"] += [res[5]]

    """ save results """

    name = "entropy_run_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    with open(f"{utils.DATA_PATH}/{name}", 'w') as f:
        json.dump(data, f)

    logger(f"saved to {name}")

