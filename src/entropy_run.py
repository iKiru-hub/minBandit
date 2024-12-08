import numpy as np
from multiprocessing import Pool, cpu_count
import os, json
from tqdm import tqdm
import time

import models as mm
import envs
import utils


logger = utils.setup_logger(level=2)
logger(f"{logger}")


class Settings:

    verbose = False
    rounds = 2
    trials = 2
    reps = 10
    K = 10
    model = None
    load = True
    plot = False
    env = "v0"
    multiple = 1
    visual = False
    save = True
    idx = 4


settings1 = Settings()
settings1.rounds = 200
settings1.trials = 2
settings1.reps = 1
settings1.verbose = False
settings1.idx = 6
settings1.load = True
settings1.env = "v0"
settings1.K = 10

NUM_BETAS = 3

probability = np.around(np.random.uniform(0.05, 0.5, settings1.K),
                        2)
probability[0] = 1.

model_params = utils.load_model(idx=settings1.idx)
model_params["K"] = settings1.K

beta_values = 1.5**(np.linspace(7, 1, NUM_BETAS))


def run_(probabilities_set, params):

    # parameters
    K = settings1.K
    nb_rounds = settings1.rounds
    nb_trials = settings1.trials
    nb_reps = 1
    verbose = settings1.verbose
    env_type = settings1.env


    # define the environment
    if env_type == "driftv0":
        env = envs.KABdriftv0(K=K,
                              probabilities_set=probabilities_set,
                              verbose=verbose,
                              tau=5)
    elif env_type == "driftv1":
        env = envs.KABdriftv1(K=K,
                              verbose=verbose,
                              tau=100,
                              normalize=True,
                              fixed_p=0.9)
    elif env_type == "sinv0":
        frequencies = np.linspace(0, 0.4, K)
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            normalize=True,
                            verbose=verbose)
    else:
        env = envs.KABv0(K=K,
                         probabilities_set=probabilities_set,
                         verbose=verbose)

    if verbose:
        logger.info(f"%env: {env}")

    # define models
    params["K"] = K


    model_list = [
        mm.ThompsonSampling(K=K),
        mm.EpsilonGreedy(K=K, epsilon=0.1),
        mm.UCB1(K=K),
        mm.Model(**params)
    ]

    # run
    results = envs.trial_multiple_models(
                         models=model_list,
                         environment=env,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                         nb_reps=nb_reps,
                         entropy_calc=True,
                         verbose=settings1.verbose)
    return results


def softmax(z, beta):
    return np.exp(beta*z) / np.exp(beta*z).sum()


def calculate_for_beta(beta):

    """Encapsulate the per-beta computation."""
    # Define probabilities
    p = softmax(probability, beta)
    probabilities_set = np.array([p.tolist()])
    prob_entropy = utils.calc_entropy(p)

    # Run simulation
    results = run_(settings1, probabilities_set, params)

    model_reward = results["scores"]
    model_reward_std = results["score_list"].mean(axis=2).std(axis=1)
    upper_list = results["upper_bound_list"]

    model_entropy = []
    model_entropy_std = []
    for i in range(settings1.trials):
        model_entropy += [results["entropy_list"][:, :, i, :].mean(axis=2).mean(axis=1)]
        model_entropy_std += [results["entropy_list"][:, :, i, :].mean(axis=2).std(axis=1)]

    return prob_entropy, model_reward, model_reward_std, model_entropy, model_entropy_std, upper_list


def calculation_over_betas(empty):

    """Encapsulate the per-beta computation."""


    prob_entropy = []
    model_entropy = []
    model_reward = []
    model_entropy_std = []
    model_reward_std = []
    upper_list = []

    for beta in tqdm(beta_values):

        #logger(f"running {beta=:.4f}")

        # define proababilities 
        p = softmax(probability, beta)
        probabilities_set = np.array([p.tolist()])
        prob_entropy += [utils.calc_entropy(p)]

        """ run """
        results = run_(probabilities_set, model_params)
        model_reward += [results["score_list"].tolist()]
        model_reward_std += [results["score_list"].tolist()] 
        upper_list += [results["upper_bound_list"].tolist()]
        for i in range(settings1.trials):
            model_entropy += [results["entropy_list"][:, :, i, :].mean(axis=2).mean(axis=1).tolist()]
            model_entropy_std += [results["entropy_list"][:, :, i, :].mean(axis=2).std(axis=1)]

    model_reward_std = np.stack(model_reward_std).tolist()
    model_entropy_std = np.stack(model_entropy_std).tolist()

    return prob_entropy, model_reward, model_reward_std, model_entropy, model_entropy_std, upper_list


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
    logger(f"{NUM_BETAS=}")
    logger(f"{names=}")
    logger(f"running...")

    with Pool(processes=NUM_CORES) as pool:
        results = list(
            tqdm(pool.imap(calculation_over_betas, [None] * NUM_REPS,
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
    with open(f"data/{name}", 'w') as f:
        json.dump(data, f)

    logger(f"saved to {name}")

