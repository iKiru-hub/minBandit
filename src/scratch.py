import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from multiprocessing import Pool, cpu_count
from scipy.ndimage import convolve1d
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
settings1.rounds = 2000
settings1.trials = 2
settings1.reps = 1
settings1.verbose = False
settings1.idx = 6
settings1.load = True
settings1.env = "v0"
settings1.K = 10

NUM_BETAS = 10

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
        model_reward += [results["scores"].tolist()]
        model_reward_std += [results["score_list"].mean(axis=2).std(axis=1)] 
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
            data["prob_entropy"] += res[0]
        data["model_reward"] += res[1]
        data["model_reward_std"] += res[2]
        data["model_entropy"] += res[3]
        data["model_entropy_std"] += res[4]
        data["upper_list"] += res[5]

    """ save results """

    print(data)

    name = "entropy_run_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    with open(f"data/{name}", 'w') as f:
        json.dump(data, f)

    logger(f"saved to {name}")


#     # Collect results
#     prob_entropy, model_reward, model_entropy = [], [], []
#     model_reward_std, model_entropy_std = [], []
#     upper_list = []
#     for res in results:
#         prob_entropy.append(res[0])
#         model_reward.append(res[1])
#         model_reward_std.append(res[2])
#         model_entropy.append(res[3])  # Extend because entropy is a list
#         model_entropy_std.append(res[4])
#         upper_list.append(res[5])

#     # Convert to numpy arrays
#     model_reward = np.array(model_reward)
#     model_entropy = np.array(model_entropy)
#     model_reward_std = np.stack(model_reward_std)
#     model_entropy_std = np.stack(model_entropy_std)
#     upper_list = np.array(upper_list)

#     logger(f"reward: {model_reward.shape}")
#     logger(f"reward_std: {model_reward_std.shape}")
#     logger(f"entropy: {model_entropy.shape}")
#     logger(f"entropy std: {model_entropy_std.shape}")
#     tot = len(prob_entropy)

#     logger("done")

#     data = {
#         "model_reward": model_reward.tolist(),
#         "model_reward_std": model_reward_std.tolist(),
#         "model_entropy": model_entropy.tolist(),
#         "model_entropy_std": model_entropy_std.tolist(),
#         "prob_entropy": prob_entropy,
#         "upper_list": upper_list.tolist(),
#         "beta_values": beta_values.tolist(),
#         "names": names,
#         "rounds": settings1.rounds,
#         "trials": settings1.trials,
#         "reps": settings1.reps,
#         "K": settings1.K,
#     }

#     name = "entropy_run_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
#     with open(f"data/{name}", 'w') as f:
#         json.dump(data, f)

#     logger(f"saved to {name}")


#     """ Plotting """

#     plt.style.use('science')  # Requires matplotlib-style package
#     plt.rcParams.update({
#         'font.family': 'serif',
#         'font.serif': ['Times', 'Computer Modern Roman'],
#         'text.usetex': True,
#         'axes.labelsize': 10,
#         'font.size': 10,
#         'legend.fontsize': 8,
#         'xtick.labelsize': 8,
#         'ytick.labelsize': 8
#     })

#     # Create figure with improved layout
#     fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5),
#                                    sharex=True,
#                                    constrained_layout=False)

#     # Colorblind-friendly color palette
#     colors = sns.color_palette("tab10", len(names))

#     ax.plot(range(NUM_BETAS), upper_list, '-o',
#             color="black", alpha=0.6, label="Max")

#     # Reward plot (top subplot)
#     for i, name in enumerate(names):
#         ax.errorbar(range(NUM_BETAS),
#                     model_reward[:, i],
#                     yerr=model_reward_std[:, i],
#                     fmt='-o',
#                     color=colors[i],
#                     label=name,
#                     capsize=3,
#                     alpha=0.7,
#                     linewidth=2,
#                     markersize=5)

#     ax.legend(loc='best',
#               frameon=True,
#               fancybox=True,
#               framealpha=0.7,
#               edgecolor='gray')
#     ax.grid(alpha=0.3, linestyle='--')
#     ax.set_title(r'\textbf{Model Performance vs Distribution Entropy}')
#     ax.set_ylabel(r"Reward $R$")
#     ax.set_ylim((0, 1.1))

#     # Entropy plot (bottom subplot)
#     for i, name in enumerate(names):
#         ax1.errorbar(range(NUM_BETAS),
#                      model_entropy[:, :1, i].flatten(),
#                      yerr=model_entropy_std[:, :1, i].flatten(),
#                      fmt='-o',
#                      color=colors[i],
#                      capsize=3,
#                      alpha=0.7,
#                      linewidth=2,
#                      markersize=5)
#         ax1.errorbar(range(NUM_BETAS),
#                      model_entropy[:, 1:, i].flatten(),
#                      yerr=model_entropy_std[:, 1:, i].flatten(),
#                      fmt='--v',
#                      color=colors[i],
#                      alpha=0.7,
#                      capsize=3,
#                      linewidth=2,
#                      markersize=5)

#     ax1.set_xlabel(r"Entropy $H$")
#     ax1.set_ylabel(r"Model Entropy")
#     ax1.set_xticks(range(len(prob_entropy)))
#     ax1.set_xticklabels(np.around(prob_entropy, 3), rotation=45, ha='right')
#     ax1.set_title(r'\textbf{Model Entropy vs Distribution Entropy}')
#     ax1.grid(alpha=0.3, linestyle='--')

#     # Overall figure title
#     fig.suptitle(r'\textbf{Performance over Levels of Distribution Entropy}',
#                  fontsize=17, y=0.98)

#     # Save with high DPI for publication
#     # plt.savefig('entropy_performance_plot.pdf', dpi=300, bbox_inches='tight')
#     plt.show()
