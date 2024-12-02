import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from multiprocessing import Pool, cpu_count
from scipy.ndimage import convolve1d
import os
from tqdm import tqdm

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
settings1.trials = 1
settings1.reps = 5
settings1.verbose = False
settings1.idx = 4
settings1.load = True
settings1.env = "v0"
settings1.K = 10

probability = np.around(np.random.uniform(0.05, 0.5, settings1.K),
                        2)
probability[0] = 0.9

params = utils.load_model(idx=4)
params["K"] = settings1.K


def run_(settings, probabilities_set, params):
    
    # parameters
    K = settings1.K
    nb_rounds = settings1.rounds
    nb_trials = settings1.trials
    nb_reps = settings1.reps
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
    model_entropy = [
        results["entropy_list"][:, :, i, :].mean(axis=2).mean(axis=1)
        for i in range(settings1.trials)
    ]

    return prob_entropy, model_reward, model_entropy





if __name__ == "__main__":

    """ settings """

    NUM_CORES = 4
    NUM_BETAS = 8

    names = ['Thompson Sampling', 'Epsilon-Greedy', 'UCB1', 'Model']

    """ parameters """
    beta_values = 1.5**(np.linspace(7, 1, NUM_BETAS))

    """ parallel computation """
    chunksize = NUM_BETAS // NUM_CORES  # Divide the workload evenly

    with Pool(processes=NUM_CORES) as pool:
        results = list(
            tqdm(pool.imap(calculate_for_beta, beta_values, chunksize=chunksize), 
                 total=len(beta_values))
        )

    # Collect results
    prob_entropy, model_reward, model_entropy = [], [], []
    for res in results:
        prob_entropy.append(res[0])
        model_reward.append(res[1])
        model_entropy.extend(res[2])  # Extend because entropy is a list

    # Convert to numpy arrays
    model_reward = np.array(model_reward)
    model_entropy = np.array(model_entropy)

    logger(f"reward: {model_reward.shape}")
    logger(f"entropy: {model_entropy.shape}")
    tot = len(prob_entropy)

    logger("done")


    """ Plotting """

    plt.style.use('science')  # Requires matplotlib-style package
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Computer Modern Roman'],
        'text.usetex': True,
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })

    # Create figure with improved layout
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5), 
                                   sharex=True, 
                                   constrained_layout=False)

    # Colorblind-friendly color palette
    colors = sns.color_palette("tab10", len(names))

    # Reward plot (top subplot)
    for i, name in enumerate(names):
        ax.errorbar(range(len(prob_entropy)), 
                    model_reward[:, i], 
                    fmt='-o', 
                    color=colors[i],
                    label=name,
                    capsize=3,
                    linewidth=2,
                    markersize=5)

    ax.legend(loc='best', 
              frameon=True, 
              fancybox=True, 
              framealpha=0.7, 
              edgecolor='gray')
    ax.grid(alpha=0.2, linestyle='--')
    ax.set_title(r'\textbf{Model Performance vs Distribution Entropy}')
    ax.set_ylabel(r"Reward $R$")
    ax.set_ylim((0, 1.1))

    # Entropy plot (bottom subplot)
    for i, name in enumerate(names):
        ax1.errorbar(range(len(prob_entropy)), 
                     model_entropy[:, i], 
                     fmt='-o', 
                     color=colors[i],
                     capsize=3,
                     linewidth=2,
                     markersize=5)
        # ax1.errorbar(range(len(prob_entropy)), 
        #              model_entropy[tot:, i], 
        #              fmt='--v', 
        #              color=colors[i],
        #              capsize=3,
        #              linewidth=2,
        #              markersize=5)

    ax1.set_xlabel(r"Entropy $H$")
    ax1.set_ylabel(r"Model Entropy")
    ax1.set_xticks(range(len(prob_entropy)))
    ax1.set_xticklabels(np.around(prob_entropy, 3), rotation=45, ha='right')
    ax1.set_title(r'\textbf{Model Entropy vs Distribution Entropy}')
    ax1.grid(alpha=0.2, linestyle='--')

    # Overall figure title
    fig.suptitle(r'\textbf{Performance over Levels of Distribution Entropy}', 
                 fontsize=17, y=0.98)

    # Save with high DPI for publication
    plt.savefig('entropy_performance_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()
