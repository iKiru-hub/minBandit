import numpy as np
import matplotlib.pyplot as plt
import os, time
from tqdm import tqdm

from scipy.ndimage import convolve1d
from multiprocessing import Pool
import warnings

import envs as envs
import models as mm
import utils as utils

logger = utils.setup_logger(__name__)


def main_multiple(args):

    # parameters
    K = args.K
    nb_rounds = args.rounds
    nb_trials = args.trials
    nb_reps = args.reps
    verbose = args.verbose

    # define proababilities set
    probabilities_set = []
    for i in range(nb_trials):
        p = np.around(np.random.uniform(0.05, 0.3, K), 2)
        p[i%K] = 0.9
        # p[np.random.randint(0, K)] = 0.9
        probabilities_set += [p.tolist()]

    probabilities_set = np.array(probabilities_set)

    # define the environment
    if args.env == "simple":
        env = envs.KArmedBandit(K=K,
                                probabilities_set=probabilities_set,
                                verbose=False)
    elif args.env == "smooth2":
        env = envs.KArmedBanditSmoothII(K=K,
                                verbose=False,
                                tau=40,
                                fixed_p=0.7)
    else:
        env = envs.KArmedBanditSmooth(K=K,
                                probabilities_set=probabilities_set,
                                verbose=False,
                                tau=40)

    if verbose:
        logger.info(f"%env: {env}")

    # define models
    if args.load:
        params = utils.load_model(idx=args.idx,
                                  verbose=args.verbose)
        params["K"] = K

    else:
        params = {
            "K": K,
            "dur_pre": 2000,
            "dur_post": 2000,
            "lr": 0.1,
            "gain": 1.,
            "threshold": 0.5,
            "alpha": 0.,
            "beta": 1.,
            "mu": 0.,
            "sigma": 1.,
            "r": 1.,
            "alpha_lr": 0.1,
            "beta_lr": 0.1,
            "mu_lr": 0.1,
            "sigma_lr": 0.1,
            "r_lr": 0.1,
            "w_max": 5.,
            "value_function": "gaussian",
            "lr_function": "gaussian",
        }

    models = [
        mm.ThompsonSampling(K=K),
        mm.EpsilonGreedy(K=K, epsilon=0.1),
        mm.UCB1(K=K),
        mm.Model(**params)
    ]

    # run
    results = envs.trial_multi_model(
                         models=models,
                         environment=env,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                         nb_reps=nb_reps,
                         verbose=verbose)

    # utils.plot_multiple_regret(results, window=10)
    if args.plot:
        fig = utils.plot_multiple_reward(results, window=20)

        # save
        if args.save:
            # current time
            identifier = f"{time.strftime('%Y%m%d_%H%M%S')}"
            dirname = os.path.join(utils.MEDIA_PATH, identifier)
            os.makedirs(dirname, exist_ok=True)

            # save figure
            fig.savefig(f"{dirname}/figure.png")

            # save info
            info = {
                "nb_trials": nb_trials,
                "nb_rounds": nb_rounds,
                "nb_reps": nb_reps,
                "K": K,
                "params": params,
                "environment": f"{env}",
            }

            with open(f"{dirname}/info.json", "w") as f:
                json.dump(info, f)

            logger.info(f"Results saved in {dirname}")

    return results

class Settings:

    verbose = False
    rounds = 2
    trials = 2
    reps = 10
    K = 10
    model = None
    load = True
    plot = False
    env = "simple"
    multiple = 1
    visual = False
    save = True
    idx = 4

def run_main(settings):
    return main_multiple(settings)


def calc_reg(record, ki, mi):
    z = record[f'{ki}']['reward_list'][mi].mean(axis=0).mean(axis=1)
    upper = record['0']['upper_bound_list'][0]
    res = relu(upper - z)
    return res.sum()

def relu(x):
    return x*(x>0).astype(int)



if __name__ == "__main__":

    """ general settings """

    variable = [1, 2, 3]
    NUM_REP = 2
    SAVE = True

    """ simulation settings """

    settings_list = []

    NUM_VAR = len(variable)
    for r in variable:
        settings2 = Settings()
        settings2.trials = 600
        settings2.reps = 1
        settings2.K = 10
        settings2.rounds = r

        settings_list.append(settings2)


    """ run in parallel """

    # define number of processes
    processes = max((os.cpu_count() - 1), NUM_VAR)

    logger(f"%{NUM_REP=}")
    logger(f"%variables={variable}")
    logger(f"%CORES={processes}")
    logger(f"%{SAVE=}")

    # run
    results = np.zeros((4, NUM_REP, NUM_VAR))
    for rep in tqdm(range(NUM_REP)):
        with Pool(processes) as p:
            record = list(tqdm(p.imap(run_main, settings_list),
                               total=len(settings_list), disable=True))

        record = {f"{i}": res for i, res in enumerate(record)}

        # calculate regret
        for mi in range(4):
            res_m = []
            for ki in range(NUM_VAR):
                res_m += [calc_reg(record, ki, mi)]

            #
            results[mi, rep] = res_m

    # average
    results = results.mean(axis=1)

    """ plot """

    fig = plt.figure(figsize=(8, 6))

    colors = plt.cm.tab10(range(4))
    for mi in range(4):

        name = record['0']['names'][mi]
        plt.plot(results[mi],
                 '-o',
                 color=colors[mi], 
                 label=name)

    plt.legend(loc="upper right")
    plt.xticks(range(len(variable)), variable)
    plt.xlabel("rounds per trial")
    plt.ylabel("regret")
    plt.title(f"Regret [{NUM_REP}-average]")
    plt.grid(alpha=0.5)
    plt.show()

    if SAVE:
        name = "plot_" + time.strftime('%H%M%S_%d%m%Y')

        # check if `media` directory exists
        if os.path.exists("./media/"):
            file_path = "./media/" + name
        else:
            logger.warning(f"`./media/` not found [pwd=`{os.getcwd()}`]")
            file_path = name

        fig.savefig(f"{file_path}.png")
        logger.info(f"Results saved as {file_path}")






