import numpy as np
import matplotlib.pyplot as plt
import os, time, json
from tqdm import tqdm

from scipy.ndimage import convolve1d
from multiprocessing import Pool
import warnings
import argparse

import envs as envs
import models as mm
import utils as utils

logger = utils.setup_logger(__name__)



""" utilities """

ax_dict = {
    2: (2, 1),
    3: (3, 1),
    4: (2, 2),
    5: (3, 2),
    6: (3, 2),
    7: (4, 2),
    8: (4, 2),
}


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


def calc_reg_smooth(record: dict, mi: int) -> float:

    """
    Calculate regret for a given model and environment `smooth`

    Parameters
    ----------
    record : dict
        record of the simulation
    ki : int
        index of the run
    mi : int
        index of the model

    Returns
    -------
    float
        regret
    """

    z = record['reward_list'][mi].mean(axis=0).mean(axis=1)
    upper = record['upper_bound_list'][:, 0]
    res = relu(upper - z)
    return res.sum()


def calc_reg_simple(record: dict, mi: int) -> float:

    """
    Calculate regret for a given model and environment `simple`

    Parameters
    ----------
    record : dict
        record of the simulation
    ki : int
        index of the run
    mi : int
        index of the model

    Returns
    -------
    float
        regret
    """

    # average over repetitions and over trials
    z = record['reward_list'][mi].mean(axis=0).mean(axis=0)

    # upper bound | assuming that it is the same for all trials
    upper = record['upper_bound_list'][0]

    # calculate element-wise error
    res = relu(upper - z)

    return res.sum()


def relu(x):
    return x*(x>0).astype(int)


""" main functions """


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


def main_simple(variable: list, NUM_REP: int, SAVE: bool, SHOW: bool,
                trials: int, rounds: int, MAX_CORES: int=None):

    """ general settings """

    ENV = "simple"
    RUN_NAME = f"{ENV}_"
    NUM_VAR = len(variable)

    """ settings for parallellization """

    # define number of processes (cores)
    NUM_CORES = min((os.cpu_count() - 1), NUM_REP)
    NUM_CORES = min(NUM_CORES, MAX_CORES) if MAX_CORES is not None else NUM_CORES
    NUM_TASKS = 1

    # if the total number of cores is less than the number of repetitions
    # defined the maximum number of cores and over-repetitions
    if NUM_CORES < NUM_REP:

        # number of repetitions per core
        NUM_TASKS = int(np.ceil(NUM_REP / NUM_CORES))

        # determine the number of cores
        NUM_CORES = NUM_REP // NUM_TASKS

        # adjust the number of repetitions
        NUM_REP_new = int(NUM_CORES * NUM_TASKS)

        if NUM_REP_new != NUM_REP:
            logger.warning(f"Number of repetitions adjusted to {NUM_REP_new}")
            NUM_REP = NUM_REP_new
    else:
        logger.debug(f"Keeping {NUM_REP=}, {NUM_CORES=}, {NUM_TASKS=}")

    """ simulation settings """

    # list of unique settings
    var_settings_list = []

    # list of settings for each repetition
    rep_settings_list = []

    for r in variable:
        settings = Settings()
        settings.trials = trials
        settings.reps = 1
        settings.K = r
        settings.rounds = rounds
        settings.env = "simple"

        rep_settings_list += [[settings for _ in range(NUM_REP)]]

    logger(f"%{ENV=}")
    logger(f"%{NUM_REP=}")
    logger(f"%variables={variable} [{NUM_VAR}]")
    logger(f"%{NUM_CORES=}")
    logger(f"%{NUM_TASKS=}")
    logger(f"%{SAVE=}")
    logger(f"%{SHOW=}")

    # record
    results = np.zeros((4, NUM_VAR))
    variances = np.zeros((4, NUM_VAR, 2))

    rew_mean = np.zeros((4, NUM_VAR, trials * rounds))
    rew_std = np.zeros((4, NUM_VAR, trials * rounds))

    # loop over the variables
    for i_var in tqdm(range(NUM_VAR)):

        record = []

        # run with a single variable settings for all repetitions
        # split each cpu over tasks
        for _ in range(NUM_TASKS):
            with Pool(NUM_CORES) as p:
                record += list(tqdm(p.imap(run_main,
                                rep_settings_list[i_var]),
                                total=NUM_REP, disable=True))

        # make dict out of the repetitions
        record = {f"{i}": res for i, res in enumerate(record)}

        # calculate regret for all models
        for mi in range(4):

            # regret over all repetitions
            regret = np.zeros(NUM_REP*NUM_TASKS)
            rew = np.zeros((NUM_REP*NUM_TASKS, trials * rounds))
            for i_rep in range(NUM_REP*NUM_TASKS):
                regret[i_rep] = calc_reg_simple(
                    record=record[str(i_rep)],
                    mi=mi)

                rew[i_rep] = record[str(i_rep)]['reward_list'][mi].mean(axis=0).flatten()

            # save average regret
            mean = regret.mean() / (trials * rounds)
            results[mi, i_var] = mean
            variances[mi, i_var] = [regret.std() / (trials * rounds),
                                    regret.std() / (trials * rounds)]

            # save rewards
            rew_mean[mi, i_var] = rew.mean(axis=0)
            rew_std[mi, i_var] = rew.std(axis=0)

            # calculate variance above/below the mean
            # as the 68% value
            # var_above = np.percentile(regret[np.where(regret > mean)], 68)
            # var_below = np.percentile(regret[np.where(regret < mean)], 68)
            # variances[mi, i_var] = [var_above, var_below]
            # variances[mi, i_var] = [regret.std() / (trials * rounds),
            #                         regret.std() / (trials * rounds)]

    #
    # rew_mean = rew_mean.mean(axis=2)
    # rew_std = rew_std.mean(axis=2)

    """ plot """

    fig_reward = plot_multiple_reward(
                    means=rew_mean, variances=rew_std,
                    variable=variable,
                    title=f"Reward [{NUM_REP*NUM_TASKS}-" + \
                        "average] - variant=`slow stochastic`",
                    ax_title="arms",
                    names=record['0']['names'],
                    xlabel="all rounds",
                    ylabel="reward")

    fig_regret = plot_regret(means=results, variances=variances,
                            variable=variable,
                            title=f"Regret [{NUM_REP*NUM_TASKS}-average] - " + \
                                "variant=`slow stochastic`",
                            names=record['0']['names'],
                            xlabel="K (#arms)", ylabel="regret")

    # fig_reward = plot_reward(means=rew_mean, variances=rew_std,
    #                         variable=variable,
    #                         title=f"Reward [{NUM_REP*NUM_TASKS}-average] - " + \
    #                             "variant=`slow stochastic`",
    #                         names=record['0']['names'],
    #                         xlabel="K (#arms)", ylabel="reward")

    if SHOW:
        plt.show()

    if SAVE:
        save_run(results=results, variable=variable, RUN_NAME=RUN_NAME,
                 fig=fig_regret, fig2=fig_reward)


def main_smooth(variable: list, NUM_REP: int, SAVE: bool, SHOW: bool,
                trials: int, K: int, MAX_CORES: int=None):

    """ general settings """

    ENV = "smooth2"
    RUN_NAME = f"{ENV}_"
    NUM_VAR = len(variable)

    """ settings for parallellization """

    # define number of processes (cores)
    NUM_CORES = min((os.cpu_count() - 1), NUM_REP)
    NUM_CORES = min(NUM_CORES, MAX_CORES) if MAX_CORES is not None else NUM_CORES
    NUM_TASKS    = 1

    # if the total number of cores is less than the number of repetitions
    # defined the maximum number of cores and over-repetitions
    if NUM_CORES < NUM_REP:

        # number of repetitions per core
        NUM_TASKS = int(np.ceil(NUM_REP / NUM_CORES))

        # determine the number of cores
        NUM_CORES = NUM_REP // NUM_TASKS

        # adjust the number of repetitions
        NUM_REP_new = int(NUM_CORES * NUM_TASKS)

        if NUM_REP_new != NUM_REP:
            logger.warning(f"Number of repetitions adjusted to {NUM_REP_new}")
            NUM_REP = NUM_REP_new
    else:
        logger.debug(f"Keeping {NUM_REP=}, {NUM_CORES=}, {NUM_TASKS=}")

    """ simulation settings """

    # list of unique settings
    var_settings_list = []

    # list of settings for each repetition
    rep_settings_list = []

    for r in variable:
        settings = Settings()
        settings.trials = trials
        settings.reps = 1
        settings.K = K
        settings.rounds = r  # <<<--- 
        settings.env = "smooth2"

        rep_settings_list += [[settings for _ in range(NUM_REP)]]

    logger(f"%{ENV=}")
    logger(f"%{NUM_REP=}")
    logger(f"%variables={variable} [{NUM_VAR}]")
    logger(f"%{NUM_CORES=}")
    logger(f"%{NUM_TASKS=}")
    logger(f"%{SAVE=}")
    logger(f"%{SHOW=}")

    # run
    results = np.zeros((4, NUM_VAR))
    variances = np.zeros((4, NUM_VAR, 2))

    rew_mean = np.zeros((4, NUM_VAR, trials))
    rew_std = np.zeros((4, NUM_VAR, trials))

    # loop over the variables
    for i_var in tqdm(range(NUM_VAR)):

        record = []

        # run with a single variable settings for all repetitions
        # split each cpu over tasks
        for _ in range(NUM_TASKS):
            with Pool(NUM_CORES) as p:
                record += list(tqdm(p.imap(run_main,
                                    rep_settings_list[i_var]),
                                   total=NUM_REP, disable=True))

        # make dict out of the repetitions
        record = {f"{i}": res for i, res in enumerate(record)}

        # calculate regret for all models
        for mi in range(4):

            nb_trials = rep_settings_list[i_var][0].trials
            nb_rounds = rep_settings_list[i_var][0].rounds

            # regret over all repetitions
            regret = np.zeros(NUM_CORES*NUM_TASKS)
            rew = np.zeros((NUM_CORES*NUM_TASKS, nb_trials))
            for i_rep in range(NUM_CORES*NUM_TASKS):
                regret[i_rep] = calc_reg_smooth(
                    record=record[str(i_rep)],
                    mi=mi)
                # print(record[str(i_rep)]['reward_list'][mi].shape)
                rew[i_rep] = record[str(i_rep)]['reward_list'][mi].mean(axis=0).mean(axis=1)

            # save average regret
            results[mi, i_var] = regret.mean() / (nb_trials * nb_rounds)
            variances[mi, i_var] = [regret.std() / (nb_trials * nb_rounds),
                                    regret.std() / (nb_trials * nb_rounds)]

            # save rewards
            rew_mean[mi, i_var, :] = rew.mean(axis=0)
            rew_std[mi, i_var, :] = rew.std(axis=0)
                                  # rew.std(axis=0)]

            # calculate variance above/below the mean
            # as the 68% value
            # var_above = np.percentile(regret[np.where(regret > mean)], 68)
            # var_below = np.percentile(regret[np.where(regret < mean)], 68)
            # variances[mi, i_var] = [var_above, var_below]
            # variances[mi, i_var] = [regret.std(),
            #                         regret.std()]


    # rew_mean = rew_mean.mean(axis=1) / (NUM_TASKS * NUM_CORES)
    # rew_std = rew_std.mean(axis=1) / (NUM_TASKS * NUM_CORES)

    """ plot """

    fig_reward = plot_multiple_reward(
                    means=rew_mean, variances=rew_std,
                    variable=variable,
                    title=f"Reward [{NUM_REP*NUM_TASKS}-" + \
                        "average] - variant=`fast stochastic`",
                    ax_title="round/trial",
                    names=record['0']['names'],
                    xlabel="trial (round average)",
                    ylabel="reward")

    fig_regret = plot_regret(
                    means=results, variances=variances,
                    variable=variable,
                    title=f"Regret [{NUM_REP*NUM_TASKS}-" + \
                        "average] - variant=`fast stochastic`",
                    names=record['0']['names'],
                    xlabel="trials * rounds",
                    ylabel="regret")

#     fig_reward = plot_reward(means=rew_mean, variances=rew_std,
#                             variable=variable,
#                             title=f"Reward [{NUM_REP*NUM_TASKS}-average] - " + \
#                                 "variant=`fast stochastic`",
#                             names=record['0']['names'],
#                             xlabel="trials * rounds", ylabel="reward")


    if SHOW:
        plt.show()

    if SAVE:
        save_run(results=results,
                 variable=variable,
                 RUN_NAME=RUN_NAME,
                 fig=fig_regret, fig2=fig_reward)


def save_run(results: np.ndarray, variable: list, RUN_NAME: str,
             fig: plt.Figure, fig2: plt.Figure):

        name = RUN_NAME + time.strftime('%d%m%Y_%H%M%S')

        # check if `media` directory exists
        if os.path.exists("./media/"):
            folder_path = "./media/" + name
        else:
            logger.warning(f"`./media/` not found [pwd=`{os.getcwd()}`]")
            folder_path = name

        # create directory
        os.makedirs(folder_path, exist_ok=True)

        # convert results into a json file
        save_results = {
            "variable": variable,
            "data": results.tolist()
        }

        fig_path = os.path.join(folder_path, "figure_regret.png")
        results_path = os.path.join(folder_path, "results.json")
        fig.savefig(fig_path)

        fig_path2 = os.path.join(folder_path, "figure_reward.png")
        fig2.savefig(fig_path2)

        with open(results_path, "w") as f:
            json.dump(save_results, f)
        logger.info(f"Results saved in `{folder_path}`")


def plot_regret(means: np.ndarray, variances: np.ndarray,
                variable: list, title: str,
                names: list, xlabel: str, ylabel: str):

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(range(len(names)))

    for mi in range(4):

        name = names[mi]

        # plot variances above/below the mean as a shaded area
        ax.fill_between(range(len(variable)),
                         means[mi] - variances[mi, :, 1],
                         means[mi] + variances[mi, :, 0],
                         color=colors[mi],
                         alpha=0.15)

        # plot mean
        ax.plot(means[mi],
                 '-o',
                 alpha=0.4,
                 color=colors[mi],
                 label=name)

    ax.legend(loc="upper right")
    ax.set_xticks(range(len(variable)), variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.01)

    ax.set_title(title)
    ax.grid(alpha=0.5)

    return fig


def plot_reward(means: np.ndarray, variances: np.ndarray,
                variable: list, title: str,
                names: list, xlabel: str, ylabel: str):


    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(range(len(names)))

    for mi in range(len(names)):

        name = names[mi]

        # plot mean
        ax.plot(means[mi],
                 '-o',
                 alpha=0.4,
                 color=colors[mi],
                 label=name)

        # plot variances above/below the mean as a shaded area
        ax.fill_between(range(len(variable)),
                         means[mi].mean() - variances[mi, :, 1],
                         means[mi].mean() + variances[mi, :, 0],
                         color=colors[mi],
                         alpha=0.15)

    ax.legend(loc="upper right")
    ax.set_xticks(range(len(variable)), variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(alpha=0.5)

    return fig


def plot_multiple_reward(means: np.ndarray, variances: np.ndarray,
                         variable: list, title: str, ax_title: str,
                        names: list, xlabel: str, ylabel: str):

    assert (means.shape[0], means.shape[1]) == (len(names), len(variable)), \
        "Shape of means does not match the number of models and variables"

    tot = means.shape[1]
    x_size = means.shape[2]

    rows, cols = ax_dict[tot]

    fig, ax = plt.subplots(rows, cols, figsize=(8*cols, 6*rows),
                           sharex=True, sharey=True)
    ax = ax.flatten()
    colors = plt.cm.tab10(range(len(names)))

    k_average = 50

    for i in range(tot):

        # top reward
        ax[i].axhline(0.9, color="black", linestyle="--", alpha=0.35,
                      label="upper bound")

        for mi in range(len(names)):

            name = names[mi]

            # plot mean
            m = convolve1d(means[mi, i], np.ones(k_average), mode="constant")/k_average
            ax[i].plot(m,
                     '-',
                     alpha=0.4,
                     color=colors[mi],
                     label=name)

            # plot variances above/below the mean as a shaded area
            v1 = convolve1d(means[mi, i] - variances[mi, i],
                                np.ones(k_average), mode="constant")/k_average
            v2 = convolve1d(means[mi, i] + variances[mi, i],
                                np.ones(k_average), mode="constant")/k_average
            ax[i].fill_between(range(x_size),
                               v1, v2,
                             color=colors[mi],
                             alpha=0.15)

        if i == 0:
            ax[i].legend(loc="upper right")
            ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel("$R$")
        ax[i].set_title(f"{ax_title}={variable[i]}")
        ax[i].grid(alpha=0.5)

    fig.suptitle(title)

    return fig






if __name__ == "__main__":

    """ arguments """

    parser = argparse.ArgumentParser(description="Run simulations")
    parser.add_argument("--run", type=str, default="simple",
                        help="run type: simple or smooth")
    parser.add_argument('--save', action='store_true',
                        help='save',
                        default=False)
    parser.add_argument('--show', action='store_true',
                        help='show', default=False)
    parser.add_argument('--test', action='store_true',
                        help='small run', default=False)
    parser.add_argument('--max_cores', type=int, default=None,
                        help='maximum number of cores')
    args = parser.parse_args()


    """ run """

    # run simple : K
    if args.run == "simple":
        if args.test:
            main_simple(variable=[5, 700],  # K
                        NUM_REP=int(4),
                        SAVE=args.save,
                        SHOW=args.show,
                        MAX_CORES=args.max_cores,
                        trials=2,
                        rounds=3000)
        else:
            main_simple(variable=[3, 6, 12, 25, 50, 100, 200, 400, 600, 1000, 1500, 2100],  # K
                        NUM_REP=int(3*128),
                        SAVE=args.save,
                        SHOW=args.show,
                        MAX_CORES=args.max_cores,
                        trials=3,
                        rounds=3000)

    # run smooth : rounds
    else:
        if args.test:
            main_smooth(variable=[1, 2],  # rounds
                    NUM_REP=int(4),
                    SAVE=args.save,
                    SHOW=args.show,
                    MAX_CORES=args.max_cores,
                    trials=400,
                    K=10)
        else:
            main_smooth(variable=[1, 2, 3],  # rounds
                        NUM_REP=int(2*128),
                        SAVE=args.save,
                        SHOW=args.show,
                        MAX_CORES=args.max_cores,
                        trials=600,
                        K=10)
