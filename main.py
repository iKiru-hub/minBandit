import numpy as np
import time, argparse, os, json

import src.envs as envs
import src.models as mm
import src.utils as utils

logger = utils.setup_logger(__name__)



def main(args) -> dict:

    # parameters
    K = args.K
    nb_rounds = args.rounds
    nb_trials = args.trials
    env_type = args.env
    verbose = args.verbose
    visual = args.visual

    # define proababilities set
    probabilities_set = utils.make_probability_set(K=K,
                                                   nb_trials=nb_trials,
                                                   fixed_p=None,
                                                   normalize=False)
    # define the environment
    if env_type == "simple":
        env = envs.KArmedBandit(K=K,
                                probabilities_set=probabilities_set,
                                verbose=False)
    elif env_type == "smooth":
        env = envs.KArmedBanditSmooth(K=K,
                                probabilities_set=probabilities_set,
                                verbose=False,
                                tau=5)
    else:
        env = envs.KArmedBanditSmoothII(K=K,
                                verbose=False,
                                tau=100,
                                fix_p=0.9)

    # define the model
    model_name = args.model
    if model_name == "thompson":
        model = mm.ThompsonSampling(K=K)
    elif model_name == "epsilon":
        model = mm.EpsilonGreedy(K=K, epsilon=0.1)
    elif model_name == "ucb":
        model = mm.UCB1(K=K)
    else:
        # define models
        if args.load:
            idx = args.idx if args.idx >= 0 else None
            params = utils.load_model(idx=idx)
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

        model = mm.Model(**params)

    # run
    if visual:
        envs.visual_trial(model=model,
                          environment=env,
                          nb_rounds=nb_rounds)
        results = None
    else:
        results = envs.trial(model=model,
                             environment=env,
                             nb_trials=nb_trials,
                             nb_rounds=nb_rounds,
                             verbose=verbose)

    return results


def main_multiple(args, **kwargs) -> dict:

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
                                  verbose=verbose)
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
        fig = utils.plot_multiple_reward(
            stats=results,
            window=20,
            title=f" - $K=${K}, $round/trial={nb_rounds}$",
            show=args.show,
        )

        # save
        if args.save:

            if kwargs.get("dirpath", None) is None and \
                kwargs.get("run_id", None) is None:

                # current time
                identifier = f"{time.strftime('rewards_%Y_%m_%d_%H%M%S')}"
                # dirname = os.path.join(utils.MEDIA_PATH, identifier)
                # os.makedirs(dirname, exist_ok=True)
                try:
                    dirname = os.path.join(utils.MEDIA_PATH, identifier)
                    os.makedirs(dirname, exist_ok=True)
                except:
                    dirname = os.path.join(utils.MEDIA_PATH_2, f"{identifier}_1")
                    os.makedirs(dirname, exist_ok=True)
                # os.system(f"mkdir {dirname}")
                run_id = ""

            else:
                assert os.path.exists(kwargs["dirpath"]), "The provided directory does not exist"
                assert kwargs.get("run_id", None) is not None, "Please provide a 'run_id'"
                dirname = kwargs["dirpath"]
                run_id = kwargs["run_id"]

            # save figure
            fig.savefig(f"{dirname}/figure{run_id}.png")

            # save info
            info = {
                "nb_trials": nb_trials,
                "nb_rounds": nb_rounds,
                "nb_reps": nb_reps,
                "K": K,
                "params": params,
                "environment": f"{env}",
            }

            # make dict of the results
            save_results = {
                "rewards": results["reward_list"].tolist(),
                "names": tuple(results["names"]),
                "scores": results["scores"].tolist(),
                "chance": results["chance_list"].tolist(),
                "upper": results["upper_bound_list"].tolist(),
            }

            with open(f"{dirname}/info{run_id}.json", "w") as f:
                json.dump(info, f)

                json.dump(save_results, f)

            if verbose:
                logger.info(f"Results saved in {dirname}")

    return results


def multiple_main_multiple(args):

    logger.info("Running multiple simulations with " + \
        "different parameters")

    # define variables to vary
    variables = [1, 2, 3, 5, 7, 10]
    args.save = True
    args.plot = True
    args.verbose = False
    args.show = False

    # make folder
    if args.save:
        identifier = f"{args.env}_{time.strftime('%Y_%m_%d_%H%M%S')}"
        try:
            dirpath = os.path.join(utils.MEDIA_PATH, identifier)
            os.makedirs(dirpath, exist_ok=True)
        except:
            dirpath = os.path.join(utils.MEDIA_PATH_2, f"{identifier}_1")
            os.makedirs(dirpath, exist_ok=True)
        logger.info(f"Results will be saved in `{dirpath}`")

    # run
    for i, var in enumerate(variables):

        logger.info(f"running `var={var}`...")

        # update the arguments
        if args.env == "simple":
            args.K = var
        else:
            args.rounds = var

        args.idx = i

        # run
        _ = main_multiple(args, dirpath=dirpath, run_id=f"_{var}")

    logger.info("<<< done >>>")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='Navigation memory')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose',
                        default=False)
    parser.add_argument('--rounds', type=int,
                        help='number of rounds in a trial',
                        default=50)
    parser.add_argument('--trials', type=int,
                        help='number of trials',
                        default=1)
    parser.add_argument('--reps', type=int,
                        help='number of repetitions',
                        default=1)
    parser.add_argument('--K', type=int,
                        help='number of arms of the bandit',
                        default=3)
    parser.add_argument('--model', type=str,
                        help='model to run: `ucb`, `thompson`, ' + \
        '`epsilon`; if nothing specified or wrong name, ' + \
        'the default is the custom model',
                        default="model")
    parser.add_argument('--load', action='store_true',
                        help='load saved model',
                        default=False)
    parser.add_argument('--plot', action='store_true',
                        help='plot at the end of the simulation',
                        default=False)
    parser.add_argument('--show', action='store_true',
                        help='whether to show the plot or not',
                        default=False)
    parser.add_argument('--env', type=str,
                        help='type of environment: `simple`, `smooth`',
                        default="simple")
    parser.add_argument('--multiple', type=int,
                        help='run multiple models: 0 (single) or 1 (multiple)',
                        default=1)
    parser.add_argument('--visual', action='store_true',
                        help='visualize the trial',
                        default=False)
    parser.add_argument('--save', action='store_true',
                        help='save the results in a folder',
                        default=False)
    parser.add_argument('--idx', type=int,
                        help='the index of the model to load',
                        default=-1)

    args = parser.parse_args()

    if args.multiple == 1:
        main_multiple(args=args)
    elif args.multiple == 2:
        multiple_main_multiple(args=args)
    else:
        main(args=args)

