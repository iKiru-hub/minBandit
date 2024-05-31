import numpy as np
import time, argparse, os

import src.envs as envs
import src.models as mm
import src.utils as utils

logger = utils.setup_logger(__name__)



def main(args) -> dict:

    # parameters
    K = args.K
    nb_rounds = args.rounds
    nb_trials = args.trials
    verbose = args.verbose

    # define proababilities set
    probabilities_set = []
    for i in range(nb_trials):
        p = np.around(np.random.uniform(0.05, 0.3, K), 2)
        # p[i%K] = 0.9
        p[np.random.randint(0, K)] = 0.9
        probabilities_set += [p.tolist()]

    probabilities_set = np.array(probabilities_set)

    # define the environment
    # env = envs.KArmedBandit(K=K,
    #                         probabilities_set=probabilities_set,
    #                         verbose=False)
    env = envs.KArmedBanditSmooth(K=K,
                            probabilities_set=probabilities_set,
                            verbose=False,
                            tau=5)

    # define the model
    model_name = args.model
    if model_name == "thompson":
        model = mm.ThompsonSampling(K=K)
    elif model_name == "epsilon":
        model = mm.EpsilonGreedy(K=K, epsilon=0.1)
    elif model_name == "ucb":
        model = mm.UCB1(K=K)
    else:
        dur_pre = 2000
        dur_post = 2000
        model = mm.Model(K=K, dur_pre=dur_pre,
                         dur_post=dur_post,
                         lr=0.1)

    # run
    results = envs.trial(model=model,
                         environment=env,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                         verbose=verbose)

    return results



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
    env = envs.KArmedBandit(K=K,
                            probabilities_set=probabilities_set,
                            verbose=False)

    # env = envs.KArmedBanditSmooth(K=K,
    #                         probabilities_set=probabilities_set,
    #                         verbose=False,
    #                         tau=5)

    # define models
    dur_pre = 2000
    dur_post = 2000
    models = [
        mm.ThompsonSampling(K=K),
        mm.EpsilonGreedy(K=K, epsilon=0.1),
        mm.UCB1(K=K),
        mm.Model(K=K, dur_pre=dur_pre,
                 dur_post=dur_post,
                 lr=0.1)
    ]

    # run
    results = envs.trial_multi_model(
                         models=models,
                         environment=env,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                         nb_reps=nb_reps,
                         verbose=verbose)

    utils.plot_multiple(results)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Navigation memory')
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

    args = parser.parse_args()

    # main(args=args)

    main_multiple(args=args)


