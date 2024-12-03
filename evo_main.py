import numpy as np
import time, os, sys
import random
import argparse, yaml
from deap import base, creator, tools, cma

import tools.evolutions as me

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time, warnings
import src.envs as envs
import src.models as mm

from src.utils import setup_logger, make_probability_set
logger = setup_logger(__name__)




""" Env """

# define the environment
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
        frequencies = np.linspace(0.02, 0.4, K)
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


def calc_fitness2(stats: dict) -> float:

    """
    Calculate the fitness value.

    Parameters
    ----------
    stats : dict
        The statistics of the agent.

    Returns
    -------
    fitness : float
        The fitness value.
    """

    fitness = stats["mean_scores"][0].mean(axis=0).mean(axis=0)
    assert isinstance(fitness, float), f"fitness is not a float: {fitness}"

    return (fitness,)



class Env2:

    """
    The game class for a k-arm bandit task.
    """

    def __init__(self, protocol: dict,
                 verbose: bool,
                 env_type: str="simple"):

        """
        The game class.

        Parameters
        ----------
        game : envs.Environment
            The environment.
        protocol : dict
            The protocol for a trial.
        probabilities_set : list
            The set of probabilities for each round.
        verbose : bool
            Whether to print information.
        env_type: str
            type of kbandit environment, options are
            `simple` or `smooth`.
            Default `simple`.
        """

        #
        self.fitness_size = 1
        self.protocol = protocol
        self.verbose = verbose
        self.env_type = env_type

        self.K = protocol["K"]
        self.tau = protocol["tau"]
        self.nb_trials = protocol["nb_trials"]
        self.nb_rounds = protocol["nb_rounds"]
        self.nb_reps = protocol["nb_reps"]

    def __repr__(self):

        return f"Env({self.nb_reps}:{self.nb_trials}:{self.nb_rounds}, K={self.K}, env_type={self.env_type})"

    def _make_env(self, env_type: str, nb_trials: int, K: int=None) -> object:

        """
        Make the environment.

        Parameters
        ----------
        env_type : str
            The type of environment.
        nb_trials : int
            The number of trials.
        K : int
            The number of bandits.

        Returns
        -------
        env : object
            The environment object.
        """

        if K is None:
            K = self.K

        normalize = False
        fixed_p = 0.7

        probabilities_set = make_probability_set(K=K,
                                                 nb_trials=nb_trials,
                                                 fixed_p=fixed_p,
                                                 normalize=normalize)

        if env_type == "simple":
            return envs.KArmedBandit(K=K,
                                     probabilities_set=probabilities_set)
        elif env_type == "smooth2":
            return envs.KArmedBanditSmoothII(
                             K=K,
                             verbose=False,
                             tau=self.tau,
                             fixed_p=fixed_p,
                             normalize=normalize)
        else:
            return envs.KArmedBanditSmooth(
                             K=K,
                             probabilities_set=probabilities_set,
                             tau=self.tau)

    def run(self, agent: object) -> float:

        """
        Evaluate the agent on the k-arm bandit task.

        Parameters
        ----------
        agent : object
            The agent object.

        Returns
        -------
        fitness : float
            The fitness value.
        """

        ### environment ###
        # environment & game
        # env = self._make_env()

        ### model ###
        # params = agent.get_genome()
        # params['K'] = self.K
        # model = mm.Model(**params)

        ### fit ###
        # stats = envs.trial_multi_model(models=[model],
        #                                environment=env,
        #                                nb_trials=self.nb_trials,
        #                                nb_rounds=self.nb_rounds,
        #                                nb_reps=self.nb_reps)

        # return calc_fitness(stats=stats)

        # variant: multiple K
        # Ks = [5, 15, 40]
        Ks = [5, 35]

        fitness = 0.
        for env_type in ("simple", "smooth", "smooth2"):

            # config-overwriting settings <<<< ! >>>>
            nb_trials = 3 if env_type == "simple" else self.nb_trials
            nb_rounds = 200 if env_type == "simple" else self.nb_rounds

            for K in Ks:
                env = self._make_env(env_type=env_type,
                                     nb_trials=nb_trials,
                                     K=K)
                params = agent.get_genome()
                params['K'] = K
                model = mm.Model(**params)
                stats = envs.trial_multi_model(models=[model],
                                               environment=env,
                                               nb_trials=nb_trials,
                                               nb_rounds=nb_rounds,
                                               nb_reps=self.nb_reps)

                fitness += calc_fitness(stats=stats)[0]

        return (fitness / (len(Ks) + 3),)


class Env:

    """
    The game class for a k-arm bandit task.
    """

    def __init__(self, protocol: dict,
                 verbose: bool,
                 env_type: str="simple"):

        """
        The game class.

        Parameters
        ----------
        game : envs.Environment
            The environment.
        protocol : dict
            The protocol for a trial.
        probabilities_set : list
            The set of probabilities for each round.
        verbose : bool
            Whether to print information.
        env_type: str
            type of kbandit environment, options are
            `simple` or `smooth`.
            Default `simple`.
        """

        #
        self.fitness_size = 1
        self.protocol = protocol
        self.verbose = verbose
        self.env_type = env_type

        self.K = protocol["K"]
        self.tau = protocol["tau"]
        self.nb_trials = protocol["nb_trials"]
        self.nb_rounds = protocol["nb_rounds"]
        self.nb_reps = protocol["nb_reps"]

    def __repr__(self):

        return f"Env({self.nb_reps}:{self.nb_trials}:{self.nb_rounds}, K={self.K}, env_type={self.env_type})"

    def _make_env(self, env_type: str, nb_trials: int, K: int=None) -> object:

        """
        Make the environment.

        Parameters
        ----------
        env_type : str
            The type of environment.
        nb_trials : int
            The number of trials.
        K : int
            The number of bandits.

        Returns
        -------
        env : object
            The environment object.
        """

        if K is None:
            K = self.K

        normalize = False
        fixed_p = 0.7

        probabilities_set = make_probability_set(K=K,
                                        nb_trials=self.nb_trials,
                                          fixed_p=0.9,
                                                 normalize=normalize)

        return make_env(K=K,
                       probabilities_set=probabilities_set,
                       env_type=env_type,
                       tau=self.tau,
                       normalize=False)

    def run(self, agent: object) -> float:

        """
        Evaluate the agent on the k-arm bandit task.

        Parameters
        ----------
        agent : object
            The agent object.

        Returns
        -------
        fitness : float
            The fitness value.
        """

        # variant: multiple K
        Ks = [10, 200]

        fitness = 0.
        for env_type in ("v0", "driftv0", "sin0"):

            # config-overwriting settings <<<< ! >>>>
            nb_trials = 2 if env_type == "v0" else self.nb_trials
            nb_rounds = 500 if env_type == "v0" else self.nb_rounds

            for K in Ks:
                env = self._make_env(env_type=env_type,
                                     nb_trials=nb_trials,
                                     K=K)
                params = agent.get_genome()
                params['K'] = K
                model = mm.Model(**params)
                stats = envs.trial_multiple_models(models=[model],
                                               environment=env,
                                               nb_trials=nb_trials,
                                               nb_rounds=nb_rounds,
                                               nb_reps=self.nb_reps)

                fitness += stats["score_list"][0].mean(axis=1).mean(axis=0)

        return (fitness / (len(Ks) + 3),)



"""
GENOME SETUP
------------

K - lr - tau - dur_pre - dur_post - *value function*

"""

FIXED_PARAMETERS = {
    # 'w_max': 5.,
    # 'dur_pre': 100,
    # 'dur_post': 100,
    # 'lr': 0.01,
    # 'tau': 10,

    # 'alpha': 0.1,
    # 'beta': 1.,
    # 'mu': 0.,
    # 'sigma': 1.,
    # 'r': 0.5,
    'value_function': "gaussian",
}

PARAMETERS = {
    'tau_u': lambda: random.randint(1, 300),
    'tau_v': lambda: random.randint(1, 300),
    'gain': lambda: random.randint(1, 500) / 10,
    'threshold': lambda: round(random.uniform(0.1, 10), 2),

    'alpha': lambda: round(random.uniform(-5, 5), 1),
    'beta': lambda: round(random.uniform(0.1, 10), 1),
    'mu': lambda: round(random.uniform(-5, 5), 1),
    'sigma': lambda: round(random.uniform(0.01, 10), 1),
    'r': lambda: round(random.uniform(.0, 1.0), 2),

    'alpha_lr': lambda: round(random.uniform(-5, 5), 1),
    'beta_lr': lambda: round(random.uniform(0.1, 10), 1),
    'mu_lr': lambda: round(random.uniform(-5, 5), 1),
    'sigma_lr': lambda: round(random.uniform(0.01, 10), 1),
    'r_lr': lambda: round(random.uniform(.0, 1.0), 2),

    'w_max': lambda: round(random.uniform(3, 4), 1),
    'lr': lambda: round(random.uniform(0.001, 0.1), 3),
    # 'tau': lambda: round(random.uniform(1, 100), 1),
    'dur_pre': lambda: random.randint(100, 3000),
    'dur_post': lambda: random.randint(100, 3000),
    'value_function': lambda: random.choice(["gaussian", "none"]),
    'lr_function': lambda: random.choice(["gaussian", "none"]),
}



if __name__ == "__main__" :


    """ import yaml configs """

    with open("configs/config_evo.yaml", 'r') as stream:
        try:
            config_evo = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            sys.exit(1)

    with open("configs/config_model.yaml", 'r') as stream:
        try:
            config_model = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            sys.exit(1)


    """ Arguments """

    parser = argparse.ArgumentParser(description='K-armed bandit')
    parser.add_argument('--verbose', action='store_true', help='verbose',
                        default=False)
    parser.add_argument('--duration', type=int,
                        help='max duration of the simulation',
                        default=-1)
    parser.add_argument('--nosave', action='store_true', help='dont save',
                        default=False)
    parser.add_argument('--noplot', action='store_true', help='plot evolution results',
                        default=False)

    args = parser.parse_args()

    verbose = args.verbose
    np.random.seed = config_model["seed"]


    """ Game initialization """

    model = me.Model
    me.USE_TQDM = False
    me.FORCE_POOL = False

    # trial settings
    env_type = "v0" #"smooth"  # "simple" or "smooth"
    K = config_model["bandits"]["K"]
    nb_trials = config_model["trial"]["nb_trials"]
    nb_rounds = config_model["trial"]["nb_rounds"]
    protocol = {
        "nb_reps": config_model["trial"]["nb_reps"],
        "nb_trials": nb_trials,
        "nb_rounds": nb_rounds,
        "K": K,
        "tau": config_model["bandits"]["tau"],
    }

    # environment & game
    env = Env(protocol=protocol,
              verbose=verbose,
              env_type=env_type)


    """ Evolution initialization """
    

    fitness_weights = (1.,)
    NPOP = config_evo["evolution"]["NPOP"]
    NGEN = config_evo["evolution"]["NGEN"]
    NUM_CORES = config_evo["evolution"]["NUM_CORES"]  # out of 8
    path = r"src/_evo_cache/"

    # Ignore runtime warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # CMA-ES
    # parameters
    N_param = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters
    MEAN = np.random.rand(N_param)  # Initial mean, could be randomized
    SIGMA = config_evo["evolution"]["SIGMA"]  # Initial standard deviation
    lambda_ = 4 + np.floor(3 * np.log(N_param))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA,
                            lambda_=lambda_)

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=env,
                              model=model,
                              strategy=strategy,
                              FIXED_PARAMETERS=FIXED_PARAMETERS.copy(),
                              fitness_weights=fitness_weights,
                              verbose=verbose)

    # ---| Run |---
    settings = {
        "NPOP": NPOP,
        "NGEN": NGEN,
        "CXPB": 0.6,
        "MUTPB": 0.7,
        "NLOG": 1,
        "TARGET": (1.,),
        "TARGET_ERROR": 0.,
        "NUM_CORES": NUM_CORES,
    }

    # ---| Visualisation |---

    title_text = f"{protocol['nb_reps']}:{protocol['nb_trials']}:{protocol['nb_rounds']} "

    if not args.noplot:
        visualizer = me.Visualizer(settings=settings, online=True,
                                   target=None,
                                   k_average=10,
                                   fitness_size=len(fitness_weights),
                                   ylims=[0., 1.1],
                                   title_text=title_text,
                                   path=path)
    else:
        visualizer = None

    """ simulation initialization """

    # ---| save |---
    save = not args.nosave
    if verbose:
        logger(f"%env : {env}")
        logger.info(f"%save: {save}")

    # save = True

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) \
        if os.path.isfile(os.path.join(path, f))])
    filename = str(n_files+1) + "_best"

    # extra information
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": env.__repr__(),
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
        "data": "k-armed bandit",
        "other": "all (2 Ks and 3 variants)",
    }

    # ---| Run |---
    max_duration = args.duration if args.duration > 0 else None
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       verbose=verbose,
                       max_duration=max_duration,
                       save_figure=save,
                       path=path)


