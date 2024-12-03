


"""
FROM src.envs
"""


class KArmedBandit:

    def __init__(self, K: int, probabilities_set: list,
                 verbose: bool=False):

        """
        Parameters
        ----------
        K : int
            The number of arms of the bandit.
        probabilities_set : list
            The set of probabilities to use
        verbose : bool, optional. Default False.
        """

        assert len(probabilities_set[0]) == K, \
            "The number of probabilities must be equal to K"

        self.K = K
        self._probabilities_set_or = probabilities_set
        self.probabilities_set = probabilities_set
        self.probabilities = probabilities_set[0]
        self.nb_sets = len(probabilities_set)
        self.counter = 0

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)

        self.verbose = verbose

    def __repr__(self):

        return f"KArmedBandit(K={self.K}, nb_sets={self.nb_sets})"

    def sample(self, k: int) -> int:

        """
        Sample the reward of the k-th arm

        Parameters
        ----------
        k : int
            The index of the arm

        Returns
        -------
        int
            The reward
        """

        return np.random.binomial(1, p=self.probabilities[k])

    def update(self):

        """
        Renew the reward distribution
        """

        self.counter += 1
        self.probabilities = self.probabilities_set[self.counter % \
            self.nb_sets]
        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")

    def get_info(self):

        return {
            "probabilities": self.probabilities.tolist(),
            "chance": self.chance_level.tolist(),
            "upper_bound": self.upper_bound.tolist()
        }

    def reset(self):

        self.probabilities_set = self._probabilities_set_or
        self.probabilities = self.probabilities_set[0]
        # self.trg_probabilities = self.probabilities_set[1]
        self.counter = 0

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)


class KArmedBanditSmooth:

    """
    concept drift occuring smoothly 
    """

    def __init__(self, K: int, probabilities_set: list,
                 tau: float=5.,
                 verbose: bool=False):

        """
        Parameters
        ----------
        K : int
            The number of arms of the bandit.
        probabilities_set : list
            The set of probabilities to use
        tau : float
            time constant of the distribution update
        verbose : bool, optional. Default False.
        """

        self.K = K
        self.probabilities_set = probabilities_set if isinstance(
            probabilities_set, np.ndarray) else np.array(
                    probabilities_set)

        self._probabilities_set_or = self.probabilities_set.copy()
        self.probabilities = self.probabilities_set[0]
        self.trg_probabilities = self.probabilities_set[1]
        self.nb_sets = len(probabilities_set)
        self.counter = 0
        self._tau = tau

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)

        self.verbose = verbose

    def __repr__(self):

        return f"KArmedBanditSmooth(K={self.K}, tau={self._tau}, nb_sets={self.nb_sets})"

    def sample(self, k: int) -> int:

        """
        Sample the reward of the k-th arm

        Parameters
        ----------
        k : int
            The index of the arm

        Returns
        -------
        int
            The reward
        """

        return np.random.binomial(1, p=self.probabilities[k])

    def update(self):

        """
        Renew the reward distribution
        """

        self.probabilities += (self.trg_probabilities - \
            self.probabilities) / self._tau
        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)

        # check if the target distribution has been reached
        if np.abs(self.probabilities - self.trg_probabilities).sum() < 0.01:
            self.trg_probabilities = self.probabilities_set[self.counter % self.nb_sets]
            self.counter += 1

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")

    def get_info(self):

        return {
            "probabilities": self.probabilities.tolist(),
            "chance": self.chance_level.tolist(),
            "upper_bound": self.upper_bound.tolist()
        }

    def reset(self):

        self.probabilities_set = self._probabilities_set_or
        self.probabilities = self.probabilities_set[0]
        self.trg_probabilities = self.probabilities_set[1]
        self.counter = 0

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)


class KArmedBanditSmoothII:

    """
    concept drift occuring smoothly but each time with a
    random probability distribution
    """

    def __init__(self, K: int, tau: float=5., verbose: bool=False,
                 fixed_p: float=False, seed: int=None, normalize: bool=False):

        """
        Parameters
        ----------
        K : int
            The number of arms of the bandit.
        tau : float
            time constant of the distribution update
        verbose : bool, optional.
            Default False.
        fixed_p : float, optional.
            Default False.
        seed : int, optional.
            Default None.
        normalize : bool, optional.
            Default False.
        """

        if seed is not None:
            np.random.seed(seed)

        self.K = K
        self._fixed_p = fixed_p
        self._tau = tau
        self._normalize = normalize

        self.probabilities = self._new_distribution()
        self.trg_probabilities = self._new_distribution()
        self.counter = 0

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)

        self.verbose = verbose

    def __repr__(self):

        return f"KArmedBanditSmooth(K={self.K}, tau={self._tau}, fixed_p={self._fixed_p})"

    def _new_distribution(self):

        """
        Generate a new distribution
        """

        if self._fixed_p:
            p = np.random.uniform(0, 0.3, size=self.K)
            p[np.random.randint(0, self.K)] = self._fixed_p
        else:
            p = np.random.uniform(0, 1, size=self.K)

        if self._normalize:
            p /= p.sum()
        return p

    def sample(self, k: int) -> int:

        """
        Sample the reward of the k-th arm

        Parameters
        ----------
        k : int
            The index of the arm

        Returns
        -------
        int
            The reward
        """

        return np.random.binomial(1, p=self.probabilities[k])

    def update(self):

        """
        Renew the reward distribution
        """

        # check if the target distribution has been reached
        if np.abs(self.probabilities - self.trg_probabilities).sum() < 0.01:
            self.trg_probabilities = self._new_distribution()
            self.counter += 1

        self.probabilities += (self.trg_probabilities - \
            self.probabilities) / self._tau
        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")

    def get_info(self):

        return {
            "probabilities": self.probabilities.tolist(),
            "chance": self.chance_level.tolist(),
            "upper_bound": self.upper_bound.tolist()
        }

    def reset(self):

        self.probabilities = self._new_distribution()
        self.trg_probabilities = self._new_distribution()
        self.counter = 0

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)
        self.best_arm = np.argmax(self.probabilities)



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



