import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import time, warnings
from abc import ABC, abstractmethod
from statistics import mode

try:
    from src.utils import tqdm_enumerate, setup_logger, calc_entropy, cosine_similarity
except ModuleNotFoundError:
    from utils import tqdm_enumerate, setup_logger, calc_entropy, cosine_similarity


logger = setup_logger(__name__)



""" Game """


class KAB(ABC):

    """
    abstract class for all k-armed bandit problems
    """

    def __init__(self, K: int, probabilities_set: list,
                 record: bool=False, verbose: bool=False):

        """
        Parameters
        ----------
        K : int
            The number of arms of the bandit.
        probabilities_set : list
            The set of probabilities to use
        record : bool, optional
            Record the results, by default False
        verbose : bool, optional. Default False.
        """

        self.K = K

        if probabilities_set is not None:
            assert len(probabilities_set[0]) == K, \
                "The number of probabilities must be equal to K"
            self._probabilities_set_or = probabilities_set.copy()
            self.probabilities_set = probabilities_set if isinstance(
                probabilities_set, np.ndarray) else np.array(
                        probabilities_set)
            self.probabilities = self.probabilities_set[0]
            self.nb_sets = len(probabilities_set)
        else:
            self.probabilities = np.ones(K) / K

        self.counter = 0

        self.chance_level_list = [np.mean(self.probabilities)]
        self.upper_bound_list = [np.max(self.probabilities)]
        self.best_arm_list = [np.argmax(self.probabilities)]
        self.probabilities_record = [self.probabilities.tolist()]
        self.verbose = verbose
        self.record = record

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.K}," + \
            f" #sets={self.nb_sets})"

    def __len__(self) -> int:
        return self.nb_sets

    def _update_record(self):

        if not self.record:
            return
        self.chance_level_list += [np.mean(self.probabilities)]
        self.upper_bound_list += [np.max(self.probabilities)]
        self.best_arm_list += [np.argmax(self.probabilities)]
        self.probabilities_record += [self.probabilities.tolist()]

    def sample(self, k: int, update_flag: bool=True) -> int:

        """
        Sample the reward of the k-th arm

        Parameters
        ----------
        k : int
            The index of the arm
        update_flag : bool, optional
            Update the record, by default True

        Returns
        -------
        int
            The reward
        """

        if update_flag:
            self._update()

        return np.random.binomial(1, p=self.probabilities[k])

    @abstractmethod
    def _update(self):
        pass

    @property
    def chance_level(self) -> float:
        return np.mean(self.chance_level_list)

    @property
    def upper_bound(self) -> float:
        return np.mean(self.upper_bound_list)

    @property
    def best_arm(self) -> int:
        return mode(self.best_arm_list)

    def get_info(self):

        return {
            "probabilities": self.probabilities.tolist(),
            "chance": self.chance_level.tolist(),
            "upper_bound": self.upper_bound.tolist()
        }

    def reset(self, complete: bool=False):

        if complete:

            self.probabilities_set = self._probabilities_set_or.copy()
            self.probabilities = self.probabilities_set[0]
            self.counter = 0

            self.chance_level_list = [np.mean(self.probabilities)]
            self.upper_bound_list = [np.max(self.probabilities)]
            self.best_arm_list = [np.argmax(self.probabilities)]


class KABv0(KAB):

    def __init__(self, K: int, probabilities_set: list,
                 record: bool=False, verbose: bool=False):

        super().__init__(K=K, probabilities_set=probabilities_set,
                         record=record, verbose=verbose)

    def _update(self):

        self._update_record()

        pass

    def reset(self, complete: bool=False):

        """
        Renew the reward distribution
        """

        super().reset(complete=complete)

        self.counter += 1
        self.probabilities = self.probabilities_set[self.counter % \
            self.nb_sets]

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")


class KABdriftv0(KAB):

    """
    concept drift occuring smoothly
    """

    def __init__(self, K: int, probabilities_set: list,
                 tau: float=100,
                 record: bool=False, verbose: bool=False):

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

        super().__init__(K=K, probabilities_set=probabilities_set,
                         record=record,
                         verbose=verbose)

        self.probabilities_set = tuple([tuple(p.tolist()) for p in self.probabilities_set])

        self.probabilities = np.array(self.probabilities_set[0])
        self.trg_probabilities = np.array(self.probabilities_set[1])

        self._tau = tau
        self.counter = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.K}," + \
            f" #sets={self.nb_sets}, tau={self._tau})"

    def _update(self):

        """
        Renew the reward distribution
        """

        self.probabilities += (self.trg_probabilities - \
            self.probabilities) / self._tau

        self._update_record()

        # check if the target distribution has been reached
        err = np.abs(self.probabilities - self.trg_probabilities).sum()
        if err < 0.01:

            self.trg_probabilities = np.array(self.probabilities_set[self.counter % self.nb_sets])
            self.counter += 1

            err = np.abs(self.probabilities - self.trg_probabilities).sum()

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")

    def reset(self, complete: bool=False):
        super().reset(complete=complete)

        # self.probabilities_set = tuple([tuple(p.tolist()) for p in self.probabilities_set])
        if complete:
            self.counter = 0
            self.probabilities = np.array(self.probabilities_set[0])
            self.trg_probabilities = np.array(self.probabilities_set[1])


class KABdriftv1(KAB):

    """
    concept drift occuring smoothly but each time with a
    random probability distribution
    """

    def __init__(self, K: int, tau: float=5., verbose: bool=False,
                 fixed_p: float=False, record: bool=False,
                 seed: int=None, normalize: bool=False):

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

        super().__init__(K=K, probabilities_set=None,
                         verbose=verbose)

        self._fixed_p = fixed_p
        self._tau = tau
        self._normalize = normalize

        self.probabilities = self._new_distribution()
        self.trg_probabilities = self._new_distribution()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.K}," + \
            f" tau={self._tau}, fixed_p={self._fixed_p})"

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

        self._update_record()

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")

    def reset(self):

        self.probabilities = self._new_distribution()
        self.trg_probabilities = self._new_distribution()
        self.counter = 0

        self.chance_level_list = [np.mean(self.probabilities)]
        self.upper_bound_list = [np.max(self.probabilities)]
        self.best_arm_list = [np.argmax(self.probabilities)]


class KABsinv0(KAB):

    """
    use sin waves to generate a eward distribution
    """

    def __init__(self, K: int, frequencies: list,
                 phases: list=None,
                 constants: list=[],
                 normalize: bool=True,
                 record: bool=False,
                 verbose: bool=False):

        """
        Parameters
        ----------
        K : int
            The number of arms of the bandit.
        frequencies : list
            The frequencies of the sin waves
            shaping the reward distribution
        phases : list, optional
            The phases of the sin waves.
            Default None.
        normalize : bool
            Normalize the probabilities.
            Default True.
        verbose : bool, optional. Default False.
        """

        super().__init__(K=K, probabilities_set=None,
                         record=record, verbose=verbose)

        self._frequencies = frequencies if isinstance(frequencies,
                                                      np.ndarray) else np.array(frequencies)
        self._normalize = normalize
        if phases is not None:
            self._phases = phases if isinstance(phases,
                                                np.ndarray) else np.array(phases)
        else:
            self._phases = np.zeros(K)

        self.constants = constants
        self.num_constants = len(constants)

        self._update()
        self.counter = 0
        if self.num_constants > 0:
            self._name = "KABsinv1"
        else:
            self._name = "KABsinv0"

    def __str__(self) -> str:
        return f"{self._name}"

    def __repr__(self) -> str:
        return f"{self._name}({self.K}," + \
            f" frequencies={self._frequencies}, " + \
            f"const={self.num_constants})"

    def _update(self):

        """
        Renew the reward distribution
        """

        # update the probabilities
        self.probabilities = 0.5*np.sin(2 * np.pi * \
            self._frequencies * self.counter / 100 + self._phases) + 0.5

        if self._normalize:
            if self.probabilities.sum() == 0:
                self.probabilities = np.ones(self.K) / self.K
            else:
                self.probabilities /= self.probabilities.sum()

        # set some to constant values
        self.probabilities[:self.num_constants] = self.constants

        # update record
        self._update_record()
        self.counter += 1

        if self.verbose:
            logger.info("---renew")
            logger.info(f"%probabilities: {self.probabilities}")
            logger.info(f"%chance level: {self.chance_level:.3f}")
            logger.info(f"%upper bound: {self.upper_bound:.3f}")

    def reset(self, complete: bool=False):
        if complete:
            self.counter = 0
            self.chance_level_list = [np.mean(self.probabilities)]
            self.upper_bound_list = [np.max(self.probabilities)]
            self.best_arm_list = [np.argmax(self.probabilities)]


def make_new_env(K: int, env_type: str, nb_trials: int=3) -> object:

    probabilities_set = np.random.normal(0.5, 0.2, (nb_trials, K)).clip(0, 1)

    # define the environment
    if env_type == "driftv0":
        env = KABdriftv0(K=K,
                              probabilities_set=probabilities_set,
                              verbose=False,
                              tau=100)
    elif env_type == "sinv0":
        frequencies = np.random.uniform(0, 0.1, K)
        phases = np.random.uniform(0, 6.28, K)
        env = KABsinv0(K=K,
                            frequencies=frequencies,
                            normalize=False,
                            phases=phases,
                            verbose=False)
    elif env_type == "sinv1":
        frequencies = np.random.uniform(0, 0.1, K)
        phases = np.random.uniform(0, 6.28, K)
        constants = np.random.uniform(0, 0.7, K//2)
        env = KABsinv0(K=K,
                            frequencies=frequencies,
                            normalize=False,
                            phases=phases,
                            constants=constants,
                            verbose=False)
    elif env_type == "v0":
        env = KABv0(K=K,
                         probabilities_set=probabilities_set,
                         verbose=False)
    else:
        raise NameError(f"\n==> {env_type=} not recognized")

    return env



""" Trial """


def trial(model: object, environment: KAB,
          nb_trials: int, nb_rounds: int,
          verbose: bool=False,
          score_only: bool=False,
          disable: bool=False) -> dict:

    """
    Run a trial of the k-armed bandit problem with the given model
    and a given environement

    Parameters
    ----------
    model : object
        The model to use
    environment : KAB
        The k-armed bandit environment
    nb_trials : int
        The number of trials
    nb_rounds : int 
        The number of rounds
    verbose : bool, optional
        Display information, by default False
    disable : bool, optional
        Disable the progress bar, by default False

    Returns
    -------
    list
        The results of the trial
    """

    # initialize
    model.reset()

    # record
    rewards_list = []
    chance_list = []
    upper_bound_list = []
    selection_list = []
    score = 0.
    chance = 0.
    upper_bound = 0.
    weights = np.zeros((model._K, nb_trials * nb_rounds))

    #
    if verbose:
        logger.info(f"%trials={nb_trials}")
        logger.info(f"%rounds={nb_rounds}")
        logger.info(f"Model: {model}")
        logger.info(f"K-armed bandit: {environment}")

    # loop over trials
    for trial_i in tqdm(range(nb_trials), disable=disable):

        # renew the reward distribution
        # if trial_i > 0:
        #     environment.update()
        if trial_i > 0:
            environment.reset()

        # record
        rewards = np.zeros(nb_rounds)
        chances = np.zeros(nb_rounds)
        upper_bounds = np.zeros(nb_rounds)
        selections = np.zeros(nb_rounds)

        # loop over rounds
        for round_i in range(nb_rounds):

            # select an arm
            k = model.select_arm()

            # sample the reward
            reward = environment.sample(k=k)

            # update the model
            model.update(k=k, reward=reward)

            # record
            rewards[round_i] = reward

            if score_only:
                continue

            chances[round_i] = environment.chance_level
            upper_bounds[round_i] = environment.upper_bound
            selections[round_i] = k
            weights[:, (trial_i+1)*round_i] = model._W.flatten()

       # ---------------------------- #

        score += np.sum(rewards) / nb_rounds / nb_trials

        if score_only:
            continue
        chance += environment.chance_level / nb_trials
        upper_bound += environment.upper_bound / nb_trials

        rewards_list.append(rewards.tolist())
        chance_list.append(chances.tolist())
        upper_bound_list.append(upper_bounds.tolist())
        selection_list.append(selections.tolist())


    # ---------------------------- #

    if verbose:
        logger.info("")
        logger.info(f">>> upper : {upper_bound:.3f}")
        logger.info(f">>> chance: {chance:.3f}")
        logger.info(f">>> score : {score:.3f}")

    if score_only:
        return {"scores": score}

    stats = {
        "rewards_list": rewards_list,
        "chance_list": chance_list,
        "optimal_list": upper_bound_list,
        "score": score,
        "chance": chance,
        "upper_bound": upper_bound,
        "selection_list": selection_list,
        "weights": weights.tolist()
    }

    return stats


def trial_multiple_models(models: list, environment: KAB,
                          nb_trials: int, nb_rounds: int, nb_reps: int=1,
                          bin_size: int=20,
                          entropy_calc: bool=False,
                          verbose: bool=False) -> list:

    """
    Run a trial of the k-armed bandit problem with the given model

    Parameters
    ----------
    model : object
        The model to use
    environment : KArmedBandit
        The k-armed bandit environment
    nb_trials : int
        The number of trials
    nb_rounds : int 
        The number of rounds
    nb_reps: int
        The number of repetitions
    verbose : bool, optional
        Display information, by default False

    Returns
    -------
    list
        The results of the trial
    """

    # initialize
    K = environment.K
    names = []
    for m in models:
        m.reset()
        names += [m.__str__()]

    if verbose:
        logger(f"%names: {names}")

    # global record
    # chance_list = np.zeros((nb_trials, nb_rounds))
    # upper_bound_list = np.zeros((nb_trials, nb_rounds))
    # best_arm_list = np.zeros((nb_trials, nb_rounds, K))
    chance_list = np.zeros(nb_trials)
    upper_bound_list = np.zeros(nb_trials)
    best_arm_list = np.zeros((nb_trials, K))

    # local record
    reward_list = np.zeros((len(models), nb_reps, nb_trials, nb_rounds))
    arm_list = np.zeros((len(models), nb_reps, nb_trials, nb_rounds, K))
    score_list = np.zeros((len(models), nb_reps, nb_trials))
    mean_score_list = np.zeros((len(models), nb_reps, nb_trials))

    if entropy_calc:
        entropy_list = np.zeros((len(models), nb_reps,
                                 nb_trials, nb_rounds-bin_size))

    #
    if verbose:
        logger.info(f"%reps={nb_reps}")
        logger.info(f"%trials={nb_trials}")
        logger.info(f"%rounds={nb_rounds}")
        for m in models:
            logger.info(f"Model: {m}")
        logger.info(f"K-armed bandit: {environment}")

    # loop over repetitions
    for rep_i in tqdm(range(nb_reps), disable=not verbose or nb_reps < 2):
        environment.reset(complete=True)

        # loop over trials
        for trial_i in tqdm(range(nb_trials),
                            disable=not verbose or nb_reps > 2):

            # renew the reward distribution
            if trial_i > 0:
                # environment.update()
                environment.reset()

            # global record update
            if rep_i == 0:
                # chance_list[trial_i, round_i] = environment.chance_level
                # upper_bound_list[trial_i, round_i] = environment.upper_bound
                # best_arm_list[trial_i, round_i, environment.best_arm] = 1.
                chance_list[trial_i] = environment.chance_level
                upper_bound_list[trial_i] = environment.upper_bound
                best_arm_list[trial_i, environment.best_arm] = 1.

            # loop over rounds
            for round_i in range(nb_rounds):

                # update only once per round
                environment._update()

                # loop over models
                for i, m in enumerate(models):

                    k = m.select_arm() # select an arm
                    reward = environment.sample(k=k, update_flag=False) # sample reward
                    m.update(k=k, reward=reward) # update model

                    # local record update
                    arm_list[i, rep_i, trial_i, round_i, k] = 1
                    reward_list[i, rep_i, trial_i, round_i] = reward

            # ---------------------------- #
            score_list[:, rep_i, trial_i] = reward_list[:, rep_i, trial_i, -int(nb_rounds/10):].mean(axis=1)
            # mean_score_list[:, rep_i, trial_i] = reward_list[:, rep_i, trial_i].mean(axis=1)

            # calculate entropy for the trial
            if entropy_calc:
                for i, m in enumerate(models):
                    arms = arm_list[i, rep_i, trial_i]
                    entropy = np.zeros(nb_rounds-bin_size)
                    for l in range(0, nb_rounds-bin_size):
                        entropy_list[i, rep_i, trial_i, l] = calc_entropy(
                            arms[l: l+bin_size].sum(axis=0))

        # ---------------------------- #
        if verbose and nb_reps < 2:
            logger.info("")
            logger.info(f"trial {trial_i}")
            logger.info(f">>> upper : {upper_bound_list[trial_i]:.3f}")
            logger.info(f">>> chance: {chance_list[trial_i]:.3f}")
            for i, name in enumerate(names):
                logger.info(f">>> {name} : {score_list[i, rep_i].mean():.3f}")

    # ---------------------------- #
    # calculate averages over all simulations
    scores = score_list.mean(axis=2).mean(axis=1)

    stats = {
        "reward_list": reward_list,
        "chance_list": chance_list,
        "upper_bound_list": upper_bound_list,
        "best_arm_list": best_arm_list,
        "arm_list": arm_list,
        "scores": scores,
        "score_list": score_list,
        # "mean_scores": mean_score_list,
        "names": names
    }

    if entropy_calc:
        stats["entropy_list"] = entropy_list

    return stats


def visual_trial(model: object,
                 environment: object,
                 nb_rounds: int,
                 nb_trials: int,
                 t_update: int=100,
                 style: str="tape",
                 plot: bool=True,
                 online: bool=True):

    """
    Visualize the trial

    Parameters
    ----------
    model : object
        The model to use
    environment : object
        The environment
    nb_rounds : int
        The number of rounds
    nb_trials : int
        The number of trials
    t_update : int, optional
        The time to wait before updating the plot.
        Default 100[ms].
    style : str, optional
        The style of the plot.
        Default "3d"
    plot : bool, optional
        Display the plot.
        Default True
    """

    logger.info("%visual trial")

    # initialize
    model.reset()
    K = environment.K

    if K != 3:
        warnings.warn("The visual trial is only available for K=3")
        # return

    # 3D plot

    p = environment.probabilities
    idx_p = np.argmax(p) + 1
    if online:
        fig = plt.figure()
        if len(p) < 7:
            fig.suptitle("$\mathbf{\pi}=$" + \
                f"{environment.probabilities} [$i=${idx_p}]")
        else:
            fig.suptitle("$\mathbf{\pi}_{\\text{max}}=$" + \
                f"{environment.probabilities.max()} [$i=${idx_p}]")

        if style == "3d":
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        ax = None

    all_choices = np.zeros((nb_trials * nb_rounds, K))
    all_top = []
    total_reward = 0.

    # run
    for trial_i in tqdm(range(nb_trials)):

        # env
        environment.update()
        p = environment.probabilities
        # all_top += [p.argmax()]

        idx_p = np.argmax(p) + 1
        if online:
            if len(p) < 7:
                fig.suptitle("$\mathbf{\pi}=$" + \
                    f"{np.around(environment.probabilities, 2)} [$i=${idx_p}]")
            else:
                fig.suptitle("$\mathbf{\pi}_{\\text{max}}=$" + \
                    f"{environment.probabilities.max()} [$i=${idx_p}]")

        #
        model.reset(complete=True)
        reward_round = 0.
        for round_i in tqdm(range(nb_rounds)):

            all_top += [environment.best_arm]

            title = f" [{round_i+1}/{trial_i+1}] - "
            title += "$\\tilde{R}=$"
            title += f"{reward_round/(round_i+1):.3f}"

            # select an arm
            k = model.select_arm(visualize=round_i % 20 == 0 and online,
                                 ax=ax,
                                 t_update=t_update,
                                 style=style,
                                 title=title)

            # sample the reward
            reward = environment.sample(k=k)
            reward_round += reward
            all_choices[trial_i * nb_rounds + round_i, k] = 1

            # update the model
            model.update(k=k, reward=reward)
            environment.update()

        #
        total_reward = reward_round / nb_rounds

    # ---------------------------- #
    plt.close()

    if plot:
        plot_choices(all_choices=all_choices, all_top=all_top,
                     nb_rounds=nb_rounds, nb_trials=nb_trials, K=K,
                     title=f"Model: {model}, Environment: {environment}")
    else:
        return all_choices, all_top, total_reward


def plot_choices(all_choices: np.ndarray, all_top: list,
                 nb_rounds: int, nb_trials: int, K: int,
                 ax: object=None, title: str="",
                 show: bool=True,
                 xlab: bool=True):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.imshow(all_choices.T, cmap="Greys",
               aspect="auto", interpolation="None",
               label="selections")
    ax.set_yticks(range(K))
    ax.set_yticklabels(range(1, K+1))
    ax.set_ylabel("arms", fontsize=15)
    if xlab:
        ax.set_xlabel("rounds", fontsize=15)

    for i in range(nb_trials):

        if i == 0:
            for t in range(nb_rounds):
                if t == 0:
                    ax.plot(t, all_top[t], 'r', marker="v",
                            markersize=5, label="best")
                    continue
                ax.plot(t, all_top[t], 'r', marker="v",
                        markersize=5)

            # ax.plot([i * nb_rounds, (i+1) * nb_rounds],
            #          [all_top[i], all_top[i]], 'r',
            #          label="best")
            continue

        for t in range(nb_rounds):
            ax.plot(i * nb_rounds + t, all_top[t], 'r', marker="v",
                    markersize=5)
        # ax.plot([i * nb_rounds, (i+1) * nb_rounds],
        #          [all_top[i], all_top[i]], 'r')

        if i == 1:
            ax.axvline(i * nb_rounds, color="grey",
                        linestyle="--", linewidth=1.,
                        label="trial onset")
            continue
        ax.axvline(i * nb_rounds, color="grey",
                    linestyle="--", linewidth=1.)

    # plt.title(f"Selections over time", fontsize=15)
    if xlab:
        ax.legend(loc="lower right", fontsize=15)
    ax.set_title(title, fontsize=15)

    if show:
        plt.show()


""" other functions """


@jit(nopython=True)
def eval_prediction_jit(K: int, prediction: np.ndarray,
                        probabilities: np.ndarray, fixed: bool) -> bool:

    """
    Evaluate the prediction

    Parameters
    ----------
    K : int
        The number of classes
    prediction : np.ndarray
        The prediction of the network
    probability : np.ndarray
        The probability distribution
    fixed : bool
        Fixed rewards

    Returns
    -------
    bool
        eventual reward
    """

    # average activity for each neuronal class
    values = np.zeros(K)
    for k in range(K):
        values[k] = np.mean(prediction[k])

    # select the class with the highest activity
    predicted_k = np.argmax(values)

    if fixed:
        return True

    return bool(np.random.binomial(1, probabilities[predicted_k]))


def parse_results_to_regret(results: dict):

    """
    extract measures and calculates regret
    """

    return (np.array(results["optimal_list"]) - \
        np.array(results["rewards_list"])).mean()


def parse_results_to_reward(results: dict):

    """
    extract measures and calculates regret
    """

    return np.array(results["scores"]).mean()



if __name__ == "__main__":


    K = 5
    Np = 3
    probabilities_set = np.random.uniform(0, 1, size=(Np, K)).tolist()

    mab = KArmedBandit(K=K, probabilities_set=probabilities_set,
                       verbose=False)
    # mab = KArmedBanditSmooth(K=K,
    #                          probabilities_set=probabilities_set,
    #                          tau=100,
    #                          verbose=False)

    # mab = KArmedBanditSmoothII(K=K,
    #                          tau=80,
    #                          verbose=False)

    T = 1000
    rec = np.zeros((T, K))
    upp = np.zeros(T)
    for t in range(T):
        # print(f"p={np.around(mab.probabilities, 2),} (trg: {np.around(mab.trg_probabilities, 2)})" )
        mab.update()
        rec[t] = mab.probabilities
        upp[t] = mab.probabilities.max()

    plt.plot(upp, 'o-')
    plt.title(np.around(probabilities_set, 2).max(axis=1))
    plt.ylim(0, 1)
    # plt.imshow(rec.T, cmap="Greys")
    plt.show()
