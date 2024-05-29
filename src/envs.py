import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import time, warnings

try:
    from src.utils import tqdm_enumerate, setup_logger
except ModuleNotFoundError:
    from utils import tqdm_enumerate, setup_logger


logger = setup_logger(__name__)



""" Game """


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

        self.K = K
        self.probabilities_set = probabilities_set
        self.probabilities = probabilities_set[0]
        self.nb_sets = len(probabilities_set)
        self.counter = 0

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)

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
        self.probabilities = self.probabilities_set[self.counter % self.nb_sets]
        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)

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


class KArmedBanditSmooth:

    """
    concept drift occuring smoothly 
    """

    def __init__(self, K: int, probabilities_set: list, tau: float=5.,
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
        self.probabilities_set = probabilities_set if isinstance(probabilities_set,
                                            np.ndarray) else np.array(probabilities_set)
        self.probabilities = self.probabilities_set[0]
        self.trg_probabilities = self.probabilities_set[1]
        self.nb_sets = len(probabilities_set)
        self.counter = 0
        self._tau = tau

        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)

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

        # check if the target distribution has been reached
        if np.abs(self.probabilities - self.trg_probabilities).sum() < 0.001:
            self.trg_probabilities = self.probabilities_set[self.counter % self.nb_sets]
            # print(f"%trg: {np.around(self.trg_probabilities, 2)}")
            self.counter += 1

        self.probabilities += (self.trg_probabilities - self.probabilities) / self._tau
        self.chance_level = np.mean(self.probabilities)
        self.upper_bound = np.max(self.probabilities)

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



""" Trial """


def trial(model: object, environment: KArmedBandit,
              nb_trials: int, nb_rounds: int, verbose: bool=False) -> list:

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
    verbose : bool, optional
        Display information, by default False

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
    score = 0.
    chance = 0.
    upper_bound = 0.

    #
    if verbose:
        logger.info(f"%trials={nb_trials}")
        logger.info(f"%rounds={nb_rounds}")
        logger.info(f"Model: {model}")
        logger.info(f"K-armed bandit: {environment}")

    # run
    for trial_i in tqdm(range(nb_trials)):

        # if verbose:
        #     logger.info(f"Trial {trial_i}")

        # renew the reward distribution
        if trial_i > 0:
            environment.update()

        # record
        rewards = np.zeros(nb_rounds)
        chances = np.zeros(nb_rounds)
        upper_bounds = np.zeros(nb_rounds)

        for round_i in range(nb_rounds):

            # select an arm
            k = model.select_arm()

            # sample the reward
            reward = environment.sample(k=k)

            # update the model
            model.update(k=k, reward=reward)

            # record
            rewards[round_i] = reward
            chances[round_i] = environment.chance_level
            upper_bounds[round_i] = environment.upper_bound

        # ---------------------------- #

        rewards_list.append(rewards.tolist())
        chance_list.append(chances.tolist())
        upper_bound_list.append(upper_bounds.tolist())

        score += np.sum(rewards) / nb_rounds / nb_trials
        chance += environment.chance_level / nb_trials
        upper_bound += environment.upper_bound / nb_trials

    # ---------------------------- #

    if verbose:
        logger.info("")
        logger.info(f">>> upper : {upper_bound:.3f}")
        logger.info(f">>> chance: {chance:.3f}")
        logger.info(f">>> score : {score:.3f}")


    stats = {
        "rewards_list": rewards_list,
        "chance_list": chance_list,
        "optimal_list": upper_bound_list,
        "score": score,
        "chance": chance,
        "upper_bound": upper_bound
    }

    return stats



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



if __name__ == "__main__":


    K = 5
    Np = 3
    probabilities_set = np.random.uniform(0, 1, size=(Np, K)).tolist()

    # mab = KArmedBandit(K=K, probabilities_set=probabilities_set, verbose=False)
    mab = KArmedBanditSmooth(K=K, probabilities_set=probabilities_set, tau=10,
                              verbose=False)

    T = 200
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
