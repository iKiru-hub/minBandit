import numpy as np
from numba import jit
from abc import ABC, abstractmethod



class MBsolver(ABC):

    def __init__(self, K: int):

        self._K = K

    def __str__(self) -> str:
        return "MBsolver"

    @abstractmethod
    def select_arm(self) -> int:

        pass

    @abstractmethod
    def update(self, k: int, reward: int):

        pass

    @abstractmethod
    def reset(self):

        pass


class Model(MBsolver):

    def __init__(self, K: int, lr: float=0.01, tau: float=10.,
                 dur_pre: int=100, dur_post: int=100,
                 alpha: float=0., beta: float=1.,
                 mu: float=0., sigma: float=1.,
                 r: float=1., w_max: float=5.,
                 alpha_lr: float=0., beta_lr: float=1.,
                 mu_lr: float=0., sigma_lr: float=1.,
                 r_lr: float=1.,
                 value_function: str="gaussian",
                 lr_function: str="none"):

        super().__init__(K)

        # parameters
        self._options = np.arange(K).astype(int)
        # self._lr = lr
        self._tau = tau
        self._w_max = w_max
        self._value_function_name = value_function
        self._lr_function_name = lr_function
        # if self._lr_function_name == "gaussian":
        self._lr = np.ones(K) * lr

        # roll parameters
        self._dur_pre = dur_pre
        self._dur_post = dur_post

        # variables
        self._u = np.zeros((K, 1))  # memory var
        self._v = np.zeros((K, 1))  # pfc var
        self._W = np.zeros((K, 1))
        self._choice = None
        self.is_random = False

        # value function
        self._alpha = alpha
        self._beta = beta
        self._mu = mu
        self._sigma = sigma
        self._r = r

        # learning rate function
        self._alpha_lr = alpha_lr
        self._beta_lr = beta_lr
        self._mu_lr = mu_lr
        self._sigma_lr = sigma_lr
        self._r_lr = r_lr

    def __str__(self):
        return "`Model`"

    def __repr__(self):

        return f"Model(K={self._K}, dur_pre={self._dur_pre}," + \
            f" dur_post={self._dur_post})"

    def _make_decison(self) -> int:

        """
        Make a decision based on the current state of the model
        """

        self._u = np.around(self._u, 3)

        choice = np.argmax(self._u)

        if choice == np.argmax(self._v) and self._u[choice] > 0.:
            self._choice = choice
            self.is_random = False
        else:
            self._choice = np.random.choice(self._options)
            self.is_random = True

        return self._choice

    def _value_function(self):

        if self._value_function_name == "gaussian":

            return gaussian_sigmoid(x=self._W, alpha=self._alpha,
                                    beta=self._beta, mu=self._mu,
                                    sigma=self._sigma, r=self._r)

        return self._W

    def _lr_function(self):

        if self._lr_function_name == "gaussian":

            return gaussian_sigmoid(x=self._W, alpha=self._alpha_lr,
                                               beta=self._beta_lr, mu=self._mu_lr,
                                               sigma=self._sigma_lr, r=self._r_lr).flatten()

        return self._lr

    def _step(self, Iext: np.ndarray=0):

        """
        Step the model forward in time

        Parameters
        ----------
        Iext: np.ndarray
            external input to the model
        """

        # update
        self._u += (- self._u + self._v + Iext) / self._tau
        self._v += (- self._v + self._value_function() * self._u) / self._tau

    def select_arm(self) -> int:

        """
        Rollout the model and obtain a response
        """

        Iext = np.ones((self._K, 1))

        self.reset()

        # pre-decision period
        for _ in range(self._dur_pre):
            self._step(Iext=Iext)

        # post-decision period
        for _ in range(self._dur_post):
            self._step()

        return self._make_decison()

    def update(self, reward: int, k=None):

        """
        Update the model based on the feedback

        Parameters
        ----------
        choice: int
            the choice made by the model
        reward: int
            the reward received by the model
        """

        self._W[self._choice] += self._lr_function()[self._choice] * \
            (self._w_max * reward - self._W[self._choice])

    def get_values(self) -> np.ndarray:

        """
        Get the values of the model
        """
        return self._value_function().flatten().copy()

    def reset(self):

        self._u = np.zeros((self._K, 1))
        self._v = np.zeros((self._K, 1))
        self._choice = None



""" from the literature """


class ThompsonSampling(MBsolver):

    """
    Thompson Sampling bandit algorithm.
    """

    def __init__(self, K: int):

        """
        Parameters
        ----------
        K : int
            Number of arms of the bandit.
        """

        super().__init__(K)
        self.alpha = np.ones(K)
        self.beta = np.ones(K)

    def __str__(self):
        return "`Thompson Sampling`"

    def select_arm(self) -> int:

        """
        Sample an arm from the bandit.

        Returns
        -------
        int
            Index of the arm to pull.
        """

        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, k: int, reward: int):

        """
        Update the parameters of the bandit.

        Parameters
        ----------
        k : int
            Index of the arm that was pulled.
        reward : int
            Reward obtained by pulling the arm.
        """

        self.alpha[k] += reward
        self.beta[k] += 1 - reward

    def reset(self):

        """
        Reset the parameters of the bandit.
        """

        self.alpha = np.ones(self._K)
        self.beta = np.ones(self._K)


class EpsilonGreedy(MBsolver):

    """
    Epsilon-Greedy bandit algorithm.
    """

    def __init__(self, K: int, epsilon: float):

        """
        Parameters
        ----------
        K : int
            Number of arms of the bandit.
        epsilon : float
            Probability of selecting a random arm.
        """

        super().__init__(K)
        self.epsilon = epsilon
        self.q = np.zeros(K)
        self.n = np.zeros(K)

    def __str__(self):
        return f"`Epsilon-Greedy`"

    def select_arm(self) -> int:

        """
        Sample an arm from the bandit.

        Returns
        -------
        int
            Index of the arm to pull.
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(self._K)
        return np.argmax(self.q)

    def update(self, k: int, reward: int):

        """
        Update the parameters of the bandit.

        Parameters
        ----------
        k : int
            Index of the arm that was pulled.
        reward : int
            Reward obtained by pulling the arm.
        """

        self.n[k] += 1
        self.q[k] += (reward - self.q[k]) / self.n[k]

    def reset(self):

        """
        Reset the parameters of the bandit.
        """

        self.q = np.zeros(self._K)
        self.n = np.zeros(self._K)


class UCB1(MBsolver):

    """
    UCB1 bandit algorithm.
    """

    def __init__(self, K: int):

        """
        Parameters
        ----------
        K : int
            Number of arms of the bandit.
        """

        super().__init__(K)
        self.q = np.zeros(K)
        self.n = np.zeros(K)
        self.t = 0

    def __str__(self):
        return "`UCB1`"

    def select_arm(self) -> int:

        """
        Sample an arm from the bandit.

        Returns
        -------
        int
            Index of the arm to pull.
        """

        if 0 in self.n:
            return np.random.choice(np.where(self.n == 0)[0])

        ucb = self.q + np.sqrt(2 * np.log(self.t) / self.n)
        return np.argmax(ucb)

    def update(self, k: int, reward: int):

        """
        Update the parameters of the bandit.

        Parameters
        ----------
        k : int
            Index of the arm that was pulled.
        reward : int
            Reward obtained by pulling the arm.
        """

        self.n[k] += 1
        self.q[k] += (reward - self.q[k]) / self.n[k]
        self.t += 1

    def reset(self):

        """
        Reset the parameters of the bandit.
        """

        self.q = np.zeros(self._K)
        self.n = np.zeros(self._K)
        self.t = 0




@jit(nopython=True)
def softmax(x: np.ndarray, beta: float=1.) -> np.ndarray:

    """
    Compute the softmax of an array

    Parameters
    ----------
    x: np.ndarray
        the array to compute the softmax of
    beta: float
        the inverse temperature of the softmax

    Returns
    -------
    np.ndarray
        the softmax of the array
    """

    return np.exp(beta * x) / np.sum(np.exp(beta * x))


@jit(nopython=True)
def max_normalize(x: np.ndarray) -> np.ndarray:

    """
    Normalize an array to the maximum value

    Parameters
    ----------
    x: np.ndarray
        the array to normalize

    Returns
    -------
    np.ndarray
        the normalized array
    """

    return x / np.max(x)


@jit#(no_python=True)
def gaussian_sigmoid(x: np.ndarray, alpha: float, beta: float,
                     mu: float, sigma: float, r: float) -> np.ndarray:

    return r / (1 + np.exp(-beta*(x - alpha))) + \
        (1 - r) * np.exp(- ((x - mu)**2) / sigma)


@jit(nopython=True)
def sigmoid(x: np.ndarray, alpha: float=0., beta: float=1.) -> np.ndarray:

    return 1 / (1 + np.exp(-beta*(x - alpha)))


@jit(nopython=True)
def mlp(x: float, y: np.ndarray, param1: float, param2: float,
        param3: float, param4: float, param5: float, param6: float,
        param7: float, param8: float) -> float:

    return param7 * sigmoid(
        param1 * x + param2 * y + param3,
    ) + param8 * sigmoid(
        param4 * x + param5 * y + param6
    )



