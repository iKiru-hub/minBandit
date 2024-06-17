import numpy as np
from numba import jit
from abc import ABC, abstractmethod
from src.utils import sigmoid, gaussian_sigmoid, generalized_sigmoid
import matplotlib.pyplot as plt



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

    def __init__(self, K: int, lr: float=0.01, tau_u: float=10.,
                 tau_v: float=10., tau: int=None,
                 dur_pre: int=100, dur_post: int=100,
                 alpha: float=0., beta: float=1.,
                 mu: float=0., sigma: float=1.,
                 r: float=1., w_max: float=5.,
                 alpha_lr: float=0., beta_lr: float=1.,
                 mu_lr: float=0., sigma_lr: float=1.,
                 r_lr: float=1., gain: float=1.,
                 threshold: float=0.,
                 value_function: str="gaussian",
                 lr_function: str="none"):

        super().__init__(K)

        # parameters
        self._options = np.arange(K).astype(int)
        # self._lr = lr
        self._tau_u = tau_u
        self._tau_v = tau_v
        if tau is not None:
            self._tau_u = tau
            self._tau_v = tau
        self._w_max = w_max
        self._value_function_name = value_function
        self._lr_function_name = lr_function
        # if self._lr_function_name == "gaussian":
        self._lr = np.ones(K) * lr

        self._gain = gain
        self._threshold = threshold

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
        self._u += (- self._u + generalized_sigmoid(
                            self._v, self._gain, self._threshold) + \
                            Iext) / self._tau_u
        self._v += (- self._v + self._value_function() * self._u) / self._tau_v

    def _visualize_selection(self, ax: plt.Axes=None):

        """
        Visualize the selection made by the model
        """

        Iext = np.ones((self._K, 1))
 
        U = np.zeros((self._dur_pre + self._dur_post, self._K))
        V = np.zeros((self._dur_pre + self._dur_post, self._K))

        # figure
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # pre-decision period
        for t in range(self._dur_pre):
            self._step(Iext=Iext)
            U[t] = self._u.flatten()
            V[t] = self._v.flatten()

            if t % 20 != 0:
                continue

            # plot
            ax.clear()
            # plot U as a black line in 3D
            ax.plot(U[:t, 0], U[:t, 1], U[:t, 2], 'k-',
                    alpha=0.5)
            # plot V as a red line in 3D
            ax.plot(V[:t, 0], V[:t, 1], V[:t, 2], 'r-',
                    alpha=0.5)
            # plot the current point in 3D
            ax.plot([self._u[0]], [self._u[1]], [self._u[2]], 'ko', label="u")
            ax.plot([self._v[0]], [self._v[1]], [self._v[2]], 'ro', label="v")
            ax.set_title(f"Pre - t={t}ms")

            ax.set_xlim(0., 2.)
            ax.set_ylim(0., 2.)
            ax.set_zlim(0., 2.)

            ax.set_xlabel("1")
            ax.set_ylabel("2")
            ax.set_zlabel("3")

            ax.legend()

            plt.pause(0.001)

        # post-decision period
        for t in range(t, t+self._dur_post):

            self._step()
            U[t] = self._u.flatten()
            V[t] = self._v.flatten()

            if t % 20 != 0:
                continue

            # plot
            ax.clear()
            # plot U as a black line in 3D
            ax.plot(U[:t, 0], U[:t, 1], U[:t, 2], 'k-',
                    alpha=0.5)
            # plot V as a red line in 3D
            ax.plot(V[:t, 0], V[:t, 1], V[:t, 2], 'r-',
                    alpha=0.5)
            # plot the current point in 3D
            ax.plot([self._u[0]], [self._u[1]], [self._u[2]], 'ko', label="u")
            ax.plot([self._v[0]], [self._v[1]], [self._v[2]], 'ro', label="v")
            ax.set_title(f"Post - t={t}ms")

            ax.set_xlim(0., 2.)
            ax.set_ylim(0., 2.)
            ax.set_zlim(0., 2.)

            ax.set_xlabel("1")
            ax.set_ylabel("2")
            ax.set_zlabel("3")

            ax.legend()

            plt.pause(0.001)


    def select_arm(self, visualize: bool=False,
                   ax: plt.Axes=None) -> int:

        """
        Rollout the model and obtain a response

        Parameters
        ----------
        visualize: bool
            whether to visualize the model.
            Default is False
        ax: plt.Axes
            the axis to plot on (3D).
            Default is None

        Returns
        -------
        int
            the choice made by the model
        """

        self.reset()

        if visualize:
            self._visualize_selection(ax=ax)
        else:
            Iext = np.ones((self._K, 1))

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

    def get_activations(self) -> tuple:

        """
        Get the activations of the model
        """

        return self._u.flatten().copy(), self._v.flatten().copy()

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


