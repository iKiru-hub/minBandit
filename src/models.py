import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from abc import ABC, abstractmethod

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import sigmoid, gaussian_sigmoid, generalized_sigmoid, neural_response_func
    from utils import plot_online_3d, plot_online_2d, plot_online_tape, plot_online_choices
except ImportError or ModuleNotFoundError:
    from src.utils import sigmoid, gaussian_sigmoid, generalized_sigmoid, neural_response_func
    from src.utils import plot_online_3d, plot_online_2d, plot_online_tape#, plot_online



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
                 lr_function: str="none",
                 track_weights: bool=False,
                 weights_history: bool=False):

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

        # record
        self.choices = []
        self.dw_record = []
        self.w_record = []
        self.track_weights = track_weights
        self.weights_history = weights_history

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

        self.choices.append(self._choice)
        return self._choice

    def _value_function(self):

        if self._value_function_name == "gaussian":

            return gaussian_sigmoid(x=self._W, alpha=self._alpha,
                                    beta=self._beta, mu=self._mu,
                                    sigma=self._sigma, r=self._r)

        return None

    def _lr_function(self):

        if self._lr_function_name == "gaussian":

            return gaussian_sigmoid(x=self._W, alpha=self._alpha_lr,
                                    beta=self._beta_lr, mu=self._mu_lr,
                                    sigma=self._sigma_lr, r=self._r_lr).flatten()

        return None

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

    def _visualize_selection(self, ax: plt.Axes=None,
                             t_update: int=1, title: str="",
                             style: str="3d"):

        """
        Visualize the selection made by the model

        Parameters
        ----------
        ax: plt.Axes
            the axis to plot on (3D).
            Default is None
        t_update: int
            the time interval to update the plot
            Default is 1
        title: str
            the title of the plot
            Default is ""
        style: str
            the style of the plot,
            "3d", "2d", "tape"
            Default is "3d"
        """

        Iext = np.ones((self._K, 1))
 
        U = np.zeros((self._dur_pre + self._dur_post, self._K))
        V = np.zeros((self._dur_pre + self._dur_post, self._K))

        # figure
        if ax is None:
            if style == "3d":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots()
                # ax = fig.add_subplot(111)

        # pre-decision period
        for t in range(self._dur_pre):
            self._step(Iext=Iext)
            U[t] = self._u.flatten()
            V[t] = self._v.flatten()

            if t % t_update != 0:
                continue

            # plot
            if style == "3d":
                plot_online_3d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Pre - t={t}ms" + title)
            elif style == "2d":
                plot_online_2d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Pre - t={t}ms" + title)
            elif style == "tape":
                plot_online_tape(ax=ax,
                                 x=self._u,
                                 title="$u_{pre}$"+\
                                 f" - t={t}ms" + title,
                                 lsty="o--")
            plt.pause(0.001)

        # post-decision period
        for t in range(t, t+self._dur_post):

            self._step()
            U[t] = self._u.flatten()
            V[t] = self._v.flatten()

            if t % t_update != 0:
                continue

            # plot
            if style == "3d":
                plot_online_3d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Post - t={t}ms" + title)
            elif style == "2d":
                plot_online_2d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Post - t={t}ms" + title)
            elif style == "tape":
                plot_online_tape(ax=ax,
                                 x=self._u,
                                 title="$u_{post}$"+\
                                 f" - t={t}ms" + title,
                                 lsty="o-")

            plt.pause(0.001)

        if style == "choice":
            plot_online_choices(ax=ax, K=self._K,
                                choices=self.choices,
                                title=title)

            plt.pause(0.001)

    def select_arm(self, visualize: bool=False,
                   ax: plt.Axes=None,
                   style: str="3d",
                   t_update: int=20,
                   title: str="") -> int:

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
        t_update: int
            the time interval to update the plot
            Default is 20
        titile: str
            the text to display in the plot
            Default ""
        style: str
            the style of the plot,
            "3d", "2d", "tape"
            Default is "3d"

        Returns
        -------
        int
            the choice made by the model
        """

        self.reset()

        if visualize:
            self._visualize_selection(ax=ax,
                                      t_update=t_update,
                                      title=title,
                                      style=style)
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

        dw = self._lr_function()[self._choice] * \
            (self._w_max * reward - self._W[self._choice])
        self._W[self._choice] += dw

        if self.track_weights:
            self.dw_record.append(dw.item())

        if self.weights_history:
            self.w_record.append(self._W.flatten().copy())

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

    def get_delta_record(self) -> np.ndarray:

        """
        Get the record of the delta weights
        """

        return np.array(self.dw_record)

    def get_weights_record(self) -> np.ndarray:

        """
        Get the record of the weights
        """

        return np.array(self.w_record)

    def reset(self, complete: bool=False):

        self._u = np.zeros((self._K, 1))
        self._v = np.zeros((self._K, 1))
        self._choice = None

        if complete:
            self.choices = []


class Modelv2(MBsolver):

    def __init__(self, K: int, lr: float=0.01, tau_u: float=10.,
                 tau_v: float=10., tau: int=None,
                 dur_pre: int=100, dur_post: int=100,
                 alpha: float=0., beta: float=1.,
                 mu: float=0., sigma: float=1.,
                 r: float=1., w_max: float=5.,
                 alpha_lr: float=0., beta_lr: float=1.,
                 mu_lr: float=0., sigma_lr: float=1.,
                 r_lr: float=1.,
                 gain_v: float=1.,
                 offset_v: float=0.,
                 threshold_v: float=0.,
                 gain_u: float=1.,
                 offset_u: float=0.,
                 threshold_u: float=0.,
                 value_function: str="gaussian",
                 lr_function: str="gaussian",
                 track_weights: bool=False,
                 weights_history: bool=False):

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

        # neural reponse parameters
        self._gain_v = gain_v
        self._offset_v = offset_v
        self._threshold_v = threshold_v

        self._gain_u = gain_u
        self._offset_u = offset_u
        self._threshold_u = threshold_u

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

        # record
        self.choices = []
        self.dw_record = []
        self.w_record = []
        self.track_weights = track_weights
        self.weights_history = weights_history

    def __str__(self):
        return "`Model.v2`"

    def __repr__(self):

        return f"Model.v2(K={self._K}, dur_pre={self._dur_pre}," + \
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

        self.choices.append(self._choice)
        return self._choice

    def _value_function(self):

        if self._value_function_name == "gaussian":

            return gaussian_sigmoid(x=self._W, alpha=self._alpha,
                                    beta=self._beta, mu=self._mu,
                                    sigma=self._sigma, r=self._r)

        return None

    def _lr_function(self):

        if self._lr_function_name == "gaussian":

            return gaussian_sigmoid(x=self._W, alpha=self._alpha_lr,
                                    beta=self._beta_lr, mu=self._mu_lr,
                                    sigma=self._sigma_lr, r=self._r_lr).flatten()

        return None

    def _step(self, Iext: np.ndarray=0):

        """
        Step the model forward in time

        Parameters
        ----------
        Iext: np.ndarray
            external input to the model
        """

        # update
        self._u += (- self._u + neural_response_func(
                        self._v, self._gain_v,
                        self._offset_v, self._threshold_v) + \
                        Iext) / self._tau_u
        self._v += (- self._v + self._value_function() * \
                             neural_response_func(
                        self._u, self._gain_u,
                        self._offset_u, self._threshold_u)) / self._tau_v

    def _visualize_selection(self, ax: plt.Axes=None,
                             t_update: int=1, title: str="",
                             style: str="3d"):

        """
        Visualize the selection made by the model

        Parameters
        ----------
        ax: plt.Axes
            the axis to plot on (3D).
            Default is None
        t_update: int
            the time interval to update the plot
            Default is 1
        title: str
            the title of the plot
            Default is ""
        style: str
            the style of the plot,
            "3d", "2d", "tape"
            Default is "3d"
        """

        Iext = np.ones((self._K, 1))
 
        U = np.zeros((self._dur_pre + self._dur_post, self._K))
        V = np.zeros((self._dur_pre + self._dur_post, self._K))

        # figure
        if ax is None:
            if style == "3d":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots()
                # ax = fig.add_subplot(111)

        # pre-decision period
        for t in range(self._dur_pre):
            self._step(Iext=Iext)
            U[t] = self._u.flatten()
            V[t] = self._v.flatten()

            if t % t_update != 0:
                continue

            # plot
            if style == "3d":
                plot_online_3d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Pre - t={t}ms" + title)
            elif style == "2d":
                plot_online_2d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Pre - t={t}ms" + title)
            elif style == "tape":
                plot_online_tape(ax=ax,
                                 x=self._u,
                                 title="$u_{pre}$"+\
                                 f" - t={t}ms" + title,
                                 lsty="o--")
            plt.pause(0.001)

        # post-decision period
        for t in range(t, t+self._dur_post):

            self._step()
            U[t] = self._u.flatten()
            V[t] = self._v.flatten()

            if t % t_update != 0:
                continue

            # plot
            if style == "3d":
                plot_online_3d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Post - t={t}ms" + title)
            elif style == "2d":
                plot_online_2d(ax=ax, t=t, U=U, V=V,
                               u=self._u, v=self._v,
                               title=f"Post - t={t}ms" + title)
            elif style == "tape":
                plot_online_tape(ax=ax,
                                 x=self._u,
                                 title="$u_{post}$"+\
                                 f" - t={t}ms" + title,
                                 lsty="o-")

            plt.pause(0.001)

        if style == "choice":
            plot_online_choices(ax=ax, K=self._K,
                                choices=self.choices,
                                title=title)

            plt.pause(0.001)

    def select_arm(self, visualize: bool=False,
                   ax: plt.Axes=None,
                   style: str="3d",
                   t_update: int=20,
                   title: str="") -> int:

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
        t_update: int
            the time interval to update the plot
            Default is 20
        titile: str
            the text to display in the plot
            Default ""
        style: str
            the style of the plot,
            "3d", "2d", "tape"
            Default is "3d"

        Returns
        -------
        int
            the choice made by the model
        """

        self.reset()

        if visualize:
            self._visualize_selection(ax=ax,
                                      t_update=t_update,
                                      title=title,
                                      style=style)
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

        dw = self._lr_function()[self._choice] * \
            (self._w_max * reward - self._W[self._choice])
        self._W[self._choice] += dw

        if self.track_weights:
            self.dw_record.append(dw.item())

        if self.weights_history:
            self.w_record.append(self._W.flatten().copy())

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

    def get_delta_record(self) -> np.ndarray:

        """
        Get the record of the delta weights
        """

        return np.array(self.dw_record)

    def get_weights_record(self) -> np.ndarray:

        """
        Get the record of the weights
        """

        return np.array(self.w_record)

    def reset(self, complete: bool=False):

        self._u = np.zeros((self._K, 1))
        self._v = np.zeros((self._K, 1))
        self._choice = None

        if complete:
            self.choices = []



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

    def select_arm(self, **kwargs) -> int:

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

    def reset(self, **kwargs):

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

    def select_arm(self, **kwargs) -> int:

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

    def reset(self, **kwargs):

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

    def select_arm(self, **kwargs) -> int:

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

    def reset(self, **kwargs):

        """
        Reset the parameters of the bandit.
        """

        self.q = np.zeros(self._K)
        self.n = np.zeros(self._K)
        self.t = 0


