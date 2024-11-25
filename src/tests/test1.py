import pytest
import numpy as np

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import src.envs as envs
import main



""" setup """

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_v0():

    # --- init
    K = 5
    N = 3
    tau = 40
    probabilities_set = np.random.rand(N, K)
    probabilities_set = probabilities_set / np.sum(probabilities_set, axis=1)[:, None]

    # --- v0
    env1 = envs.KABv0(K=K,
                     probabilities_set=probabilities_set)

    one_sample = env1.sample(k=0)
    assert isinstance(one_sample, int), "sample() should" + \
        f" return an integer but got {type(s)}"

    uppers = [env1.upper_bound]
    for _ in range(N):
        env1.update()
        uppers.append(env1.upper_bound)

    assert uppers[0] == uppers[-1], "the probability sets" + \
        f" should be cyclical but got {uppers[0]} " + \
        f"and {uppers[-1]}"

    # --- run
    args = {
        "verbose": False,
        "rounds": 2,
        "trials": 1,
        "reps": 1,
        "K": 5,
        "model": "model",
        "load": False,
        "idx":None,
        "plot": False,
        "show": False,
        "env": None,
        "multiple": 0,
        "save": False,
        "visual": False
    }

    args = Args(**args)
    main.main(args=args)


def test_driftv0():

    # --- init
    K = 5
    N = 3
    tau = 40
    probabilities_set = np.random.rand(N, K)
    probabilities_set = probabilities_set / np.sum(probabilities_set, axis=1)[:, None]
    env2 = envs.KABdriftv0(K=K,
                           probabilities_set=probabilities_set,
                           tau=tau)
    env2.update()

    # --- run
    args = {
        "verbose": False,
        "rounds": 2,
        "trials": 2,
        "reps": 1,
        "K": 5,
        "model": "model",
        "load": False,
        "idx":None,
        "plot": False,
        "show": False,
        "env": "driftv0",
        "multiple": 0,
        "save": False,
        "visual": False
    }

    args = Args(**args)
    main.main(args=args)


def test_driftv1():

    # --- init
    K = 5
    N = 3
    tau = 40
    probabilities_set = np.random.rand(N, K)
    probabilities_set = probabilities_set / np.sum(probabilities_set, axis=1)[:, None]

    env3 = envs.KABdriftv1(K=K, tau=tau, fixed_p=0.5,
                           normalize=True)
    new_distribution = env3._new_distribution()
    assert len(new_distribution) == K, "new_distribution" + \
        f" should have length {K} but got {len(new_distribution)}"

    # --- run
    args = {
        "verbose": False,
        "rounds": 2,
        "trials": 2,
        "reps": 1,
        "K": 5,
        "model": "model",
        "load": False,
        "idx":None,
        "plot": False,
        "show": False,
        "env": "driftv1",
        "multiple": 0,
        "save": False,
        "visual": False
    }

    args = Args(**args)
    main.main(args=args)


def test_sinv0():

    # --- init
    K = 5
    N = 3
    tau = 40
    probabilities_set = np.random.rand(N, K)
    probabilities_set = probabilities_set / np.sum(probabilities_set, axis=1)[:, None]

    env4 = envs.KABsinv0(K=K, frequencies=[1, 2, 3, 4, 5],
                              normalize=True)
    assert env4.probabilities[0] == env4.probabilities[-1], "the" + \
        " initially should be equal"
    env4.update()
    env4.update()
    assert env4.probabilities[0] != env4.probabilities[-1], "the" + \
        " probabilities should have changed after two steps"

    # --- run
    args = {
        "verbose": False,
        "rounds": 2,
        "trials": 1,
        "reps": 1,
        "K": 5,
        "model": "model",
        "load": False,
        "idx":None,
        "plot": False,
        "show": False,
        "env": "sinv0",
        "multiple": 0,
        "save": False,
        "visual": False
    }

    args = Args(**args)
    main.main(args=args)

