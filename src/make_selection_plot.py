import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import time, warnings

try:
    from src.utils import tqdm_enumerate, setup_logger
except ModuleNotFoundError:
    import utils
    import models as mm
    import envs


logger = utils.setup_logger(__name__)

""" settings """
K = 5
nb_rounds = 300
nb_trials = 2
env_type = "sinv0"
verbose = True 

""" environment """
# define proababilities set
probabilities_set = utils.make_probability_set(K=K,
                                               nb_trials=nb_trials,
                                               fixed_p=0.9,
                                               normalize=False)
# define the environment
def make_current_env(kind: str, probabilities_set: list):

    if env_type == "driftv0":
        env = envs.KABdriftv0(K=K,
                              probabilities_set=probabilities_set,
                              verbose=verbose,
                              tau=5)
    elif env_type == "driftv1":
        env = envs.KABdriftv1(K=K,
                              verbose=verbose,
                              tau=100,
                              normalize=True,
                              fixed_p=0.9)
    elif env_type == "sinv0":
        frequencies = np.arange(1, K+1)
        phases = np.random.uniform(0, 2*np.pi, K)
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            phases=phases,
                            normalize=True,
                            verbose=verbose)
    else:
        env = envs.KABv0(K=K,
                         probabilities_set=probabilities_set,
                         verbose=verbose)

    return env


""" model run """

# define the model
idx = 5
params = utils.load_model(idx=idx)
params["K"] = K

model = mm.Model(**params)

logger(f"%{model}")

# run
env = make_current_env(env_type, probabilities_set)
out = envs.visual_trial(
                  model=model,
                  environment=env,
                  nb_rounds=nb_rounds,
                  nb_trials=nb_trials,
                  t_update=200,
                  style="choice",
                  online=False,
                  plot=True)

if out is not None:
    choices, tops, reward = out
else:
    import sys
    sys.exit(0)

record = {"Model": [choices, tops, reward]}

# run the other models
names = ("Thompson sampling",
         "Epsilon greedy",
         "UCB1")
models = (
    mm.ThompsonSampling(K=K),
    mm.EpsilonGreedy(K=K, epsilon=0.1),
    mm.UCB1(K=K)
)
for name, model in zip(names, models):

    env = make_current_env(env_type, probabilities_set)
    choices, tops, reward = envs.visual_trial(
                        model=model,
                        environment=env,
                        nb_rounds=nb_rounds,
                        nb_trials=nb_trials,
                        t_update=200,
                        style="choice",
                        online=False,
                        plot=False)

    record[name] = [choices, tops, reward]


""" plot """


fig, axs = plt.subplots(4, 1, figsize=(10, 10),
                        sharex=True)

for i, (name, (choices, tops, reward)) in enumerate(record.items()):

    envs.plot_choices(all_choices=choices,
                      all_top=tops,
                      nb_rounds=nb_rounds,
                      nb_trials=nb_trials,
                      K=K,
                      title=f"{name} - $R=${reward:.2f}",
                      ax=axs[i],
                      show=False,
                      xlab=False)

axs[-1].set_xlabel("rounds", fontsize=15)
axs[0].legend()
plt.tight_layout()
plt.show()

