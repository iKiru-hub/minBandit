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
K = 10
nb_rounds = 300
nb_trials = 3
env_type = "simple"
verbose = True

""" environment """
# define proababilities set
probabilities_set = utils.make_probability_set(K=K,
                                               nb_trials=nb_trials,
                                               fixed_p=0.9,
                                               normalize=False)
# define the environment
def make_current_env(kind: str, probabilities_set: list):

    if env_type == "simple":
        env = envs.KArmedBandit(K=K,
                                probabilities_set=probabilities_set,
                                verbose=False)
    elif env_type == "smooth":
        env = envs.KArmedBanditSmooth(K=K,
                                probabilities_set=probabilities_set,
                                verbose=False,
                                tau=5)
    else:
        env = envs.KArmedBanditSmoothII(K=K,
                                verbose=False,
                                tau=100,
                                fix_p=0.9)

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
choices, tops, reward = envs.visual_trial(
                  model=model,
                  environment=env,
                  nb_rounds=nb_rounds,
                  nb_trials=nb_trials,
                  t_update=200,
                  style="choice",
                  online=False,
                  plot=False)

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

