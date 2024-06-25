# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import src.utils as utils
from tqdm import tqdm
# os.chdir("minBandit/")
print(os.getcwd())
import main
from scipy.ndimage import convolve1d

# %%
class Settings:

    verbose = False
    rounds = 2
    trials = 2
    reps = 10
    K = 10
    model = None
    load = True
    plot = False
    env = "simple"
    multiple = 1
    visual = False
    save = True
    idx = 4


# %% [markdown]
# -------------------------
# ## **Simple**

# %%
""" settings """
settings1 = Settings()
settings1.rounds = 200
settings1.trials = 3
settings1.reps = 10
settings1.verbose = True

# %%
""" run """
record = {}

Ks = [5, 15, 45, 95]
for i, k in tqdm(zip(range(len(Ks)), Ks)):
    settings1.K = k
    results = main.main_multiple(args=settings1)
    record[str(i)] = results

print("done")

# %%

def plot(idx, color, m=0, theta=0.0125):

    # array of mean reward for each round over reps for a given trial
    z = record[f'{m}']['reward_list'][idx][:, 0].mean(axis=0)
    name = record['0']['names'][idx]

    # steady-state reward : last ~50
    mu_ref = z[-50:].mean()

    # residuals
    res = (z - mu_ref)**2

    # 1d convolution over y
    kc = 80
    y = convolve1d(res, np.ones(kc), mode="constant")/kc

    # point of stability
    cy_idx = np.where(y<theta)[0].min()
    # plt.axvline(cy_idx, color=color, alpha=0.5, linestyle='-.',
    #             label=f"{cy_idx}")

    plt.plot(convolve1d(z, np.ones(kc), mode="constant")/kc,
             color=color,
             alpha=0.5, label=f"{name} residuals")
    # plt.plot(y, alpha=0.5, linestyle='--', color=color,
    #          label=f"{name} rewards")

theta = 0.01
plt.axhline(y=theta, color='red', alpha=0.5)
plot(idx=2, color='blue')
plot(idx=3, color='green')
plt.legend(loc="lower right")
plt.show()


# %%
""" make the plot over different `K` """

def calc_stability_points(record, idx):

    points = []

    for _, k_record in record.items():

        # array of mean reward for each round over reps for a given trial
        z = k_record['reward_list'][idx][:, 0].mean(axis=0)
        name = k_record['names'][idx]

        # steady-state reward : last ~50
        mu_ref = z[-50:].mean()

        # residuals
        res = (z - mu_ref)**2

        # 1d convolution over y
        kc = 50
        y = convolve1d(res, np.ones(kc), mode="constant")/kc

        # point of stability
        p = np.where(y<0.015)[0]
        if len(p) == 0:
            points += [200]
        else:
            points += [p.min()]

    return points, name


# %%
""" make plot """

colors = plt.cm.tab10(range(4))
for i in range(4):

    points, name = calc_stability_points(record, idx=i)

    plt.plot(points, '-o', color=colors[i], label=name)

# plt.yticks(np.arange(0, 1, 0.01), ())
plt.legend(loc="upper left")
plt.xticks(range(len(rounds)), rounds)
plt.grid()
plt.show()

# %%



# %% [markdown]
# -------------------------
# ## **Smooth**

# %%
""" settings """
settings2 = Settings()
settings2.rounds = 1
settings2.trials = 600
settings2.reps = 2
settings2.K = 10

# %%
""" run """
record2 = {}

rounds = [1, 2, 3, 5, 10]
for i, k in tqdm(zip(range(len(rounds)), rounds)):
    print(f"{i=} {k=}")
    settings2.rounds = k
    results2 = main.main_multiple(args=settings2)
    record2[str(i)] = results2

print("done")

# %%
record.keys()



