import wandb
import numpy as np
from tqdm import tqdm

import models
import utils
import envs

import argparse

import sys

logger = utils.setup_logger(__name__)


""" sweep settings """

# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "param_search minb",
    "metric": {
        "goal": "minimize",
        "name": "regret"
    },

    "parameters": {
        "gain": {"distribution": "uniform",
                  "min": 0.01,
                  "max": 50.},
        "threshold": {"distribution": "uniform",
                 "min": 0.1,
                 "max": 10.},
        "alpha": {"distribution": "uniform",
                  "min": -3,
                  "max": 3},
        "beta": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 10},
        "mu": {"distribution": "uniform",
                  "min": -3,
                  "max": 3},
        "sigma": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 10},
        "r": {"distribution": "uniform",
                  "min": 0.,
                  "max": 1.},
        "sigma": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 10},
        "alpha_lr": {"distribution": "uniform",
                  "min": -3,
                  "max": 3},
        "beta_lr": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 10},
        "mu_lr": {"distribution": "uniform",
                  "min": -3,
                  "max": 3},
        "sigma_lr": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 10},
        "r_lr": {"distribution": "uniform",
                  "min": 0.,
                  "max": 1.},
        "lr": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 10.},
        "w_max": {"distribution": "uniform",
                  "min": 0.1,
                  "max": 9.},
    }
}


""" data and model settings """

# define the environment
def make_env(K: int,
             env_type: str,
             probabilities_set: list,
             tau: int,
             normalize=True):

    if env_type == "driftv0":
        env = envs.KABdriftv0(K=K,
                              probabilities_set=probabilities_set,
                              verbose=False,
                              tau=tau)
    elif env_type == "driftv1":
        env = envs.KABdriftv1(K=K,
                              verbose=False,
                              tau=tau,
                              normalize=normalize,
                              fixed_p=0.9)
    elif env_type == "sinv0":
        frequencies = np.linspace(0.1, 0.4, K)
        phases = np.random.uniform(0, 2*np.pi, K)
        env = envs.KABsinv0(K=K,
                            frequencies=frequencies,
                            phases=phases,
                            normalize=normalize,
                            verbose=False)
    else:
        env = envs.KABv0(K=K,
                         probabilities_set=probabilities_set,
                         verbose=False)

    return env


def train_model(model_params: dict,
                envs_list: list,
                nb_trials: int,
                nb_rounds: int,
                nb_reps: int) -> float:

    """
    train a model over multiple environments
    and return the average regret
    """

    regret = 0.

    for _ in range(nb_reps):
        for env in envs_list:
            model = models.Model(**model_params)
            results = envs.trial(model=model,
                                 environment=env,
                                 nb_trials=nb_trials,
                                 nb_rounds=nb_rounds,
                                 verbose=False,
                                 disable=True)
            regret += envs.parse_results_to_regret(results=results)

    return regret / (len(envs_list) * nb_reps)


""" training """

def main():

    logger("<<< ---------------------- >>>")

    # --- settings ---
    K = 100
    nb_trials = 2
    nb_rounds = 400
    nb_reps = 2

    # --- environment ---
    probabilities_set = utils.make_probability_set(K=K,
                                                   nb_trials=nb_trials,
                                                   fixed_p=0.9,
                                                   normalize=False)
    envs_list = [
            make_env(K=K,
                     env_type="v0",
                     probabilities_set=probabilities_set,
                     tau=None),
            # make_env(K=K,
            #          env_type="driftv0",
            #          probabilities_set=probabilities_set,
            #          tau=10),
            # make_env(K=K,
            #          kind="driftv1",
            #          probabilities_set=probabilities_set,
            #          tau=100),
            # make_env(K=K,
            #          env_type="sinv0",
            #          probabilities_set=probabilities_set,
                     # tau=None),
    ]

    names = ""
    for env in envs_list:
        names += f"`{env}`"

    logger(f"environment list: {names}")
    logger("<<< ---------------------- >>>")

    # --- wandb sweep ---
    run = wandb.init()

    # make model parameters
    model_params = {
        "K": K,
        "dur_pre": 2000,
        "dur_post": 2000,
        "lr": wandb.config.lr,
        "gain": wandb.config.gain,
        "threshold": wandb.config.threshold,
        "alpha": wandb.config.alpha,
        "beta": wandb.config.beta,
        "mu": wandb.config.mu,
        "sigma": wandb.config.sigma,
        "r": wandb.config.r,
        "alpha_lr": wandb.config.alpha_lr,
        "beta_lr": wandb.config.beta_lr,
        "mu_lr": wandb.config.mu_lr,
        "sigma_lr": wandb.config.sigma_lr,
        "r_lr": wandb.config.r_lr,
        "w_max": wandb.config.w_max,
        "value_function": "gaussian",
        "lr_function": "gaussian",
    }

    # model
    regret = train_model(model_params=model_params,
                         envs_list=envs_list,
                         nb_trials=nb_trials,
                         nb_rounds=nb_rounds,
                          nb_reps=nb_reps)

    logger(f"regret: {regret:.3f}")
    wandb.log({"regret": regret})



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="param search for minBandit model")
    parser.add_argument('--count', type=int,
                        help='number of iterations',
                        default=10)
    args = parser.parse_args()

    # Initialize sweep by passing in config.
    # (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           project="minBandit")

    logger.info(f"%sweep id: {sweep_id}")
    logger.info(f"%count: {args.count}")

    # Start sweep job.
    wandb.agent(sweep_id,
                function=main,
                count=args.count)



