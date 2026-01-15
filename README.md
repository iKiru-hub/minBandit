# minBandit
A minimal rate neural model for solving a stochastic multi-armed bandit problem

The repository is organized as follows:

* **configs/**: YAML files containing hyperparameters for the model and evolutionary search
* **media/**: plots (.svg, .png) generated from the simulations
* **notebooks/**: jupyter notebooks
* **src/**: The core source code, including:
    * `main.py`: principal simulations and visualizations for one or more models on different environments
    * `evo_main.py`: evolution search over defined parameters and conditions, possibly using multiple cores
    * `analysis/`: focused experiments to test specific features (entropy, single models, sensitivity)
    * `core/`: backend utils functions and classes


**Python dependancies**
In order to have the right necessary packaged, assuming ```pip``` is installed, run the command ```pip install -r requirements.txt```

**Example usage**
For a quick simulation:
```python3 main.py --rounds 1000 --trials 2 --K 100 --multiple 1 --env --verbose```

all models (UCB, ThompsonSampling, EpsilonGreedy, NSA (our model)) are run on a piecewise stationary environment (MAB-D, the default one) for 2 trial 2000 rounds and 100 arms.

All available options for the main simulations can be printed with the command ```python3 main.py --help```
```txt
  -h, --help           show this help message and exit
  --verbose            verbose
  --rounds ROUNDS      number of rounds in a trial
  --trials TRIALS      number of trials
  --reps REPS          number of repetitions
  --K K                number of arms of the bandit
  --model MODEL        model to run: `ucb`, `thompson`, `epsilon`; if nothing specified or wrong name, the default
                       is the custom `model`
  --load               load saved model from evolution search
  --plot               plot at the end of the simulation
  --show               whether to show the plot or not
  --env ENV            type of environment: `driftv0`, `driftv1`, `sinv0`, or nothing for `v0`
  --style STYLE        style of online plotting: `choice`, `3d`, `2d`,, `tape`
  --multiple MULTIPLE  run multiple models: 0 (single) or 1 (multiple)
  --visual             visualize the trial
  --save               save the results in a folder
  --idx IDX            the index of the model to load, default 5 (good parameters)
```
 

