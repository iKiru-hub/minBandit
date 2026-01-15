# minBandit
A minimal rate neural model for solving a stochastic multi-armed bandit problem

The repository is organized as follows:

* **configs/**: YAML files containing hyperparameters for the model and evolutionary search
* **media/**: plots (.svg, .png) generated from the simulations
* **notebooks/**: jupyter notebooks
* **src/**: The core source code, including:
    * `main.py`: principal simulations and visualizations for one or more models on different environments
    * `evo_main.py`: evolution search over defined parameters and conditions
    * `analysis/`: focused experiments to test specific features (entropy, single models, sensitivity)
    * `core/`: backend utils functions and classes
* **requirements.txt**: python dependencies, run ```pip install -r requirements.txt``` to install them.

---
