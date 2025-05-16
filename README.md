
Welcome to Data Intelligence Challenge-2AMC15!
This is the repository containing the challenge environment code.

## Quickstart

1. Create a virtual environment for this course with Python >= 3.10. Using conda, you can do: `conda create -n dic2025 python=3.11`. Use `conda activate dic2025` to activate it `conda deactivate` to deactivate it.
2. Clone this repository into the local directory you prefer `git clone https://github.com/DataIntelligenceChallenge/2AMC15-2025.git`.
3. Install the required packages `pip install -r requirements.txt`. Now, you are ready to use the simulation environment! :partying_face:	
4. Run `$ python train.py grid_configs/example_grid.npy` to start training!

`train.py` is just an example training script. Inside this file, initialize the agent you want to train and evaluate. Feel free to modify it as necessary. Its usage is:

```bash
usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] 
                GRID [GRID ...]

DIC Reinforcement Learning Trainer.

positional arguments:
  GRID                  Paths to the grid file to use. There can be more than
                        one.
options:
  -h, --help                 show this help message and exit
  --no_gui                   Disables rendering to train faster (boolean)
  --sigma SIGMA              Sigma value for the stochasticity of the environment. (float, default=0.1, should be in [0, 1])
  --fps FPS                  Frames per second to render at. Only used if no_gui is not set. (int, default=30)
  --iter ITER                Number of iterations to go through. Should be integer. (int, default=1000)
  --random_seed RANDOM_SEED  Random seed value for the environment. (int, default=0)
```

## Code guide

The code is made up of 2 modules: 

1. `agent`
2. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents you may want to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritence, skip the following section:

#### `BaseAgent` as an abstract class
Here you can find an explanation about abstract classes [Geeks for Geeks](https://www.geeksforgeeks.org/abstract-classes-in-python/).

Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `update()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `world` module

The world module contains:
1. `grid_creator.py`
2. `environment.py`
3. `grid.py`
4. `gui.py`

#### Grid creator
Run this file to create new grids.

```bash
$ python grid_creator.py
```

This will start up a web server where you create new grids, of different sizes with various elements arrangements.
To view the grid creator itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.


#### The Environment

The `Environment` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment
- `Environment().evaluate_agent()` to evaluate the agent after training.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the the actual representation of the world on which the agent moves. It is a 2D Numpy array.

#### The GUI

The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.

## Folder Structure

The project is organized as follows:

- **agent/**: All RL agent classes and logic.
- **world/**: Code for the environment, grid, GUI, and grid creation tools.
- **utils/**: Helper functions for training, evaluation, plotting, and custom rewards.
- **grid_configs/**: Stores grid files used for experiments.
- **train_main.py**: Main entry point for training a specific agent on a specific grid with specific environment and experiment settings.
- **train_logic/**: Contains the logic for training individual algorithms.
- **run_experiments.py**: Define a series of experiment settings (algorithm, environment, and experiment settings) and call train_main logic to run each experiment.
- **experimental_results_2/**: Stores results generated for the final report and experiments.
- **deprecated/**: Deprecated scripts and code no longer in active use.
- **requirements.txt**: Lists required Python packages.
- **README.md**: Project documentation and usage instructions.


## How to Run Experiments

To run experiments, you should use the `run_experiments.py` script. This script allows you to define a series of experiments by specifying all relevant environment, agent, and experiment settings in a structured way. Each experiment is configured by calling the `run_evaluation` function with your desired parameters.

### Example Experiment Definition

Below is an example of how to define an experiment in `run_experiments.py`:

```python
run_evaluation(
    # Environment-specific settings
    grid=grid_A1,
    sigma=0.02,
    gamma=0.9,
    reward_func=custom_reward_function,
    agent_start_pos_col=9,
    agent_start_pos_row=13,
    # Agent-specific settings
    algorithm="q_learning",
    epsilon=1,
    epsilon_min=0.1,
    delta=1e-6,
    alpha=0.1,
    episodes=15000,
    iters=500,
    early_stopping=-1,
    # General experimental setup
    random_seed_full_experiment=0,
    number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results/experiment_1/grid_A1/q_learning/"
)
```

## Environment-specific

- Define the grid, stochasticity (sigma), discount factor (gamma), reward function, and agent starting position.

## Agent-specific

- Choose the algorithm and its hyperparameters.

## General setup

- Set the random seed and number of repetitions for statistical robustness.

## experiment_path

- Specify where the results for this experiment will be saved. This should be a unique path for each experiment.

## Running the Experiments

1. Open `run_experiments.py` and define your experiments using the format above.
2. Save the file.
3. Run the experiments from the terminal:
   ```bash
   python run_experiments.py
   ```

## Output Structure

All experiment results are saved in the `experimental_results/` directory. The structure will look like this:

experimental_results/
└── experiment_1/
    └── grid_A1/
        └── q_learning/
            ├── results.json
            ├── plots/
            │   ├── reward_curve.png
            │   └── q_value_convergence.png
            ├── q_table.npy
            └── config.yaml

Each experiment generates the following output in the output folder:
- `numerical_results.csv`: Summary of experiment results and metrics.
- `cumulative_reward.png`: Visualization of the cumulative reward (AUC) plot.
- `policy_heatmap.png`: The visualized agent policy enacted on the grid.

You can find all outputs for your experiments in the corresponding subfolder under `experimental_results/`.
