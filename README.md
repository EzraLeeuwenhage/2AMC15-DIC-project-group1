Adjusted readme.md for our project. Note that it is important to have a working environment with all dependencies as noted in the requirements.txt. 

## Folder Structure

The project is organized as follows:

- **agent/**: All RL agent classes and logic.
- **world/**: Code for the environment, grid, GUI, and grid creation tools.
- **utils/**: Helper functions for training, evaluation, plotting, and custom rewards.
- **grid_configs/**: Stores grid files used for experiments.
- **train_logic/**: Contains the logic for training individual algorithms.
- **experimental_results/**: Stores results generated for the final report and experiments.
- **deprecated/**: Deprecated scripts and code no longer in active use.
- **run_experiments.py**: Define a series of experiment settings (algorithm, environment, and experiment settings) and call evaluate_metrics.py logic to run each experiment.
- **train_main.py**: Main entry point for training a specific agent on a specific grid with specific environment and experiment settings.
- **evaluation_metrics.py**: Basically a wrapper for the train_main.py file, which allows us to run all evaluation metrics for the experiments. 
- **requirements.txt**: Lists required Python packages.
- **README.md**: Project documentation and usage instructions.

## How to Run an Agent

Training a single agent is efficient, it will be done in a couple of minutes maximum. For this we need to call the train_main.py file. This is easiest to do from the command prompt, and to give it the necessary arguments. 

Example command: ``python train_main.py grid_configs/A1_grid.npy --episodes=15000 --algorithm=q_learning --n_eps_gui=5000 --iter=1000 --agent_start_pos_col=9 --agent_start_pos_row=13``

For a full list of command options see the table below:

| Flag                         | Type      | Default                     | Description                                                                                                   |
|------------------------------|-----------|-----------------------------|---------------------------------------------------------------------------------------------------------------|
| `GRID`                       | `Path`    | —                           | One or more grid file(s) to use for training (e.g. `grid_configs/A1_grid.npy`).                               |
| `--algorithm`                | `str`     | `None`                      | Which RL algorithm to train (e.g. `q_learning`, `mc`, `dp`).                                                  |
| `--agent_start_pos_col`      | `int`     | `9`                         | Starting column of the agent in the GUI. If `None`, starts at a random position.                              |
| `--agent_start_pos_row`      | `int`     | `10`                        | Starting row of the agent in the GUI. If `None`, starts at a random position.                                 |
| `--sigma`                    | `float`   | `0.1`                       | Slip probability in the environment dynamics.                                                                 |
| `--reward_func`              | `callable`| `custom_reward_function`    | Reward function for the environment (must be a Python callable).                                              |
| `--fps`                      | `int`     | `30`                        | Frames per second for the GUI rendering.                                                                      |
| `--random_seed`              | `int`     | `0`                         | Random seed for reproducibility.                                                                              |
| `--gamma`                    | `float`   | `0.9`                       | Discount factor for future rewards.                                                                           |
| `--delta`                    | `float`   | `1e-6`                      | Threshold for Q-value updates for early stopping (convergence criterion).                                     |
| `--alpha`                    | `float`   | `0.1`                       | Learning rate (for Q-learning algorithms).                                                                    |
| `--epsilon`                  | `float`   | `1.0`                       | Starting ε value for ε-greedy exploration.                                                                    |
| `--epsilon_min`              | `float`   | `0.1`                       | Minimum ε value after decay.                                                                                  |
| `--episodes`                 | `int`     | `10000`                     | Total number of training episodes.                                                                            |
| `--iters`                    | `int`     | `1000`                      | Maximum number of iterations (steps) per episode.                                                             |
| `--early_stopping`           | `int`     | `-1`                        | Number of episodes with no improvement to wait before stopping early (`-1` = disabled).                       |
| `--n_eps_gui`                | `int`     | `100`                       | Show the GUI every N episodes, -1 if you want to disable the GUI completely                                   |
| `--output_plots`             | `boolean` | `False`                     | Show plots convergence plot and policy heatmap at the end of training.                                        |

## How to Run Experiments

To run experiments, you should use the `run_experiments.py` script. This script allows you to define a series of experiments by specifying all relevant environment, agent, and experiment settings in a structured way. Each experiment is configured by calling the `run_evaluation` function with your desired parameters. No further command line arguments are necessary, as it is defined in the code. Note that the full set of experiments might take hours to complete.

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

#### Environment-specific

- Define the grid, stochasticity (sigma), discount factor (gamma), reward function, and agent starting position.

#### Agent-specific

- Choose the algorithm and its hyperparameters.

#### General setup

- Set the random seed and number of repetitions for statistical robustness.

#### Saving results

- Specify where the results for this experiment will be saved. This should be a unique path for each experiment.

## Running the Experiments

1. Open `run_experiments.py` and define your experiments using the format above.
2. Save the file (if you made changes).
3. Run the experiments from the terminal or directly from code editor:
   ```bash
   python run_experiments.py
   ```

## Output Structure

All experiment results are saved in the `experimental_results/` directory. The structure will look like this:

experimental_results/
└── experiment_1/
    └── grid_A1/
        └── q_learning/
            ├── numerical_results.csv
            ├── cumulative_reward.png
            └── policy_heatmap.png

Each experiment generates the following output in the output folder:
- `numerical_results.csv`: Summary of experiment results and metrics.
- `cumulative_reward.png`: Visualization of the cumulative reward (AUC) plot.
- `policy_heatmap.png`: The visualized agent policy enacted on the grid.

You can find all outputs for your experiments in the corresponding subfolder under `experimental_results/`.
