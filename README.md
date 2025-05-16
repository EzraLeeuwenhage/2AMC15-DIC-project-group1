Adjusted readme.md for our project. Note that it is important to have a working environment with all dependencies. 

## Folder Structure

The project is organized as follows:

- **agent/**: All RL agent classes and logic.
- **world/**: Code for the environment, grid, GUI, and grid creation tools.
- **utils/**: Helper functions for training, evaluation, plotting, and custom rewards.
- **grid_configs/**: Stores grid files used for experiments.
- **train_main.py**: Main entry point for training a specific agent on a specific grid with specific environment and experiment settings.
- **train_logic/**: Contains the logic for training individual algorithms.
- **run_experiments.py**: Define a series of experiment settings (algorithm, environment, and experiment settings) and call train_main logic to run each experiment.
- **experimental_results/**: Stores results generated for the final report and experiments.
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
            ├── numerical_results.csv
            ├── cumulative_reward.png
            └── policy_heatmap.png

Each experiment generates the following output in the output folder:
- `numerical_results.csv`: Summary of experiment results and metrics.
- `cumulative_reward.png`: Visualization of the cumulative reward (AUC) plot.
- `policy_heatmap.png`: The visualized agent policy enacted on the grid.

You can find all outputs for your experiments in the corresponding subfolder under `experimental_results/`.
