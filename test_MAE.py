from train_main import main
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.reward_functions import custom_reward_function
import csv
from evaluation_metrics import run_evaluation
from utils.reward_functions import custom_reward_function

grid_A1 = "grid_configs/A1_grid.npy"

sigma=0.02
gamma=0.9
reward_func=custom_reward_function
agent_start_pos_col=9
agent_start_pos_row=13
# Agent-specific
algorithm="q_learning"
epsilon=1
epsilon_min=0.1
delta=1e-6
alpha=0.1
episodes=15000
iters=500
early_stopping=-1,
# General to experimental setup
random_seed_full_experiment=0
# Saving results

cumulative_rewards_for_episode, q_table, trained_agent, grid_ = main(
    # Specific to the experiment
    grid=[grid_A1], algorithm=algorithm, agent_start_pos_col=agent_start_pos_col,
    agent_start_pos_row=agent_start_pos_row, sigma=sigma, reward_func=reward_func,
    gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, epsilon_min=epsilon_min,
    episodes=episodes, iters=iters, early_stopping=early_stopping,
    # General across experiments
    random_seed=random_seed_full_experiment, no_gui=True, n_eps_gui=-1, fps=30, output_plots=False
)

grid_A1 = "grid_configs/A1_grid.npy"

grid=grid_A1
sigma=0.02
gamma=0.9
reward_func=custom_reward_function
agent_start_pos_col=9
agent_start_pos_row=13
# Agent-specific
algorithm="dp"
epsilon=1
epsilon_min=0.1
delta=1e-6
alpha=0.1
episodes=15000
iters=500
early_stopping=-1,
# General to experimental setup
random_seed_full_experiment=0
# Saving results

cumulative_rewards_for_episode_dp, q_table_dp, trained_agent_dp, grid_dp = main(
    # Specific to the experiment
    grid=[grid_A1], algorithm=algorithm, agent_start_pos_col=agent_start_pos_col,
    agent_start_pos_row=agent_start_pos_row, sigma=sigma, reward_func=reward_func,
    gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, epsilon_min=epsilon_min,
    episodes=episodes, iters=iters, early_stopping=early_stopping,
    # General across experiments
    random_seed=random_seed_full_experiment, no_gui=True, n_eps_gui=-1, fps=30, output_plots=False
)

errors = []
for key in q_table_dp.keys():
    action = np.argmax(q_table_dp[key])
    value_vi = np.max(q_table_dp[key])
    try:
        value = q_table[key][action]
    except:
        value = 0
    error = abs(value - value_vi)
    errors.append(error)
print(np.mean(errors))