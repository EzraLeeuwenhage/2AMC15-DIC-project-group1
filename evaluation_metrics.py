from train_main import main
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.reward_functions import custom_reward_function
import csv
from utils.plots import plot_policy_heatmap


def calc_normalized_auc(reward_list, min_reward, max_reward):
    # area above the min‐baseline
    area_above_min = np.sum(reward_list - min_reward)
    # total possible area above the min‐baseline
    total_area = (max_reward - min_reward) * len(reward_list)
    # the fraction you want
    auc_normalized = area_above_min / total_area
    return auc_normalized


def run_evaluation(
        # Specific to the experiment
        # Enironment-specific
        grid="grid_configs/A1_grid.npy", sigma=0, gamma=0.9, reward_func=None, agent_start_pos_col=9, agent_start_pos_row=13,
        # Agent-specific
        algorithm="q_learning", epsilon=0.15, epsilon_min=0.1, delta=1e-6, alpha=0.01, episodes=15000, iters=500, early_stopping=-1,
        # General to experimental setup
        random_seed_full_experiment=0, number_of_repititions=100,
        # Saving results
        experiment_path="experimental_results"
):
    """
    Alternative reward function is custom_reward_func
    decay rate = 0 gives fixed epsilon
    """
    grid = [grid]  # Needs to be supplied as list
    cumulative_rewards_for_all_episodes = []
    print(experiment_path)

    # Obtain VI Q-table for MAE comparison
    _, vi_q_table, _, _ = main(
        # Specific to the experiment
        grid=grid, algorithm="dp", agent_start_pos_col=agent_start_pos_col, 
        agent_start_pos_row=agent_start_pos_row, sigma=sigma, reward_func=reward_func, 
        gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, epsilon_min=epsilon_min,
        episodes=episodes, iters=iters, early_stopping=early_stopping,
        # General across experiments
        random_seed=random_seed_full_experiment, no_gui=True, n_eps_gui=-1, fps=30, output_plots=False
    )

    for repitition in range(number_of_repititions):
        random_seed_run = random_seed_full_experiment + repitition

        cumulative_rewards_for_episode, q_table, trained_agent, grid_ = main(
            # Specific to the experiment
            grid=grid, algorithm=algorithm, agent_start_pos_col=agent_start_pos_col, 
            agent_start_pos_row=agent_start_pos_row, sigma=sigma, reward_func=reward_func, 
            gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, epsilon_min=epsilon_min,
            episodes=episodes, iters=iters, early_stopping=early_stopping,
            # General across experiments
            random_seed=random_seed_run, no_gui=True, n_eps_gui=-1, fps=30, output_plots=False
        )
        
        cumulative_rewards_for_all_episodes.append(cumulative_rewards_for_episode)

    # Convert to numpy for faster computations
    cumulative_rewards_for_all_episodes = np.array(cumulative_rewards_for_all_episodes)

    # Compute per episode mean to get the expected cumulative reward
    expected_cumulative_reward = np.mean(cumulative_rewards_for_all_episodes, axis=0) 

    # Compute lower confidence bound.
    z = 1.96
    std = np.std(cumulative_rewards_for_all_episodes, axis=0, ddof=1)
    half_width_interval = (z * std / np.sqrt(expected_cumulative_reward.shape[0]))
    lower_bound_cumulative_reward = expected_cumulative_reward - half_width_interval

    # Minimal reward (or biggest cost is -5) --> minimal reward that could be obtained is -5 * number of steps per episode
    minimal_reward = -5 * iters
    # Maximal reward can be analytically computed for a given grid: maximal_reward = number of steps to reach the target * -1 + reward for reaching the target
    if grid[0] == "grid_configs/A1_grid.npy":
        grid_name = "A1 grid"
        maximal_reward = (22 * -1) + 1000
    else:
        grid_name = "Custom grid"
        maximal_reward = (34 * -1) + 1000
        
    exp_n = experiment_path.split("/")[1].split("_")[1]
    exp_name = f"E{exp_n}"
    suffix = f"{exp_name} on {grid_name}"
    # Create Q-learning abbreviation
    if algorithm == "q_learning":
        algorithm = "QL"
    # Uppercase algorithm name, mainly for MC and DP
    algorithm = algorithm.upper()

    # Creating cumulative reward plot
    episodes = np.arange(1, cumulative_rewards_for_all_episodes.shape[1]+1)
    plt.plot(episodes, expected_cumulative_reward, label="Expected cumulative reward", color='blue', alpha=0.5, linewidth=1)
    plt.plot(episodes, lower_bound_cumulative_reward, label="LCB cumulative reward", color='red', alpha=0.5, linewidth=1)
    # plot theoretical maximum reward as dotted horizontal line
    plt.axhline(y=maximal_reward, color='green', linestyle='--', label="Theoretical maximum reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title(f"Cumulative reward per episode for {algorithm} agent - {suffix}")
    plt.legend()
    plt.savefig(experiment_path + "cumulative_reward.png", dpi=600)

    # Compute summarizing statistics
    auc_exp = calc_normalized_auc(expected_cumulative_reward, minimal_reward, maximal_reward)
    auc_low = calc_normalized_auc(lower_bound_cumulative_reward, minimal_reward, maximal_reward)
    max_exp = np.max(expected_cumulative_reward)
    max_low = np.max(lower_bound_cumulative_reward)

    # Saving all statistics to a CSV file
    fieldnames = ['auc_exp', 'auc_low', 'max_exp', 'max_low', 'maximal_reward']
    values = [auc_exp,  auc_low,  max_exp,  max_low,  maximal_reward]
    file_for_results = experiment_path + "numerical_results.csv"
    with open(file_for_results, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)  # header
        writer.writerow(values)      # one row of values

    # We also want to plot the policy heatmap plot and compare the values of the Q-table agents to the values via VI. 
    # Policy plot
    title = f"Policy heatmap for {algorithm} agent - {suffix}"
    image = plot_policy_heatmap(q_table, trained_agent.visit_counts, grid_.cells, title, show_image=False)
    image.savefig(experiment_path + "policy_heatmap.png", dpi=600)
    # Optimal policy? Make sure that we check Q-table optimal policy on all positions of optimal path
    # TODO: VI different approach --> how do we even evaluate this?
    # Q-table comparison? with MAE?



if __name__ == "__main__":

    # Get the name of the current directory
    current_dir_name = os.path.basename(os.getcwd())
    print(current_dir_name)

    if current_dir_name != '2AMC15-DIC-project-group1':
        try:
            os.chdir("2AMC15-DIC-project-group1")
            print(f"Changed directory to 2AMC15-DIC-project-group1")
        except OSError as e:
            print(f"Error: could not change to 2AMC15-DIC-project-group1: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Already in 'test' directory.")

    run_evaluation(
        # Enironment-specific
        grid="grid_configs/A1_grid.npy", sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
        # Agent-specific
        algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
        # General to experimental setup
        random_seed_full_experiment=0, number_of_repititions=2,
        # Saving results
        experiment_path="experimental_results/experiment_1/grid_A1/q_learning/"
    )
        

