from train_main import main
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.reward_functions import custom_reward_function


# TODO: create experiments file 


def run_evaluation(
        # Specific to the experiment
        # Enironment-specific
        grid="grid_configs/A1_grid.npy", sigma=0, gamma=0.9, reward_func=None, agent_start_pos_col=9, agent_start_pos_row=13,
        # Agent-specific
        algorithm="q_learning", epsilon=0.15, epsilon_min=0.1, epsilon_decay_rate=0, delta=1e-6, alpha=0.01, episodes=15000, iters=500, early_stopping=-1,
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

    for repitition in range(number_of_repititions):
        random_seed_run = random_seed_full_experiment + repitition

        cumulative_rewards_for_episode = main(
            # Specific to the experiment
            grid=grid, algorithm=algorithm, agent_start_pos_col=agent_start_pos_col, 
            agent_start_pos_row=agent_start_pos_row, sigma=sigma, reward_func=reward_func, 
            gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, epsilon_min=epsilon_min,
            episodes=episodes, iters=iters, 
            # General across experiments
            random_seed=random_seed_run, no_gui=True, n_eps_gui=-1, fps=30, output_plots=False
        )
        
        cumulative_rewards_for_all_episodes.append(cumulative_rewards_for_episode)
        # Convert to numpy for faster computations
        cumulative_rewards_for_all_episodes = np.array(cumulative_rewards_for_all_episodes)

        # Compute per episode mean to get the expected cumulative reward
        expected_cumulative_reward = np.mean(cumulative_rewards_for_all_episodes, axis=0) 

        # Compute lower confidence bound. For this to be reliable we need to repeat the experiment a reasonable number of times (e.g. Repititions > 30)
        z = 1.96
        std = np.std(cumulative_rewards_for_all_episodes, axis=0, ddof=1)
        half_width_interval = (z * std / np.sqrt(expected_cumulative_reward.shape[0]))
        lower_bound_cumulative_reward = expected_cumulative_reward - half_width_interval

    episodes = np.arange(1, cumulative_rewards_for_all_episodes.shape[1]+1)
    plt.plot(episodes, expected_cumulative_reward, label="mean")
    plt.fill_between(episodes,
                     lower_bound_cumulative_reward,
                     expected_cumulative_reward + half_width_interval,
                     alpha=0.3,
                     label="95% CI")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()


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
        grid="grid_configs/A1_grid.npy", sigma=0, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
        algorithm="q_learning", epsilon=0.25, epsilon_min=0.01, delta=1e-6, alpha=0.01, episodes=15000, iters=500, early_stopping=-1,
        random_seed_full_experiment=0, number_of_repititions=10
    )
        

