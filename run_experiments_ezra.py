from evaluation_metrics import run_evaluation
from utils.reward_functions import custom_reward_function

grid_A1 = "grid_configs/A1_grid.npy"
grid_custom = "grid_configs/long_distance_narrow.npy"
number_of_reps = 10


######################### experiments grid A1 ###########################################

# ### Experiment 1 (Baseline) ####
# run_evaluation(
#         # Enironment-specific
#         grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#         # Agent-specific
#         algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#         # General to experimental setup
#         random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#         # Saving results
#         experiment_path="experimental_results_2/experiment_1/grid_A1/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_1/grid_A1/mc_control/"
# )


# ### Experiment 2 (gamma=0.99) #####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.99, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_2/grid_A1/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.99, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_2/grid_A1/mc_control/"
# )


# ### Experiment 3 (sigma=0.2) #####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.2, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_3/grid_A1/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.2, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_3/grid_A1/mc_control/"
# )


# ### Experiment 4 (epsilon=fixed) #####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="q_learning", epsilon=0.2, epsilon_min=0.2, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_4/grid_A1/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="mc", epsilon=0.2, epsilon_min=0.2, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_4/grid_A1/mc_control/"
# )

# ### Experiment 5 (learning_rate q_learning=0.5) #####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.5, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_5/grid_A1/q_learning/"
# )

# ### Experiment 6 (n_episodes = 30000) #####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=30000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_6/grid_A1/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_A1, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=9, agent_start_pos_row=13,
#     # Agent-specific
#     algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=30000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_6/grid_A1/mc_control/"
# )

# ######################### experiments grid custom ###########################################

# ### Experiment 1 (Baseline) ####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
#     # Agent-specific
#     algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_1/grid_custom/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
#     # Agent-specific
#     algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_1/grid_custom/mc_control/"
# )


# ### Experiment 2 (gamma=0.99) #####
# run_evaluation(
#     # Enironment-specific
#     grid=grid_custom, sigma=0.02, gamma=0.99, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
#     # Agent-specific
#     algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_2/grid_custom/q_learning/"
# )
# run_evaluation(
#     # Enironment-specific
#     grid=grid_custom, sigma=0.02, gamma=0.99, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
#     # Agent-specific
#     algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
#     # General to experimental setup
#     random_seed_full_experiment=0, number_of_repititions=number_of_reps,
#     # Saving results
#     experiment_path="experimental_results_2/experiment_2/grid_custom/mc_control/"
# )


### Experiment 3 (sigma=0.2) #####
run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.2, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_3/grid_custom/q_learning/"
)
run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.2, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_3/grid_custom/mc_control/"
)


### Experiment 4 (epsilon=fixed) #####
run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="q_learning", epsilon=0.2, epsilon_min=0.2, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_4/grid_custom/q_learning/"
)
run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="mc", epsilon=0.2, epsilon_min=0.2, delta=1e-6, alpha=0.1, episodes=15000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_4/grid_custom/mc_control/"
)

### Experiment 5 (learning_rate q_learning=0.5) #####
run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.5, episodes=15000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_5/grid_custom/q_learning/"
)

### Experiment 6 (n_episodes = 30000) #####
run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="q_learning", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=30000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_6/grid_custom/q_learning/"
)

run_evaluation(
    # Enironment-specific
    grid=grid_custom, sigma=0.02, gamma=0.9, reward_func=custom_reward_function, agent_start_pos_col=1, agent_start_pos_row=18,
    # Agent-specific
    algorithm="mc", epsilon=1, epsilon_min=0.1, delta=1e-6, alpha=0.1, episodes=30000, iters=500, early_stopping=-1,
    # General to experimental setup
    random_seed_full_experiment=0, number_of_repititions=number_of_reps,
    # Saving results
    experiment_path="experimental_results_2/experiment_6/grid_custom/mc_control/"
)
