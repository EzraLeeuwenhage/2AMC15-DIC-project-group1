from tqdm import trange
from agents.value_iteration_agent import ValueIterationAgent
from agents.q_learning_agent import QLearningAgent
from agents.monte_carlo_agent_v2 import MonteCarloAgent
from train_logic.train_DP_logic import train_DP
from train_logic.train_q_learning_logic import train_q_learning
from train_logic.train_mc_v2_logic import train_mc_control
from utils.plots import plot_time_series, plot_policy_heatmap, calc_auc, calc_normalized_auc, extract_VI_agent_optimal_path


def init_agent(algorithm, grid_shape, alpha, gamma, delta):
    match algorithm:
        case 'q_learning':
            return QLearningAgent(grid_shape=grid_shape, actions=[0, 1, 2, 3], alpha=alpha, gamma=gamma)
        case 'mc':
            return MonteCarloAgent(grid_shape=grid_shape, actions=[0, 1, 2, 3], alpha=alpha, gamma=gamma)
        case 'dp':
            return ValueIterationAgent(n_actions=4, gamma=gamma, delta_threshold=delta)


def set_agent_start_pos(column, row, grid_cells):
    if column is None or row is None:
        return None

    position = (column, row)
    assert grid_cells[position] == 0, f"Starting position {position} is not empty in the grid."
    return position


def train_agent(algorithm, agent, env, episodes, iters, delta, epsilon, epsilon_min, n_eps_gui, early_stopping):
    max_diff_list = []
    cumulative_reward_list = []

    for episode in trange(episodes):
        env_gui = episode % n_eps_gui == 0 if n_eps_gui != -1 else False
        state = env.reset(no_gui=not env_gui)

        if algorithm == 'dp':
            agent, _, _, max_diff_list = train_DP(agent, env, max_iterations=1000)
            return agent, max_diff_list, None

        if algorithm == 'q_learning':
            agent, max_diff_list, cumulative_reward_list, flag_break = train_q_learning(agent, state, env, iters, max_diff_list, delta, episode, episodes, epsilon, epsilon_min, early_stopping, cumulative_reward_list)
            if flag_break: 
                break

        if algorithm == 'mc':
            agent, max_diff_list, cumulative_reward_list, flag_break = train_mc_control(agent, state, env, iters, max_diff_list, delta, episode, episodes, epsilon, epsilon_min, cumulative_reward_list)
            if flag_break: 
                break

    return agent, max_diff_list, cumulative_reward_list


def evaluate_and_plot(agent, algorithm, grid, env, max_diff_list, cumulative_reward_list):
    if algorithm == 'dp':
        visit_counts, _ = extract_VI_agent_optimal_path(agent, env)
        plot_policy_heatmap(agent.q_table, visit_counts, grid.cells)

    elif algorithm in ('q_learning', 'mc'):
        print(f'AUC: {calc_auc(cumulative_reward_list)}')
        print(f'Normalized AUC: {calc_normalized_auc(cumulative_reward_list)}')
        agent.epsilon = 0  # Turn off exploration
        plot_time_series(max_diff_list, y_label='Max difference in Q-value', title='Convergence')
        plot_time_series(cumulative_reward_list, y_label='Cumulative reward', title='Reward per Episode')
        plot_policy_heatmap(agent.q_table, agent.visit_counts, grid.cells)
