"""
Train your RL Agent in this file. 
"""
from agents.value_iteration_agent import ValueIterationAgent
from agents.q_learning_agent import QLearningAgent
from agents.monte_carlo_agent import MonteCarloAgent
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from tqdm import trange
import numpy as np
from world.grid import Grid  # Import the Grid class
from utils.plots import plot_time_series, plot_policy_heatmap, plot_V, calc_auc, calc_normilized_auc
from train_q_learning_logic import train_q_learning
from train_mc_logic import train_mc2
from train_DP_logic import train_DP
try:
    from world import Environment
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent

def parse_args():
    p = ArgumentParser(description="Monte Carlo RL Trainer.")
    p.add_argument("GRID",        type=Path, nargs="+",
                   help="Grid file(s) to use for training.")
    p.add_argument("--algorithm", type=str,
                   help="algortihm to train.")
    p.add_argument("--no_gui",    action="store_true",
                   help="Disable rendering to train faster.")
    p.add_argument("--sigma",     type=float, default=0.1,
                   help="Slip probability in the environment.")
    p.add_argument("--fps",       type=int,   default=30,
                   help="Frames per second for GUI.")
    p.add_argument("--episodes",  type=int,   default=10000,
                   help="Number of training episodes.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    p.add_argument("--epsilon",      type=float, default=1.0,
                   help="Starting epsilon for ε-greedy.")
    p.add_argument("--epsilon_min",  type=float, default=0.1,
                   help="Minimum epsilon after decay.")
    p.add_argument("--decay_rate",   type=float, default=0.01,
                   help="Decay rate for exponential epsilon.")
    p.add_argument("--gamma",      type=float, default=0.9,
                   help="Discount factor.")
    p.add_argument("--eval_steps", type=int,   default=200,
                   help="Steps for final evaluation.")
    p.add_argument("--n_eps_gui", type=int, default=100,
                   help="percetage of GUI onnn of episodes to enable GUI (e.g., every N episodes).")
    p.add_argument("--delta", type=int, default=1e-6,
                   help="A threshold for the Q-value updates for early stopping")
    return p.parse_args()


def main(grid_paths, algorithm, no_gui, sigma, fps, episodes, iters, random_seed,
         epsilon, epsilon_min, decay_rate, gamma, eval_steps, n_eps_gui, delta):
    """Main loop of the program."""

    for grid in grid_paths:  # not yet used, because Q-learning per grid world

        max_diff_list = []  # For tracking convergence and convergence plot
        cumulative_reward_list = []
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed)
        grid_ = Grid.load_grid(grid)

        if algorithm=='q_learning':
            # Initialize agent
            agent = QLearningAgent(grid_shape=(grid_.n_rows, grid_.n_cols))
        if algorithm == 'mc':
            agent = MonteCarloAgent(n_actions=4,
                                    epsilon=epsilon,
                                    gamma=gamma)
        if algorithm=='dp':
            agent = ValueIterationAgent(n_actions=4, gamma=gamma, delta_threshold=delta)

        for episode in trange(episodes):
            #print(episode)
            if n_eps_gui == -1:
                no_gui = True
            elif episode % n_eps_gui == 0:
                no_gui = False
            else:
                no_gui = True

            # Set up the environment
            state = env.reset(no_gui=no_gui)

            if algorithm=='q_learning':
                agent, max_diff_list, cumulative_reward_list, flag_break = train_q_learning(agent, state, env, iters, max_diff_list, delta, episode, cumulative_reward_list)
                if flag_break:

                    break

            if algorithm == 'mc':
                agent, max_diff_list = train_mc2(agent, state, env, iters, epsilon, epsilon_min, decay_rate, episode, max_diff_list)

            if algorithm == 'dp':
                agent, value_function, optimal_policy, max_diff_list = train_DP(agent, env, max_diff_list)
                break


        if algorithm=='dp':
            plot_V(agent)
            visit_counts = (grid_.cells == 3).astype(int)
            plot_policy_heatmap(optimal_policy, visit_counts, grid_.cells)

        elif algorithm=='q_learning':
            print(f'AUC under the learning curve: {calc_auc(cumulative_reward_list)}')
            print(f'normilized AUC under the learning curve: {calc_normilized_auc(cumulative_reward_list)}')
            agent.epsilon = 0
            plot_time_series(max_diff_list, y_label='Max difference in Q-value', title = 'Convergence: Max Difference per Episode')
            plot_time_series(cumulative_reward_list, y_label='Cumulative reward', title = 'Convergence: Cumulative reward per episode')
            plot_policy_heatmap(agent.q_table, agent.visit_counts, grid_.cells)

        else:
            agent.epsilon = 0
            plot_time_series(max_diff_list, 'Convergence: Max Difference per Episode')
            plot_policy_heatmap(agent.q_table, agent.visit_counts, grid_.cells)

        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.algorithm, args.no_gui, args.sigma, args.fps, args.episodes, args.iter, args.random_seed,
         args.epsilon, args.epsilon_min, args.decay_rate, args.gamma, args.eval_steps, args.n_eps_gui, args.delta)