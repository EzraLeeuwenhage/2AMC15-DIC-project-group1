"""
Train your RL Agent in this file. 
"""
from argparse import ArgumentParser
from pathlib import Path
from world.grid import Grid
from utils.train_utils import init_agent, train_agent, evaluate_and_plot, set_agent_start_pos
from utils.reward_functions import custom_reward_function
import numpy as np

try:
    from world import Environment
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

def parse_args():
    p = ArgumentParser(description="Main RL Algorithm Trainer.")

    # arguments for environment
    p.add_argument("GRID",        type=Path, nargs="+",
                   help="Grid file(s) to use for training.")
    p.add_argument("--algorithm", type=str,
                   help="algortihm to train.")
    p.add_argument("--no_gui",    action="store_true",
                   help="Disable rendering to train faster.")
    p.add_argument("--agent_start_pos_col",    type=int,   default=None,
                   help="Starting position column of the agent in the gui representation of the grid. If None then random start position.")
    p.add_argument("--agent_start_pos_row",    type=int,   default=None,
                   help="Starting position row of the agent in the gui representation of the grid. If None then random start position.")
    p.add_argument("--sigma",     type=float, default=0.1,
                   help="Slip probability in the environment.")
    p.add_argument("--reward_func", type=callable, default=custom_reward_function,
                   help="Reward function of the environment.")
    p.add_argument("--fps",       type=int,   default=30,
                   help="Frames per second for GUI.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    
    # arguments for all agents
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor.")
    p.add_argument("--delta", type=int, default=1e-6,
                   help="A threshold for the Q-value updates for early stopping")
    
    # arguments for MC & Q-learning
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Learning rate.")
    p.add_argument("--epsilon",      type=float, default=1.0,
                   help="Starting epsilon for ε-greedy.")
    p.add_argument("--epsilon_min",  type=float, default=0.1,
                   help="Minimum epsilon after decay.")
    p.add_argument("--episodes",  type=int,   default=10000,
                   help="Number of training episodes.")
    p.add_argument("--iters", type=int, default=1000,
                   help="Number of iterations per episode.")
    p.add_argument("--early_stopping", type=int, default=-1,
                   help="Number of episodes to wait for early stopping. If -1, no early stopping is used.")
    
    # arguments for output and evaluation settings
    p.add_argument("--n_eps_gui", type=int, default=100,
                   help="percetage of GUI onnn of episodes to enable GUI (e.g., every N episodes).")
    p.add_argument("--output_plots", action="store_true",
                   help="Activate plot output.")
    
    return p.parse_args()


def main(grid, algorithm, no_gui, agent_start_pos_col, agent_start_pos_row, sigma, reward_func, fps, random_seed,
        gamma, delta, 
        alpha, epsilon, epsilon_min, episodes, iters, early_stopping,
        n_eps_gui, output_plots):
    """Main program for training a specific agent on a specific grid and extracting performance measures."""
    # init environment
    grid_fp = Path(grid[0])
    grid_ = Grid.load_grid(grid_fp)
    agent_start_pos = set_agent_start_pos(agent_start_pos_col, agent_start_pos_row, grid_.cells)
    env = Environment(grid_fp, no_gui, agent_start_pos=agent_start_pos, sigma=sigma, target_fps=fps,
                        random_seed=random_seed, reward_fn=reward_func)

    # train the agent and collect performance measures
    np.random.seed(random_seed)
    agent = init_agent(algorithm, grid_.cells.shape, alpha, gamma, delta)
    trained_agent, max_diff_list, cumulative_reward_list = train_agent(
            algorithm, agent, env, episodes, iters, delta, epsilon, epsilon_min, n_eps_gui, early_stopping)
            
    # Optionally plot agent performance
    if output_plots:
        evaluate_and_plot(trained_agent, algorithm, grid_, env, max_diff_list, cumulative_reward_list)

    return cumulative_reward_list


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.algorithm, args.no_gui, args.agent_start_pos_col, args.agent_start_pos_row, args.sigma, 
        args.reward_func, args.fps, args.random_seed,
        args.gamma, args.delta,
        args.alpha, args.epsilon, args.epsilon_min, args.episodes, args.iters, args.early_stopping,
        args.n_eps_gui, args.output_plots
    )
