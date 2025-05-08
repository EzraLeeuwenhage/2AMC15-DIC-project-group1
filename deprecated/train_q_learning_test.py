"""
Train your RL Agent in this file. 
"""

from agents.q_learning_agent3 import QLearningAgent
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
from world.grid import Grid  # Import the Grid class
from utils.plots import plot_max_diff, plot_policy_heatmap

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
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--n_eps", type=int, default=1000,
                   help="Number of episodes to run the Q-learning algorithm.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--delta", type=int, default=1e-6,
                   help="A threshold for the Q-value updates for early stopping")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, n_eps: int, iters: int, delta: float, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:  # not yet used, because Q-learning per grid world
        
        grid_ = Grid.load_grid(grid)

        # Initialize agent
        agent = QLearningAgent(grid_shape=(grid_.n_rows, grid_.n_cols))
        max_diff_list = []  # For tracking convergence and convergence plot

        for episode in range(n_eps):

            agent._count_increase()

            if episode == 0:
                no_gui = True
            elif (episode % 100 == 0) and (episode != 0):
                agent._dynamic_params()
                print(episode)
                no_gui = True
            else:
                print(episode)
                no_gui = True

            # Set up the environment
            env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                            random_seed=random_seed)
            
            q_table_old = {
                state: values.copy()   # if `values` is a NumPy array; or list(values) if it’s a list
                for state, values in agent.q_table.items()
            }
            
            # Always reset the environment to initial state
            state = env.reset()
            for _ in trange(iters):
                
                # Agent takes an action based on the latest observation and info.
                action = agent.take_action(state)

                # The action is performed in the environment
                next_state, reward, terminated, info = env.step(action)
                
                # If the final state is reached, stop.
                if terminated:
                    break

                agent.update(state, reward, info["actual_action"], next_state)

                state = next_state
            
            # max difference in q values
            max_diff = 0
            all_in_common_keys = set(agent.q_table.keys()) & set(q_table_old.keys())
            for key in all_in_common_keys:
                abs_diff = np.max((np.abs(agent.q_table[key] - q_table_old[key])))
                if abs_diff > max_diff:
                    max_diff = abs_diff
            if len(all_in_common_keys) == 0:
                max_diff = 1 # np.inf
            
            max_diff_list.append(max_diff)
            print(max_diff)
            # Stopping criterion
            if max_diff < delta:
                break

        print(len(max_diff_list))

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)
    
        plot_policy_heatmap(agent.q_table, agent.visit_counts, grid_.cells)
        plot_max_diff(max_diff_list)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.n_eps, args.delta, args.fps, args.sigma, args.random_seed)