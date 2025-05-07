"""
Train your Monte Carlo RL Agent in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import trange
from world.grid import Grid  # Import the Grid class

try:
    from world import Environment
    from agents.monte_carlo_agent import MonteCarloAgent
except ModuleNotFoundError:
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.append(root)
    from world import Environment
    from agents.monte_carlo_agent import MonteCarloAgent


def parse_args():
    p = ArgumentParser(description="Monte Carlo RL Trainer.")
    p.add_argument("GRID",        type=Path, nargs="+",
                   help="Grid file(s) to use for training.")
    p.add_argument("--no_gui",    action="store_true",
                   help="Disable rendering to train faster.")
    p.add_argument("--sigma",     type=float, default=0.1,
                   help="Slip probability in the environment.")
    p.add_argument("--fps",       type=int,   default=30,
                   help="Frames per second for GUI.")
    p.add_argument("--episodes",  type=int,   default=10000,
                   help="Number of training episodes.")
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
    p.add_argument("--gui_percentage", type=int, default=40,
                   help="percetage of GUI onnn of episodes to enable GUI (e.g., every N episodes).")
    return p.parse_args()


def main(GRID, no_gui, sigma, fps, episodes, random_seed,
         epsilon, epsilon_min, decay_rate, gamma, eval_steps, gui_percentage):
    for grid in GRID:
        env = Environment(grid,
                          no_gui=no_gui,
                          sigma=sigma,
                          target_fps=fps,
                          random_seed=random_seed)

        agent = MonteCarloAgent(n_actions=4,
                                epsilon=epsilon,
                                gamma=gamma)
        
        grid2 = Grid.load_grid(grid) # Load the grid from the file
        grid_height, grid_width = grid2.n_rows, grid2.n_cols
        grid_size = round((grid_height * grid_width ) ** 0.5) * 2 # needs work
        #print(f'The grid dimensions are {grid_height}x{grid_width} (size: {grid_size})')
        gui_percentage = episodes * (gui_percentage / 100)
        for ep in trange(episodes, desc="Episodes"):
            # exponential decay of epsilon
            agent.epsilon = epsilon_min + (
                epsilon - epsilon_min
            ) * np.exp(-decay_rate * ep)

            episode = []
            use_gui = ep % gui_percentage == 0
            state = env.reset(no_gui=not use_gui)
            done = False
            step = 0
            max_steps = int(grid_size * (1 + np.log1p(ep*0.1))) #needs work

            while not done and step < max_steps:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, reward))
                state = next_state
                step += 1

            agent.update_episode(episode)

        # final evaluation
        Environment.evaluate_agent(grid, agent,
            max_steps=eval_steps,
            sigma=sigma,
            random_seed=random_seed
        )
        agent.plot_q((grid2.n_rows, grid2.n_cols))


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.sigma, args.fps, args.episodes,args.random_seed,
         args.epsilon,args.epsilon_min, args.decay_rate, args.gamma, args.eval_steps, args.gui_percentage)
