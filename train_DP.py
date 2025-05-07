"""
Train Dynamic Programming agent with this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world import Environment
    from agents.value_iteration_agent import ValueIterationAgent
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
    from agents.value_iteration_agent import ValueIterationAgent

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
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--eval_steps", type=int,   default=200,
                   help="Steps for final evaluation.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, fps: int,
         sigma: float, random_seed: int, eval_steps: int):
    """Main loop of the program."""

    for grid in grid_paths:
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed, agent_start_pos=(1,1))
        
        _ = env.reset()
        grid = np.copy(env.grid)
        reward_fn = env.reward_fn

        agent = ValueIterationAgent(n_actions=4)
        states, P = agent.extract_transition_model(grid, env.sigma)

        # # Print the transition probabilities
        # for state, actions in P.items():
        #     print(f"{state}:")
        #     for action, tuples in enumerate(actions):
        #         print(f"Action {action}: {tuples}")

        # # Print the grid
        # for row in initial_grid:
        #     print(row)

        value_function, optimal_policy = agent.value_iteration(grid, reward_fn, states, P)

        Environment.evaluate_agent(
            env.grid_fp, 
            agent,
            max_steps=eval_steps,
            sigma=sigma,
            random_seed=random_seed
        )
        
        agent.plot_policy((grid.shape[0], grid.shape[1]))
        agent.plot_V()


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.fps, args.sigma, args.random_seed, args.eval_steps)