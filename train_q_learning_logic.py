from agents.q_learning_agent2 import QLearningAgent
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
from world.grid import Grid  # Import the Grid class
from utils.plots import plot_max_diff

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


def train_q_learning(agent, state, env, iters, max_diff_list, delta, episode):
    """Main loop of the program."""

    agent._count_increase()

    if (episode % 100 == 0) and (episode != 0):
        agent._dynamic_params()

    q_table_old = {
        state: values.copy()   # if `values` is a NumPy array; or list(values) if it’s a list
        for state, values in agent.q_table.items()
    }

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
        return agent, max_diff_list, True
    #TODO: this stopping criterion only works if the agent starts from the same place each episode.
    # This is because if on a path that is seen before the max diff is very small
    # However this does not mean it learned all paths correctly
    # furthermore, the max diff plot makes more sense when start position is always the same
    # if start position is different each time the max diff plot becomes less insightful
    
    return agent, max_diff_list, False