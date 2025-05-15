from agents.q_learning_agent import QLearningAgent
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
from world.grid import Grid  # Import the Grid class


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


def train_q_learning(agent, state, env, iters, max_diff_list, delta, episode, episodes, epsilon, epsilon_min, early_stopping, cumulative_reward_list):
    """Main loop of the program."""
    agent.initialize_epsilon(episode, episodes, epsilon, epsilon_min)

    q_table_old = {
        state: values.copy()   # if `values` is a NumPy array; or list(values) if it’s a list
        for state, values in agent.q_table.items()
    }

    for _ in range(iters):

        # Agent takes an action based on the latest observation and info.
        action = agent.take_action(state)

        # The action is performed in the environment
        next_state, reward, terminated, info = env.step(action)

        # If the final state is reached, stop.
        if terminated:
            break

        agent.update(state, reward, info["actual_action"], next_state)

        state = next_state
    cumulative_reward_list.append(env.world_stats["cumulative_reward"])
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

    if early_stopping != -1:
        # Stopping criterion --> no significant change for N=20 episodes in a row.
        if max_diff < delta:
            agent._closer_to_termination()
            if agent.nr_consecutive_eps_no_change >= early_stopping:
                return agent, max_diff_list, cumulative_reward_list, True
        else:
            agent._significant_change_to_q_values()  # resetting counter of consecutive episodes of no change
    
    return agent, max_diff_list, cumulative_reward_list, agent.q_table, False