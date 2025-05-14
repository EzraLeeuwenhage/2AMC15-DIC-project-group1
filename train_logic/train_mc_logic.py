from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import trange

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



def train_mc(agent, env, iters, epsilon, epsilon_min, decay_rate, episode, max_diff_list):

    # exponential decay of epsilon
    agent.epsilon = epsilon_min + (
            epsilon - epsilon_min
    ) * np.exp(-decay_rate * episode)

    episode_list = []
    state = env.reset()
    done = False
    step = 0

    while not done and step < iters:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_list.append((state, action, reward))
        state = next_state
        step += 1

    agent.update_episode(episode_list)

    #TODO: implement max diff list

    return agent, max_diff_list

def train_mc2(agent, state, env, iters, epsilon, epsilon_min, decay_rate, episode, max_diff_list):
    # exponential decay of epsilon
    agent.epsilon = epsilon_min + (
            epsilon - epsilon_min
    ) * np.exp(-decay_rate * episode)

    episode_list = []
    done = False
    step = 0
    grid_size = env.grid.shape[0] * env.grid.shape[1]
    max_steps = int(grid_size * (1 + np.log1p(episode*0.1))) #needs work

    while not done and step < max_steps:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_list.append((state, action, reward))
        state = next_state
        step += 1

    agent.update_episode(episode_list)
    #TODO: implement max diff list

    return agent, max_diff_list

def train_mc3(agent, state, env, iters, nr_of_episodes, delta, max_diff_list):
    agent._dynamic_params(nr_of_episodes)

    q_table_old = {
        state: values.copy()   # if `values` is a NumPy array; or list(values) if it’s a list
        for state, values in agent.q_table.items()
    }

    episode_list = []
    done = False
    step = 0

    while not done and step < iters:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        if reward >0:
            reward = 1000
        episode_list.append((state, action, reward))
        state = next_state
        step += 1

    agent.update_episode(episode_list)

    max_diff = 0
    all_in_common_keys = set(agent.q_table.keys()) & set(q_table_old.keys())
    for key in all_in_common_keys:
        abs_diff = np.max((np.abs(agent.q_table[key] - q_table_old[key])))
        if abs_diff > max_diff:
            max_diff = abs_diff
    if len(all_in_common_keys) == 0:
        max_diff = 1 # np.inf

    max_diff_list.append(max_diff)

    # Stopping criterion --> no significant change for N=20 episodes in a row.
    if max_diff < delta:
        agent._closer_to_termination()
        if agent.nr_consecutive_eps_no_change >= 20:
            return agent, max_diff_list, True
    else:
        agent._significant_change_to_q_values()  # resetting counter of consecutive episodes of no change

    return agent, max_diff_list, False
