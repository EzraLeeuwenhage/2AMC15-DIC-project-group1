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