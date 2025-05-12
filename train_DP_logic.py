import numpy as np

def one_hot_encode(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec

def train_DP(agent, env, max_diff_list):
    _ = env.reset()
    grid = np.copy(env.grid)
    reward_fn = env.reward_fn
    states, P = agent.extract_transition_model(grid, env.sigma)
    value_function, optimal_policy = agent.value_iteration(grid, reward_fn, states, P)
    optimal_policy = {key: one_hot_encode(optimal_policy[key], 4) for key in optimal_policy.keys()}
    return agent, value_function, optimal_policy, max_diff_list