import numpy as np

def train_DP(agent, env, max_diff_list):
    _ = env.reset()
    grid = np.copy(env.grid)
    reward_fn = env.reward_fn
    states, P = agent.extract_transition_model(grid, env.sigma)
    value_function, optimal_policy = agent.value_iteration(grid, reward_fn, states, P)
    return agent, value_function, optimal_policy, max_diff_list