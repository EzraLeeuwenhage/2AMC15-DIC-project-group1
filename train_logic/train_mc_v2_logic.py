import numpy as np

def train_mc_control(agent, state, env, iters, max_diff_list, episode, episodes, 
                     epsilon_max, epsilon_min, cumulative_reward_list):
    """Train a Monte Carlo control agent for one episode and track Q-value changes."""
    # initialize current episode history and epsilon values for MC agent
    agent.initialize_episode_history()
    agent.initialize_epsilon(episode, episodes, epsilon_max, epsilon_min)

    q_table_old = {
        state: values.copy()
        for state, values in agent.q_table.items()
    }

    for step in range(iters):

        action = agent.take_action(state)
        next_state, reward, terminated, info = env.step(action)
        agent.update(state, reward, info["actual_action"])
        
        if terminated: # If the final state is reached, stop the episode
            break

        state = next_state
        
    # Cumulative reward update for AUC Curve
    cumulative_reward_list.append(env.world_stats["cumulative_reward"])
    agent.mc_update()

    # max difference in q values
    max_diff = 0
    all_in_common_keys = set(agent.q_table.keys()) & set(q_table_old.keys())
    for key in all_in_common_keys:
        abs_diff = np.max((np.abs(agent.q_table[key] - q_table_old[key])))
        if abs_diff > max_diff:
            max_diff = abs_diff
    if len(all_in_common_keys) == 0:
        max_diff = 1

    max_diff_list.append(max_diff)
    
    return agent, max_diff_list, cumulative_reward_list, agent.q_table, False
