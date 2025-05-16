import numpy as np

def train_q_learning(agent, state, env, iters, max_diff_list, delta, episode, episodes, 
                     epsilon, epsilon_min, early_stopping, cumulative_reward_list):
    """Train a Q-learning agent for one episode and track Q-value changes with early stopping."""
    agent.initialize_epsilon(episode, episodes, epsilon, epsilon_min)

    q_table_old = {
        state: values.copy()
        for state, values in agent.q_table.items()
    }

    for _ in range(iters):
        action = agent.take_action(state)
        next_state, reward, terminated, info = env.step(action)
        agent.update(state, reward, info["actual_action"], next_state)
        state = next_state

        if terminated: # If the final state is reached, stop the episode
            break

    # Cumulative reward update for AUC Curve
    cumulative_reward_list.append(env.world_stats["cumulative_reward"])
    max_diff = 0
    all_in_common_keys = set(agent.q_table.keys()) & set(q_table_old.keys())
    for key in all_in_common_keys:
        abs_diff = np.max((np.abs(agent.q_table[key] - q_table_old[key])))
        if abs_diff > max_diff:
            max_diff = abs_diff
    if len(all_in_common_keys) == 0:
        max_diff = 1

    max_diff_list.append(max_diff)

    if early_stopping != -1:
        # Stopping criterion --> no significant change for N=20 episodes in a row.
        if max_diff < delta:
            agent._closer_to_termination()
            if agent.nr_consecutive_eps_no_change >= early_stopping:
                return agent, max_diff_list, cumulative_reward_list, True
        else:
            agent._significant_change_to_q_values()  # resetting counter of episodes without change
    
    return agent, max_diff_list, cumulative_reward_list, agent.q_table, False