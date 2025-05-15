import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np


# Plotting the max_diffs over time to track convergence

def plot_time_series(value_list, y_label = '', title=''):
    """
    Plot value per episode.

    Parameters:
    -----------
    value_list : list of float
        value_list[i] is the value in episode i.
    """
    episodes = range(len(value_list))
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, value_list, linestyle='-', linewidth=1)  # thin line, no markers
    plt.xlabel('Episode/Iteration')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calc_normalized_auc(reward_list):
    episodes = np.arange(len(reward_list))
    auc = np.trapz(reward_list, episodes)
    min_reward = min(reward_list)
    max_reward = max(reward_list)
    min_auc = min_reward * (len(reward_list) - 1)
    max_auc = max_reward * (len(reward_list) - 1)
    normalized_auc = (auc - min_auc) / (max_auc - min_auc) if max_auc != min_auc else 0
    return normalized_auc

def calc_auc(reward_list):
    episodes = np.arange(len(reward_list))
    auc = np.trapz(reward_list, episodes)
    return auc

def plot_policy_heatmap(q_table: dict, visit_counts: np.ndarray, layout_matrix: np.ndarray, title=None, show_image=True):
    """
    Plots a heatmap of visit frequencies (viridis), marks obstacles in dark grey,
    and overlays arrows for the optimal action in each visited state.

    Args:
        layout_matrix (2D array of ints): different integers indicate different entities.
        "empty": 0,
        "boundary": 1,
        "obstacle": 2,
        "target": 3,
        "charger": 4
    """
    # The environment works with (column, row) indexing. The GUI works with (row, column), 
    # Therefore we convert the (c, r) to (r, c), to make the plot aligned with the GUI visualization
    layout_matrix = layout_matrix.T
    # visit_counts is maintained in the more customary (r, c) format immediately to make this plotting easier. Therefore no transposing is necessary
    # Q-table is indirectly transposed when creating the annotation grid using grid[r, c] indexing
    nrows, ncols = layout_matrix.shape

    fig, ax = plt.subplots()  # initiate plot
    # Heatmap of visits
    img = ax.imshow(visit_counts, cmap='viridis', interpolation='nearest', origin='upper')
    fig.colorbar(img, ax=ax, label='Visit Count')

    # Overlay obstacle cells
    for i in range(nrows):
        for j in range(ncols):
            if layout_matrix[i, j] in [1, 2]:  # grey color for borders and obstacles
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightgray')
                ax.add_patch(rect)

    # Overlay targets --> we can add one for chargers later
    for i in range(nrows):
        for j in range(ncols):
            if layout_matrix[i, j] in [3]:
                circle = Circle(xy=(j, i), radius=0.3, color='#92e000')  
                # add green circle for target, but at same time color in background to see how often it was reached
                ax.add_patch(circle)

    grid = np.full((nrows, ncols), ' ', dtype=object)
    arrow_map = {0: '↓', 1: '↑', 2: '←', 3: '→'}

    # Find best action per state
    best_actions = {}
    for state in q_table:
        best_action = np.argmax(q_table[state])
        best_actions[state] = (best_action, q_table[state][best_action])

    # Fill in arrows only where we have data
    for (c, r), (action, _) in best_actions.items():
        if 0 <= c < ncols and 0 <= r < nrows:  # Ensure within bounds
            grid[r, c] = arrow_map.get(action, '?')  # By indexing with [r, c] we sort of transpose the Q-table values

    for (r, c), label in np.ndenumerate(grid):
        if label != ' ':  # Only add annotation if there is an arrow
            ax.text(
                c,               
                r,               
                label,            
                ha='center', 
                va='center',
                color='white',    
                fontsize=12
            )

    # Remove axes labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if show_image:
        plt.show()
    return fig


def extract_VI_agent_optimal_path(agent, env):
    """
    Executes the optimal policy (from the value iteration agent) in the environment
    and returns a visit_counts array highlighting the optimal path.

    Args:
        env: environment which agent was trained on.

    Returns:
        visit_counts: 2D array with 1s along the optimal path, 0s elsewhere.
    """
    state = env.reset()
    env.no_gui = True
    # env.sigma = 0

    optimal_path = []
    optimal_path.append(state)

    ncols, nrows = env.grid.shape
    visit_counts = np.zeros((nrows, ncols), dtype=int)

    r, c = state[1], state[0]
    visit_counts[r, c] = 1

    terminated = False
    while not terminated:
        action = agent.take_action(state)
        state, _, terminated, _ = env.step(action)

        optimal_path.append(state)

        r, c = state[1], state[0]
        visit_counts[r, c] = 1

    return visit_counts, optimal_path
