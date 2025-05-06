import matplotlib.pyplot as plt
plt.ion()


# Plotting the max_diffs over time to track convergence

def plot_max_diff(max_diff_list):
    """
    Plot the max Q-value difference per episode.

    Parameters:
    -----------
    max_diff_list : list of float
        max_diff_list[i] is the max Q-value change in episode i.
    """
    episodes = range(len(max_diff_list))
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, max_diff_list, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-value Difference')
    plt.title('Convergence: Max Difference per Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig("max_diff_plot.png")
    # plt.close()
