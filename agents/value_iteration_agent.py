from agents.base_agent import BaseAgent
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ValueIterationAgent(BaseAgent):
    def __init__(self, n_actions: int, gamma: float = 0.9, delta_threshold: float = 1e-4):
        """Value Iteration Agent.
        Args:
            n_actions: Number of possible actions.
            gamma: Discount factor.
            delta_threshold: Threshold for convergence.
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.delta_threshold = delta_threshold
        self.V = defaultdict(float)
        self.policy = {} 
        self.delta_history = [] # per step maximum change over all state values

    def take_action(self, state: tuple[int, int]) -> int:
        pass

    def update(self, state: tuple[int, int], reward: float, action: int):
        pass

    def plot_policy(self, grid_shape=(8, 8)):
        """ Plot a 2D heatmap showing the best action for each state (x, y) using arrows.
        Empty cells indicate states not visited (e.g., walls or unreachable).
        
        Parameters:
            grid_shape (tuple): (width, height) of the environment grid. """
        arrow_map = {0: '↓', 1: '↑', 2: '←', 3: '→'}
        grid = np.full(grid_shape[::-1], ' ', dtype=object)

        for (x, y), action in self.policy.items():
            if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
                grid[y, x] = arrow_map.get(action, '?')

        plt.figure(figsize=(grid_shape[0], grid_shape[1]))
        sns.heatmap(np.zeros_like(grid, dtype=float),
                    cbar=False,
                    annot=grid,
                    fmt='',
                    linewidths=0.5,
                    linecolor='gray',
                    square=True,
                    cmap="Greys")
        plt.title("Policy (best action per state)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

    def plot_V(self):
        """Plot convergence (max delta V over states per step)."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.delta_history, label='Max delta V')
        plt.xlabel('Iterations')
        plt.ylabel('Max Value Change')
        plt.title('Value Function Convergence')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
