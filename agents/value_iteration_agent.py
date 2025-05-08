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
        self.optimal_policy = []


        ### dynamic programming: VI/ PI -> assume model of environment: transition probability matrix uit env krijgen
        # MC + Q learning -> dont assume model, use environment to get statistical idea of true probabilities and iterate on that

        # P(new_state | state + action), state = [links midden rechts] ik zit in links actions = [links rechts]
        # P(s' | (s, a))

        # matrix for action rechts
        # links: [0, 1, 0] - dit moet altijd naar 1 summen
        # midden: [0, 0, 1]
        # rechts: [0, 0, 1]

        # matrix for action links
        # bellmann equation: R + future rewards * chance of reaching them -> final value for ecery state
        # policy is dan het kiezen van de hoogste values

    def take_action(self, state: tuple[int, int]) -> int:
        # given state -> do action according to learned optimal policy
        pass

    def update(self, state: tuple[int, int], reward: float, action: int):
        pass

    def extract_transition_model(self, grid):
        """Builds a full transition model for every state and action.

        Each state has 4 actions, and each action lists 4 possible next states
        (up, down, left, right) with their respective probabilities.

        Returns:
            dict[state] = [list of 4 lists], where each inner list contains up to 4
            (next_state, probability) tuples for that action.
        """
        directions = {
            0: (0, 1),   # down
            1: (0, -1),  # up
            2: (-1, 0),  # left
            3: (1, 0)    # right
        }
        direction_list = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # for generating all possible next states

        valid_values = {0, 3, 4}
        n_rows, n_cols = grid.shape

        P = {}

        for x in range(n_cols):
            for y in range(n_rows):
                if grid[y, x] not in valid_values:
                    continue  # skip unreachable states

                state = (x, y)
                P[state] = []

                for action in range(4):
                    dx, dy = directions[action]
                    ax, ay = x + dx, y + dy

                    if 0 <= ax < n_cols and 0 <= ay < n_rows and grid[ay, ax] in valid_values:
                        intended_next_state = (ax, ay)
                    else:
                        intended_next_state = (x, y)  # bump, stay

                    action_transitions = []
                    for ndx, ndy in direction_list:
                        nx, ny = x + ndx, y + ndy
                        if 0 <= nx < n_cols and 0 <= ny < n_rows and grid[ny, nx] in valid_values:
                            neighbor = (nx, ny)
                        else:
                            neighbor = (x, y)

                        if neighbor == intended_next_state:
                            action_transitions.append((neighbor, 1.0))
                        else:
                            action_transitions.append((neighbor, 0.0))

                    P[state].append(action_transitions)

        return P

    def value_iteration():
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
