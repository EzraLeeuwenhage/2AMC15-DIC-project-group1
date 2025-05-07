from agents.base_agent import BaseAgent
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ValueIterationAgent(BaseAgent):
    def __init__(self, n_actions: int, gamma: float = 0.9, delta_threshold: float = 1e-4, ):
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
        """Builds the transition model based on the provided grid."""
        P = defaultdict(lambda: defaultdict(dict))
        directions = {
            0: (0, 1),   # down
            1: (0, -1),  # up
            2: (-1, 0),  # left
            3: (1, 0)    # right
        }
        valid_values = {0, 3, 4}
        n_rows, n_cols = grid.shape
        
        for x in range(n_cols):
            for y in range(n_rows):
                if grid[y, x] not in valid_values:
                    continue  # skip obstacles and walls

                state = (x, y)
                P[state] = []
                neighbor_state= [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

                for x_loop in range(1, n_cols - 1):
                    for y_loop in range(1, n_rows - 1):
                        state_loop = (x_loop, y_loop)
                        if grid[x_loop, y_loop] in valid_values and state != state_loop:
                            P[state].append((state_loop, 1.0 if state_loop in neighbor_state else 0)) 

                    ''''
                    if 0 <= nx < n_cols and 0 <= ny < n_rows and grid[ny, nx] in valid_values:
                        next_state = (nx, ny)
                    else:
                        next_state = (x, y)  # bump into wall/obstacle, stay in place

                    # Deterministic transition: only one possible outcome with prob 1
                    P[state][action] = [(next_state, 1.0)]
                    '''
        return P

    def value_iteration(self, probs, gamma=0.99, theta=1e-6):
        states = probs.keys()
        
        V = {s: 0 for s in states}  # Initialize value function to 0
        policy = {}

        while True:
            delta = 0
            for s in states:
                v = V[s]
                action_values = []

                for a in actions:
                    total = 0
                    for prob, next_state, reward in transition_prob(s, a):
                        total += prob * (reward + gamma * V[next_state])
                    action_values.append(total)

                best_action_value = max(action_values)
                V[s] = best_action_value
                delta = max(delta, abs(v - best_action_value))

            if delta < theta:
                break

        # Derive policy from the final value function
        for s in states:
            best_action = None
            best_value = float('-inf')
            for a in actions:
                total = 0
                for prob, next_state, reward in transition_prob(s, a):
                    total += prob * (reward + gamma * V[next_state])
                if total > best_value:
                    best_value = total
                    best_action = a
            policy[s] = best_action

        return V, policy
    
        

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
