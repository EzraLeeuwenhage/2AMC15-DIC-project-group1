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
        return self.policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        pass

    def extract_transition_model(self, grid, sigma):
        """Builds a transition model for each state and action.

        Each action results in one intended next state (prob=1.0) and
        three other adjacent states (prob=0.0) for easy future stochastic updates.

        Returns:
            states: List of valid states.
            P: dict[state] = list of 4 lists (one per action), each containing 4 tuples
            (next_state, prob) representing the 4 adjacent positions.
        """
        directions = {
            0: (0, 1),   # down
            1: (0, -1),  # up
            2: (-1, 0),  # left
            3: (1, 0)    # right
        }

        direction_list = list(directions.values())  # [(0,1), (0,-1), (-1,0), (1,0)]
        valid_values = {0, 3, 4}
        n_rows, n_cols = grid.shape

        states = []
        P = {}

        for x in range(n_rows):
            for y in range(n_cols):
                if grid[x, y] not in valid_values:
                    continue

                state = (x, y)
                states.append(state)
                P[state] = []

                for action_index, (dx, dy) in directions.items():
                    intended_next_state = (x + dx, y + dy)
                    action_transitions = []

                    for direction in direction_list:
                        new_x, new_y = x + direction[0], y + direction[1]
                        next_state = (new_x, new_y)

                        if next_state == intended_next_state:
                            prob = (1 - sigma) + sigma / 4
                        else:
                            prob = sigma / 4
                        
                        action_transitions.append((next_state, prob))
                    
                    P[state].append(action_transitions)

        return states, P

    
    def value_iteration(self, grid, reward_fn, states, P, max_iterations=1000):
        """Performs Value Iteration given transition model and reward function."""
        self.V = {state: 0 for state in states}
        
        for i in range(max_iterations):
            delta = 0
            new_V = self.V.copy()

            for state in states:
                action_values = []

                for action in range(self.n_actions):
                    value = 0

                    for intended_next_state, prob in P[state][action]:
                        reward = reward_fn(grid, intended_next_state)
                        
                        # handle case where 'intended' next_state is illegal state
                        if grid[intended_next_state] in {1, 2}:  
                            actual_next_state = state
                        else:
                            actual_next_state = intended_next_state
    
                        value += prob * (reward + self.gamma * self.V[actual_next_state])

                    action_values.append(value)

                new_V[state] = max(action_values)
                delta = max(delta, abs(self.V[state] - new_V[state]))

            self.V = new_V
            self.delta_history.append(delta)
            if delta < self.delta_threshold:
                break

        # Get optimal policy
        self.policy = {}
        for state in states:
            best_action = None
            best_value = float('-inf')

            for action in range(self.n_actions):
                value = 0
                for intended_next_state, prob in P[state][action]:
                    reward = reward_fn(grid, intended_next_state)

                    # handle case where 'intended' next_state is illegal state
                    if grid[intended_next_state] in {1, 2}:  
                        actual_next_state = state
                    else:
                        actual_next_state = intended_next_state

                    value += prob * (reward + self.gamma * self.V[actual_next_state])

                if value > best_value:
                    best_value = value
                    best_action = action

            self.policy[state] = best_action
        
        return self.V, self.policy


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
