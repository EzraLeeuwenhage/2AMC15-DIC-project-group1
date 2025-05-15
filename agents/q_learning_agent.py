# import os
# print(os.getcwd)

from agents.base_agent import BaseAgent
import numpy as np


class QLearningAgent(BaseAgent):
    """
    State is just a tuple with two integer coordinates. 
    From environment _move_agent it appears that you always provide all possible actions to choose from. 
    The agent learns in which states which actions are good and which actions lead to failed moves (e.g. running into a wall)
    Actions:
        - 0: Move down
        - 1: Move up
        - 2: Move left
        - 3: Move right
    """

    def __init__(self, grid_shape, actions, alpha, gamma, epsilon=0.5):
        super().__init__()
        self.q_table = {}  # Layout of Q_table is this dictionary structure: {(state): [action_values]}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nr_consecutive_eps_no_change = 0  # only used for stopping criterion
        self.visit_counts = np.zeros(grid_shape, dtype=int)  # Only used for plotting, does not break Markov Property

    def _closer_to_termination(self):
        """Keep track in how many consecutive episodes the Q-values did not change significantly. I.e. max_diff of Q-values below some delta."""
        self.nr_consecutive_eps_no_change += 1

    def _significant_change_to_q_values(self):
        """If Q-values did change significantly, we need to reset the maintained number of consecutive episodes without change."""
        self.nr_consecutive_eps_no_change = 0

    def _dynamic_params(self):
        """Halve the exploration rate"""
        self.epsilon /= 2

    def _ensure_state_exists(self, state):
        """If state does not exist (is not in the dictionary) yet, create an entry for this state initializing the Q-value for all actions at 0."""
        if state not in self.q_table:
            self.q_table[state] = np.array([0.0 for _ in self.actions])

    def initialize_epsilon(self, episode, episodes, epsilon_max, epsilon_min):
        '''Initialize epsilon for current episode based on epsilon decay from epsilon to epsilon_min'''
        # Exponential epsilon decay from epsilon=1.0 to epsilon_min=0.2
        decay_constant = 5.0 * (np.log(epsilon_max / epsilon_min) / episodes)
        self.epsilon_mc = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_constant * episode)

    def take_action(self, state: tuple[int, int]) -> int:
        """Choose some action using epsilon greedy, and before an action is chosen we record the visit to a state."""
        self._ensure_state_exists(state)
        # Record visit of being in a state when taking an action from that state
        c, r = state  # Environment uses the (c, r) indexing which is a bit odd
        self.visit_counts[r, c] += 1  # We maintain the visit counts in the more usual (r, c) as this is more usual, and better for plotting
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # explore
        return int(np.argmax(self.q_table[state]))  # exploit

    def update(self, state: tuple[int, int], reward: float, action: int, next_state: tuple[int, int]):
        """Execute update to the Q-values, following the Bellman Equation."""
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        best_next_q = max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * ((reward + self.gamma * best_next_q) - self.q_table[state][action])
