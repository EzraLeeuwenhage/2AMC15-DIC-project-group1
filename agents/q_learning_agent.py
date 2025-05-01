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

    def __init__(self, actions = [0, 1, 2, 3], alpha=0.1, gamma=0.9, epsilon=0.5):
        super().__init__()
        self.q_table = {}  # Layout of Q_table is this dictionary structure: {(state): [action_values]}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_count = 0

    def _count_increase(self):
        self.episode_count += 1

    def _dynamic_params(self):
        self.epsilon /= 2

    def _ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]

    def take_action(self, state: tuple[int, int]) -> int:
        self._ensure_state_exists(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # explore
        return int(np.argmax(self.q_table[state]))  # exploit

    def update(self, state: tuple[int, int], reward: float, action: int, next_state: tuple[int, int]):
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        best_next_q = max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * ((reward + self.gamma * best_next_q) - self.q_table[state][action])
