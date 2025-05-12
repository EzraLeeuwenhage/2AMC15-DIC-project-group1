from collections import defaultdict
import numpy as np
from agents.base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """On-Policy First-Visit Monte Carlo control agent."""
    def __init__(self, grid_shape, n_actions: int, epsilon: float = 0.3, gamma: float = 0.9):
        """Initialize the Monte Carlo agent."""
        self.grid_shape = grid_shape
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = {}         # action-value estimates
        self.returns_sum = {}
        self.returns_count = {}
        self.visit_counts = np.zeros(grid_shape, dtype=int)  # Only used for plotting, does not break Markov Property

    def _ensure_state_exists_returns_sum(self, state):
        """If state does not exist (is not in the dictionary) yet, create an entry for this state initializing the Q-value for all actions at 0."""
        if state not in self.returns_sum:
            self.returns_sum[state] = np.array([0.0 for _ in range(self.n_actions)])

    def _ensure_state_exists_returns_count(self, state):
        """If state does not exist (is not in the dictionary) yet, create an entry for this state initializing the Q-value for all actions at 0."""
        if state not in self.returns_count:
            self.returns_count[state] = np.array([0.0 for _ in range(self.n_actions)])

    def _ensure_state_exists_q_table(self, state):
        """If state does not exist (is not in the dictionary) yet, create an entry for this state initializing the Q-value for all actions at 0."""
        if state not in self.q_table:
            self.q_table[state] = np.array([0.0 for _ in range(self.n_actions)])

    def _dynamic_params(self):
        """Halve the exploration rate"""
        self.epsilon = max(self.epsilon-0.05, 0.0)

    def _closer_to_termination(self):
        """Keep track in how many consecutive episodes the Q-values did not change significantly. I.e. max_diff of Q-values below some delta."""
        self.nr_consecutive_eps_no_change += 1

    def _significant_change_to_q_values(self):
        """If Q-values did change significantly, we need to reset the maintained number of consecutive episodes without change."""
        self.nr_consecutive_eps_no_change = 0


    def take_action(self, state: tuple[int,int]) -> int:
        """Return action chosen by ε-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        self._ensure_state_exists_q_table(state)
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(self, state: tuple[int,int], reward: float, action: int):
        """No per-step update for Monte Carlo."""
        pass

    def update_episode(self, episode: list[tuple[tuple, int, float]]):
        # """Perform first‐visit MC update from a complete episode via backward return."""
        # G = 0.0
        # for state, action, reward in reversed(episode):
        #     r, c = state
        #     self.visit_counts[r, c] += 1
        #     self._ensure_state_exists_q_table(state)
        #     self._ensure_state_exists_returns_sum(state)
        #     self._ensure_state_exists_returns_count(state)
        #     G = self.gamma * G + reward
        #     self.returns_sum[state][action] += G
        #     self.returns_count[state][action] += 1
        #     self.q_table[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
        """
            Perform first‐visit MC update from a complete episode.

            episode: [(state, action, reward), …] with length T
            """
        T = len(episode)

        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            returns[t] = G

        seen = set()
        for t, (state, action, _) in enumerate(episode):
            if (state, action) in seen:
                continue
            seen.add((state, action))

            self._ensure_state_exists_q_table(state)
            self._ensure_state_exists_returns_sum(state)
            self._ensure_state_exists_returns_count(state)

            G_t = returns[t]
            self.returns_sum[state][action] += G_t
            self.returns_count[state][action] += 1
            self.q_table[state][action] = self.returns_sum[state][action]/ self.returns_count[state][action]

            r, c = state
            self.visit_counts[r, c] += 1


