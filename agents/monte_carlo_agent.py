from collections import defaultdict
import numpy as np
from agents.base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """On-Policy First-Visit Monte Carlo control agent."""
    def __init__(self, n_actions: int, epsilon: float = 0.3, gamma: float = 0.9):
        """Initialize the Monte Carlo agent."""
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(float)           # action-value estimates
        self.returns = defaultdict(list)      # list of returns for each (s,a)

    def take_action(self, state: tuple[int,int]) -> int:
        """Return action chosen by ε-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = [self.Q[(state,a)] for a in range(self.n_actions)]
        return int(np.argmax(q_values))

    def update(self, state: tuple[int,int], reward: float, action: int):
        """No per-step update for Monte Carlo."""
        pass

    def update_episode(self, episode: list[tuple[tuple,int,float]]):
        """Perform first-visit MC update from a complete episode."""
        for t,(state,action,_) in enumerate(episode):
            # check first visit of (state, action)
            first = next(i for i,(s,a,_) in enumerate(episode) if (s,a)==(state,action))
            if first != t:
                continue
            # compute return G for this first visit
            G = 0.0
            for k,(_,_,reward) in enumerate(episode[t:], start=0):
                G += (self.gamma**k) * reward
            # update returns and Q-value
            self.returns[(state,action)].append(G)
            self.Q[(state,action)] = float(np.mean(self.returns[(state,action)]))
