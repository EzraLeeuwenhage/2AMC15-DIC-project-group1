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


    def plot_q(self, grid_shape=(8,8)):
        """ Plot a 2D heatmap showing the best action for each state (x, y) using arrows.
        Empty cells indicate states not visited (e.g., walls or unreachable).
        
        Parameters:
            grid_shape (tuple): (width, height) of the environment grid. """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        arrow_map = {
            0: '↓',  # Down
            1: '↑',  # Up
            2: '←',  # Left
            3: '→',  # Right
        }

        width, height = grid_shape
        grid = np.full((height, width), ' ', dtype=object)  # default to blank for unvisited states

        # Find best action per state
        best_actions = {}
        for (state, action), q in self.Q.items():
            if state not in best_actions or q > best_actions[state][1]:
                best_actions[state] = (action, q)

        # Fill in arrows only where we have data
        for (x, y), (action, _) in best_actions.items():
            if 0 <= x < width and 0 <= y < height:  # Ensure within bounds
                grid[y, x] = arrow_map.get(action, '?')

        plt.figure(figsize=(width, height))
        sns.heatmap(np.zeros_like(grid, dtype=float), 
                    cbar=False, 
                    annot=grid, 
                    fmt='', 
                    linewidths=0.5,
                    linecolor='gray',
                    square=True,
                    cmap="Greys")

        plt.title("Best Action per State (as Arrows)")
        plt.xlabel("x")
        plt.ylabel("y")
        #plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
