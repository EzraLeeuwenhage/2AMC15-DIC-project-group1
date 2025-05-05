from collections import defaultdict
import numpy as np
from agents.base_agent import BaseAgent


class MonteCarloAgent_EV(BaseAgent):
    """On-Policy Every-Visit Monte Carlo control agent."""
    def __init__(self, n_actions: int, epsilon: float = 0.3, gamma: float = 0.9):
        """Initialize the Monte Carlo agent."""
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(float)           # action-value estimates
        self.N_sa = defaultdict(int)          # visit count for state-action pair
        self.valuefunction = []               # Check change in value function
        self.check = 0

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
        """Perform every-visit MC update from a complete episode."""
        # Compute returns G and update Q
        G = 0.0
        for (state, action, reward) in reversed(episode):

            G = self.gamma * G + reward
            
            # If visited state before:
            if (state, action) in self.N_sa: 
                self.N_sa[(state, action)] += 1 
                self.Q[(state, action)] += (1 / self.N_sa[(state, action)]) * (G - self.Q[(state, action)])  # MC update
            
            # If visiting first time
            else:
                self.N_sa[(state, action)] = 1
                self.Q[(state, action)] = G  # MC update
        
        # Check if value_function's state is decided
        if self.check == 0:
            self.check, _ = next(iter(self.Q.items()))
            self.check = self.check[0]
        self.valuefunction.append(np.min([self.Q[(self.check,a)] for a in range(self.n_actions)]))


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

    def plot_V(self):

        import matplotlib.pyplot as plt
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(self.valuefunction, label='V(arbitrary_state) for every visit MC Control'+ str(self.check))

        plt.xlabel('Episodes')
        plt.ylabel('V(arbitrary_state) per episode')
        plt.title('Value Function Convergence for arbitrary_state' + str(self.check))
        plt.legend()
        plt.grid(True)
        # Save the figure with aim in the filename
        plt.savefig('MC_every_visit.png')
        plt.close()