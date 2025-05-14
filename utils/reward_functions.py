def custom_reward_function(grid, agent_pos) -> float:
        """This function encodes the rewards for the agent upon reaching a 
        certain state in the grid.
        This function differs from the default reward function,
        by assigning a much higher reward to reaching the target state.
        Doing this shifts the optimal policy for environments with higher
        stochasticity from "avoid bumping into walls" to "actually reach
        the target state". 

        Args:
            grid: The grid the agent is moving on, in case that is needed by
                the reward function.
            agent_pos: The position the agent is moving to.

        Returns:
            A single floating point value representing the reward for a given
            state.
        """

        match grid[agent_pos]:
            case 0:  # Moved to an empty tile
                reward = -1
            case 1 | 2:  # Moved to a wall or obstacle
                reward = -5
                pass
            case 3:  # Moved to a target tile
                reward = 1000
            case _:
                raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}.",
                                 f"at position {agent_pos}")
        return reward