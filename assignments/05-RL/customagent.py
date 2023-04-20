import numpy as np
import gymnasium as gym


class Agent:
    """
    An agent that learns to play a game using Q-learning.
    """

    def __init__(
        self, action_space, observation_space, learning_rate=0.01, discount_factor=0.99
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize the Q-table to zeros
        self.q_table = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation):
        """
        Select an action based on the current Q-values.
        """
        q_values = self.q_table[observation]
        action = np.argmax(q_values)
        return action

    def learn(self, observation, reward, done, info):
        """
        Update the Q-table based on the observed reward and transition.
        """
        if done:
            td_error = reward - self.q_table[observation]
        else:
            td_error = (
                reward
                + self.discount_factor * np.max(self.q_table[observation])
                - self.q_table[observation]
            )

        self.q_table[observation] += self.learning_rate * td_error
