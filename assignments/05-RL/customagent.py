import numpy as np
import gymnasium as gym


class Agent:
    """
    It is a model created by Zepu
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate=0.01,
        discount_factor=0.99,
    ):
        """
        It is a model created by Zepu
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize the policy weights to zeros
        self.weights = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        It is a model created by Zepu haha

        """
        # Compute the probabilities of each action based on the current policy
        logits = np.dot(observation, self.weights)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Choose an action based on the probabilities
        action = np.random.choice(self.action_space.n, p=probs)
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        action: gym.spaces.Discrete,
        reward: float,
        next_observation: gym.spaces.Box,
        done: bool,
    ) -> None:
        """
        It is a model created by Zepu
        """
        # Compute the TD error
        td_error = (
            reward
            + self.discount_factor * np.max(np.dot(next_observation, self.weights))
            - np.dot(observation, self.weights)[action]
        )

        # Update the policy weights using the TD error and the current observation
        self.weights[:, action] += self.learning_rate * td_error * observation
