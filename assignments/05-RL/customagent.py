import numpy as np
import gymnasium as gym


class Agent:
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize the policy weights to zeros
        self.weights = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation: np.ndarray) -> int:
        # Compute the probabilities of each action based on the current policy
        logits = np.dot(observation, self.weights)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Choose an action based on the probabilities
        action = np.random.choice(self.action_space.n, p=probs)
        return action

    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        # Compute the TD error
        td_error = (
            reward
            + self.discount_factor * np.max(np.dot(next_observation, self.weights))
            - np.dot(observation, self.weights)[:, action]
        )

        # Update the policy weights using the TD error and the current observation
        self.weights[:, action] += self.learning_rate * td_error * observation
