import numpy as np
import gymnasium as gym


class Agent:
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate=0.01,
        discount_factor=0.99,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize the policy weights to zeros
        self.weights = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        logits = np.dot(observation, self.weights)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        action = np.random.choice(self.action_space.n, p=probs)
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        if terminated:
            return

        if truncated:
            # If the episode has ended prematurely due to the maximum number of steps being reached,
            # assume the state-value of the next observation is zero
            next_state_value = 0
        else:
            # Compute the state-value of the next observation
            next_state_value = np.max(np.dot(observation, self.weights))

        # Compute the TD error
        td_error = (
            reward
            + self.discount_factor * next_state_value
            - np.dot(observation, self.weights)[0]
        )

        # Update the policy weights using the TD error and the current observation
        self.weights[:, observation[0]] += self.learning_rate * td_error * observation
