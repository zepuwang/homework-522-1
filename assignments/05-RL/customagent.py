import numpy as np
import gymnasium as gym


class Agent:
    """
    A reinforcement learning agent that uses a simple policy gradient algorithm called REINFORCE.
    """

    def __init__(self, env, learning_rate=0.01, discount_factor=0.99):
        """
        Initialize the agent.

        Args:
            env (gym.Env): The environment to interact with.
            learning_rate (float): The learning rate for the policy update.
            discount_factor (float): The discount factor for future rewards.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize the policy weights to zeros
        self.weights = np.zeros(
            (self.env.observation_space.shape[0], self.env.action_space.n)
        )

    def act(self, observation):
        """
        Select an action based on the current policy.

        Args:
            observation (np.ndarray): The current observation.

        Returns:
            The selected action.
        """
        # Compute the probabilities of each action based on the current policy
        logits = np.dot(observation, self.weights)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Choose an action based on the probabilities
        action = np.random.choice(self.env.action_space.n, p=probs)
        return action

    def learn(self, episode):
        """
        Update the policy weights based on the given episode.

        Args:
            episode (list): A list of (observation, action, reward) tuples for one episode.
        """
        # Compute the returns for each time step in the episode
        returns = []
        G = 0
        for t, (observation, action, reward) in enumerate(reversed(episode)):
            G = self.discount_factor * G + reward
            returns.append(G)
        returns.reverse()

        # Compute the policy gradient for each time step in the episode
        for t, (observation, action, reward) in enumerate(episode):
            # Compute the TD error
            td_error = returns[t] - np.dot(observation, self.weights)[action]

            # Update the policy weights using the TD error and the current observation
            grad_log_pi = np.zeros_like(self.weights)
            grad_log_pi[:, action] = observation
            self.weights += self.learning_rate * td_error * grad_log_pi
