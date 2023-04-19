import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class QNetwork(nn.Module):
    """
    It is a model created by Zepu
    """

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    It is a model created by Zepu
    """

    def __init__(self, buffer_size):
        """
        It is a model created by Zepu
        """

        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, observation, action, reward, next_observation, done):
        """
        It is a model created by Zepu
        """

        self.buffer.append((observation, action, reward, next_observation, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        """
        It is a model created by Zepu
        """

        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        observations, actions, rewards, next_observations, dones = zip(
            *[self.buffer[i] for i in batch]
        )
        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(next_observations),
            np.array(dones),
        )


class Agent:
    """
    It is a model created by Zepu
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate=0.001,
        discount_factor=0.99,
        buffer_size=10000,
        batch_size=32,
        update_interval=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        """
        It is a model created by Zepu
        """

        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.target_network = QNetwork(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.update_counter = 0
        self.epsilon = self.epsilon_start

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        It is a model created by Zepu
        """

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            observation = (
                torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                q_values = self.q_network(observation)
            action = torch.argmax(q_values, dim=1).item()
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

        self.replay_buffer.add(observation, action, reward, next_observation, done)
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        observations = torch.from_numpy(observations).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_observations = torch.from_numpy(next_observations).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        q_values = self.q_network(observations).gather(1, actions)
        target_q_values = self.target_network(next_observations).max(1)[0].unsqueeze(1)
        target_q_values[dones] = 0
        target_q_values = rewards + self.discount_factor * target_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
