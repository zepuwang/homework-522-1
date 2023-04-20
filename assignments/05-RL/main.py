"""


"""

import gymnasium as gym
from customagent import Agent

SHOW_ANIMATIONS = True

env = gym.make("LunarLander-v2", render_mode="human" if SHOW_ANIMATIONS else "none")
observation, info = env.reset(seed=42)


# Import the Agent class


# Create the CartPole-v1 environment
env = gym.make("CartPole-v1")

# Create an instance of the Agent class
agent = Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    learning_rate=0.01,
    discount_factor=0.99,
)

# Train the agent for 100 episodes
for episode in range(100):
    # Reset the environment for a new episode
    done = False
    total_reward = 0

    while not done:
        # Select an action using the agent's policy
        action = agent.act(observation)

        # Take a step in the environment using the selected action
        next_observation, reward, done, info = env.step(action)

        # Update the agent's policy based on the observed transition
        agent.learn(observation, action, reward, next_observation, done)

        # Update the current observation and reward
        observation = next_observation
        total_reward += reward

    print(f"Episode {episode}: Total reward = {total_reward}")

# Close the environment
env.close()
