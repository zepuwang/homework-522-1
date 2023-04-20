"""


"""

import gymnasium as gym
from customagent import Agent

SHOW_ANIMATIONS = True

env = gym.make("LunarLander-v2", render_mode="human" if SHOW_ANIMATIONS else "none")
observation, info = env.reset(seed=42)

agent = Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
)

total_reward = 0
last_n_rewards = []
for _ in range(100000):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    agent.learn(observation, reward, terminated, truncated)
    total_reward += reward

    if terminated or truncated:
        observation, info = env.reset()
        last_n_rewards.append(total_reward)
        n = min(30, len(last_n_rewards))
        avg = sum(last_n_rewards[-n:]) / n
        # improvement_emoji = "🔥" if (total_reward > avg) else "😢"
        # print(
        #   f"{improvement_emoji} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
        # )
        # if avg > 0:
        #   print("🎉 Nice work! You're ready to submit the leaderboard! 🎉")
        total_reward = 0

env.close()


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
    observation = env.reset()
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
