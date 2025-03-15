import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.executable)

from ale_py import ALEInterface
ale = ALEInterface()

# Create the environment
env = gym.make("ALE/Pacman-v5", full_action_space=True) 

# Reset environment
obs, info = env.reset()

# Run the environment for a few frames
for _ in range(1000):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)  # Perform action

    # Display the game screen
    plt.imshow(obs)
    plt.axis("off")
    plt.title(f"Reward: {reward}")
    plt.show(block=False)
    plt.pause(0.1)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# Function to create environment:
def make_pacman_env(render_mode="human"):
    env = gym.make("ALE/Pacman-v5", render_mode=render_mode)
    return env