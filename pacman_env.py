import gymnasium as gym
import time
import random
from ale_py import ALEInterface

# Create the Ms. Pac-Man environment

def make_pacman_env(render_mode, obs_type):
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode, obs_type=obs_type)
    return env

env = make_pacman_env("human", "grayscale")

# Reset the environment to start a new game
observation, info = env.reset()

done = False
for _ in range(5000):  # You can adjust the number of steps you want
    # Choose a random action from the action space
    action = env.action_space.sample()

    # Take the action and get the next observation and reward
    observation, reward, done, truncated, info = env.step(action)

    # Optionally render the screen (this will open a window showing the game)
    env.render()

    # If the game ends or is truncated, print Game Over and reset
    if done or truncated:
        print("Game Over!")
        break

    # Optionally, add a sleep to slow down the game loop
    time.sleep(0.05)

# Close the environment after the game is done
env.close()
