import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
print(sys.executable)

from ale_py import ALEInterface
ale = ALEInterface()

# Create the environment
env = gym.make("ALE/Pacman-v5", full_action_space=True) 

# Reset environment
obs, info = env.reset()

#enable interactive mode
#plt.ion()

# Run the environment for a few frames
for _ in range(1000):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)  # Perform action

<<<<<<< Updated upstream
    # Display the game screen
    plt.imshow(obs)
    plt.axis("off")
    plt.title(f"Reward: {reward}")
    plt.show(block=False)
    plt.pause(0.1)
=======
    # Convert image from RGB to BGR for OpenCV
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("Pac-Man", obs)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to close
        break
>>>>>>> Stashed changes

    if terminated or truncated:
        obs, info = env.reset()

# close the environment
env.close()
<<<<<<< Updated upstream

# Function to create environment:
def make_pacman_env(render_mode="human", obs_type="grayscale"):
    env = gym.make("ALE/Pacman-v5", render_mode=render_mode, obs_type=obs_type)
    return env
=======
cv2.destroyAllWindows()

'''   # Display the game screen
    plt.clf()
    plt.imshow(obs)
    plt.axis("off")
    plt.title(f"Reward: {reward}")
    #plt.show(block=False)
    plt.draw()
    plt.pause(0.1)

    if terminated or truncated:
        obs, info = env.reset()

# close the environment        
plt.ioff()
plt.close()
env.close()
'''
>>>>>>> Stashed changes
