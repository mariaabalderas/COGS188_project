import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import ale_py
import matplotlib.pyplot as plt
from collections import deque
import torch.nn.functional as F

# Hyperparameters
LEARNING_RATE = 1e-5
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
TARGET_UPDATE = 1000

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Ms. Pac-Man environment
def make_pacman_env():
    return gym.make("ALE/MsPacman-v5", render_mode="rgb_array", obs_type="grayscale")

env = make_pacman_env()

# Neural network for the DQN
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(sample_input).view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions))

    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = F.interpolate(x, size=(84, 84))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Select action using epsilon-greedy policy
def select_action(state, policy_net, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).item()

# Train the model
def optimize_model(policy_net, target_net, buffer, optimizer):
    if len(buffer) < BATCH_SIZE:
        return

    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), device=device, dtype=torch.float32)
    dones = torch.tensor(dones, device=device, dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize networks and optimizer
input_shape = (1, 84, 84)
num_actions = env.action_space.n
policy_net = DQN(input_shape, num_actions).to(device)
target_net = DQN(input_shape, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(BUFFER_SIZE)

# Training loop
epsilon = EPSILON_START
steps_done = 0
rewards_list = []

for episode in range(20):
    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)  # Add channel dimension
    total_reward = 0

    for t in range(10000):
        epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
        action = select_action(state, policy_net, epsilon, num_actions)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        optimize_model(policy_net, target_net, buffer, optimizer)

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

        if done or truncated:
            break

    rewards_list.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

# Plot rewards
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.savefig("rewards_pacman7.png")
plt.show()

env.close()
