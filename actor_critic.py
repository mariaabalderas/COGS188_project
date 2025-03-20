import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as T
from itertools import count
from pacman_env import make_pacman_env

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Create the Pac-Man Environment**
env = make_pacman_env(render_mode="human", obs_type="grayscale")

# **Preprocess Frames (Grayscale, Resize, Normalize)**
class FrameProcessor:
    def __init__(self):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])

    def process(self, frame):
        return self.transform(frame).to(device)  # Return tensor on device

processor = FrameProcessor()

# **Define Actor-Critic Neural Network**
class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()

        # Shared CNN Layers (Feature Extraction)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # **Actor (Policy Network)**
        self.actor_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
            nn.Softmax(dim=-1)  # Probability distribution over actions
        )

        # **Critic (Value Network)**
        self.critic_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Outputs a single state-value estimate
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        policy = self.actor_fc(x)
        value = self.critic_fc(x)
        return policy, value  # Actor outputs action probabilities, Critic outputs state-value

# **Initialize Actor-Critic Model**
action_size = env.action_space.n
model = ActorCritic(action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# **Training Hyperparameters**
GAMMA = 0.99  # Discount factor
num_episodes = 20

# **Train the Actor-Critic Agent**
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = processor.process(state).unsqueeze(0)  # Add batch dimension

    log_probs = []
    values = []
    rewards = []

    total_reward = 0
    for t in count():
        # **Forward pass: Get action probabilities and value estimate**
        policy, value = model(state)

        # **Select action based on policy**
        action_distribution = torch.distributions.Categorical(policy)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        # **Take action in the environment**
        next_state, reward, terminated, truncated, _ = env.step(action.item())

        # **Preprocess next state**
        next_state = processor.process(next_state).unsqueeze(0)

        # **Store transition data**
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        total_reward += reward

        # **Update state**
        state = next_state

        if terminated or truncated:
            break

    # **Compute Returns and Advantage Estimates**
    returns = []
    discounted_sum = 0
    for r in reversed(rewards):
        discounted_sum = r + GAMMA * discounted_sum
        returns.insert(0, discounted_sum)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    values = torch.cat(values)
    log_probs = torch.cat(log_probs)

    # **Compute Advantage Estimate**
    advantages = returns - values.squeeze()

    # **Compute Actor-Critic Loss**
    actor_loss = -torch.mean(log_probs * advantages.detach())  # Policy gradient loss
    critic_loss = torch.mean(advantages.pow(2))  # MSE loss for value function
    loss = actor_loss + critic_loss

    # **Update Network**
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}, Reward: {total_reward}")

# **Plot Training Progress**
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Actor-Critic Training Progress in Pac-Man")
plt.legend()
plt.show()

# **Close Environment**
env.close()