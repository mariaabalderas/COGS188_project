import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from dm_control import suite
from dataclasses import dataclass
from pacman_env import make_pacman_env
from stable_baselines3.common.buffers import ReplayBuffer

env = make_pacman_env(render_mode="human")

state, _ = env.reset()

Transition = namedtuple("Transition", ("obs", "reward", "terminated", "truncated", "info"))

# Define the Deep Q-Network, inheriting from torch.nn.Module
class DQN(nn.Module):
    # Intake action_size (total number of acitons)
    def __init__(self, action_size):
        super(DQN, self).__init__() # This initializes a PyTorch model
        """
        This step approximates the Q-function, predicting the Q-value for each
        possible action, given the current state. 

        Convolutional neural networks are good for extracting visual, 
        spatial features and patterns, which is why we will use them to
        analyze our observation space states.

        Since the pacman space is 2d, we will use Conv2d layers, which intakes 2d data, 
        has 2d kernel, and has a 2d output.

        For input Conv layer:
            - 4 input channels
            - 32 output channels; this layer learns 32 different filters
            - 8x8 kernel
            - stide of 4 for downsampling
            - ReLU for non-linearity
        
        For second Conv layer:
            - 32 input feature maps from previous layer
            - 64 output feature maps
            - 4x4 kernel
            - stride of 2
        
        For third Conv layer:
            - 64 input feature maps
            - 64 output feature maps
            - 3x3 kernel
            - stride of 1 (no downsampling to retain details)
        
        The output of the CNN is a feature map, which we flatten and pass through
        fully connected layers for decision-making, using the fc linear network.
        
        """
        self.conv = nn.Sequential( # Use Sequential to set up pipeline of layers
            nn.Conv2d(4, 32, kernel_size=8, stride=4), 
            nn.ReLU(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Process image inputs for DQN: grayscale, resized, stacked frames

class FrameProcessor:
    def __init__(self):
        self.transform = T.compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])
    
    def process(self, frame):
        return self.transform(frame).numpy()


class DQNAgent:
    def __init__(self, action_size, lr=1e-4, gamma=0.99, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995):
        """
        action_size: Contains all possible actions in environment
        lr: Learning rate
        gamma: 
        epsilon: Our ratio for greedy vs random actions
        epsilon_min: 
        epsilon_decay:
        batch_size:
        update_target_frequency: 

        optimizer:
            optimizer:
            loss_fn:
            memory: Experiences stored here using Replay Buffer
        """
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = 32
        self.update_target_frequency = 1000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        self.policy_net = DQN(action_size).to(self.device)
        self.target_net = DQN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()

    def select_action(self, state):
        """
        Choose action using epsilon-greedy formula.
        Args:
            state (3d NumPy Array): The current observation state, given by an RGB image
        
        Returns:
            our action (tensor), selected by epsilon-greedy
        """
        
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Convert state to a tensor, then pass through the cnn to determine action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad(): # Prevents gradient from being computed and stored, which would take time and memory
                return torch.argmax(self.policy_net(state_tensor)).item()
            
    def train(self):
        """
        In this step, our agent interacts with the environment, and the resulting reedback
        (reward, next state, and condition) are given. The resulting attributes are 
        stored in memory.

        idk if this is rt --> Next, we randomly sample a batch of experiences from replay buffer, and update
        the q-network, using the Bellman equation to check for convergence. 

        """
        # Checks if memory is smaller than the necessary batch size. It does not run
        # if it is too small. 
        if len(self.memory) < self.batch_size:
            return
        
        # Get a sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert all resulting attributes to a PyTorch Tensor and move to local device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute targeet Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1-dones) *self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target neetwork
        if np.random.randint(0, self.update_target_frequency) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay






