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

env = make_pacman_env(render_mode="human")

state, _ = env.reset()

Transition = namedtuple("Transition", ("obs", "reward", "terminated", "truncated", "info"))

# Define the Deep Q-Network, inheriting from torch.nn.Module
class DQN(nn.Module):
    # Intake action_size (total number of acitons)
    def __init__(self, action_size):
        super(DQN, self).__init__() # This initializes a PyTorch model
        """
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
