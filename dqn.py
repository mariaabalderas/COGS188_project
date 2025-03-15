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
        
        """
        self.conv = nn.Sequential( # Use Sequential to set up pipeline of layers
            nn.Conv2d(4, 32, kernel_size=8, stride=4), 
            nn.ReLU(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
