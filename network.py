import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class FeedForwardNN(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(input_dimension, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dimension)

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)

        activation1 = F.relu(self.layer1(observation))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
