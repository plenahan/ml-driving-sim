import torch
from torch import nn
import numpy as np


def _orthogonal_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class FeedForwardNN(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_size=256, output_gain=np.sqrt(2)):
        super().__init__()

        self.layer1 = _orthogonal_init(nn.Linear(input_dimension, hidden_size), gain=np.sqrt(2))
        self.layer2 = _orthogonal_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2))
        self.layer3 = _orthogonal_init(nn.Linear(hidden_size, output_dimension), gain=output_gain)

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)

        activation1 = torch.tanh(self.layer1(observation))
        activation2 = torch.tanh(self.layer2(activation1))
        return self.layer3(activation2)
