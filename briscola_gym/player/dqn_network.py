import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50


# TODO Change the network

class Network(nn.Module):
    '''
    Class used to define the network structure
    '''

    def __init__(self, observation_space):
        super(Network, self).__init__()

        # Define the input size
        input_size = observation_space['my_points'].n + observation_space['other_points'].n + \
                     observation_space['hand_size'].n + observation_space['other_hand_size'].n + \
                     observation_space['remaining_deck_cards'].n + \
                     observation_space['hand'][0].n + observation_space['hand'][1].n + observation_space['hand'][2].n + \
                     observation_space['table'][0].n + observation_space['table'][1].n + \
                     observation_space['my_discarded'][0].n + observation_space['my_discarded'][1].n + ... + \
                     observation_space['my_discarded'][39].n + \
                     observation_space['other_discarded'][0].n + observation_space['other_discarded'][1].n + ... + \
                     observation_space['other_discarded'][39].n + \
                     observation_space['turn'].n + observation_space['briscola'].n + observation_space['order'].n

        # Define the hidden layer sizes
        hidden_sizes = [64, 32]

        # Define the output size
        output_size = 3

        # Define the MLP layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x