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

    def __init__(self, output_size = 1, input_channels = 4, lr = 1e-4,
                device=torch.device('cpu')):
        super(Network, self).__init__()

        # Number of channels of the input
        self.input_size = input_channels

        # Define the networks
        '''
        Both the network have structure: 3x(Conv+ReLU+MaxPoo) + 2x(Linear)
        '''
        self.net = nn.Sequential(
                        # Convolution
                        nn.Conv2d(self.input_size, 16, kernel_size=4, stride=2),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(16),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(16, 32, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        
                        nn.Conv2d(32, 64, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        # Linear
                        nn.Flatten(),
                        nn.Linear(1024, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, output_size)
                    ).to(device)
        
        # Initialize weights
        self.init_weights()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        '''
        forward of the network given the input element x
        '''

        if torch.is_tensor(x) and x.shape[0] == self.input_size:
            # x is a tensor but it doesn't have the batch dimension
            x = x.unsqueeze(0) 
        elif x.shape[0] == self.input_size:
            # x is not a tensor and also is missing the batch dimension
            x = torch.FloatTensor(x).unsqueeze(0)
        elif not torch.is_tensor(x):
            # x is of the right shape but not a tensor
            x = torch.FloatTensor(x)

        # x is a tensor of the right size, pass it through the network
        output = self.net(x)
        # return the output
        return output

