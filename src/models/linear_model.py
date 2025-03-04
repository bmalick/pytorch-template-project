import torch
from torch import nn

class LinearRegressionFromScratch(nn.Module):
    def __init__(self, num_inputs):
        super(LinearRegressionFromScratch, self).__init__()
        self.net = nn.Linear(num_inputs, 1)

    def forward(self, x):
        return self.net(x)
