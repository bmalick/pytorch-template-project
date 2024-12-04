import torch
from torch import nn

class LinearRegressionFromScratch(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.w = torch.normal(mean=0, std=0.01, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def parameters(self): return [self.w, self.b]
