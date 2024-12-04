import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def training_step(self, batch):
        return self.loss(*batch[:-1], batch[-1])
    
    def validation_step(self, batch):
        return self.loss(*batch[:-1], batch[-1])

    def configure_optimizers(self):
        pass
