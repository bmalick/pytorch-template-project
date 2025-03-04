import torch

class CustomPolynomialLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, end_lr=1e-6, power=1.0, last_epoch=-1):
        self.max_epochs = max_epochs
        self.end_lr = float(end_lr)
        self.power = float(power)
        super(CustomPolynomialLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        epoch = self.last_epoch
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        return [(base_lr - self.end_lr) * decay + self.end_lr for base_lr in self.base_lrs]
