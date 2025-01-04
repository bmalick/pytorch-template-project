#!/home/malick/miniconda3/envs/pt/bin/python

import torch
import matplotlib.pyplot as plt

# Local imports
from src.data import data_module

class LinearDataset(data_module.DataModule):
    def __init__(self, w, b, noise=0.1, n: int = 1000, root=None, train_batch_size=8, eval_batch_size=4, num_workers=4):
        super().__init__(root=root, train_batch_size=train_batch_size,
                         eval_batch_size=eval_batch_size, num_workers=num_workers)
        self.w = torch.tensor(w)
        self.b = torch.tensor(b)

        train_size = int(n*0.8)
        noise = torch.randn(n, 1) * noise
        x = torch.randn(n, len(self.w))
        y = torch.matmul(x, self.w.reshape(-1, 1)) + self.b + noise

        self.train = torch.utils.data.TensorDataset(x[:train_size], y[:train_size])
        self.eval = torch.utils.data.TensorDataset(x[train_size:], y[train_size:])
        

if __name__ == "__main__":
    data = LinearDataset(w=[1.], b=[1.])
    train_data = data.train
    train_x, train_y = [], []
    for x,y in train_data:
        train_x.append(x.item())
        train_y.append(y.item())
    plt.scatter(train_x,train_y, s=10)
    plt.show()
