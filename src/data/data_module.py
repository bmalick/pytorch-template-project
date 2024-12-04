import torch

class DataModule:
    def __init__(self, root="../data", train_batch_size=8, eval_batch_size=4, num_workers=4):
        self.root = root
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def get_dataloader(self, train: bool):
        data = self.train if train else self.eval
        batch_size = self.train_batch_size if train else self.eval_batch_size
        
        return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size,
                                           shuffle=train, num_workers=self.num_workers,
                                           # pin_memory=True
                                           )

    def train_dataloader(self): return self.get_dataloader(train=True)

    def eval_dataloader(self): return self.get_dataloader(train=False)

