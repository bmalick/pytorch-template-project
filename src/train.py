# Standard imports
import os
import sys
import yaml
import logging
import logging.config


import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Local imports
from . import models
from . import data
from . import utils
# from . import optimizers
# from . import metrics
# from . import callbacks

class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        # self.data_collator  = data_collator
        logdir = utils.get_logdir(self.config["logdir"])
        os.makedirs("logs", exist_ok=True)
        os.makedirs(logdir, exist_ok=True)

        with open("configs/logs.yaml", "r") as f:
            log_config_dict = yaml.safe_load(f)
        log_config_dict["handlers"]["file"]["filename"] = os.path.join(logdir, "training.log")
        logging.config.dictConfig(log_config_dict)
        self.logger = logging.getLogger("train")

        self.logger.info(f"Logging into {logdir}")
        self.logger.info(f"Save config file into {logdir}")
        with open(os.path.join(logdir, "config.yaml"), 'w') as f:
            yaml.safe_dump(self.config, f)

        self.logdir = logdir

        seed = self.config.get("seed")
        # if seed is not None: utils.set_seed(seed) # TODO: Uncomment for reproductibilty

        self.writter = SummaryWriter(log_dir=logdir)

    def get_config_item(self, name: str):
        args = self.config.get(name)
        assert args is not None, f"{name} does not appear in config file"
        name = args.get("class")
        params = args.get("params", {})
        return name, params

    def fit(self):
        self.prepare_data()
        self.prepare_model()
        self.configure_optimizers()
        self.configure_metrics()
        self.configure_callbacks()

        self.get_summary()
        # TODO: add metrics
        self.results = {"loss": []}

        self.epoch = 0
        self.logger.info("Start of training")
        for _ in range(self.config["epochs"]):
            self.fit_epoch()
            self.epoch += 1
        self.logger.info("End of training")

    def prepare_data(self):
        self.logger.info("Building dataloaders")
        klass_name, params = self.get_config_item(name="data")
        klass = getattr(data, klass_name, None) # data package
        assert klass is not None, f"Class {klass_name} is not defined in data"
        self.data = klass(**params)
        self.train_dataloader = self.data.train_dataloader()
        self.eval_dataloader = self.data.eval_dataloader()

    def prepare_model(self):
        self.logger.info("Building loss")
        klass_name, params = self.get_config_item(name="loss")
        klass = getattr(nn, klass_name, None) # nn package
        assert klass is not None, f"Class {klass_name} is not defined in nn"
        self.loss = klass(**params)

        self.logger.info("Building model")
        klass_name, params = self.get_config_item(name="model")
        klass = getattr(models, klass_name, None) # models package
        assert klass is not None, f"Class {klass_name} is not defined in models"
        self.model = klass(**params)

    def configure_optimizers(self):
        self.logger.info("Building optimizers")
        klass_name, params = self.get_config_item(name="optimizer")
        params = {
            "params": self.model.parameters(),
            **params
        }
        klass = getattr(optimizers, klass_name, None) # optimizers package
        assert klass is not None, f"Class {klass_name} is not defined in optimizers"
        self.optimizer= klass(**params)

    def configure_metrics(self):
        pass

    def configure_callbacks(self):
        pass


    def fit_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_dataloader):
            loss = self.loss(self.model(*batch[:-1]), batch[-1])
            self.optimizer.zero_grad()
            loss.backward()
            # with torch.no_grad(): self.optimizer.step()
            self.optimizer.step()

            epoch_loss += loss.item()
            self.writter.add_scalar(tag="Loss/train",
                                    scalar_value=epoch_loss / (i+1),
                                    global_step=self.epoch * len(self.train_dataloader) + i)


        epoch_loss /= len(self.train_dataloader)

        # if self.eval_dataloader is None: # TODO: review this variable
        #     return
        self.model.eval()
        eval_epoch_loss = 0
        for i, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                loss = self.loss(self.model(*batch[:-1]), batch[-1])
            eval_epoch_loss += loss.item()
            self.writter.add_scalar(tag="Loss/eval",
                                    scalar_value=eval_epoch_loss / (i+1),
                                    global_step=self.epoch * len(self.eval_dataloader) + i)

        self.logger.info("Epoch: %3d, loss: %5.3f, val_loss: %5.3f" % (self.epoch+1, epoch_loss, eval_epoch_loss))
        save_results(trainer, "loss", epoch_loss, eval_epoch_loss)

