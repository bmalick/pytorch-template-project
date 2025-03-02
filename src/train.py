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
import neptune

# Local imports
from src import models
from src import data
from src import utils
# from src import optimizers
from src import metrics
# from src import callbacks

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
from src import models
from src import data
from src import utils
# from src import optimizers
# from src import metrics
# from src import callbacks

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

        self.logger.info(f"Logdir: {logdir}")
        with open(os.path.join(logdir, "config.yaml"), 'w') as f:
            yaml.safe_dump(self.config, f)

        self.logdir = logdir

        seed = self.config.get("seed")
        # if seed is not None: utils.set_seed(seed) # TODO: Uncomment for reproductibilty

        self.writer = SummaryWriter(log_dir=logdir)
        if "neptune" in self.config:
            self.neptune_run = neptune.init_run(project=self.config["neptune"])
            self.neptune_run["config"] = self.config
            self.logger.info(f"Neptune: {self.config['neptune']}")


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

        utils.get_summary(self)
        # TODO: add metrics
        self.results = {"loss": []}

        self.epoch = 0
        self.logger.info("Start of training")
        for _ in range(self.config["epochs"]):
            self.fit_epoch()
            self.epoch += 1
        self.logger.info("End of training")

    def prepare_data(self):
        klass_name, params = self.get_config_item(name="data")
        klass = getattr(data, klass_name, None) # data package
        assert klass is not None, f"Class {klass_name} is not defined in data"
        self.data = klass(**params)
        self.logger.info(f"Dataset: {self.data}")
        # TODO: Uncomment for custom dataset
        # self.data.split()
        self.train_dataloader = self.data.train_dataloader()
        self.eval_dataloader = self.data.eval_dataloader()

    def prepare_model(self):
        klass_name, params = self.get_config_item(name="loss")
        klass = getattr(nn, klass_name, None) # nn package
        assert klass is not None, f"Class {klass_name} is not defined in nn"
        self.criterion = klass(**params)
        self.logger.info(f"Loss: {self.criterion}")

        klass_name, params = self.get_config_item(name="model")
        klass = getattr(models, klass_name, None) # models package
        assert klass is not None, f"Class {klass_name} is not defined in models"
        self.model = klass(**params)
        self.logger.info(f"Model:\n{self.model}")

    def configure_optimizers(self):
        klass_name, params = self.get_config_item(name="optimizer")
        params = {
            "params": self.model.parameters(),
            **params
        }
        klass = getattr(optimizers, klass_name, None) # optimizers package
        assert klass is not None, f"Class {klass_name} is not defined in optimizers"
        self.optimizer= klass(**params)
        self.logger.info(f"Optimizers: {self.optimizer}")

    def configure_metrics(self):
        names = self.config.get("metrics")
        self.metric_funcs = {n: getattr(metrics, n, None) for n in names}
        self.compute_metrics = lambda y_true, y_pred: {n: self.metric_funcs[n](y_true=y_true, y_pred=y_pred).item() for n in self.metric_funcs}
        self.logger.info(f"Metrics: {self.metric_funcs}")

    def configure_callbacks(self):
        pass

    def fit_epoch(self):
        self.model.train()
        epoch_loss = 0
        num_samples = 0
        # TODO: take account the last batch may not have the same ammount of examples
        # DO the same for metrics
        total_metrics = {n:0. for n in self.metric_funcs}
        for i, batch in enumerate(self.train_dataloader):
            output = self.model(*batch[:-1])
            loss = self.criterion(output, batch[-1])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()*batch[0].size(0)
            num_samples += batch[0].size(0)
            # TODO: change output with pred (we may need to apply sigmoid for example)
            train_metrics = self.compute_metrics(y_true=batch[-1], y_pred=output)
            for k in train_metrics:
                total_metrics[k] += batch[0].size(0)*train_metrics[k]

            self.writer.add_scalar(tag="Loss/train",
                                    scalar_value=epoch_loss / num_samples,
                                    global_step=self.epoch * len(self.train_dataloader) + i)
            if hasattr(self, "neptune_run"):
                self.neptune_run["train/Loss"].append(epoch_loss / num_samples)

            metrics_summary = ""
            for k in total_metrics:
                v = total_metrics[k]/num_samples
                self.writer.add_scalar(tag=f"{k.title()}/train",
                                       scalar_value=v,
                                       global_step=self.epoch * len(self.train_dataloader) + i)
                if hasattr(self, "neptune_run"):
                    self.neptune_run[f"train/{k.title()}"].append(v)
                metrics_summary += f", train_{k}: {v:.3f}"

            self.logger.info("[Epoch %d/%d] [step %d/%d], train_loss: %5.3f%s" % (
                self.epoch+1, self.config["epochs"], i+1,
                len(self.train_dataloader), epoch_loss/num_samples, metrics_summary))

        epoch_loss /= num_samples
        total_metrics = {k: v / num_samples for k, v in total_metrics.items()}

        if self.eval_dataloader is None: # TODO: review this variable
            return

        self.model.eval()
        eval_epoch_loss = 0
        num_samples = 0
        total_metrics = {n:0. for n in self.metric_funcs}
        for i, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                output = self.model(*batch[:-1])
                # pred = (torch.sigmoid(output) > 0.5).int()
                loss = self.criterion(output, batch[-1])
            eval_epoch_loss += loss.item()*batch[0].shape[0]
            num_samples += batch[0].shape[0]
            # TODO: change output with pred (we may need to apply sigmoid for example)
            eval_metrics = self.compute_metrics(y_true=batch[-1], y_pred=output)
            for k in eval_metrics:
                total_metrics[k] += batch[0].size(0)*eval_metrics[k]
            self.writer.add_scalar(tag="Loss/eval",
                                    scalar_value=eval_epoch_loss / num_samples,
                                    global_step=self.epoch * len(self.eval_dataloader) + i)
            if hasattr(self, "neptune_run"):
                self.neptune_run["eval/Loss"].append(eval_epoch_loss / num_samples)
            for k in total_metrics:
                v = total_metrics[k]/num_samples
                self.writer.add_scalar(tag=f"{k.title()}/eval",
                                       scalar_value=v,
                                       global_step=self.epoch * len(self.train_dataloader) + i)
                if hasattr(self, "neptune_run"):
                    self.neptune_run[f"eval/{k.title()}"].append(v)

        eval_epoch_loss /= num_samples
        total_metrics = {k: v / num_samples for k, v in total_metrics.items()}

        utils.save_results(self, "loss", epoch_loss, eval_epoch_loss)
        self.logger.info("[End epoch %d/%d], eval_loss: %5.3f, %s" % (self.epoch+1, self.config["epochs"], eval_epoch_loss,
                                                                      ", ".join([f"eval_{k}: {v:.3f}" for k,v in total_metrics.items()])))
