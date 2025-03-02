import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import torch

def get_logdir(logdir):
    # TODO: add time
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    return logdir+"--"+timestamp

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_summary(trainer):
    # TODO: use torchinfo fro model summary
    summary = (f"## Logdir: {trainer.logdir}\n\n" + "## Dataset:\n" +
        str(getattr(trainer, "data", None)) + "\n\n" + "## Model:\n" + str(getattr(trainer, "model", None)) +
        "## Loss:\n" + str(getattr(trainer, "loss", None)) + "\n\n" +
        "## Optimizer:\n" + str(getattr(trainer, "optimizer", None)) +
        "## Metrics:\n" + str(getattr(trainer, "metric_funcs", None)) +
        f"\n\n## Seed: {trainer.config['seed']}" if "seed" in trainer.config else "" +
        f"\n\n## Epochs: {trainer.config['epochs']}"
        # "## Model architecture\n" + f"{trainer.model.__str__}\n\n"
    )

    with open(os.path.join(trainer.logdir, "summary.txt"), 'w') as f:
        f.write(summary)

    trainer.neptune_run["summary"] = summary

def save_results(trainer, name: str, train_val: float, eval_val: float):
    trainer.results[name].append({"train": train_val, "eval": eval_val})

