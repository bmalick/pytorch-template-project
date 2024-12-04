import os
import time
import random
import numpy as np
import torch


def get_logdir(logdir):
    # TODO: add time
    i = 0
    while True:
        log_path = logdir + "_" + str(i)
        # log_path = logdir + "-" + str(i)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_summary(trainer):
    # TODO: use torchinfo fro model summary
    summary = (f"## Logdir: {trainer.logdir}\n" + "## Dataset: " +
        trainer.config['data']['class'] + "\nParameters:\n" +
        "\n".join([''.ljust(11)+f"{k}".ljust(20) + f"{v}" for k,v in trainer.config["data"]["params"].items()]) +
        "\n\n" + "## Loss: " + trainer.config['loss']['class'] + "\n" +
        "## Optimizer: " + trainer.config['optimizer']['class'] + "\nParameters:\n" +
        "\n".join([''.ljust(11)+f"{k}".ljust(20) + f"{v}" for k,v in trainer.config["optimizer"]["params"].items()]) +
        f"\n## Seed: {trainer.config['seed']}" if "seed" in trainer.config else "" +
        f"\n## Epochs: {trainer.config['epochs']}"

        # "## Model architecture\n" + f"{trainer.model.__str__}\n\n"
    )

    with open(os.path.join(trainer.logdir, "summary.txt"), 'w') as f:
        f.write(summary)

def save_results(trainer, name: str, train_val: float, eval_val: float):
    trainer.results[name].append({"train": train_val, "eval": eval_val})

def save_figures(trainer):
    for name, results in trainer.results.items()
        plt.plot([r["train"] for r in results], label="train")
        plt.plot([r["eval"] for r in results], label="eval")
        plt.title(name)
        plt.savefig(os.path.join(trainer.logdir, "loss-train-vs-eval.jpg"))
        plt.legend()
        plt.close()
