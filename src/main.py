#!/home/malick/miniconda3/envs/pt/bin/python

import sys
import yaml
from src import train

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage : python -m src.main <config file path>")
        sys.exit(-1)
    
    config = yaml.safe_load(open(sys.argv[1], "r"))
    train.Trainer(config).fit()

