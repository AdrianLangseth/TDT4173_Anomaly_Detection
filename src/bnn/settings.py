import torch
import os
from pathlib import Path

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

CURRENT_DIR = Path(os.path.abspath(__file__)).parents[0]
SRC_DIR = CURRENT_DIR.parents[0]
ROOT_DIR = CURRENT_DIR.parents[1]

MODELS_DIR = os.path.join(ROOT_DIR, os.path.join("models", "bnn"))
DATA_DIR = os.path.join(ROOT_DIR, "data")
