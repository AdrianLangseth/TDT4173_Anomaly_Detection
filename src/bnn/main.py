import torch
import pyro
import os
from pathlib import Path

import Net
import data
from settings import ROOT_DIR, MODELS_DIR


def setup_model(training_set_index=0):
    training_set_size = data.training_set_sizes[training_set_index]
    data.set_train_size(training_set_size)
    
    lr = 0.001
    Net.optim = pyro.optim.Adam({"lr": lr})
    Net.num_epochs = 1_000
    Net.num_samples = 50
    Net.min_certainty = 0.45

    model_path = os.path.join(MODELS_DIR, f"{training_set_size}_train.model")
    Net.load_model(model_path)
