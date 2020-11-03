import numpy as np
import torch
import pyro
import pprint
import os
import random
from glob import glob
from PIL import Image

import torch.nn.functional as F
from pathlib import Path

from data import train_loader, test_loader, val_loader, batch_size, training_set_size, normalize
from Net import train, guide, model, num_epochs, predict_labels, \
    get_prediction_confidence, metadata, accuracy_all, accuracy_exclude_uncertain, lr

normstr = "" if normalize else "_nonorm"
fname = f"{training_set_size}t_{num_epochs}eps_{lr}lr{normstr}"
model_path = f"../models/bnn/{fname}.model"
metadata_path = f"../meta/bnn/{fname}.txt"

notmnist_path = "../data/notMNIST_small/"
notmnist_files = [
    os.path.join(notmnist_path, directory, fn)
    for directory in os.listdir(notmnist_path)
    for fn in os.listdir(os.path.join(notmnist_path, directory))
]
notmnist_iter = iter(random.sample(notmnist_files, len(notmnist_files)))
def get_notmnist_batch(batch_size=32):
    return np.array([
        np.asarray(Image.open(next(notmnist_iter)))
        for _ in range(batch_size)
    ])


if __name__ == '__main__':
    assert False
    if Path(model_path).is_file():
        pyro.get_param_store().load(model_path)
        print(accuracy_all())
        pprint.pprint(accuracy_exclude_uncertain())
    else:
        train()
        pprint.pprint(metadata)
        pyro.get_param_store().save(model_path)
        with open(metadata_path, mode="a+") as f:
            f.write(pprint.pformat(metadata))

    

