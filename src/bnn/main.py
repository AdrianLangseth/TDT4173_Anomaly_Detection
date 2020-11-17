import torch
import pyro
import pprint

import torch.nn.functional as F
from pathlib import Path

from src.bnn.data import training_set_size, normalize, notmnist_loader, batch_size
from src.bnn.Net import train, num_epochs, accuracy_all, accuracy_exclude_uncertain, lr

normstr = "" if normalize else "_nonorm"
fname = f"{training_set_size}t_{num_epochs}eps_{lr}lr_{batch_size}bs{normstr}"
model_path = f"../../models/bnn/{fname}.model"


if __name__ == '__main__':
    print("Using model:", fname)
    if Path(model_path).is_file():
        print("found pretrained model!")
        pyro.get_param_store().load(model_path)
    else:
        print("Found no model. Training one now...")
        train()
        pyro.get_param_store().save(model_path)

    accuracy_all()
    accuracy_exclude_uncertain()

    

