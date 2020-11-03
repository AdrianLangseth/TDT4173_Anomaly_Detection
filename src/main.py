import torch
import pyro
import pprint

import torch.nn.functional as F
from pathlib import Path

from data import training_set_size, normalize, notmnist_loader
from Net import train, num_epochs, metadata, accuracy_all, accuracy_exclude_uncertain, lr

normstr = "" if normalize else "_nonorm"
fname = f"{training_set_size}t_{num_epochs}eps_{lr}lr{normstr}"
model_path = f"../models/bnn/{fname}.model"
metadata_path = f"../meta/bnn/{fname}.txt"


if __name__ == '__main__':
    print("Using model:", fname)
    if Path(model_path).is_file():
        pyro.get_param_store().load(model_path)
        print("Model accuracy:", accuracy_all())
        print("Model accuracy with uncertainty:")
        pprint.pprint(accuracy_exclude_uncertain())
    else:
        train()
        pprint.pprint(metadata)
        pyro.get_param_store().save(model_path)
        with open(metadata_path, mode="a+") as f:
            f.write(pprint.pformat(metadata))
    # print(accuracy_all(mnistloader))
    # pprint.pprint(accuracy_exclude_uncertain())
    # pprint.pprint(accuracy_exclude_uncertain(notmnist_loader))

    

