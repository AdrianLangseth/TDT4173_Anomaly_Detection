import pyro
import os
from pathlib import Path

import Net
import data


models_dir = os.path.join(Path(os.path.abspath(__file__)).parents[2], os.path.join("models", "bnn"))
def setup_model(training_set_index=0):
    training_set_size = data.training_set_sizes[training_set_index]
    data.set_train_size(training_set_size)
    
    lr = 0.0075
    Net.optim = pyro.optim.Adam({"lr": lr})
    Net.num_epochs = 25 + 10*training_set_index
    Net.num_samples = 50
    Net.min_certainty = 0.45

    model_name = f"{training_set_size}_train.model"
    model_path = os.path.join(models_dir, model_name)
    Net.load_model(model_path)


if __name__ == '__main__':
    for i in range(len(data.training_set_sizes)):
        setup_model(i)
        print(Net.accuracy_all())
        print(Net.accuracy_exclude_uncertain())
