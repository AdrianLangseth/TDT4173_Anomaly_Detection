import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import SGD, Adam

from data import train_loader, test_loader, val_loader, batch_size, training_set_size, test_set_size

class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.out = Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        output = self.out(hidden)
        return output

hl_size = 512
net = NN(28*28, hl_size, 10)
lr = 0.005
optim = Adam({"lr": lr})
num_epochs = 20
num_samples = 20
min_certainty = 0.02

metadata = {
    "training_set_size": training_set_size,
    "test_set_size": test_set_size,
    "num_epochs": num_epochs,
    "lr": lr,
    "num_samples": num_samples,
    "min certainty": min_certainty,
    "losses": [],
    "accuracies": [],
    "accuracies_with_uncertainty": {}
}


log_softmax = torch.nn.LogSoftmax(dim=1)
softplus = torch.nn.Softplus()

def model_normal(like):
    return dist.Normal(loc=torch.zeros_like(like), scale=torch.ones_like(like))

def model(x_data, y_data):
    priors = dict(zip(
        ('fc1.weight', 'fc1.bias', 'out.weight', 'out.bias'),
        map(model_normal, (net.fc1.weight, net.fc1.bias, net.out.weight, net.out.bias))
    ))
    lifted_module = pyro.random_module("module", net, priors)()
    lhat = log_softmax(lifted_module(x_data))
    
    pyro.sample("obs", dist.Categorical(logits=lhat), obs=y_data)

def guide_normal(name, like):
    return dist.Normal(
        loc=pyro.param(f"{name}_exp", torch.randn_like(like)),
        scale=softplus(pyro.param(f"{name}_var", torch.randn_like(like)))
    )

def guide(x_data=None, y_data=None):
    priors = {
        'fc1.weight': guide_normal("fc1_w", net.fc1.weight),
        'fc1.bias': guide_normal("fc1_b", net.fc1.bias),
        'out.weight': guide_normal("out_w", net.out.weight).independent(1),
        'out.bias': guide_normal("out_b", net.out.bias)
    }
    return pyro.random_module("module", net, priors)()

svi = SVI(model, guide, optim, loss=Trace_ELBO())
def train():
    for i in range(num_epochs):
        loss = sum(
            svi.step(X.view(-1, 28*28), y) 
            for X, y in train_loader
        )
        total_epoch_loss = loss / training_set_size
        metadata["losses"].append(f"{total_epoch_loss:.2f}")
        metadata["accuracies"].append(accuracy_all())
        metadata["accuracies_with_uncertainty"][i] = accuracy_exclude_uncertain()
        print("Epoch ", i, " Loss ", total_epoch_loss)

def predict_labels(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)

def get_prediction_confidence(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)

def accuracy_all(data_loader=test_loader):
    num_correct = sum(
        np.sum(predict_labels(images.view(-1, 28*28)) == labels.numpy())
        for images, labels in data_loader
    )
    accuracy = num_correct / len(data_loader.dataset)
    return f"{accuracy:.2%}"

def accuracy_exclude_uncertain(data_loader=test_loader):
    num_items = len(data_loader.dataset)
    correct_predictions = total_predictions = 0
    for images, labels in data_loader:    
        correct, total = get_confident_predictions(images, labels)
        total_predictions += total
        correct_predictions += correct
    accuracy = correct_predictions / total_predictions if total_predictions else 0
    return {
        "total": num_items, "skipped": num_items - total_predictions, 
        "skip%": f"{1 - total_predictions/num_items:.2%}", "predicted": total_predictions, 
        "correct": correct_predictions, "accuracy": f"{accuracy:.2%}"
    }

def get_confident_predictions(images, labels, min_certainty=min_certainty):
    y = get_prediction_confidence(images).transpose(1, 2, 0) # row: sample. col: class. third: model
    predictions = np.median(np.exp(y), axis=2)  # get median of model predictions for each class
    certain = np.max(predictions, axis=1) > min_certainty # index of all predictions more certain than min threshold
    num_correct = np.sum(
        np.argmax(predictions[certain, :], axis=1) == labels.numpy()[certain]
    )
    return num_correct, np.sum(certain)