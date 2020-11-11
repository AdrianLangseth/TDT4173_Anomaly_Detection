import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
import pyro
from scipy.stats import entropy
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import SGD, Adam

from data import train_loader, test_loader, val_loader, batch_size, training_set_size, test_set_size
from utils import *

# Base network that we're building on
class FFNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(28*28, 512)
        self.out = Linear(512, 10)
        
    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        output = self.out(hidden)
        return output

lr = 0.0075
optim = Adam({"lr": lr})
num_epochs = 35
num_samples = 50
min_certainty = 0.5

stats_during_training = False

print_area(
    "Model Params", 
    f"{training_set_size = }, {test_set_size = }, {lr = }, {num_epochs = }, {num_samples = }, {min_certainty = }"
)

"""
    BNN Model setup
"""
net = FFNN()
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


"""
    Model training
"""
svi = SVI(model, guide, optim, loss=Trace_ELBO())
def train():
    print_area_start("Training")
    for i in range(num_epochs):
        loss = sum(
            svi.step(X.view(-1, 28*28), y) 
            for X, y in train_loader
        )
        total_epoch_loss = loss / training_set_size
        
        if stats_during_training:
            print_area_start("Epoch info")
            print_area_content(f"Epoch {i}, Loss: {total_epoch_loss}")
            accuracy_all()
            accuracy_exclude_uncertain()
            print_area_end()

    print_area_end()


"""
    Prediction functions
"""
def get_prediction_confidence(x):
    sample_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x.view(-1, 28*28)).data for model in sample_models]
    confidences = np.exp(F.log_softmax(torch.stack(yhats), 2))
    return np.asarray(confidences)

def predict_all(x):
    confidences = np.mean(get_prediction_confidence(x), 0)
    return np.argmax(confidences, axis=1)

"""
    Accuracy functions (prediction interpreters)
"""
def accuracy_all(data_loader=val_loader):
    num_correct = sum(
        np.sum(predict_all(images) == labels.numpy())
        for images, labels in data_loader
    )
    accuracy = num_correct / len(data_loader.dataset)
    print_area("Forced Prediction Accuracy", f"{accuracy = :.2%}")

def accuracy_exclude_uncertain(data_loader=val_loader):
    items_total = len(data_loader.dataset)
    correct_predictions = predictions = 0
    for images, labels in data_loader:
        batch_correct_predictions, batch_predictions = predict_confident(images, labels)
        correct_predictions += batch_correct_predictions
        predictions += batch_predictions
    accuracy = correct_predictions / predictions if predictions else 0
    skipped = items_total - predictions
    skip_percent = skipped / items_total

    print_area(
        "Accuracy With Uncertainty", 
        f"{items_total = }, {skipped = }, {skip_percent = :.2%}",
        f"{predictions = }, {correct_predictions = }, {accuracy = :.2%}"    
    )

def predict_confident(images, labels, min_certainty=min_certainty):
    labels = labels if type(labels) == np.ndarray else labels.view(-1).numpy()
    confidences = np.mean(get_prediction_confidence(images), axis=0)
    certain = np.max(confidences, axis=1) > min_certainty
    confident_predictions = np.argmax(confidences[certain, :], axis=1)
    
    num_confident = np.sum(certain)
    num_correct = np.sum(confident_predictions == labels[certain])

    return num_correct, num_confident

def prediction_data(images, labels):
    labels = labels if type(labels) == np.ndarray else labels.view(-1).numpy()
    confidences = np.mean(get_prediction_confidence(images), axis=0)
    certain = np.max(confidences, axis=1) > min_certainty

    all_predictions = np.argmax(confidences, axis=1)
    confident_predictions = np.where(certain, all_predictions, -1)

    num_confident = np.sum(confident_predictions != -1)
    num_correct_confident = np.sum(confident_predictions == labels)
    
    entropies = entropy(confidences, axis=1)

    return all_predictions, confident_predictions, num_confident, num_correct_confident, entropies 