import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
import pyro
from scipy.stats import entropy
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pathlib import Path
import warnings

import src.bnn.data as data
from src.bnn.data import device

warnings.filterwarnings("ignore", category=FutureWarning) # suppress deprecation warnings for pyro.random_module

# Base network that we're building on
class FFNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(28*28, 512)
        self.out = Linear(512, 10)
        
    def forward(self, x):
        hidden = F.softplus(self.fc1(x))
        output = self.out(hidden)
        return output

net = FFNN().to(device)
sample_models = None
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

def load_model(model_path):
    print("attempting to load model:", model_path)
    pyro.clear_param_store()
    if Path(model_path).is_file():
        print("found pretrained model!")
        pyro.get_param_store().load(model_path, map_location=device)
    else:
        print("Found no model. Training one now...")
        train_model()
        pyro.get_param_store().save(model_path)
    
    global sample_models
    sample_models = [guide(None, None) for _ in range(num_samples)]

"""
    Model training
"""
def train_model():
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    for i in range(num_epochs):
        loss = sum(
            svi.step(X, y) 
            for X, y in data.train_loader
        )
        total_epoch_loss = loss / len(data.train_loader.dataset)
        print("epoch:", i, "loss:", total_epoch_loss)
    
"""
    Prediction functions
"""
def get_prediction_confidence(x):
    yhats = [model(x).data for model in sample_models]
    confidences = torch.exp(F.log_softmax(torch.stack(yhats), 2))
    return confidences

def predict_all(x):
    confidences = torch.mean(get_prediction_confidence(x), 0)
    return torch.argmax(confidences, axis=1)

"""
    Accuracy functions (prediction interpreters)
"""
def accuracy_all():
    images, labels = data.MNISTData("val").data[:]
    accuracy = torch.sum(predict_all(images) == labels) / len(labels)
    return accuracy

def accuracy_exclude_uncertain():
    images, labels = data.MNISTData("val").data[:]
    n_predictions, n_correct_predictions = predict_confident(images, labels)
    
    accuracy = n_correct_predictions / n_predictions if n_predictions else 0
    skip_percent = 1 - n_predictions / len(labels)
    return accuracy, skip_percent

def predict_confident(images, labels):
    confidences = torch.mean(get_prediction_confidence(images), axis=0)
    certain = torch.max(confidences, axis=1).values > min_certainty
    confident_predictions = torch.argmax(confidences[certain, :], axis=1)
    
    num_confident = torch.sum(certain)
    num_correct = torch.sum(confident_predictions == labels[certain])

    return num_confident, num_correct

def prediction_data(images, labels):
    confidences = torch.mean(get_prediction_confidence(images), axis=0)
    certain = torch.max(confidences, axis=1).values > min_certainty

    all_predictions = torch.argmax(confidences, axis=1)
    confident_predictions = torch.where(certain, all_predictions, torch.tensor(-1, dtype=torch.int64))

    num_confident = torch.sum(confident_predictions != -1)
    num_correct_confident = torch.sum(confident_predictions == torch.tensor(labels))
    
    entropies = entropy(confidences.cpu(), axis=1)

    return all_predictions, confident_predictions, num_confident.item(), num_correct_confident.item(), entropies 