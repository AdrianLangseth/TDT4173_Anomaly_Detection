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
from src.bnn.settings import device

# suppress deprecation warnings for pyro.random_module
warnings.filterwarnings("ignore", category=FutureWarning) 


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

def prediction_data(images, labels):
    confidences = torch.mean(get_prediction_confidence(images), axis=0)
    all_predictions = torch.argmax(confidences, axis=1)
    entropies = entropy(confidences.cpu(), axis=1)
    accuracy = torch.mean(all_predictions == labels, dtype=float).item()

    return all_predictions, entropies, accuracy
