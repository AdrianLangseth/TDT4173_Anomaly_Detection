import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import SGD, Adam

from Net import NN
from data import train_loader, test_loader, val_loader, batch_size

net = NN(28*28, 512, 10)
lr = 0.01
num_samples = 50
num_epochs = 25


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

# optim = SGD({'lr': lr})
optim = Adam({"lr": lr})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
def train():
    for i in range(num_epochs):
        loss = sum(
            svi.step(X.view(-1, 28*28), y) 
            for X, y in train_loader
        )
        total_epoch_loss = loss / len(train_loader.dataset)
        print("Epoch ", i, " Loss ", total_epoch_loss)

def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)

def force_decision_accuracy():
    num_correct = sum(
        np.sum(predict(images.view(-1, 28*28)) == labels.numpy())
        for images, labels in test_loader
    )
    accuracy = num_correct / len(test_loader.dataset)
    print(f"accuracy: {accuracy:.2%}")

def allow_uncertainty_accuracy():
    num_samples = len(test_loader.dataset)
    correct_predictions = total_predictions = 0
    for images, labels in test_loader:    
        correct, total = test_batch(images, labels)
        total_predictions += total
        correct_predictions += correct
    accuracy = correct_predictions / total_predictions

    print("Total images: ", num_samples)
    print("Skipped: ", num_samples - total_predictions)
    print("Predicted:", total_predictions, "Correct:", correct_predictions)
    print(f"Accuracy without uncertain predictions: {accuracy:.2%}")

def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)

def test_batch(images, labels, min_certainty=0.25):
    y = give_uncertainities(images).transpose(1, 2, 0) # row: sample. col: class. third: model
    predictions = np.median(np.exp(y), axis=2)  # get median of model predictions for each class
    certain = np.max(predictions, axis=1) > min_certainty # index of all predictions more certain than min threshold
    num_correct = np.sum(
        np.argmax(predictions[certain, :], axis=1) == labels.numpy()[certain]
    )
    return num_correct, np.sum(certain)


train()
force_decision_accuracy()
allow_uncertainty_accuracy()