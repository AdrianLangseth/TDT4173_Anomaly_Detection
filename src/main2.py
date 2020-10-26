import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import SGD
# from: https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd

from Net import NN
from data import train_loader, test_loader, val_loader

lr = 0.01

input_size = 28*28
fc1_size = 1024
out_size = 10

net = NN(input_size, fc1_size, out_size)

log_softmax = torch.nn.LogSoftmax(dim=1)
softplus = torch.nn.Softplus()

def zeros_ones_normal(like):
    return dist.Normal(loc=torch.zeros_like(like), scale=torch.ones_like(like))

def model(x_data, y_data):
    priors = dict(zip(
        ('fc1.weight', 'fc1.bias', 'out.weight', 'out.bias'),
        map(zeros_ones_normal, (net.fc1.weight, net.fc1.bias, net.out.weight, net.out.bias))
    ))
    lifted_module = pyro.random_module("module", net, priors)
    lhat = log_softmax(lifted_module()(x_data))
    
    pyro.sample("obs", dist.Categorical(logits=lhat), obs=y_data)

def make_prior_normal(name, like):
    return dist.Normal(
        loc=pyro.param(f"{name}_exp", torch.randn_like(like)),
        scale=softplus(pyro.param(f"{name}_var", torch.randn_like(like)))
    )

def guide(x_data, y_data):
    priors = {
        'fc1.weight': make_prior_normal("fc1_w", net.fc1.weight),
        'fc1.bias': make_prior_normal("fc1_b", net.fc1.bias),
        'out.weight': make_prior_normal("out_w", net.out.weight).independent(1),
        'out.bias': make_prior_normal("out_b", net.out.bias)
    }

    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()


svi = SVI(model, guide, SGD({'lr': lr}), loss=Trace_ELBO())
num_iterations = 5
loss = 0
for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        X, y = data
        loss += svi.step(X.view(-1, 28*28), y)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)