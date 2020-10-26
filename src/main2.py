import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import SGD, Adam
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
    lifted_module = pyro.random_module("module", net, priors)()
    lhat = log_softmax(lifted_module(x_data))
    
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
    return pyro.random_module("module", net, priors)()

# optim = SGD({'lr': lr})
optim = Adam({"lr": lr})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_epochs = 3
loss = 0
for i in range(num_epochs):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        if batch_id >= 40: break
        X, y = data
        loss += svi.step(X.view(-1, 28*28), y)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", i, " Loss ", total_epoch_loss_train)


num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)

print('Prediction when network is forced to predict')
correct = 0
total = 0
for i, data in enumerate(test_loader):
    if i >= 20: break
    images, labels = data
    predicted = predict(images.view(-1,28*28))
    total += labels.size(0)
    correct += sum(p == l for p, l in zip(predicted, labels))
print("accuracy: %d %%" % (100 * int(correct) / int(total)))