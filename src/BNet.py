import torch
import torch.nn.functional as F
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.contrib.bnn as bnn
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO

# Mye fra: https://alsibahi.xyz/snippets/2019/06/15/pyro_mnist_bnn_kl.html

class BayesNet(PyroModule):
    def __init__(self, n_hidden=1024, n_classes=10):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def model(self, images, labels):
        images = images.view(-1, 28*28)
        n_images = images.size(0)
        with pyro.plate("data", size=n_images):
            hl = pyro.sample("hl", bnn.HiddenLayer(
                    X=images, 
                    A_mean=torch.zeros(28*28, self.n_hidden),
                    A_scale=torch.ones(28*28, self.n_hidden)
            ))
            out = pyro.sample("out", bnn.HiddenLayer(
                    X=hl,
                    A_mean=torch.zeros(self.n_hidden + 1, self.n_classes),
                    A_scale=torch.ones(self.n_hidden + 1, self.n_classes),
                    non_linearity=lambda x: F.log_softmax(x, dim=-1),
                    include_hidden_bias=False
            ))
            labels = F.one_hot(labels)
            return pyro.sample("label", dist.OneHotCategorical(logits=out), obs=labels)

    def guide(self, images, labels=None):
        images = images.view(-1, 28*28)
        n_images = images.size(0)

        hl_mean = pyro.param('hl_mean', 0.01 * torch.randn(784, self.n_hidden))
        hl_scale = pyro.param('hl_scale', 0.1 * torch.ones(784, self.n_hidden), constraint=dist.constraints.greater_than(0.01))

        out_mean = pyro.param('out_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_classes))
        out_scale = pyro.param('out_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_classes), constraint=dist.constraints.greater_than(0.01))

        with pyro.plate("data", size=n_images):
            hl = pyro.sample("hl", bnn.HiddenLayer(
                X=images, A_mean=hl_mean, A_scale=hl_scale, 
            ))
            out = pyro.sample("out", bnn.HiddenLayer(
                X=hl, A_mean=out_mean, A_scale=out_scale, 
                non_linearity=lambda x: F.log_softmax(x, dim=-1),
                include_hidden_bias=False
            ))

    def infer_parameters(self, loader, lr=0.01, num_epochs=5):
        optim = pyro.optim.SGD({"lr": lr})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)
        for i in range(num_epochs):
            total_loss = total = correct = 0
            for images, labels in loader:
                loss = svi.step(images, labels)
                pred = self.forward(images, n_samples=1).mean(0)
                total_loss += loss / len(loader.dataset)
                total += labels.size(0)
                correct += (pred.argmax(-1) == labels).sum().item()
            print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")

    def forward(self, images, n_samples):
        return torch.stack([
            pyro.poutine.trace(self.guide).get_trace(images).nodes['out']['value']  # out should be logits?
            for _ in range(n_samples) 
        ], dim=0)

    