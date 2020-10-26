import torch
import torchvision
import torchvision.transforms as transforms

batch_size = 32

# TODO: Do we want to normalize the data? We probably do. If so, agree on common parameters
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# Training set is 60k ims, test set is 10k ims. 
# TODO: Agree on where validation set will come from and its size. I propose 10k from train
test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(
    torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform),
    [50_000, 10_000]
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)