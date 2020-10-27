import torch
import torchvision
import torchvision.transforms as transforms

# TODO: Agree on batch size
batch_size = 256  # going higher than 43 results in NaN results if using SGD

# TODO: Do we want to normalize the data? We probably do. If so, agree on common parameters
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, ), (0.5, ))
])

val_set_size = 10_000
# Training set is 60k ims, test set is 10k ims. 
# TODO: Agree on where validation set will come from and its size. I propose 10k from train
test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(
    torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform),
    [60_000 - val_set_size, val_set_size]
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)