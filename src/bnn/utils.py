import torch

def get_loader_data(data_loader):
    image_batches, label_batches = zip(*data_loader)
    images = torch.cat(tuple(batch.view(-1, 28*28) for batch in image_batches), dim=0)
    labels = torch.cat(tuple(batch.view(-1) for batch in label_batches), dim=0).numpy()
    return images, labels