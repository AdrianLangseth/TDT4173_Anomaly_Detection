import torch
import numpy as np

from data import test_loader, test_set_size
from Net import predict_all


image_batches, label_batches = zip(*test_loader)
images = torch.cat(tuple(im_batch.view(-1, 28*28) for im_batch in image_batches), dim=0)
labels = torch.cat(label_batches).numpy()
predictions = predict_all(images)
accuracy = np.sum(predictions == labels) / test_set_size
