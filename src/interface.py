import torch
import pyro
import numpy as np

from data import test_loader, test_set_size, notmnist_loader
from Net import min_certainty, predict_all, prediction_data
from main import model_path


pyro.get_param_store().load(model_path)

# =====
# MNIST
# =====
image_batches, label_batches = zip(*test_loader)
images = torch.cat(tuple(im_batch.view(-1, 28*28) for im_batch in image_batches), dim=0)
labels = torch.cat(label_batches).numpy()

all_predictions,\
confident_predictions,\
num_confident_predictions,\
num_correct_predictions = prediction_data(images, labels)

num_skipped = test_set_size - num_confident_predictions
skip_percent = num_skipped / test_set_size
accuracy = np.sum(all_predictions == labels) / test_set_size
confident_accuracy = num_correct_predictions / num_confident_predictions

# =====
# notMNIST
# =====
notmnist_image_batches, notmnist_label_batches = zip(*notmnist_loader)
notmnist_images = torch.cat(tuple(im_batch.view(-1, 28*28) for im_batch in notmnist_image_batches), dim=0)
notmnist_labels = torch.cat(tuple(im_batch.view(-1) for im_batch in notmnist_label_batches), dim=0)
notmnist_test_set_size = len(notmnist_loader.dataset)

notmnist_all_predictions,\
notmnist_confident_predictions,\
notmnist_num_confident_predictions,\
notmnist_num_correct_predictions = prediction_data(notmnist_images, notmnist_labels)

notmnist_num_skipped = notmnist_test_set_size - notmnist_num_confident_predictions
notmnist_skip_percent = notmnist_num_skipped / notmnist_test_set_size
notmnist_accuracy = np.sum(notmnist_all_predictions == notmnist_labels) / notmnist_test_set_size
notmnist_confident_accuracy = notmnist_num_correct_predictions / notmnist_num_confident_predictions


if __name__ == '__main__':
    print(f"MNIST: {accuracy = :.2%}, {confident_accuracy = :.2%}, {skip_percent = :.2%}")
    print(f"notMNIST: {notmnist_skip_percent = :.2%}")
    print(notmnist_images.shape)