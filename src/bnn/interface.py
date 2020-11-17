import torch
import pyro
import numpy as np

from src.bnn.data import test_loader, test_set_size, notmnist_loader
from src.bnn.Net import prediction_data, predict_confident
from src.bnn.main import model_path


pyro.get_param_store().load(model_path)

# =====
# MNIST
# =====
image_batches, label_batches = zip(*test_loader)
images = torch.cat(tuple(im_batch.view(-1, 28*28)
                         for im_batch in image_batches), dim=0)
labels = torch.cat(label_batches).numpy()

all_predictions,\
    confident_predictions,\
    num_confident_predictions,\
    num_correct_predictions,\
    entropies = prediction_data(images, labels)

num_skipped = test_set_size - num_confident_predictions
skip_percent = num_skipped / test_set_size
accuracy = np.sum(all_predictions == labels) / test_set_size
confident_accuracy = num_correct_predictions / num_confident_predictions

# =====
# notMNIST
# =====
notmnist_image_batches, notmnist_label_batches = zip(*notmnist_loader)
notmnist_images = torch.cat(tuple(im_batch.view(-1, 28*28)
                                  for im_batch in notmnist_image_batches), dim=0)
notmnist_labels = torch.cat(tuple(im_batch.view(-1)
                                  for im_batch in notmnist_label_batches), dim=0)
notmnist_test_set_size = len(notmnist_loader.dataset)

notmnist_all_predictions,\
    notmnist_confident_predictions,\
    notmnist_num_confident_predictions,\
    notmnist_num_correct_predictions,\
    notmnist_entropies = prediction_data(notmnist_images, notmnist_labels)

notmnist_num_skipped = notmnist_test_set_size - notmnist_num_confident_predictions
notmnist_skip_percent = notmnist_num_skipped / notmnist_test_set_size
notmnist_accuracy = np.sum(notmnist_all_predictions ==
                           notmnist_labels) / notmnist_test_set_size
notmnist_confident_accuracy = notmnist_num_correct_predictions / \
    notmnist_num_confident_predictions


# returns pairs (acc, skip) for each min confidence. Both in [0, 1]
def acc_and_skip(min_confidences):
    def get_data(p_data): return (
        p_data[0]/p_data[1], 1 - p_data[1]/test_set_size)
    return np.asarray([
        get_data(predict_confident(images, labels, min_conf))
        for min_conf in min_confidences
    ])


if __name__ == '__main__':
    print(
        f"MNIST: accuracy = {accuracy:.2%}, confident_accuracy = {confident_accuracy:.2%}, skip_percent = {skip_percent:.2%}")
    print(f"notMNIST: notmnist_skip_percent = {notmnist_skip_percent:.2%}")
    print(np.mean(entropies), np.mean(notmnist_entropies))
    print(acc_and_skip([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]))
