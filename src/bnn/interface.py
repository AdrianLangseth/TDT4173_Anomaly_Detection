from pprint import pprint

from src.bnn.data import get_loader_data, notmnist_loader, MNISTData
from src.bnn.Net import prediction_data
from src.bnn.main import setup_model


mnist_images, mnist_labels = MNISTData("test").data.tensors
notmnist_images, notmnist_labels = get_loader_data(notmnist_loader)


def get_prediction_data(train_set_i=0, dataset="test"):
    assert dataset in ("train", "val", "test", "notmnist")
    if dataset == "notmnist":
        ims, labs = notmnist_images, notmnist_labels
    else:
        ims, labs = MNISTData(dataset).data[:]
    setup_model(train_set_i)
        
    return dict(zip(
        ("all_predictions", "entropies", "accuracy"),
        prediction_data(ims, labs)
    ))
    

if __name__ == '__main__':
    pprint(get_prediction_data(train_set_i=4))
    pprint(get_prediction_data(train_set_i=4, dataset="train"))
    pprint(get_prediction_data(train_set_i=0, dataset="notmnist"))
