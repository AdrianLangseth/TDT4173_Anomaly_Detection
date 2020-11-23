from pprint import pprint

from data import get_loader_data, notmnist_loader, MNISTData
from Net import prediction_data
from main import setup_model


mnist_images, mnist_labels = MNISTData("test").data.tensors
notmnist_images, notmnist_labels = get_loader_data(notmnist_loader)

def get_prediction_data(train_set_i=0, dataset="test"):
    """
    :param train_set_i: training set size index. Must be in [0, 4]. 
        Determines which model is used. 
        0 => biggest model (trained on 50k instances)
        4 => smallest model (trained on 1k instances)
    :param dataset: Must be either a string indicating one of the premade datasets
        ("train", "val", "test", "notmnist") or a new dataset in the 
        form of a tuple (instances, targets)
    :return: {
        "all_predictions": <torch.tensor of every predicted target>, 
        "entropies": <torch.tensor of entropy for every prediction>,
        "accuracy": <float of accuracy of predictions>
    }
    """
    if dataset in ("train", "val", "test", "notmnist"):
        if dataset == "notmnist":
            ims, labs = notmnist_images, notmnist_labels
        else:
            ims, labs = MNISTData(dataset).data[:]
    else:
        ims, labs = dataset

    setup_model(train_set_i)
        
    return dict(zip(
        ("all_predictions", "entropies", "accuracy"),
        prediction_data(ims, labs)
    ))
    

if __name__ == '__main__':
    pprint(get_prediction_data(train_set_i=4))
    pprint(get_prediction_data(train_set_i=4, dataset="train"))
    pprint(get_prediction_data(train_set_i=0, dataset="notmnist"))
