from src.ffnn.visualization_interface import entropy_mnist_train, entropy_mnist_test, entropy_not_mnist, mnist, webcode
from src.ffnn.data_load import get_high_entropy_mnist_test
import src.visualization.comparison as comparison
import src.visualization.metrics as metrics
import src.bnn.interface as bnn_vis
from src.bnn.website_data import get_high_ffnn_entropy_instances_data
import numpy as np
import matplotlib.pyplot as plt

train = entropy_mnist_train()
test = entropy_mnist_test()
not_mnist = entropy_not_mnist()
pred = mnist()

y_true_mnist = pred['y']
y_pred_ffnn_mnist = pred['ffnn_models/model_50000']
y_pred_dropout_mnist = pred['dropout_models/model_50000']

bnn_50 = bnn_vis.get_prediction_data(0)
bnn_19 = bnn_vis.get_prediction_data(1)
bnn_7 = bnn_vis.get_prediction_data(2)
bnn_2 = bnn_vis.get_prediction_data(3)
bnn_1 = bnn_vis.get_prediction_data(4)

bnn_50_not_mnist = bnn_vis.get_prediction_data(0, "notmnist")
bnn_19_not_mnist = bnn_vis.get_prediction_data(1, "notmnist")
bnn_7_not_mnist = bnn_vis.get_prediction_data(2, "notmnist")
bnn_2_not_mnist = bnn_vis.get_prediction_data(3, "notmnist")
bnn_1_not_mnist = bnn_vis.get_prediction_data(4, "notmnist")

bnn_50_train = bnn_vis.get_prediction_data(0, "train")

bnn_all = [bnn_1, bnn_2, bnn_7, bnn_19, bnn_50]


# Accuracy
def print_accuracies():
    acc_ffnn = metrics.compute_accuracy(y_true_mnist, y_pred_ffnn_mnist)
    acc_dropout = metrics.compute_accuracy(y_true_mnist, y_pred_dropout_mnist)
    acc_bnn = bnn_50['accuracy']
    print('Accuracy, FFNN:', round(acc_ffnn, 2))
    print('Accuracy, dropout:', round(acc_dropout, 2))
    print('Accuracy, BNN:', round(acc_bnn, 2))


# Precision, recall, F1-score, support and accuracy
def print_reports():
    metrics.print_classification_report(
        y_true_mnist, y_pred_ffnn_mnist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'FFNN')
    metrics.print_classification_report(
        y_true_mnist, y_pred_dropout_mnist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'Dropout')
    metrics.print_classification_report(
        bnn_vis.mnist_labels, bnn_50["all_predictions"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'BNN')


# Confusion matrix
def make_confusion_matrices():
    fig = metrics.make_multiple_heatmaps(
        [y_true_mnist, y_true_mnist, bnn_vis.mnist_labels],
        [y_pred_ffnn_mnist, y_pred_dropout_mnist, bnn_50["all_predictions"]],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['FFNN', 'FFNN w/dropout', 'BNN'], 20, 6, 1, 3)
    fig.savefig('../visualization/imgs/Heatmaps')


# Compare FFNN, FFNN w/dropout and BNN on their accuracy on the MNIST test set
# when different training set sizes are used
def make_accuracy_line_chart():
    data_sizes = np.array([1000, 2500, 7000, 19000, 50000], dtype=np.int32)
    ffnn_accuracy = np.zeros(data_sizes.size)
    dropout_accuracy = np.zeros(data_sizes.size)
    bnn_accuracy = np.zeros(data_sizes.size)

    for i in range(data_sizes.size):
        ffnn_accuracy[i] = metrics.compute_accuracy(y_true_mnist, pred['ffnn_models/model_' + str(data_sizes[i])])
        dropout_accuracy[i] = metrics.compute_accuracy(y_true_mnist, pred['dropout_models/model_' + str(data_sizes[i])])
        bnn_accuracy[i] = bnn_all[i]["accuracy"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    comparison.make_line_chart(
        ax, data_sizes,
        [ffnn_accuracy, dropout_accuracy, bnn_accuracy],
        ['FFNN', 'FFNN w/dropout', 'BNN'],
        'Size of training set', 'Accuracy', 'Accuracy')
    plt.xticks(data_sizes)

    fig.savefig('../visualization/imgs/Accuracy_plot')


# Violin plot of entropies
def make_entropy_plots_sets():
    # Comparison of entropies between training set, test set, and the set consisting of anomalies from notMNIST
    # for FFNN, FFNN w/dropout, and BNN, respectively
    fig6 = comparison.make_violinplot_for_comparing_sets(
        'Violin plot of entropies for the FFNN model trained on 50 000 examples',
        train['f50000'][:10000],
        test['f50000'],
        not_mnist['f50000'])
    fig6.savefig('../visualization/imgs/Entropy_FFNN_sets')

    fig7 = comparison.make_violinplot_for_comparing_sets(
        'Violin plot of entropies for the FFNN w/dropout model trained on 50 000 examples',
        train['d50000'][:10000],
        test['d50000'],
        not_mnist['d50000'])
    fig7.savefig('../visualization/imgs/Entropy_Dropout_sets')

    fig8 = comparison.make_violinplot_for_comparing_sets(
        'Violin plot of entropies for the BNN model trained on 50 000 examples',
        bnn_50_train['entropies'][:10000],
        bnn_50['entropies'],
        bnn_50_not_mnist['entropies'])
    fig8.savefig('../visualization/imgs/Entropy_BNN_sets')


def make_entropy_plots_mnist():
    # Comparison of entropies between FFNN, FFNN w/dropout and BNN on the MNIST test set
    fig = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 50 000 examples',
        test['f50000'],
        test['d50000'],
        bnn_50["entropies"])
    fig.savefig('../visualization/imgs/Entropy_50')

    fig2 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 19 000 examples',
        test['f19000'],
        test['d19000'],
        bnn_19["entropies"])
    fig2.savefig('../visualization/imgs/Entropy_19')

    fig3 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 7 000 examples',
        test['f7000'],
        test['d7000'],
        bnn_7["entropies"])
    fig3.savefig('../visualization/imgs/Entropy_7')

    fig4 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 2 500 examples',
        test['f2500'],
        test['d2500'],
        bnn_2["entropies"])
    fig4.savefig('../visualization/imgs/Entropy_2')

    fig5 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 1 000 examples',
        test['f1000'],
        test['d1000'],
        bnn_1["entropies"])
    fig5.savefig('../visualization/imgs/Entropy_1')

    # Comparison of entropies between the different training sizes for each model on the MNIST test set
    fig9 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN models',
        test['f50000'],
        test['f19000'],
        test['f7000'],
        test['f2500'],
        test['f1000'])
    fig9.savefig('../visualization/imgs/Entropy_FFNN_sizes')

    fig10 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN w/dropout models',
        test['d50000'],
        test['d19000'],
        test['d7000'],
        test['d2500'],
        test['d1000'])
    fig10.savefig('../visualization/imgs/Entropy_Dropout_sizes')

    fig11 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different BNN models',
        bnn_50['entropies'],
        bnn_19['entropies'],
        bnn_7['entropies'],
        bnn_2['entropies'],
        bnn_1['entropies'])
    fig11.savefig('../visualization/imgs/Entropy_BNN_sizes')


def make_entropy_plots_notmnist():
    # Comparison of entropies between FFNN, FFNN w/dropout and BNN on the set of anomalies
    fig = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 50 000 examples',
        not_mnist['f50000'],
        not_mnist['d50000'],
        bnn_50_not_mnist["entropies"])
    fig.savefig('../visualization/imgs/Entropy_50_notMNSIT')

    fig2 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 19 000 examples',
        not_mnist['f19000'],
        not_mnist['d19000'],
        bnn_19_not_mnist["entropies"])
    fig2.savefig('../visualization/imgs/Entropy_19_notMNSIT')

    fig3 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 7 000 examples',
        not_mnist['f7000'],
        not_mnist['d7000'],
        bnn_7_not_mnist["entropies"])
    fig3.savefig('../visualization/imgs/Entropy_7_notMNSIT')

    fig4 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 2 500 examples',
        not_mnist['f2500'],
        not_mnist['d2500'],
        bnn_2_not_mnist["entropies"])
    fig4.savefig('../visualization/imgs/Entropy_2_notMNSIT')

    fig5 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 1 000 examples',
        not_mnist['f1000'],
        not_mnist['d1000'],
        bnn_1_not_mnist["entropies"])
    fig5.savefig('../visualization/imgs/Entropy_1_notMNSIT')

    # Comparison of entropies between the different training sizes for each model on set of anomalies
    fig9 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN models',
        not_mnist['f50000'],
        not_mnist['f19000'],
        not_mnist['f7000'],
        not_mnist['f2500'],
        not_mnist['f1000'])
    fig9.savefig('../visualization/imgs/Entropy_FFNN_sizes_notMNSIT')

    fig10 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN w/dropout models',
        not_mnist['d50000'],
        not_mnist['d19000'],
        not_mnist['d7000'],
        not_mnist['d2500'],
        not_mnist['d1000'])
    fig10.savefig('../visualization/imgs/Entropy_Dropout_sizes_notMNSIT')

    fig11 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different BNN models',
        bnn_50_not_mnist['entropies'],
        bnn_19_not_mnist['entropies'],
        bnn_7_not_mnist['entropies'],
        bnn_2_not_mnist['entropies'],
        bnn_1_not_mnist['entropies'])
    fig11.savefig('../visualization/imgs/Entropy_BNN_sizes_notMNSIT')


### Webcode ###
def create_MNIST_pictures():
    ret = get_high_entropy_mnist_test()
    for i in range(len(ret)):
        image = ret[i][0]
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.grid(b=None)
        plt.axis('off')
        plt.savefig('../visualization/imgs/MNIST_' + str(i), bbox_inches='tight')


def print_webdata():
    data = webcode()
    true_values = data['y']
    d_data = data['d']
    f_data = data['f']

    # FFNN
    for i in range(10):
        r_conf = [round(x, 2) for x in f_data[i][1]]
        print(
            "Iteration", str(i) + ", FFNN:\n",
            "Confidences:", r_conf,
            "Prediction:", f_data[i][0],
            "True value:", true_values[i]
        )

    # FFNN w/dropout
    for i in range(10):
        r_conf = [round(x, 2) for x in d_data[i][1]]
        print(
            "Iteration", str(i) + ", Dropout:\n",
            "Confidences:", r_conf,
            "Prediction:", d_data[i][0],
            "True value:", true_values[i]
        )

    # BNN
    conf, pred, y = get_high_ffnn_entropy_instances_data()
    for i in range(10):
        r_conf = [round(x, 2) for x in conf[i].tolist()]
        print(
            "Iteration", str(i) + ", BNN:\n",
            "Confidences:", r_conf,
            "Prediction:", pred[i],
            "True value:", y[i]
        )


if __name__ == "__main__":
    print_accuracies()
    print_reports()
    make_accuracy_line_chart()
    # make_confusion_matrices()
    make_entropy_plots_sets()
    make_entropy_plots_mnist()
    make_entropy_plots_notmnist()
    # create_MNIST_pictures()
    # print_webdata()
