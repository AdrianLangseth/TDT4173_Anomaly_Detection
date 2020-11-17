from src.ffnn.visualization_interface import entropy_mnist_train, entropy_mnist_test, entropy_not_mnist, mnist
import src.visualization.comparison as comparison
import src.visualization.metrics as metrics
import src.bnn.interface as bnn_vis
import numpy as np
import matplotlib.pyplot as plt

train = entropy_mnist_train()
test = entropy_mnist_test()
# notmnist = entropy_not_mnist()
pred = mnist()

y_true_mnist = pred['y']
y_pred_ffnn_mnist = pred['ffnn_models/model_50000']
y_pred_dropout_mnist = pred['dropout_models/model_50000']


# Accuracy
def print_accuracies():
    acc_ffnn = metrics.compute_accuracy(y_true_mnist, y_pred_ffnn_mnist)
    acc_dropout = metrics.compute_accuracy(y_true_mnist, y_pred_dropout_mnist)
    acc_bnn = metrics.compute_accuracy(bnn_vis.labels, bnn_vis.confident_predictions)
    print('Accuracy, FFNN:', round(acc_ffnn, 2))
    print('Accuracy, dropout:', round(acc_dropout, 2))
    print('Accuracy, BNN:', round(acc_bnn, 2))


# Precision, recall, F1-score, and support
def print_reports():
    metrics.print_classification_report(
        y_true_mnist, y_pred_ffnn_mnist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'FFNN')
    metrics.print_classification_report(
        y_true_mnist, y_pred_dropout_mnist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'Dropout')
    metrics.print_classification_report(
        bnn_vis.labels, bnn_vis.confident_predictions, [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'BNN')


# Confusion matrix
def make_confusion_matrices():
    fig = metrics.make_multiple_heatmaps(
        [y_true_mnist, y_true_mnist, bnn_vis.labels],
        [y_pred_ffnn_mnist, y_pred_dropout_mnist, bnn_vis.confident_predictions],
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['FFNN', 'FFNN w/dropout', 'BNN'], 20, 6, 1, 3)
    fig.savefig('Heatmaps')


# Compare FFNN, FFNN w/dropout and BNN on accuracy when different training set sizes are used
def make_accuracy_line_chart():
    data_sizes = np.array([1000, 2500, 7000, 19000, 50000], dtype=np.int32)
    ffnn_accuracy = np.zeros(data_sizes.size)
    dropout_accuracy = np.zeros(data_sizes.size)
    # bnn_accuracy = np.zeros(data_sizes.size)
    for i in range(data_sizes.size):
        ffnn_accuracy[i] = metrics.compute_accuracy(y_true_mnist, pred['ffnn_models/model_' + str(data_sizes[i])])
        dropout_accuracy[i] = metrics.compute_accuracy(y_true_mnist, pred['dropout_models/model_' + str(data_sizes[i])])
        # bnn_accuracy[i] = metrics.compute_accuracy(y_true, y_pred_test)

    fig, ax = plt.subplots(nrows=1, ncols=1,
                            figsize=(10, 5))
    plt.subplots_adjust(hspace=0.5)
    comparison.make_line_chart(ax, data_sizes, [ffnn_accuracy, dropout_accuracy], ['FFNN', 'FFNN w/dropout'],
                                   'Size of training set', 'Accuracy (%)', 'Accuracy')

    fig.savefig('Accuracy_plot')


# Violin plot of entropies
def make_entropy_plots():
    fig = comparison.make_violinplot('Entropy', test['f50000'],
                                      test['d50000'],
                                      bnn_vis.entropies)
    fig.savefig('Entropy')


if __name__ == "__main__":
    # print_accuracies()
    # print_reports()
    # make_accuracy_line_chart()
    # make_confusion_matrices()
    make_entropy_plots()
