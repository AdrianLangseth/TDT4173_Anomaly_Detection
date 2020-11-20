from src.ffnn.visualization_interface import entropy_mnist_train, entropy_mnist_test, entropy_not_mnist, mnist
import src.visualization.comparison as comparison
import src.visualization.metrics as metrics
import src.bnn.interface as bnn_vis
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

# bnn_50_train = bnn_vis.get_prediction_data(0, "train")

bnn_all = [bnn_1, bnn_2, bnn_7, bnn_19, bnn_50]


# Accuracy
def print_accuracies():
    acc_ffnn = metrics.compute_accuracy(y_true_mnist, y_pred_ffnn_mnist)
    acc_dropout = metrics.compute_accuracy(y_true_mnist, y_pred_dropout_mnist)
    acc_bnn = bnn_50['accuracy']
    acc_conf_bnn_all = metrics.compute_accuracy(bnn_vis.mnist_labels, bnn_50['confident_predictions'])
    acc_conf_bnn_within = bnn_50["confident_accuracy"]
    print('Accuracy, FFNN:', round(acc_ffnn, 2))
    print('Accuracy, dropout:', round(acc_dropout, 2))
    print('Accuracy, BNN w/o uncertainty:', round(acc_bnn, 2))
    print('Accuracy, BNN w/uncertainty:', round(acc_conf_bnn_all, 2))
    print('Confident accuracy, BNN w/uncertainty:', round(acc_conf_bnn_within, 2))


# Precision, recall, F1-score, and support
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
    metrics.print_classification_report(
        bnn_vis.mnist_labels, bnn_50["confident_predictions"], [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'BNN, conf')


# Confusion matrix
def make_confusion_matrices():
    fig = metrics.make_multiple_heatmaps(
        [y_true_mnist, y_true_mnist, bnn_vis.mnist_labels, bnn_vis.mnist_labels],
        [y_pred_ffnn_mnist, y_pred_dropout_mnist, bnn_50["all_predictions"], bnn_50["confident_predictions"]],
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ['FFNN', 'FFNN w/dropout', 'BNN, all predictions', 'BNN, confident predictions'], 20, 6, 1, 4)
    fig.savefig('Heatmaps')


# Compare FFNN, FFNN w/dropout and BNN on accuracy when different training set sizes are used
def make_accuracy_line_chart():
    data_sizes = np.array([1000, 2500, 7000, 19000, 50000], dtype=np.int32)
    ffnn_accuracy = np.zeros(data_sizes.size)
    dropout_accuracy = np.zeros(data_sizes.size)
    bnn_accuracy = np.zeros(data_sizes.size)
    bnn_accuracy_conf_whole = np.zeros(data_sizes.size)
    bnn_accuracy_conf_conf = np.zeros(data_sizes.size)

    for i in range(data_sizes.size):
        ffnn_accuracy[i] = metrics.compute_accuracy(y_true_mnist, pred['ffnn_models/model_' + str(data_sizes[i])])
        dropout_accuracy[i] = metrics.compute_accuracy(y_true_mnist, pred['dropout_models/model_' + str(data_sizes[i])])
        bnn_accuracy[i] = bnn_all[i]["accuracy"]
        bnn_accuracy_conf_whole[i] = metrics.compute_accuracy(bnn_vis.mnist_labels, bnn_all[i]["confident_predictions"])
        bnn_accuracy_conf_conf[i] = bnn_all[i]['confident_accuracy']

    fig, ax = plt.subplots(nrows=1, ncols=1,
                            figsize=(15, 5))
    comparison.make_line_chart(
        ax, data_sizes,
        [ffnn_accuracy, dropout_accuracy, bnn_accuracy, bnn_accuracy_conf_whole, bnn_accuracy_conf_conf],
        ['FFNN', 'FFNN w/dropout', 'BNN, w/o uncertainty',
         'BNN, w/uncertainty, all', 'BNN, w/uncertainty, confident'],
        'Size of training set', 'Accuracy (%)', 'Accuracy')
    plt.xticks(data_sizes)

    fig.savefig('Accuracy_plot')
    return {"FFNN": ffnn_accuracy, "Dropout": dropout_accuracy, "BNN_all": bnn_accuracy,
            "BNN_conf": bnn_accuracy_conf_whole, "BNN_conf_within": bnn_accuracy_conf_conf}


# Violin plot of entropies
def make_entropy_plots_sets():
    # Comparison of entropies between training set, test set, and the set consisting of anomalies from notMNIST
    # for FFNN, FFNN w/dropout, and BNN, respectively
    fig6 = comparison.make_violinplot_for_comparing_sets(
        'Violin plot of entropies for the FFNN model trained on 50 000 examples',
        train['f50000'][:10000],
        test['f50000'],
        not_mnist['f50000'])
    fig6.savefig('Entropy_FFNN_sets')

    fig7 = comparison.make_violinplot_for_comparing_sets(
        'Violin plot of entropies for the FFNN w/dropout model trained on 50 000 examples',
        train['d50000'][:10000],
        test['d50000'],
        not_mnist['d50000'])
    fig7.savefig('Entropy_Dropout_sets')

    # TODO: Change first bnn_50['entropies'] to bnn_50_train['entropies']
    fig8 = comparison.make_violinplot_for_comparing_sets(
        'Violin plot of entropies for the BNN model trained on 50 000 examples',
        bnn_50['entropies'],
        bnn_50['entropies'],
        bnn_50_not_mnist['entropies'])
    fig8.savefig('Entropy_BNN_sets')


def make_entropy_plots_mnist():
    # Comparison of entropies between FFNN, FFNN w/dropout and BNN on the MNIST test set
    fig = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 50 000 examples',
        test['f50000'],
        test['d50000'],
        bnn_50["entropies"])
    fig.savefig('Entropy_50')

    fig2 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 19 000 examples',
        test['f19000'],
        test['d19000'],
        bnn_19["entropies"])
    fig2.savefig('Entropy_19')

    fig3 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 7 000 examples',
        test['f7000'],
        test['d7000'],
        bnn_7["entropies"])
    fig3.savefig('Entropy_7')

    fig4 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 2 500 examples',
        test['f2500'],
        test['d2500'],
        bnn_2["entropies"])
    fig4.savefig('Entropy_2')

    fig5 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 1 000 examples',
        test['f1000'],
        test['d1000'],
        bnn_1["entropies"])
    fig5.savefig('Entropy_1')

    # Comparison of entropies between the different training sizes for each model on the MNIST test set
    fig9 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN models',
        test['f50000'],
        test['f19000'],
        test['f7000'],
        test['f2500'],
        test['f1000'])
    fig9.savefig('Entropy_FFNN_sizes')

    fig10 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN w/dropout models',
        test['d50000'],
        test['d19000'],
        test['d7000'],
        test['d2500'],
        test['d1000'])
    fig10.savefig('Entropy_Dropout_sizes')

    fig11 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different BNN models',
        bnn_50['entropies'],
        bnn_19['entropies'],
        bnn_7['entropies'],
        bnn_2['entropies'],
        bnn_1['entropies'])
    fig11.savefig('Entropy_BNN_sizes')


def make_entropy_plots_notmnist():
    # Comparison of entropies between FFNN, FFNN w/dropout and BNN on the set of anomalies
    fig = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 50 000 examples',
        not_mnist['f50000'],
        not_mnist['d50000'],
        bnn_50_not_mnist["entropies"])
    fig.savefig('Entropy_50_notMNSIT')

    fig2 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 19 000 examples',
        not_mnist['f19000'],
        not_mnist['d19000'],
        bnn_19_not_mnist["entropies"])
    fig2.savefig('Entropy_19_notMNSIT')

    fig3 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 7 000 examples',
        not_mnist['f7000'],
        not_mnist['d7000'],
        bnn_7_not_mnist["entropies"])
    fig3.savefig('Entropy_7_notMNSIT')

    fig4 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 2 500 examples',
        not_mnist['f2500'],
        not_mnist['d2500'],
        bnn_2_not_mnist["entropies"])
    fig4.savefig('Entropy_2_notMNSIT')

    fig5 = comparison.make_violinplot_for_comparing_models(
        'Violin plot of entropies for models trained on 1 000 examples',
        not_mnist['f1000'],
        not_mnist['d1000'],
        bnn_1_not_mnist["entropies"])
    fig5.savefig('Entropy_1_notMNSIT')

    # Comparison of entropies between the different training sizes for each model on set of anomalies
    fig9 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN models',
        not_mnist['f50000'],
        not_mnist['f19000'],
        not_mnist['f7000'],
        not_mnist['f2500'],
        not_mnist['f1000'])
    fig9.savefig('Entropy_FFNN_sizes_notMNSIT')

    fig10 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different FFNN w/dropout models',
        not_mnist['d50000'],
        not_mnist['d19000'],
        not_mnist['d7000'],
        not_mnist['d2500'],
        not_mnist['d1000'])
    fig10.savefig('Entropy_Dropout_sizes_notMNSIT')

    fig11 = comparison.make_violinplot_for_comparing_sizes(
        'Violin plot of entropies for the different BNN models',
        bnn_50_not_mnist['entropies'],
        bnn_19_not_mnist['entropies'],
        bnn_7_not_mnist['entropies'],
        bnn_2_not_mnist['entropies'],
        bnn_1_not_mnist['entropies'])
    fig11.savefig('Entropy_BNN_sizes_notMNSIT')


if __name__ == "__main__":
    print_accuracies()
    print_reports()
    make_accuracy_line_chart()
    # make_confusion_matrices()
    make_entropy_plots_sets()
    make_entropy_plots_mnist()
    make_entropy_plots_notmnist()
