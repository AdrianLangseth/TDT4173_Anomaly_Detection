import numpy as np

import visualization.metrics as metrics
import visualization.comparison as comparison

# Make in- and out-of-sample predictions
# TODO: remove line below and use y_train/y_test
y_true = np.array([0, 0, 0, 1, 1, 0, 1, 2, 2, 1])
# TODO: replace line below with true predication, i.e.: model.predict(X_train)
y_pred_train = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 1])
# TODO: replace line below with true predication, i.e.: model.predict(X_test)
y_pred_test = np.array([0, 1, 0, 1, 1, 1, 0, -1, 2, 2])


def main():
    # TODO: Remove since accuracy is given in report
    # Accuracy
    acc_train = metrics.compute_accuracy(y_true, y_pred_train)
    acc_test = metrics.compute_accuracy(y_true, y_pred_test)
    print('Accuracy on training set:', round(acc_train, 2))
    print('Accuracy on testing set:', round(acc_test, 2))

    # Precision, recall, F1-score, support and accuracy
    metrics.print_classification_report(
        y_true, y_pred_train, [-1, 0, 1, 2], ['-1', '0', '1', '2'], 'Train')
    metrics.print_classification_report(
        y_true, y_pred_test, [-1, 0, 1, 2], ['-1', '0', '1', '2'], 'Test')

    # Heatmaps
    fig = metrics.make_multiple_heatmaps(
        [y_true, y_true], [y_pred_train, y_pred_test], [-1, 0, 1, 2], ['Train', 'Test'], 16, 6, 1, 2)
    fig.savefig('Heatmaps')

    # Compare FFNN and BNN on accuracy and anmomaly detection when different training set sizes are used
    data_sizes = np.array([100, 1000, 10000, 30000, 60000], dtype=np.float32)
    ffnn_accuracy = np.zeros(data_sizes.size)
    bnn_accuracy = np.zeros(data_sizes.size)
    for i in range(data_sizes.size):
        ffnn_accuracy[i] = metrics.compute_accuracy(y_true, y_pred_train)
        bnn_accuracy[i] = metrics.compute_accuracy(y_true, y_pred_test)

    ffnn_anomalies_detected = np.array(
        [0.1, 0.2, 0.2, 0.22, 1/3], dtype=np.float32)
    bnn_anomalies_detected = np.array(
        [0.2, 1/3, 0.66, 0.7, 0.91], dtype=np.float32)

    fig = comparison.make_line_charts(data_sizes, [[ffnn_accuracy, bnn_accuracy], [
                                      ffnn_anomalies_detected, bnn_anomalies_detected]],
                                      [['FFNN', 'BNN'], ['FFNN', 'BNN']],
                                      'Size of training set',
                                      ['Accuracy (%)',
                                       'Detected anomalies (%)'],
                                      ['Accuracy', 'Anomaly Detection'],
                                      10, 5, 2, 1)
    # fig.suptitle('Master piiecoo', fontsize=16)
    fig.savefig('Dobbel_plots')

    # Violin plot
    fig2 = comparison.make_violinplot('Entropy', [1, 1, 1, 2, 2, 2, 3, 3, 3], [
        1, 2, 3, 1, 2, 3, 1, 2, 3], [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1])
    fig2.savefig('Entropy')


main()
