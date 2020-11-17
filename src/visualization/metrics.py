import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


def compute_accuracy(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)


# Print out classification results and plot confusion matrices
def print_classification_report(y_true, y_pred, labels, target_names, title):
    report = sklearn.metrics.classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print('{0:->53}\n{0:>25}%s{0:<22}\n{0:->53}'.format("") % title)
    print(report)


def make_heatmap(y_true, y_pred, ax, title, target_names):
    ax.set_title(title)
    cf_matrix_counts = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=target_names)
    cf_matrix_percentage = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=target_names, normalize='pred')
    labels = [f'{v1}\n{int(round(v2, 2)*100)}%' for v1, v2 in
              zip(cf_matrix_counts.flatten(), cf_matrix_percentage.flatten())]
    labels = np.asarray(labels).reshape(len(target_names), len(target_names))
    cf_matrix = pd.DataFrame(
        data=cf_matrix_counts, index=target_names, columns=target_names)
    hm = sns.heatmap(cf_matrix, annot=labels,
                     fmt='', ax=ax, cmap='Blues')
    hm.set_xlabel('Predicted')
    hm.set_ylabel('Target')


def make_multiple_heatmaps(y_true: list, y_pred: list, target_names: list, titles: list, fig_width: float, fig_height: float, nrows: int, ncols: int):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(fig_width, fig_height))
    for i in range(len(axs)):
        make_heatmap(y_true[i], y_pred[i], axs[i], titles[i], target_names)
    return fig
