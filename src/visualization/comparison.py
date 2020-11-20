import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')


def make_line_chart(ax: axes.Axes, x: np.ndarray or list, y: np.ndarray or list, label: list, x_label: str, y_label: str, titel: str):
    for i in range(len(y)):
        ax.plot(x, y[i], label=label[i])
    ax.set_title(titel)
    ax.legend(loc='best')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def make_line_charts(x: np.ndarray, y: list, legend_labels: list, x_label: str, y_labels: list, titles: list, fig_width: float, fig_height: float, nrows: int, ncols: int):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(fig_width, fig_height))
    for i in range(len(axs)):
        make_line_chart(axs[i], x, y[i], legend_labels[i],
                        x_label, y_labels[i], titles[i])
    return fig


def make_violinplot_for_comparing_models(title: str, ffnn_data: np.ndarray or list, ffnn_dropout_data: np.ndarray or list, bnn_data: np.ndarray or list):
    df = pd.DataFrame({
        'FFNN': ffnn_data,
        'FFNN w/dropout': ffnn_dropout_data,
        'BNN': bnn_data,
    })
    colors = sns.color_palette("hls", 8)
    fig, ax = plt.subplots(1, 1)
    sns.violinplot(
        data=df, palette={'FFNN': colors[2], 'FFNN w/dropout': colors[3], 'BNN': colors[4]}, ax=ax)
    plt.ylabel('Entropy')
    fig.suptitle(title, fontsize=16)
    return fig


def make_violinplot_for_comparing_sets(title: str, train_data: np.ndarray or list, test_data: np.ndarray or list, not_mnist_data: np.ndarray or list):
    df = pd.DataFrame({
        'Training set': train_data,
        'Test set': test_data,
        'Anomalies set': not_mnist_data,
    })
    colors = sns.color_palette("hls", 8)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.violinplot(
        data=df, palette={'Training set': colors[2], 'Test set': colors[3], 'Anomalies set': colors[4]}, ax=ax)
    plt.ylabel('Entropy')
    fig.suptitle(title, fontsize=16)
    return fig


def make_violinplot_for_comparing_sizes(title: str, data_50: np.ndarray or list, data_19: np.ndarray or list, data_7: np.ndarray or list, data_2: np.ndarray or list, data_1: np.ndarray or list):
    df = pd.DataFrame({
        '1 000': data_1,
        '2 500': data_2,
        '7 000': data_7,
        '19 000': data_19,
        '50 000': data_50
    })
    colors = sns.color_palette("hls", 8)
    fig, ax = plt.subplots(1, 1)
    sns.violinplot(
        data=df,
        palette={'1 000': colors[2], '2 500': colors[3], '7 000': colors[4], '19 000': colors[7], '50 000': colors[6]},
        ax=ax)
    plt.ylabel('Entropy')
    plt.xlabel('Size of training set')
    fig.suptitle(title, fontsize=16)
    return fig
