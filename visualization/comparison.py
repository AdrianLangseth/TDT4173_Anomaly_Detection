import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')


def make_line_chart(ax: axes.Axes, x: list, y: list, label: list, x_label: str, y_label: str, titel: str):
    for i in range(len(y)):
        ax.plot(x, y[i], label=label[i])
    ax.set_title(titel)
    ax.legend(loc='best')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def make_line_charts(x: np.ndarray, y: list, legend_labels: list, x_label: str, y_labels: list, titles: list, fig_width: float, fig_height: float, nrows: int, ncols: int):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(fig_width, fig_height))
    plt.subplots_adjust(hspace=0.5)
    for i in range(len(axs)):
        make_line_chart(axs[i], x, y[i], legend_labels[i],
                        x_label, y_labels[i], titles[i])
    return fig


def make_violinplot(title: str, ffnn_data: list, ffnn_dropout_data: list, bnn_data: list):
    df = pd.DataFrame({
        'FFNN': ffnn_data,
        'FFNN w/dropout': ffnn_dropout_data,
        'BNN': bnn_data,
    })
    colors = sns.color_palette("hls")
    fig, ax = plt.subplots(1, 1)
    sns.violinplot(
        data=df, palette={'FFNN': colors[2], 'FFNN w/dropout': colors[3], 'BNN': colors[5]}, ax=ax)
    fig.suptitle(title, fontsize=16)
    return fig
