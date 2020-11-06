import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np


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
