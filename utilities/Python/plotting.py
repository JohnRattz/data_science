import numpy as np
import pandas as pd
from scipy.stats import norm
import keras
from keras import optimizers
import matplotlib.pyplot as plt

value_text_fontsize = 12

def monte_carlo_plot_confidence_band(ax, extrapolation_dates, monte_carlo_predicted_values_ranges,
                                     label_and_ticker, color, confidence=0.95):
    """
    Plots a confidence band on `ax` (a `matplotlib.axes.Axes` object).

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axes on which to plot the confidence band.
    extrapolation_dates: numpy.ndarray
        The dates being considered.
    monte_carlo_predicted_values_ranges: pandas.DataFrame
        A `DataFrame` of `ndarray` objects containing randomly generated values for `extrapolation_dates`.
    label_and_ticker: str
        The asset being considered. A column label in `monte_carlo_predicted_values_ranges`.
    color: str
        The band's color. One of the named colors recognized by `matplotlib`.
    confidence: float
        The fractional percent confidence for which to plot the band.
    """
    current_predicted_values_ranges = monte_carlo_predicted_values_ranges[label_and_ticker]
    # The mean values for each extrapolation date for this asset.
    predicted_values_means = \
        current_predicted_values_ranges.apply(lambda values: np.mean(values))
    predicted_values_stds = \
        current_predicted_values_ranges.apply(lambda values: np.std(values))
    # A `DataFrame` of tuples containing the minimum and maximum values
    # of the interval of desired confidence for each date.
    min_max_values = pd.DataFrame(index=predicted_values_means.index)
    # Record the minimum and maximum values for the confidence interval of each date.
    for date in extrapolation_dates:
        minimum, maximum = norm.interval(alpha=1 - confidence, loc=predicted_values_means.loc[date],
                                         scale=predicted_values_stds.loc[date])
        min_max_values.loc[date, 0] = minimum
        min_max_values.loc[date, 1] = maximum
    ax.fill_between(extrapolation_dates, min_max_values[0], min_max_values[1],
                    color=color, label='Monte Carlo {:.0f}% Confidence Band'.format(confidence*100))

### Seaborn ###

def add_value_text_to_seaborn_barplot(ax, plotting_data, vertical_label, percent=False):
    """
    Adds text for the value of each bar in a seaborn bar plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes (matplotlib.axes.AxesSubplot)
        The axes returned from a call to `seaborn.barplot()`
    plotting_data: pandas.core.frame.DataFrame
        The `DataFrame` containing the plotting data.
    horizontal_label: str
        A string denoting the label of the column in `plotting_data` which spans the x axis.
    vertical_label: str
        A string denoting the label of the column in `plotting_data` which determines bar height.
    """
    x_pos = 0  # The position on the x axis at which to add the text.
    for index, row in plotting_data.iterrows():
        text = str(round(row[vertical_label], 3)) if not percent else str(round(row[vertical_label], 3) * 100) + '%'
        ax.text(x_pos, row[vertical_label], text, color='black', ha='center', fontsize=value_text_fontsize)
        x_pos += 1.0  # Add the x axis spacing between points.

### End Seaborn ###

### Keras ###

def get_optimizer_name(optimizer):
    """
    Returns an appropriate name for a given Keras optimizer.

    Parameters
    ----------
    optimizer: keras.optimizers.Optimizer
        A Keras optimizer, such as the Adam optimizer.
    """
    names = {optimizers.Adam:'adam'}
    return names[optimizer]

class KerasPlotLosses(keras.callbacks.Callback):
    """
    A plotting callback for keras **fit()** functions.
    e.g. **model.fit(..., callbacks=[KerasPlotLosses(nb_epochs)])**

    This was derived from a GitHub Gist by user "stared":
    https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
    """
    def __init__(self, optimizer, optimizer_params, nb_epochs, filepath=None):
        """
        TODO: Document this function.
        :param optimizer: 
        :param optimizer_params: 
        :param nb_epochs: 
        :param filepath: 
        """
        self.nb_epochs = nb_epochs
        optimizer_name = get_optimizer_name(optimizer)
        # Space-separated list of parameter names and their values.
        optimizer_param_names = ", ".join(["{}:{}".format(k,v) for k,v in
                                          optimizer_params.items()])
        self.title = "{} ({})".format(optimizer_name, optimizer_param_names)
        self.filename = filepath
        if filepath is not None:
            for param, param_val in optimizer_params.items():
                if isinstance(param_val, (int, float)):
                    self.filename += '__{}_{}'.format(param, param_val)
        print("Writing training results to {}".format(self.filename))

    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        plt.close('all')
        # plt.ion()
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))
        # plt.plot(self.x, self.losses, label="loss")
        # plt.title(self.title)
        # plt.legend()
        # plt.draw()
        # plt.pause(0.001)
        # plt.cla()

        if epoch == self.nb_epochs - 1: # Plot the figure after the last epoch.
            # plt.ioff()
            plt.plot(self.x, self.losses, label="loss")
            plt.title(self.title)
            plt.legend()
            if self.filename is not None:
                plt.savefig(self.filename)

### End Keras ###
