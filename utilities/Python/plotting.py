import os

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
    # TODO: Enable this to plot properly even when other matplotlib figures are created during its lifetime.
    A plotting callback for keras **fit()** functions.
    e.g. **model.fit(..., callbacks=[KerasPlotLosses(nb_epochs)])**

    This callback plots loss over training epochs for Keras models. A notable use is
    to plot loss for multiple parameter sets during a grid search.

    This was derived from a GitHub Gist by user "stared":
    https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
    """
    def __init__(self, nb_epochs, dirname=None, title_prefix="", **kwargs):
        """
        TODO: Document this function.

        Parameters
        ----------
        nb_epochs: int
            Then number of epochs that will be used in training.
        dirname: str
            The filepath to save the resulting figure to.
        **kwargs: dict
            Keyword arguments for the construction of the figure (using `matplotlib.pyplot.figure()`).
        """
        super().__init__()
        self.fig, self.axes = plt.subplots(**kwargs)
        # self.axes = plt.gca()
        self.fig_num = self.fig.number
        self.nb_epochs = nb_epochs
        self.dirname = dirname
        if dirname is not None:
            self.filename = dirname
        self.title_prefix = title_prefix
        # The title of the figure.
        self.title = None
        # Dictionary of optimizer parameters.
        self.optimizer_params = None
        # Space-separated list of parameter names and their values. Used to set labels in the figure.
        self.optimizer_param_names = None
        # Non-optimizer parameters.
        self.batch_size = None
        self.hidden_layer_sizes = None
        self.dropout_rate = None

    def set_optimizer(self, optimizer):
        self.title = "{} Keras training loss (Optimizer: {})".format(self.title_prefix, get_optimizer_name(optimizer))

    def set_optimizer_params(self, optimizer_params):
        self.optimizer_params = optimizer_params
        # print("self.optimizer_params:", self.optimizer_params)
        # print("Figure id:", id(self.fig))
        self.optimizer_param_names = ", ".join(["{}:{}".format(k, v) for k, v in
                                           optimizer_params.items()])

    ## Setting Non-Optimizer Parameters (for naming the file) ##

    def set_non_optimizer_params(self, non_optimizer_params):
        """
        Sets non-optimizer parameter values with a dictionary.
        """
        if 'batch_size' in non_optimizer_params:
            self.set_batch_size(non_optimizer_params['batch_size'])
        if 'hidden_layer_sizes' in non_optimizer_params:
            self.set_hidden_layer_sizes(non_optimizer_params['hidden_layer_sizes'])
        if 'dropout_rate' in non_optimizer_params:
            self.set_dropout_rate(non_optimizer_params['dropout_rate'])

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_hidden_layer_sizes(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate

    ## End Setting Non-Optimizer Parameters ##

    def set_filename(self):
        if (self.dirname is not None) and (self.filename == self.dirname) and \
                (self.batch_size is not None) and (self.hidden_layer_sizes is not None):
            self.filename = 'b_{}__h'.format(self.batch_size)
            for hidden_layer_size in self.hidden_layer_sizes:
                self.filename += '_{}'.format(hidden_layer_size)
            if self.dropout_rate is not None:
                self.filename += '__d_{}'.format(str(self.dropout_rate).replace('.', '_'))
            self.filename = os.path.join(self.dirname, self.filename)

    def save_figure(self):
        self.set_filename()
        plt.figure(self.fig_num) # Reset the active figure to this one.
        if self.filename is not None:
            plt.savefig(self.filename)

    ## Methods from keras.callbacks.Callback ##

    def on_train_begin(self, logs={}):
        plt.figure(self.fig_num) # Reset the active figure to this one.
        self.logs = []
        self.x = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))

        if epoch == self.nb_epochs - 1: # This is the last epoch, so plot the losses.
            plt.plot(self.x, self.losses, label=self.optimizer_param_names)
            plt.title(self.title)
            plt.legend()

    ## End Methods from keras.callbacks.Callback ##

### End Keras ###
