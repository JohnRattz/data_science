value_text_fontsize = 12

def add_value_text_to_seaborn_barplot(ax, plotting_data, horizontal_label, vertical_label):
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
    x_pos = 0 # The position on the x axis at which to add the text.
    for index, row in plotting_data.iterrows():
        ax.text(x_pos, row[vertical_label], str(round(row[vertical_label], 2)),
                color='black', ha='center', fontsize=value_text_fontsize)
        x_pos += 1.0  # Add the x axis spacing between points.

def monte_carlo_plot_confidence_band(ax, extrapolation_dates, monte_carlo_predicted_values_ranges,
                                     label_and_ticker, confidence=0.95):
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
        The cryptocurrency being considered.
    confidence: float
        The fractional percent confidence for which to plot the band.
    """
    import pandas as pd
    from scipy.stats import norm

    current_predicted_values_ranges = monte_carlo_predicted_values_ranges[label_and_ticker]
    # The mean values for each extrapolation date for this cryptocurrency.
    predicted_values_means = \
        current_predicted_values_ranges.apply(lambda values: np.mean(values))
    predicted_values_stds = \
        current_predicted_values_ranges.apply(lambda values: np.std(values))
    # A `DataFrame` of tuples containing the minimum and maximum values
    # of the interval of desired confidence for each date.
    min_max_values = pd.DataFrame(index=predicted_values_means.index)
    # Record the minimum and maximum values for the confidence interval of each date.
    for date in extrapolation_dates:
        minimum, maximum = norm.interval(alpha=1-confidence, loc=predicted_values_means.loc[date],
                                         scale=predicted_values_stds.loc[date])
        min_max_values.loc[date, 0] = minimum
        min_max_values.loc[date, 1] = maximum
    ax.fill_between(extrapolation_dates, min_max_values[0], min_max_values[1],
                    color='cyan', label='Monte Carlo {:.0f}% Confidence Band'.format(confidence*100))