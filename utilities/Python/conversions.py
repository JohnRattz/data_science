def returns_to_yearly(returns, resolution):
    """
    Compounds daily or hourly returns to yearly returns.

    Parameters
    ----------
    returns: pandas.core.series.Series or numpy.ndarray
        The returns to convert to yearly returns.
    resolution: str
        Either 'daily' or 'hourly'.
    """
    exp_fact = 365 if resolution == 'daily' else 24*365
    return (returns + 1) ** 365