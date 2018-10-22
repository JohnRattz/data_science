import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

### Financial ###

def find_Markowitz_optimal_portfolio_weights(log_returns, return_risk_free=0, markowitz_iterations=100000, plot=False):
    """
    Finds the optimal fractional composition (the "weights") for a portfolio of financial assets using the
    Markowitz portfolio optimization technique.
    # TODO: Increase `markowitz_iterations` to 100000 when done programming.
    # TODO: Store the plot image rather than showing it.

    Parameters
    ----------
    log_returns: pandas.core.frame.DataFrame
        The logarithmic returns of the financial assets being considered.
    return_risk_free: int or float
        The return of a risk-free asset.
    plot: bool
        Whether or not to plot the scattered points, showing the "efficient frontier".

    Returns
    -------
    optimal_weights: pandas.Series
        The optimal fractional composition of the financial assets contained in `log_returns.
    """
    portfolio_returns = np.empty(markowitz_iterations, dtype=np.float64)
    portfolio_volatilities = np.empty(markowitz_iterations, dtype=np.float64)
    log_returns_mean = log_returns.mean()
    log_returns_cov = log_returns.cov()
    num_assets = len(log_returns.columns)
    # Weights are the fractional amounts of assets in the hypothetical portfolio.
    # The optimal weights are the ones that maximize the Sharpe ratio for the portfolio.
    weights = np.empty((markowitz_iterations, num_assets), dtype=np.float64)
    for i in range(markowitz_iterations):
        weights[i] = np.random.random(num_assets)
        weights[i] /= np.sum(weights[i])
        portfolio_returns[i] = np.sum(weights[i] * log_returns_mean)
        portfolio_volatilities[i] = np.sqrt(np.dot(weights[i].T, np.dot(log_returns_cov, weights[i])))
    if plot: # Plot the scattered points, showing the "efficient frontier".
        portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
        portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6), s=12, alpha=0.2)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.show()
    sharpe_ratios = (portfolio_returns - return_risk_free) / np.sqrt(portfolio_volatilities)
    index_max_sharpe = np.argmax(sharpe_ratios)
    return pd.Series(data=weights[index_max_sharpe], index=log_returns.columns)

def calc_CAPM_betas(returns, market_index):
    """
    Calculate the CAPM beta values of assets based on their returns and a specified market index.

    Parameters
    ----------
    returns: pandas.core.frame.DataFrame
        The returns of the assets to acquire beta values for (index would be time).
    market_index: str
        String specifying the column name in `returns` that serves as the market index.

    Returns
    -------
    betas: pandas.Series
        A `Series` containing the beta values for each asset in `returns`.
    """
    returns_cov = returns.cov()
    betas = pd.Series(index=returns.columns)
    # The index of the column in `betas` corresponding to the market index (Bitcoin)
    market_index_index = betas.index.get_loc(market_index)
    for i, label_and_ticker in enumerate(betas.index):
        cov_with_market = returns_cov.iloc[i, market_index_index]
        market_var = returns_cov.iloc[market_index_index, market_index_index]
        betas[label_and_ticker] = cov_with_market / market_var
    return betas

def CAPM_RoR(betas, returns, market_index, return_risk_free):
    """
    # TODO: Document this method.
    """
    # print("In CAPM_RoR!")
    # print("betas:", betas)
    # print("returns:", returns)
    return_market = returns[market_index].mean()
    # print("return_market:", return_market)
    risk_premium = return_market - return_risk_free
    # print("risk_premium:", risk_premium)
    CAPM_expected_rates_of_return = pd.Series(index=betas.index)
    for i, label_and_ticker in enumerate(CAPM_expected_rates_of_return.index):
        CAPM_expected_rates_of_return[i] = return_risk_free + betas[label_and_ticker] * risk_premium
    return CAPM_expected_rates_of_return


def run_monte_carlo_financial_simulation(prices, extrapolation_dates, iterations=1000):
    """
    Runs a Monte Carlo simulation of asset values using the concept of Brownian motion.

    Parameters
    ----------
    prices: pandas.core.frame.DataFrame
        The value history for the assets being considered.
    extrapolation_dates: numpy.ndarray
        The dates for which we desire to predict asset values.
    iterations: int
        The number of values to generate for each extrapolation date. Directly impacts accuracy and runtime performance.

    Returns
    -------
    predicted_values_ranges: pandas.core.frame.DataFrame
        A `DataFrame` with cells containing 1D NumPy arrays with `iterations` predicted asset values.
    predicted_values: pandas.core.frame.DataFrame
        A `DataFrame` with cells containing the average values of the arrays in the cells of `predicted_values_ranges`.
    """
    num_extrapolation_dates = len(extrapolation_dates)
    log_returns = np.log(prices / prices.shift(1))
    log_returns_means = log_returns.mean(axis=0)
    log_returns_vars = log_returns.var(axis=0)
    # The drifts represent the directions that the logarithmic rates of return have been taking thus far.
    drifts = log_returns_means - (0.5 * log_returns_vars)
    log_returns_stds = log_returns.std(axis=0)
    predicted_log_returns = {}
    for i, label_and_ticker in enumerate(log_returns.columns):
        # `Z` is a model of Brownian motion.
        Z = norm.ppf(np.random.rand(num_extrapolation_dates, iterations))
        predicted_log_returns[label_and_ticker] = \
            pd.DataFrame(data=np.exp(np.array(drifts.loc[label_and_ticker]) +
                                     np.array(log_returns_stds.loc[label_and_ticker]) * Z),
                         index=extrapolation_dates)
    initial_values = prices.iloc[-1]
    # For each asset, for each date, there are `iterations` predicted values.
    predicted_values_ranges = pd.DataFrame(index=extrapolation_dates, columns=log_returns.columns)
    # This `DataFrame` contains the mean of predicted values for each asset, for each date.
    predicted_values = pd.DataFrame(index=extrapolation_dates, columns=log_returns.columns)
    for label_index, label_and_ticker in enumerate(log_returns.columns):
        predicted_values_ranges.iloc[0, label_index] = \
            initial_values[label_and_ticker] * predicted_log_returns[label_and_ticker].iloc[0].values
        predicted_values.iloc[0, label_index] = \
            np.mean(predicted_values_ranges.iloc[0, label_index])
        for date_index in range(1, num_extrapolation_dates):
            predicted_values_ranges.iloc[date_index, label_index] = \
                predicted_values_ranges.iloc[date_index - 1, label_index] * \
                predicted_log_returns[label_and_ticker].iloc[date_index].values
            predicted_values.iloc[date_index, label_index] = \
                np.mean(predicted_values_ranges.iloc[date_index, label_index])
    return predicted_values_ranges, predicted_values

### Financial End ###