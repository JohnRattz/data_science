import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### Financial ###

def find_optimal_portfolio_weights(log_returns, return_risk_free=0, plot=False):
    """
    Finds the optimal fractional composition (the "weights") for a portfolio of financial assets.

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
    markowitz_iterations = 1000  # TODO: Increase `markowitz_iterations` to 100000 when done programming.
    portfolio_returns = np.empty(markowitz_iterations, dtype=np.float64)
    portfolio_volatilities = np.empty(markowitz_iterations, dtype=np.float64)
    subset_log_returns_mean = log_returns.mean()
    subset_log_returns_cov = log_returns.cov()
    num_assets = len(log_returns.columns)
    # Weights are the fractional amounts of assets in the hypothetical portfolio.
    # The optimal weights are the ones that maximize the Sharpe ratio for the portfolio.
    weights = np.empty((markowitz_iterations, num_assets), dtype=np.float64)
    for i in range(markowitz_iterations):
        weights[i] = np.random.random(num_assets)
        weights[i] /= np.sum(weights[i])
        portfolio_returns[i] = np.sum(weights[i] * subset_log_returns_mean)
        portfolio_volatilities[i] = np.sqrt(np.dot(weights[i].T, np.dot(subset_log_returns_cov, weights[i])))
    if plot: # Plot the scattered points, showing the "efficient frontier".
        portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
        portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6), s=12, alpha=0.2)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.show()
    sharpe_ratios = (portfolio_returns - return_risk_free) / np.sqrt(portfolio_volatilities)
    index_max_sharpe = np.argmax(sharpe_ratios)
    return pd.Series(data=weights[index_max_sharpe], index=log_returns.columns)