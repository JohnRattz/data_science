import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV

from cryptocurrencies.Python.ETL import load_csvs, load_sql

warnings.simplefilter('ignore')
import seaborn as sns

from analysis import find_optimal_portfolio_weights
from plotting import add_value_text_to_seaborn_barplot, monte_carlo_plot_confidence_band

def pandas_dt_to_str(date):
    """
    Prints pandas `Datetime` objects as strings.
    """
    return date.strftime("%Y-%m-%d")


def numpy_dt64_to_str(date):
    """
    Prints NumPy `datetime64` objects as strings.
    """
    pd_datetime = pd.to_datetime(str(date))
    return pandas_dt_to_str(pd_datetime)


def main():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    figure_dpi = 200

    # Data Importing (SQL)
    # bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
    # monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves = load_sql()
    # Data Importing (CSV)
    bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
    monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves = load_csvs()

    currencies_labels_tickers = np.array(
        [['Bitcoin Cash', 'BCH'], ['Bitcoin', 'BTC'], ['BitConnect', 'BCC'], ['Dash', 'DASH'],
         ['Ethereum Classic', 'ETC'], ['Ethereum', 'ETH'], ['Iota', 'MIOTA'], ['Litecoin', 'LTC'],
         ['Monero', 'XMR'], ['Nem', 'XEM'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
         ['Qtum', 'QTUM'], ['Ripple', 'XRP'], ['Stratis', 'STRAT'], ['Waves', 'WAVES']])
    currencies_labels = currencies_labels_tickers[:, 0]
    currencies_tickers = currencies_labels_tickers[:, 1]
    currencies_labels_and_tickers = ["{} ({})".format(currencies_label, currencies_ticker)
                                     for currencies_label, currencies_ticker in
                                     zip(currencies_labels, currencies_tickers)]
    num_currencies = len(currencies_labels_tickers)
    currencies = pd.concat([bitcoin_cash, bitcoin, bitconnect, dash, ethereum_classic, ethereum, iota,
                            litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves],
                           axis=1, keys=currencies_labels_and_tickers)
    currencies.columns.names = ['Name', 'Currency Info']
    currencies.set_index(pd.to_datetime(currencies.index), inplace=True)
    currencies.index.name = 'Date'
    currencies.sort_index(inplace=True)
    # Retrieve the closing prices for the currencies.
    prices = currencies.xs(key='Close', axis=1, level='Currency Info')

    # Value Trends

    num_cols = 2
    num_rows = int(np.ceil(num_currencies / num_cols))
    for i, column in enumerate(prices.columns):
        currency_label = currencies_labels[i]
        currency_label_no_spaces = currency_label.replace(' ', '_')
        currency_ticker = currencies_tickers[i]
        plt.figure(figsize=(12, 6))
        plt.plot(prices[column])
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.title("{} ({}) Closing Value"
                  .format(currency_label, currency_ticker))
        currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
        if not os.path.exists(currency_figures_subdir):
            os.makedirs(currency_figures_subdir)
        plt.savefig('{}/{}_price_trend'.format(currency_figures_subdir, currency_label_no_spaces))

    # Value Correlations

    plt.figure(figsize=(12, 6))
    price_correlations = prices.corr()
    correlations = sns.clustermap(price_correlations, annot=True)
    correlations.savefig('{}/correlations.png'.format(figures_dir))
    plt.clf()

    # Removing Currencies with Short Histories

    # See where values are absent and keep only currencies with reasonably lengthy histories.
    # Convert the timestamps to strings with less length for y-axis labels.
    old_prices_index = prices.index
    prices.index = prices.index.map(lambda date: pandas_dt_to_str(date))
    is_null_prices = pd.isnull(prices)
    # Select dates every 120 days for tick labels.
    ax = sns.heatmap(is_null_prices, yticklabels=120)
    absent_values_fig = ax.get_figure()
    plt.tight_layout()
    prices.index = old_prices_index
    absent_values_fig.savefig('{}/absent_values.png'.format(figures_dir))

    currencies_labels_tickers_to_remove = np.array(
        [['Bitcoin Cash', 'BCH'], ['BitConnect', 'BCC'], ['Ethereum Classic', 'ETC'],
         ['Iota', 'MIOTA'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
         ['Qtum', 'QTUM'], ['Stratis', 'STRAT'], ['Waves', 'WAVES']])
    currencies_labels_to_remove = currencies_labels_tickers_to_remove[:, 0]
    currencies_tickers_to_remove = currencies_labels_tickers_to_remove[:, 1]
    currencies_labels_and_tickers_to_remove = ["{} ({})".format(currencies_label, currencies_ticker)
                                               for currencies_label, currencies_ticker in
                                               zip(currencies_labels_to_remove, currencies_tickers_to_remove)]
    print("Removing currencies: {}".format(currencies_labels_and_tickers_to_remove))
    subset_prices = prices.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_currencies_labels = [currency_label for currency_label in currencies_labels
                                if currency_label not in currencies_labels_to_remove]
    subset_currencies_tickers = [currency_ticker for currency_ticker in currencies_tickers
                                 if currency_ticker not in currencies_tickers_to_remove]
    subset_currencies_labels_and_tickers = subset_prices.columns.values
    subset_num_currencies = len(subset_prices.columns)
    subset_prices_nonan = subset_prices.dropna()
    print("Beginning and ending dates with data for remaining currencies: {}, {}".
          format(subset_prices_nonan.index[0], subset_prices_nonan.index[-1]))

    # Volatility Examination

    num_non_nan_days = len(subset_prices_nonan)
    print("Considering {} days of price information (as many as without NaN values)".format(num_non_nan_days))
    # Find the returns.
    returns = prices.pct_change()
    subset_returns = subset_prices_nonan.pct_change()
    log_returns = np.log(prices / prices.shift(1))
    subset_log_returns = log_returns.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)

    # Find the standard deviations in returns.
    returns_std_yearly = subset_returns.groupby(subset_prices_nonan.index.year).std()
    # Standard deviations in returns for 2017 in descending order.
    returns_std_2017 = returns_std_yearly.loc[2017]
    returns_std_2017.sort_values(ascending=False, inplace=True)
    plt.subplots(figsize=(12, 6))
    plotting_data = returns_std_2017.to_frame(name='Volatility').reset_index()
    volatility_plot = sns.barplot(x='Name', y='Volatility', data=plotting_data, palette='viridis')
    add_value_text_to_seaborn_barplot(volatility_plot, plotting_data, 'Volatility')
    plt.title('Volatility (2017)')
    plt.savefig('{}/volatility.png'.format(figures_dir), dpi=figure_dpi)

    # Find (daily) asset betas (covariance_with_market / market_variance).
    log_returns_cov = log_returns.cov()
    # print(log_returns_cov.head())
    betas = pd.Series(index=log_returns.columns)
    # The index of the column in `betas` corresponding to the market index (Bitcoin)
    market_index = "Bitcoin (BTC)"
    market_index_index = betas.index.get_loc(market_index)
    for i, label_and_ticker in enumerate(betas.index):
        cov_with_market = log_returns_cov.iloc[i, market_index_index]
        # print("cov_with_market: ", cov_with_market)
        market_var = log_returns_cov.iloc[market_index_index, market_index_index]
        # print("market_var: ", market_var)
        betas[label_and_ticker] = cov_with_market / market_var
    betas.sort_values(inplace=True)
    subset_betas = betas.drop(labels=currencies_labels_and_tickers_to_remove)
    # Create a visualization with the beta values.
    fig, ax = plt.subplots(figsize=(12, 6))
    plotting_data = subset_betas.to_frame(name='Beta').reset_index()
    beta_plot = sns.barplot(ax=ax, x='Name', y='Beta', data=plotting_data, palette='viridis')
    # Show values in the figure.
    add_value_text_to_seaborn_barplot(beta_plot, plotting_data, 'Beta')
    plt.title('Betas (Bitcoin (BTC) as Market Index)')
    plt.savefig('{}/betas.png'.format(figures_dir), dpi=figure_dpi)

    # Find assets' rates of return according to CAPM:
    return_risk_free = 0  # 0.025 # RoR of 10 year US government bond
    return_market = returns[market_index].mean()
    risk_premium = return_market - return_risk_free
    CAPM_expected_rates_of_return = pd.Series(index=subset_betas.index)
    for i, label_and_ticker in enumerate(CAPM_expected_rates_of_return.index):
        CAPM_expected_rates_of_return[i] = return_risk_free + subset_betas[label_and_ticker] * risk_premium
    CAPM_expected_rates_of_return.sort_values(ascending=False, inplace=True)
    # Create a visualization with the weights.
    fig, ax = plt.subplots(figsize=(12, 6))
    plotting_data = CAPM_expected_rates_of_return.to_frame(name='Rate of Return').reset_index()
    CAPM_plot = sns.barplot(ax=ax, x='Name', y='Rate of Return', data=plotting_data, palette='viridis')
    # Show values in the figure.
    add_value_text_to_seaborn_barplot(CAPM_plot, plotting_data, 'Rate of Return', percent=True)
    plt.title('CAPM Expected Rates of Return')
    plt.savefig('{}/CAPM_rates_of_return.png'.format(figures_dir), dpi=figure_dpi)

    # TODO: Calculate Sharpe ratios for individual assets ((return_asset - return_risk_free)/std_asset)?

    # Determine the optimal portfolio using Markowitz optimization.
    optimal_weights = find_optimal_portfolio_weights(subset_log_returns, return_risk_free)
    optimal_weights.sort_values(ascending=False, inplace=True)
    # Create a visualization with the weights.
    fig, ax = plt.subplots(figsize=(12, 6))
    plotting_data = optimal_weights.to_frame(name='Weight').reset_index()
    portfolio_weights_plot = sns.barplot(ax=ax, x='Name', y='Weight', data=plotting_data, palette='viridis')
    # Show values in the figure.
    add_value_text_to_seaborn_barplot(portfolio_weights_plot, plotting_data, 'Weight')
    plt.title('Markowitz Optimal Portfolio Weights')
    plt.savefig('{}/optimal_portfolio_weights.png'.format(figures_dir), dpi=figure_dpi)

    # Run Monte Carlo simulation to predict future values.
    return_means = log_returns.mean(axis=0)
    # print("\nreturn_means: {}\n".format(return_means))
    return_vars = log_returns.var(axis=0)
    # print("\nreturn_vars: {}\n".format(return_vars))
    drifts = return_means - (0.5 * return_vars)
    # print("\ndrifts: {}\n".format(drifts))
    return_stds = log_returns.std(axis=0)
    # print("\nreturn_stds: {}\n".format(return_stds))
    from scipy.stats import norm
    # print("Stdevs from mean for cumulative probability of 0.95 for a norm dist: ", norm.ppf(0.95))
    # print("\nreturns.tail(): {}\n".format(returns.tail()))
    last_data_date = log_returns.index[-1]
    first_extrapolation_date = last_data_date + pd.DateOffset(days=1)
    last_extrapolation_date = pd.to_datetime('2018-12-31')
    extrapolation_dates = np.array(pd.date_range(first_extrapolation_date, last_extrapolation_date))
    num_extrapolation_dates = len(extrapolation_dates)
    # num_days_to_predict = 1000  # The number of days after the last date in the data to predict currency values for.
    # TODO: Increase `monte_carlo_iterations` when finished programming.
    monte_carlo_iterations = 5  # The number of times to run the simulation. The number of columns in `predicted_returns` below.
    # print("first extrapolation date: ", numpy_dt64_to_str(extrapolation_dates[0]))
    # print("date after last available: ", pandas_dt_to_str(last_extrapolation_date + pd.Timedelta(days=1)))
    # print("last extrapolation date: ", numpy_dt64_to_str(extrapolation_dates[-1]))
    # Predicted log returns.
    monte_carlo_predicted_returns = {}  # pd.DataFrame(columns=log_returns.columns, index=extended_date_range)
    for i, label_and_ticker in enumerate(log_returns.columns):
        # `Z` is a model of Brownian motion.
        Z = norm.ppf(np.random.rand(num_extrapolation_dates, monte_carlo_iterations))
        # print("shape: ", np.exp(np.array(drifts.loc[label_and_ticker]) + np.array(return_stds.loc[label_and_ticker]) * Z).shape)
        monte_carlo_predicted_returns[label_and_ticker] = \
            pd.DataFrame(data=np.exp(np.array(drifts.loc[label_and_ticker]) +
                                     np.array(return_stds.loc[label_and_ticker]) * Z),
                         index=extrapolation_dates)
        # Row 0 contains the last date of known price information.
        # index=pd.Index(np.append(np.array(last_extrapolation_date),
        # extended_date_range)))
        # print("predicted_returns[{}].shape: {}".format(label_and_ticker, predicted_returns[label_and_ticker].shape))
    # print("\nlog_returns.tail(2): \n{}".format(log_returns.tail(2)))
    # print("\npredicted_returns.head(2): \n{}".format(predicted_returns.head(2)))
    # print("\npredicted_returns.tail(2): \n{}".format(predicted_returns.tail(2)))
    # print("predicted_returns.shape: ", predicted_returns.shape)
    initial_values = prices.iloc[-1]
    # print("initial_values:\n", initial_values)
    # For each cryptocurrency, for each date, there are `monte_carlo_iterations` predicted values.
    monte_carlo_predicted_values_ranges = pd.DataFrame(index=extrapolation_dates, columns=log_returns.columns)
    # This `DataFrame` has one value for each cryptocurrency, for each date (the mean).
    monte_carlo_predicted_values = pd.DataFrame(index=extrapolation_dates, columns=log_returns.columns)
    for label_index, label_and_ticker in enumerate(log_returns.columns):
        monte_carlo_predicted_values_ranges.iloc[0, label_index] = \
            initial_values[label_and_ticker] * monte_carlo_predicted_returns[label_and_ticker].iloc[0].values
        monte_carlo_predicted_values.iloc[0, label_index] = \
            np.mean(monte_carlo_predicted_values_ranges.iloc[0, label_index])
        for date_index in range(1, num_extrapolation_dates):
            monte_carlo_predicted_values_ranges.iloc[date_index, label_index] = \
                monte_carlo_predicted_values_ranges.iloc[date_index - 1, label_index] * \
                monte_carlo_predicted_returns[label_and_ticker].iloc[date_index].values
            monte_carlo_predicted_values.iloc[date_index, label_index] = \
                np.mean(monte_carlo_predicted_values_ranges.iloc[date_index, label_index])
    # print("predicted_values.head(2):\n", monte_carlo_predicted_values_ranges.head(2))
    subset_monte_carlo_predicted_values_ranges = \
        monte_carlo_predicted_values_ranges.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_monte_carlo_predicted_values = \
        monte_carlo_predicted_values.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    # plt.clf()
    # monte_carlo_predicted_values.plot()
    # subset_monte_carlo_predicted_values.plot()
    # for date in monte_carlo_predicted_values_ranges.index:
    #     values = np.array(monte_carlo_predicted_values_ranges.loc[date,'Bitcoin (BTC)'])
    #     plt.plot(np.repeat(date, len(values)), values)
    # plt.show()
    # exit()

    # Data Extraction

    data = subset_prices_nonan.reset_index()
    dates = data['Date'].values

    # We will predict closing prices based on these numbers of days preceding the date of prediction.
    # The max `window_size` is `num_non_nan_days`, but some lower values may result in poor models due to small test sets
    # during cross validation, and possibly even training failures due to empty test sets for even larger values.
    window_sizes = list(range(30, 361, 30))  # Window sizes measured in days - approximately 1 to 12 months.
    for window_size in window_sizes:
        print("Predicting prices with a window of {} days of preceding currency values".format(window_size))
        num_windows = len(data) - window_size
        X = np.empty((num_windows, subset_num_currencies * window_size), dtype=np.float64)
        for i in range(num_windows):
            X[i] = subset_prices_nonan[i:i + window_size].values.flatten()
        y = data.drop('Date', axis=1).values[window_size:]

        # Model Training
        load_models = True
        if not load_models:
            # Ensemble models
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            # Extra-Trees regressor
            model_extra_trees = ExtraTreesRegressor()
            params_extra_trees = {'n_estimators': [500],
                                  'min_samples_split': [2, 5, 10],
                                  'max_features': ['auto', 'sqrt', 'log2']}
            # Random forest regressor
            model_random_forest = RandomForestRegressor()
            params_random_forest = {'n_estimators': [500],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt', 'log2']}

            # Neighbors models
            from sklearn.neighbors import KNeighborsRegressor
            # KNearestNeighbors regressor
            model_knn = KNeighborsRegressor()
            params_knn = {'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto'],
                          'leaf_size': [5, 10, 20, 30, 40, 50, 60]}

            # Collect models for use in GridSearchCV.
            models = [model_extra_trees, model_random_forest, model_knn]
            param_sets = [params_extra_trees, params_random_forest, params_knn]

            # Tuples of scores and the corresponding models
            score_model_tuples = []

            # Specify the cross validation method.
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # # Create the models.
            # import multiprocessing
            # num_cores = multiprocessing.cpu_count()
            # for i in range(len(models)):
            #     grid_search = GridSearchCV(models[i], param_sets[i],
            #                                scoring=make_scorer(r2_score), cv=cv,
            #                                n_jobs=int(num_cores/2), verbose=1)
            #     grid_search.fit(X, y)
            #     model = grid_search.best_estimator_
            #     score = grid_search.best_score_
            #     score_model_tuples.append((score, model))
            # # Wait 1 second for printing from GridSearchCV to complete.
            # import time
            # time.sleep(1)
            # # Choose the model with the best score.
            # score_model_tuples.sort(key=lambda tup: tup[0], reverse=True)
            # best_score = score_model_tuples[0][0]
            # model = score_model_tuples[0][1]
            # print("Best model and score for window size {}: {}".format(window_size, (model, best_score)))
        else:
            # Model Loading
            import pickle
            model_pickle_path = "{}/model_{}.pkl".format(models_dir, window_size)
            with open(model_pickle_path, "rb") as model_infile:
                model = pickle.load(model_infile)

        # Validation and Visualization

        pred = model.predict(X)

        # Plot the actual values along with the predictions. They should overlap.
        subset_num_cols = 2
        subset_num_rows = int(np.ceil(subset_num_currencies / subset_num_cols))
        collective_fig = plt.figure(figsize=(12 * subset_num_cols, 6 * subset_num_rows))
        for i in range(subset_num_currencies):
            currency_label = subset_currencies_labels[i]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[i]
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, i + 1)
            collective_ax_current.plot(dates[window_size:], y[:, i], color='blue', alpha=0.5, label='True')
            collective_ax_current.plot(dates[window_size:], pred[:, i], color='red', alpha=0.5, label='Predicted')
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            indiv_fig_ax.plot(dates[window_size:], y[:, i], color='blue', alpha=0.5, label='True')
            indiv_fig_ax.plot(dates[window_size:], pred[:, i], color='red', alpha=0.5, label='Predicted')
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
            indiv_fig.savefig('{}/{}_validation_{}.png'.format(currency_figures_subdir,
                                                               currency_label_no_spaces, window_size), dpi=figure_dpi)
        collective_fig.savefig('{}/validation_{}.png'.format(figures_dir, window_size), dpi=figure_dpi)
        collective_fig.clf()

        # Get the model's predictions for the rest of 2017 and 2018.
        # last_data_date = data['Date'].iloc[-1]
        # first_extrapolation_date = last_data_date + pd.DateOffset(days=1)
        # last_extrapolation_date = '2018-12-31'
        # extrapolation_dates = np.array(pd.date_range(first_extrapolation_date, last_extrapolation_date))
        # num_extrapolation_dates = len(extrapolation_dates)
        extrapolation_X = np.zeros((num_extrapolation_dates, subset_num_currencies * window_size), dtype=np.float64)
        extrapolation_y = np.zeros((num_extrapolation_dates, subset_num_currencies), dtype=np.float64)
        # First `window_size` windows contain known values.
        given_prices = subset_prices_nonan[-window_size:].values.flatten()
        extrapolation_X[0] = given_prices
        extrapolation_y[0] = model.predict(extrapolation_X[0].reshape(1, -1))
        for i in range(1, window_size):
            given_prices = subset_prices_nonan[-window_size + i:].values.flatten()
            previous_predicted_prices = extrapolation_y[:i].flatten()
            extrapolation_X[i] = np.concatenate((given_prices, previous_predicted_prices))
            extrapolation_y[i] = model.predict(extrapolation_X[i].reshape(1, -1)).flatten()
        # Remaining windows contain only predicted values (predicting based on previous predictions).
        for i in range(window_size, num_extrapolation_dates):
            previous_predicted_prices = extrapolation_y[i - window_size:i].flatten()
            extrapolation_X[i] = previous_predicted_prices
            extrapolation_y[i] = model.predict(extrapolation_X[i].reshape(1, -1)).flatten()

        ### Plotting ###
        # The colors of lines for various things.
        actual_values_color = 'blue'
        actual_values_label = 'True'
        ml_model_color = 'red'
        ml_model_label = 'ML Model'
        monte_carlo_color = 'green'
        monte_carlo_label = 'Monte Carlo AVG'

        # Plot predictions for the rest of 2017 and 2018.
        for i in range(subset_num_currencies):
            currency_label = subset_currencies_labels[i]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[i]
            label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, i + 1)
            # ML model predictions
            collective_ax_current.plot(extrapolation_dates, extrapolation_y[:, i],
                                       color=ml_model_color, label=ml_model_label)
            # Monte Carlo predictions
            collective_ax_current.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                                       color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(collective_ax_current, extrapolation_dates,
                                             monte_carlo_predicted_values_ranges, label_and_ticker)
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) Predicted Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            # ML model predictions
            indiv_fig_ax.plot(extrapolation_dates, extrapolation_y[:, i],
                              color=ml_model_color, label=ml_model_label)
            # Monte Carlo predictions
            indiv_fig_ax.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                              color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(indiv_fig_ax, extrapolation_dates,
                                             monte_carlo_predicted_values_ranges, label_and_ticker)
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) Predicted Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
            indiv_fig.savefig('{}/{}_predictions_{}.png'.format(currency_figures_subdir,
                                                                currency_label_no_spaces, window_size), dpi=figure_dpi)
        collective_fig.savefig('{}/predictions_{}.png'.format(figures_dir, window_size), dpi=figure_dpi)
        collective_fig.clf()

        # Plot the predicitons for the rest of 2017 and 2018 along with the actual values for the date range used.
        for i in range(subset_num_currencies):
            currency_label = subset_currencies_labels[i]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[i]
            label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, i + 1)
            # Actual values
            collective_ax_current.plot(dates[window_size:], y[:, i],
                                       color=actual_values_color, label=actual_values_label)
            # ML model predictions
            collective_ax_current.plot(extrapolation_dates, extrapolation_y[:, i],
                                       color=ml_model_color, label=ml_model_label)
            # Monte Carlo predictions
            collective_ax_current.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                                       color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(collective_ax_current, extrapolation_dates,
                                             monte_carlo_predicted_values_ranges, label_and_ticker)
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) True + Predicted Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            # Actual values
            indiv_fig_ax.plot(dates[window_size:], y[:, i],
                              color=actual_values_color, label=actual_values_label)
            # ML model predictions
            indiv_fig_ax.plot(extrapolation_dates, extrapolation_y[:, i],
                              color=ml_model_color, label=ml_model_label)
            # Monte Carlo predictions
            indiv_fig_ax.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                              color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(indiv_fig_ax, extrapolation_dates,
                                             monte_carlo_predicted_values_ranges, label_and_ticker)
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) True + Predicted Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
            plt.savefig('{}/{}_actual_plus_predictions_{}.png'.format(currency_figures_subdir,
                                                                      currency_label_no_spaces, window_size),
                        dpi=figure_dpi)
        collective_fig.savefig('{}/actual_plus_predictions_{}.png'
                               .format(figures_dir, window_size), dpi=figure_dpi)
        plt.close('all')

        if not load_models:
            # Save the model for this window size.
            with open(model_pickle_path, "wb") as model_outfile:
                pickle.dump(model, model_outfile)


if __name__ == '__main__':
    main()
