import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import warnings

from cryptocurrencies.python.ETL import load_csvs, load_sql

warnings.simplefilter('ignore')
import seaborn as sns

def main():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    figure_dpi = 200

    # Data Importing (SQL)
    bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
    monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves = load_sql()
    # Data Importing (CSV)
    # bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
    # monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves = load_csvs()

    currencies_labels_tickers = np.array(
        [['Bitcoin Cash', 'BCH'], ['Bitcoin', 'BTC'], ['BitConnect', 'BCC'], ['Dash', 'DASH'],
         ['Ethereum Classic', 'ETC'], ['Ethereum', 'ETH'], ['Iota', 'MIOTA'], ['Litecoin', 'LTC'],
         ['Monero', 'XMR'], ['Nem', 'XEM'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
         ['Qtum', 'QTUM'], ['Ripple', 'XRP'], ['Stratis', 'STRAT'], ['Waves', 'WAVES']])
    currencies_labels = currencies_labels_tickers[:, 0]
    currencies_tickers = currencies_labels_tickers[:, 1]
    currencies_labels_and_tickers = ["{} ({})".format(currencies_label, currencies_ticker)
                                 for currencies_label, currencies_ticker in zip(currencies_labels, currencies_tickers)]
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
    print("Heatmap showing absent values:")
    # Convert the timestamps to strings with less length for y-axis labels.
    old_prices_index = prices.index
    prices.index = prices.index.map(lambda date: date.strftime("%Y-%m-%d"))
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
    returns = subset_prices_nonan.pct_change()
    # Find the standard deviations in returns.
    returns_std_yearly = returns.groupby(subset_prices_nonan.index.year).std()
    # Standard deviations in returns for 2017 in descending order.
    returns_std_2017 = returns_std_yearly.loc[2017]
    returns_std_2017.sort_values(ascending=False, inplace=True)
    plt.subplots(figsize=(12, 6))
    sns.barplot(x='Name', y='Volatility', data=returns_std_2017.to_frame(name='Volatility').reset_index(),
                palette='viridis')
    plt.title('Volatility (2017)')
    plt.savefig('{}/volatility.png'.format(figures_dir), dpi=figure_dpi)

    # Data Extraction

    data = subset_prices_nonan.reset_index()
    dates = data['Date'].values

    # We will predict closing prices based on these numbers of days preceding the date of prediction.
    # The max `window_size` is `num_non_nan_days`, but some lower values may result in poor models due to small test sets
    # during cross validation, and possibly even training failures due to empty test sets for even larger values.
    window_sizes = list(range(30, 361, 30)) # Window sizes measured in days - approximately 1 to 12 months.
    for window_size in window_sizes:
        print("Predicting prices with a window of {} days of preceding currency values".format(window_size))
        num_windows = len(data) - window_size
        X = np.empty((num_windows, subset_num_currencies * window_size), dtype=np.float64)
        for i in range(num_windows):
            X[i] = subset_prices_nonan[i:i + window_size].values.flatten()
        y = data.drop('Date', axis=1).values[window_size:]

        # Model Training

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
                      'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                      'leaf_size': [5, 10, 20, 30, 40, 50, 60]}

        # Collect models for use in GridSearchCV.
        models = [model_extra_trees, model_random_forest, model_knn]
        param_sets = [params_extra_trees, params_random_forest, params_knn]

        # Tuples of scores and the corresponding models
        score_model_tuples = []

        # Specify the cross validation method.
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Scoring

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

        # Model Loading

        import pickle
        model_pickle_path = "{}/model_{}.pkl".format(models_dir, window_size)
        with open(model_pickle_path, "rb") as model_infile:
            model = pickle.load(model_infile)

        # Validation and Visualization

        # Plot the actual values along with the predictions. They should overlap.
        pred = model.predict(X)

        subset_num_cols = 2
        subset_num_rows = int(np.ceil(subset_num_currencies / subset_num_cols))
        collective_fig = plt.figure(figsize=(12 * subset_num_cols, 6 * subset_num_rows))
        for i in range(subset_num_currencies):
            currency_label = subset_currencies_labels[i]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[i]
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, i+1)
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
        last_data_date = data['Date'].iloc[-1]
        first_extrapolation_date = last_data_date + pd.DateOffset(days=1)
        last_extrapolation_date = '2018-12-31'
        extrapolation_dates = np.array(pd.date_range(first_extrapolation_date, last_extrapolation_date))
        num_extrapolation_dates = len(extrapolation_dates)
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

        # Plot the predictions for the rest of 2017 and 2018.
        for i in range(subset_num_currencies):
            currency_label = subset_currencies_labels[i]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[i]
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, i + 1)
            collective_ax_current.plot(extrapolation_dates, extrapolation_y[:, i], color='red')
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) Predicted Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            indiv_fig_ax.plot(extrapolation_dates, extrapolation_y[:, i], color='red')
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) Predicted Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
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
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, i + 1)
            collective_ax_current.plot(dates[window_size:], y[:, i], color='blue', label='True')
            collective_ax_current.plot(extrapolation_dates, extrapolation_y[:, i], color='red', label='Predicted')
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) True + Predicted Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            indiv_fig_ax.plot(dates[window_size:], y[:, i], color='blue', label='True')
            indiv_fig_ax.plot(extrapolation_dates, extrapolation_y[:, i], color='red', label='Predicted')
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) True + Predicted Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
            plt.savefig('{}/{}_actual_plus_predictions_{}.png'.format(currency_figures_subdir,
                                                                      currency_label_no_spaces, window_size), dpi=figure_dpi)
        collective_fig.savefig('{}/actual_plus_predictions_{}.png'
                               .format(figures_dir, window_size), dpi=figure_dpi)
        plt.close('all')

        # Save the model for this window size.
        with open(model_pickle_path, "wb") as model_outfile:
            pickle.dump(model, model_outfile)


if __name__ == '__main__':
    main()