"""
The time resolution of the data for the models to train on. Either 'daily' or 'hourly'.
See `ETL.py` for more information.
"""
def main():
    ### Imports ###

    # Import miscellaneous global variables.
    import sys
    sys.path.insert(0, '../../globals/Python')
    from globals import utilities_dir, num_lgc_prcs

    import os
    import warnings
    import time

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    warnings.simplefilter('ignore')
    import seaborn as sns
    from plotting import KerasPlotLosses

    from ETL import load_data

    # Model Training
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, r2_score
    import keras
    import keras.models as ks_models
    from machine_learning import create_keras_regressor, keras_reg_grid_search

    # Custom utility functions
    sys.path.insert(0, '../../' + utilities_dir)
    from analysis import find_optimal_portfolio_weights, calc_CAPM_betas, CAPM_RoR, run_monte_carlo_financial_simulation
    from conversions import pandas_dt_to_str
    from plotting import add_value_text_to_seaborn_barplot, monte_carlo_plot_confidence_band

    # Optimizers
    from keras.optimizers import Adam

    ### End Imports ###

    ### Main Settings ###

    # Data Importing
    # Whether to load from CSV files or a local MySQL database ('csv' or 'sql').
    load_source = 'csv'

    # Machine Learning Settings
    # The resolution of data for the analysis.
    analysis_data_resolution = 'daily'
    # The resolution of data for the machine learning models.
    model_data_resolution = 'hourly'
    # The number of logical processors to use during grid search.
    num_lgc_prcs_grd_srch = min(3, num_lgc_prcs)
    # Whether or not to use cross validation in Keras grid search.
    keras_use_cv_grd_srch = False

    # Figure Variables and Paths
    figure_size = (12, 6) # The base figure size.
    figure_dpi = 300
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Models Variables and Paths
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    ### End Main Settings ###

    # Data Importing
    # bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
    # monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves = load_daily_data(source=load_source)
    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        load_data(resolution=analysis_data_resolution, date_range=('2013-04-28', '2017-12-31'), source=load_source)

    # Directory Structure Creation (Figures)
    currency_figures_subdirs = ['{}/{}'.format(figures_dir, currency_label.replace(' ', '_'))
                                for currency_label in currencies_labels]
    for currency_figures_subdir in currency_figures_subdirs:
        if not os.path.exists(currency_figures_subdir):
            os.makedirs(currency_figures_subdir)
    keras_figures_subdirs = ['{}/{}'.format(currency_figures_subdir, 'keras')
                                for currency_figures_subdir in currency_figures_subdirs]
    for keras_figures_subdir in keras_figures_subdirs:
        if not os.path.exists(keras_figures_subdir):
            os.makedirs(keras_figures_subdir)

    # Value Trends

    num_cols = 2
    num_rows = int(np.ceil(num_currencies / num_cols))
    for currency_index, column in enumerate(prices.columns):
        currency_label = currencies_labels[currency_index]
        currency_label_no_spaces = currency_label.replace(' ', '_')
        currency_ticker = currencies_tickers[currency_index]
        plt.figure(figsize=figure_size, dpi=figure_dpi)
        plt.plot(prices[column])
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.title("{} ({}) Closing Value"
                  .format(currency_label, currency_ticker))
        currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
        plt.savefig('{}/{}_price_trend'.format(currency_figures_subdir, currency_label_no_spaces))

    # Find the returns.
    returns = prices.pct_change()

    # Value Correlations
    plt.figure(figsize=figure_size, dpi=figure_dpi)
    # Price Correlations
    price_correlations = prices.corr()
    price_correlations_fig = sns.clustermap(price_correlations, annot=True)
    price_correlations_fig.savefig('{}/price_correlations.png'.format(figures_dir))
    plt.clf()
    # Return Correlations
    returns_correlations = returns.corr()
    returns_correlations_fig = sns.clustermap(returns_correlations, annot=True)
    returns_correlations_fig.savefig('{}/return_correlations.png'.format(figures_dir))
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
    print("Considering {} days of price information (as many as without NaN values).".format(num_non_nan_days))

    # Find the returns.
    subset_returns = subset_prices_nonan.pct_change()
    log_returns = np.log(prices / prices.shift(1))
    subset_log_returns = log_returns.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)

    # Find the standard deviations in returns.
    returns_std_yearly = subset_returns.groupby(subset_prices_nonan.index.year).std()
    # Standard deviations in returns for 2017 in descending order.
    returns_std_2017 = returns_std_yearly.loc[2017]
    returns_std_2017.sort_values(ascending=False, inplace=True)
    plt.subplots(figsize=figure_size)
    plotting_data = returns_std_2017.to_frame(name='Volatility').reset_index()
    volatility_plot = sns.barplot(x='Name', y='Volatility', data=plotting_data, palette='viridis')
    add_value_text_to_seaborn_barplot(volatility_plot, plotting_data, 'Volatility')
    plt.title('Volatility (2017)')
    plt.savefig('{}/volatility.png'.format(figures_dir), dpi=figure_dpi)

    # Find (daily) asset betas (covariance_with_market / market_variance).
    market_index = 'Bitcoin (BTC)'
    betas = calc_CAPM_betas(log_returns, market_index)
    betas.sort_values(ascending=False, inplace=True)
    subset_betas = betas.drop(labels=currencies_labels_and_tickers_to_remove)
    # Create a visualization with the beta values.
    fig, ax = plt.subplots(figsize=figure_size)
    plotting_data = subset_betas.to_frame(name='Beta').reset_index()
    beta_plot = sns.barplot(ax=ax, x='Name', y='Beta', data=plotting_data, palette='viridis')
    # Show values in the figure.
    add_value_text_to_seaborn_barplot(beta_plot, plotting_data, 'Beta')
    plt.title('Betas (Bitcoin (BTC) as Market Index)')
    plt.savefig('{}/betas.png'.format(figures_dir), dpi=figure_dpi)

    # Find assets' rates of return according to CAPM.
    # NOTE: The rate of return of a risk free asset is often taken to be about 2.5% (e.g. 10 year US government bond),
    # but for the time period being analyzed (2015-2017), the rate of return of the market index (taken to be Bitcoin)
    # is less than 2.5%, so the risk premium becomes negative (there would be no reason to invest in the market
    # according to CAPM).
    return_risk_free = 0
    CAPM_expected_rates_of_return = CAPM_RoR(subset_betas, returns, market_index, return_risk_free)
    CAPM_expected_rates_of_return.sort_values(ascending=False, inplace=True)
    # Create a visualization with the weights.
    fig, ax = plt.subplots(figsize=figure_size)
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
    fig, ax = plt.subplots(figsize=figure_size)
    plotting_data = optimal_weights.to_frame(name='Weight').reset_index()
    portfolio_weights_plot = sns.barplot(ax=ax, x='Name', y='Weight', data=plotting_data, palette='viridis')
    # Show values in the figure.
    add_value_text_to_seaborn_barplot(portfolio_weights_plot, plotting_data, 'Weight')
    plt.title('Markowitz Optimal Portfolio Weights')
    plt.savefig('{}/optimal_portfolio_weights.png'.format(figures_dir), dpi=figure_dpi)

    # Run Monte Carlo simulation to predict future values.
    last_data_date = log_returns.index[-1]
    first_extrapolation_date = last_data_date + pd.DateOffset(days=1)
    last_extrapolation_date = pd.to_datetime('2018-12-31')
    extrapolation_dates = np.array(pd.date_range(first_extrapolation_date, last_extrapolation_date))
    num_extrapolation_dates = len(extrapolation_dates)
    MC_predicted_values_ranges, MC_predicted_values = run_monte_carlo_financial_simulation(prices, extrapolation_dates)
    subset_monte_carlo_predicted_values_ranges = \
        MC_predicted_values_ranges.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_monte_carlo_predicted_values = \
        MC_predicted_values.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)

    dates = subset_prices_nonan.index.values

    # Directory Structure Creation (Machine Learning Models)
    currency_models_subdirs = ['{}/{}'.format(models_dir, currency_label.replace(' ', '_'))
                               for currency_label in subset_currencies_labels]
    for currency_models_subdir in currency_models_subdirs:
        if not os.path.exists(currency_models_subdir):
            os.makedirs(currency_models_subdir)

    # We will predict closing prices based on these numbers of days preceding the date of prediction.
    # The max `window_size` is `num_non_nan_days`, but even reasonably close values may result in poor models or even
    # training failures (errors) due to small or even empty test sets, respectively, during cross validation.
    window_sizes = None
    if model_data_resolution == 'daily':
        window_sizes = [7, 14] + list(range(30, 361, 30))  # Window sizes in days - 1 week, 2 weeks, and 1 to 12 months.
    else:
        window_sizes = 24*7*np.array([1, 2, 4, 8])  # Window sizes in hours - 1 week, 2 weeks, 1 month, and 2 months.
        # With just 14 days and 7 cryptocurrencies, we have 14 * 24 * 7 = 2352 features.

    # Directory Structure Creation (Keras Optimization Trend Figures)
    for keras_figures_subdir in keras_figures_subdirs:
        for window_size in window_sizes:
            keras_figures_window_subdir = os.path.join(keras_figures_subdir, 'w_{}'.format(window_size))
            if not os.path.exists(keras_figures_window_subdir):
                os.makedirs(keras_figures_window_subdir)

    # Load the hourly data for model training if such has been specified.
    if model_data_resolution=='hourly':
        num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
            load_data(resolution=model_data_resolution, date_range=('2017-07-01', '2017-12-31'), source=load_source)
        currencies_labels_and_tickers_to_remove = \
            [label_and_ticker_to_remove for label_and_ticker_to_remove in currencies_labels_and_tickers_to_remove
             if label_and_ticker_to_remove in prices.columns]
        subset_prices = prices.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
        subset_currencies_labels = [currency_label for currency_label in currencies_labels
                                    if currency_label not in currencies_labels_to_remove]
        subset_currencies_tickers = [currency_ticker for currency_ticker in currencies_tickers
                                     if currency_ticker not in currencies_tickers_to_remove]
        subset_currencies_labels_and_tickers = subset_prices.columns.values
        subset_num_currencies = len(subset_prices.columns)
        subset_prices_nonan = subset_prices.dropna()
        dates = subset_prices_nonan.index.values

    ### Main Loop ###
    for window_size in window_sizes:
        print("Predicting prices with a window of {} days of preceding currency values.".format(window_size))

        # Data Extraction
        num_windows = len(subset_prices_nonan) - window_size
        num_features = subset_num_currencies * window_size
        X = np.empty((num_windows, num_features), dtype=np.float64)
        for window_index in range(num_windows):
            X[window_index,:] = subset_prices_nonan[window_index:window_index + window_size].values.flatten()
        y = subset_prices_nonan.values[window_size:]

        # Feature Scaling
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)
        # print("X[:5,0::subset_num_currencies*2], y[:5,0]: ",
        #       X[:5,0::subset_num_currencies*2], y[:5,0])
        # print("X_scaled[:5,0::subset_num_currencies*2], y_scaled[:5,0]: ",
        #       X_scaled[:5,0::subset_num_currencies*2], y_scaled[:5,0])

        ### Model Specifications ##

        # Neural Network models
        # Multilayer Perceptron regressor
        # model_neural_net = MLPRegressor()
        # The number of neurons in the hidden layers will be around half the mean of the number of inputs and outputs.
        # hidden_layer_median_size = int(round((num_features + subset_num_currencies)/2))
        single_hidden_layer_size = 128
        neural_net_hidden_layer_sizes = [(single_hidden_layer_size, single_hidden_layer_size // 8)]
        # params_neural_net = {'hidden_layer_sizes': neural_net_hidden_layer_sizes,
        #                      'max_iter': [1000000],
        #                      'beta_1': [0.6, 0.7, 0.8],
        #                      'beta_2': [0.9, 0.95, 0.999],
        #                      'alpha': [1e-10, 1e-8]}
        from sklearn.pipeline import Pipeline
        model_neural_net = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor())
        ])
        params_neural_net = {'model__hidden_layer_sizes': neural_net_hidden_layer_sizes,
                             'model__max_iter': [5000],
                             'model__beta_1': [0.7, 0.8, 0.9],
                             'model__beta_2': [0.9, 0.95, 0.999],
                             'model__alpha': [1e-10, 1e-8, 1e-4]}

        from keras.wrappers.scikit_learn import KerasRegressor
        # regressor = KerasRegressor(build_fn=create_keras_model, epochs=100, batch_size=10) # TODO: Set verbosity?
        # regressor.fit(X,y)
        # pred = regressor.predict(y)
        # plt.plot(X[:,0::window_size], y[:,0])
        # plt.plot(X[:, 0::window_size], pred[:, 0])
        # plt.show()
        # exit()
        # model_neural_net = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('model', KerasRegressor(build_fn=create_keras_model))
        # ])
        keras_model_neural_net = "Keras_NN"
        # params_neural_net = {'model__input_dim': [subset_num_currencies],
        #                      'model__hidden_layer_sizes': neural_net_hidden_layer_sizes,
        #                      'model__output_dim': [subset_num_currencies],
        #                      'model__epochs': [1000],
        #                      'model__batch_size': [100]}
        first_hidden_layer_size = 512
        # three_hidden_layers = (first_hidden_layer_size, first_hidden_layer_size // 2, first_hidden_layer_size // 4)
        # four_hidden_layers = three_hidden_layers + (first_hidden_layer_size // 8,)
        # five_hidden_layers = four_hidden_layers + (first_hidden_layer_size // 16,)
        # six_hidden_layers = five_hidden_layers + (first_hidden_layer_size // 32,)
        # hidden_layer_sizes = [three_hidden_layers, four_hidden_layers, five_hidden_layers, six_hidden_layers]

        # hidden_layer_sizes = [(first_hidden_layer_size, first_hidden_layer_size // 2, first_hidden_layer_size // 4,
        #                        first_hidden_layer_size // 8)]
        # hidden_layer_sizes = [(first_hidden_layer_size, first_hidden_layer_size // 4, first_hidden_layer_size // 16,
        #                        first_hidden_layer_size // 64)]
        hidden_layer_sizes = [(first_hidden_layer_size, first_hidden_layer_size // 4, first_hidden_layer_size // 16)]
        # hidden_layer_sizes = [(first_hidden_layer_size, first_hidden_layer_size // 8)]
        keras_params_neural_net = \
            {'batch_size': [4, 8, 12],  # A too large batch size results in device OOMs.
             'hidden_layer_sizes': hidden_layer_sizes,
             'dropout_rate': [0.1, 0.2],
             'optimizer': [Adam],
             # Parameters for Adam optimizer.
             'lr': [1e-8, 1e-6, 1e-4],
             'beta_1': [0.7, 0.9],
             'beta_2': [0.9, 0.999]}
        # The number of epochs to train for in cross validation.
        keras_neural_net_epochs_grd_srch = 10 # TODO: Make this larger. 100?
        # The number of epochs to train for after cross validation.
        keras_neural_net_epochs = 10 # TODO: Make this larger. 100?
        # make_keras_picklable()
        # print(type(KerasRegressor(build_fn=create_keras_model)))
        # from model_saving_loading import save_keras_pipeline, load_keras_pipeline
        # model_filepath = 'keras_model'
        # keras_step_name = 'model'
        # pipeline_filepath = 'keras_pipeline'
        # save_keras_pipeline(model_neural_net, keras_step_name, model_filepath, pipeline_filepath)
        # loaded_pipeline = load_keras_pipeline(keras_step_name, model_filepath, pipeline_filepath)
        # print(loaded_pipeline)


        # Linear Models
        # TODO: Ridge (sklearn.linear_model.Ridge) regressor (if variance among price correlations
        # TODO: for individual assets tend to be high)

        # SVM models
        # TODO: Support Vector regressor
        # model_svr = SVR()
        # model_svr = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('model', SVR())
        # ])
        # params_svr = {'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        #               'model__gamma': [0.01, 0.1, 1/subset_num_currencies, 1.0, 10.0],
        #               'model__epsilon': [0.1], # Higher epsilon reduces variance
        #               'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}

        # Ensemble models
        # Extra-Trees regressor
        # model_extra_trees = ExtraTreesRegressor()
        # model_extra_trees = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('model', ExtraTreesRegressor())
        # ])
        # params_extra_trees = {'model__n_estimators': [500],
        #                       'model__min_samples_split': [2, 5, 10],
        #                       'model__max_features': ['auto', 'sqrt', 'log2']}
        # Random forest regressor
        # model_random_forest = RandomForestRegressor()
        # model_random_forest = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('model', RandomForestRegressor())
        # ])
        # params_random_forest = {'model__n_estimators': [500],
        #                         'model__min_samples_split': [2, 5, 10],
        #                         'model__max_features': ['auto', 'sqrt', 'log2']}

        # Neighbors models
        # TODO: If flat lining is less of an issue with KNN models removed, clean up these lines.
        # KNearestNeighbors regressor
        # model_knn = KNeighborsRegressor()
        # model_knn = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('model', KNeighborsRegressor())
        # ])
        # params_knn = {'model__n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #               'model__weights': ['uniform', 'distance'],
        #               'model__algorithm': ['auto'],
        #               'model__leaf_size': [5, 10, 20, 30, 40, 50, 60]}

        # Collect models for use in GridSearchCV.
        # models_to_test = [model_extra_trees, model_random_forest] # ensemble_trees
        # param_sets = [params_extra_trees, params_random_forest] # ensemble_trees
        # models_to_test = [model_extra_trees, model_random_forest, model_knn] # knn_allowed
        # param_sets = [params_extra_trees, params_random_forest, params_knn] # knn_allowed
        # models_to_test = [model_neural_net]  # neural_network
        # param_grids = [params_neural_net]  # neural_network
        models_to_test = [keras_model_neural_net]  # Keras neural_network
        param_grids = [keras_params_neural_net]  # Keras neural_network
        # models_to_test = [model_svr]  # SVR
        # param_sets = [params_svr]  # SVR

        load_models = False  # Whether or not to load models that have already been created by this program.
        # Train models on different amounts of division of the data into training and test sets.
        # Specify the cross validation method.
        cv = TimeSeriesSplit(n_splits=5)
        keras_cv = cv if keras_use_cv_grd_srch else None

        ### Model Training ###

        time_unit = 'days' if model_data_resolution=='daily' else 'hours' # Used for printing the training status.

        # Single Asset Model (consider only windows of values for one asset each (disallows learning of correlations))
        single_asset_models = []
        # Build a model for each cryptocurrency.
        for currency_index in range(subset_num_currencies):
            currency_label = subset_currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            # TODO: Replace path string formatting with `os.path.join()`.
            currency_models_subdir = '{}/{}'.format(models_dir, currency_label_no_spaces)
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            keras_figures_subdir = '{}/{}'.format(currency_figures_subdir, 'keras')
            keras_figures_window_subdir = os.path.join(keras_figures_subdir, 'w_{}'.format(window_size))

            # Paths
            model_base_path = "{}/{}_model_{}".format(currency_models_subdir, currency_label_no_spaces, window_size)
            model_pickle_path = "{}.pkl".format(model_base_path)

            # Data for only this cryptocurrency.
            X_subset = X[:, currency_index::subset_num_currencies]
            y_subset = y[:, currency_index].reshape(-1, 1)
            # print(y_subset.shape, y[:, currency_index].shape)
            # print("X_subset[:5], y_subset[:5]: ", X_subset[:5], y_subset[:5])

            # Feature Scaling
            X_subset_scaler = StandardScaler()
            X_subset_scaled = X_subset_scaler.fit_transform(X_subset)
            y_subset_scaler = StandardScaler()
            y_subset_scaled = y_subset_scaler.fit_transform(y_subset)

            if not load_models:
                currency_label_and_ticker = subset_currencies_labels_and_tickers[currency_index]
                print("Currently training a model for {} with a window size of {} {}."
                    .format(currency_label_and_ticker, window_size, time_unit))

                # Tuples of scores, the corresponding models, and the best batch sizes for Keras models.
                score_model_batch_size_tuples = []

                # Create the models.
                for i in range(len(models_to_test)):
                    model_to_test = models_to_test[i]
                    param_grid = param_grids[i]
                    best_batch_size = None
                    # print("model_to_test, param_set: ", model_to_test, param_set)
                    model = None
                    # In this case, Keras needs to be trained differently to save its models to the filesystem.
                    if model_to_test == keras_model_neural_net:
                        model, score, best_batch_size = \
                            keras_reg_grid_search(X_subset, y_subset, build_fn=create_keras_regressor, output_dim=1,
                                                  param_grid=param_grid, epochs=keras_neural_net_epochs_grd_srch,
                                                  cv=keras_cv, scoring=r2_score, scale=True, verbose=1,
                                                  plot_losses=True, plotting_dir=keras_figures_window_subdir,
                                                  figure_title_prefix='{} (window size {})'.format(currency_label, window_size),
                                                  figure_kwargs={'figsize':figure_size, 'dpi':figure_dpi})
                    else:
                        grid_search = GridSearchCV(model_to_test, param_grid, scoring=make_scorer(r2_score),
                                                   cv=cv, n_jobs=num_lgc_prcs_grd_srch)
                        grid_search.fit(X_subset_scaled, y_subset_scaled.ravel())
                        model = grid_search.best_estimator_
                        score = grid_search.best_score_
                    score_model_batch_size_tuples.append((score, model, best_batch_size))
                time.sleep(1)  # Wait 1 second for printing from GridSearchCV to complete.
                # Choose the model with the best score.
                score_model_batch_size_tuples.sort(key=lambda tup: tup[0], reverse=True)
                best_score = score_model_batch_size_tuples[0][0]
                model = score_model_batch_size_tuples[0][1]
                best_batch_size = score_model_batch_size_tuples[0][2]
                print("Best model and score for asset {} and window size {}: {}"
                      .format(currency_label_and_ticker, window_size, (model, best_score)))
                # Save the model for this window size after fitting to the whole dataset.
                if type(model) is keras.models.Sequential:
                    model.fit(X_subset_scaled, y_subset_scaled, epochs=keras_neural_net_epochs,
                              batch_size=best_batch_size, verbose=0)
                    ks_models.save_model(model, model_base_path + '.h5')
                else: # Model is refit by GridSearchCV.
                    with open(model_pickle_path, "wb") as model_outfile:
                        pickle.dump(model, model_outfile)
            else:
                # Model Loading
                try: # Try to load the neural network model first (runs on GPUs).
                    model = ks_models.load_model(model_base_path + '.h5')
                except OSError:
                    with open(model_pickle_path, "rb") as model_infile:
                        model = pickle.load(model_infile)
            single_asset_models.append(model)

        # Collective Model (consider windows of values for all assets)
        collective_assets_model = None
        model_base_path = "{}/model_{}".format(models_dir, window_size)
        model_pickle_path = "{}.pkl".format(model_base_path)
        keras_figures_subdir = '{}/{}'.format(figures_dir, 'keras')

        if not load_models:
            print("Currently training a model for all assets collectively "
                  "with a window size of {} {}.".format(window_size, time_unit))

            # Tuples of scores and the corresponding models (along with the best batch sizes for Keras models)
            score_model_batch_size_tuples = []

            # Create the models.
            for i in range(len(models_to_test)):
                model_to_test = models_to_test[i]
                param_grid = param_grids[i]
                best_batch_size = None
                # print("model_to_test, param_set: ", model_to_test, param_set)
                model = None
                # In this case, Keras needs to be trained differently to save its models to the filesystem.
                if model_to_test == keras_model_neural_net:
                    model, score, best_batch_size = \
                        keras_reg_grid_search(X, y, build_fn=create_keras_regressor, output_dim=subset_num_currencies,
                                              param_grid=params_neural_net, epochs=keras_neural_net_epochs_grd_srch,
                                              cv=keras_cv, scoring=r2_score, scale=True, verbose=1,
                                              plot_losses=True, plotting_dir=keras_figures_subdir,
                                              figure_title_prefix='Collective (window size {})'.format(window_size),
                                              figure_kwargs={'figsize': figure_size, 'dpi': figure_dpi})
                else:
                    grid_search = GridSearchCV(model_to_test, param_grid, scoring=make_scorer(r2_score),
                                               cv=cv, n_jobs=num_lgc_prcs_grd_srch)#max(1, round(num_cores/2)))
                    grid_search.fit(X_scaled, y_scaled.ravel())
                    model = grid_search.best_estimator_
                    score = grid_search.best_score_
                score_model_batch_size_tuples.append((score, model, best_batch_size, i))
            time.sleep(1)  # Wait 1 second for printing from GridSearchCV to complete.
            # Choose the model with the best score.
            score_model_batch_size_tuples.sort(key=lambda tup: tup[0], reverse=True)
            best_score = score_model_batch_size_tuples[0][0]
            collective_assets_model = score_model_batch_size_tuples[0][1]
            best_batch_size = score_model_batch_size_tuples[0][2]
            best_param_grid = param_grids[score_model_batch_size_tuples[0][3]]
            print("Best collective model and score for window size {}: {}"
                  .format(window_size, (collective_assets_model, best_score)))
            # Save the model for this window size after fitting to the whole dataset.
            if type(collective_assets_model) is keras.models.Sequential:
                collective_assets_model.fit(X_scaled, y_scaled, epochs=keras_neural_net_epochs,
                                            batch_size=best_batch_size, verbose=0)
                ks_models.save_model(collective_assets_model, model_base_path + '.h5')
            else: # Model is refit by GridSearchCV.
                with open(model_pickle_path, "wb") as model_outfile:
                    pickle.dump(collective_assets_model, model_outfile)
        else:
            # Model Loading
            try: # Try to load the neural network model first (runs on GPUs).
                collective_assets_model = ks_models.load_model(model_base_path + '.h5')
            except OSError:
                with open(model_pickle_path, "rb") as model_infile:
                    collective_assets_model = pickle.load(model_infile)

        # Validation and Visualization

        # Single Asset Model Predictions
        def single_asset_models_predict(X):
            """
            Returns predictions for the separate single asset models in the format of
            predictions from the collective assets model.

            Parameters
            ----------
            X: numpy.ndarray
                The feature vectors to predict for.
            """
            single_asset_models_pred = np.empty((len(X), subset_num_currencies), dtype=np.float64)
            for currency_index in range(subset_num_currencies):
                single_asset_models_pred[:, currency_index] = \
                    single_asset_models[currency_index].predict(X[:, currency_index::subset_num_currencies]).flatten()
            return single_asset_models_pred

        single_asset_models_pred = y_scaler.inverse_transform(single_asset_models_predict(X_scaled))

        # Collective Model Predictions
        collective_assets_model_pred = y_scaler.inverse_transform(collective_assets_model.predict(X_scaled))

        ### Plotting ###
        # The colors of lines for various things.
        actual_values_color = 'blue'
        actual_values_label = 'True'
        ml_model_single_asset_color = 'red'
        ml_model_single_asset_label = 'ML Model (Single Asset Predictors)'
        ml_model_collective_color = 'orange'
        ml_model_collective_label = 'ML Model (Collective Predictor)'
        monte_carlo_color = 'green'
        monte_carlo_label = 'Monte Carlo AVG'

        # Plot the actual values along with the predictions. They should overlap.
        subset_num_cols = 2
        subset_num_rows = int(np.ceil(subset_num_currencies / subset_num_cols))
        collective_fig = plt.figure(figsize=(12 * subset_num_cols, 6 * subset_num_rows))
        for currency_index in range(subset_num_currencies):
            currency_label = subset_currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[currency_index]
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
            # Actual values
            collective_ax_current.plot(dates[window_size:], y[:, currency_index],
                                       color=actual_values_color, alpha=0.5, label=actual_values_label)
            # Single Asset Model Predictions
            collective_ax_current.plot(dates[window_size:], single_asset_models_pred[:, currency_index],
                                       color=ml_model_single_asset_color, alpha=0.5, label=ml_model_single_asset_label)
            # Collective Model Predictions
            collective_ax_current.plot(dates[window_size:], collective_assets_model_pred[:, currency_index],
                                       color=ml_model_collective_color, alpha=0.5, label=ml_model_collective_label)
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            # Actual values
            indiv_fig_ax.plot(dates[window_size:], y[:, currency_index],
                              color=actual_values_color, alpha=0.5, label=actual_values_label)
            # Single Asset Model Predictions
            indiv_fig_ax.plot(dates[window_size:], single_asset_models_pred[:, currency_index],
                              color=ml_model_single_asset_color, alpha=0.5, label=ml_model_single_asset_label)
            # Collective Model Predictions
            indiv_fig_ax.plot(dates[window_size:], collective_assets_model_pred[:, currency_index],
                              color=ml_model_collective_color, alpha=0.5, label=ml_model_collective_label)
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            indiv_fig.savefig('{}/{}_validation_{}.png'
                              .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                              dpi=figure_dpi)
        collective_fig.savefig('{}/validation_{}.png'
                               .format(figures_dir, window_size), dpi=figure_dpi)
        collective_fig.clf()

        # Get the models' predictions for the rest of 2017 and 2018.
        # Single Asset Model Predictions
        single_asset_models_extrapolation_X = \
            np.zeros((num_extrapolation_dates, subset_num_currencies * window_size), dtype=np.float64)
        single_asset_models_extrapolation_y = \
            np.zeros((num_extrapolation_dates, subset_num_currencies), dtype=np.float64)
        # Collective Model Predictions
        collective_assets_model_extrapolation_X = \
            np.zeros((num_extrapolation_dates, subset_num_currencies * window_size), dtype=np.float64)
        collective_assets_model_extrapolation_y = \
            np.zeros((num_extrapolation_dates, subset_num_currencies), dtype=np.float64)

        # First `window_size` windows contain known values.
        given_prices = subset_prices_nonan.values[-window_size:].flatten()
        # Single Asset Model Predictions
        single_asset_models_extrapolation_X[0] = given_prices
        single_asset_models_extrapolation_y[0] = \
            single_asset_models_predict(single_asset_models_extrapolation_X[0].reshape(1, -1))
        # Collective Model Predictions
        collective_assets_model_extrapolation_X[0] = given_prices
        collective_assets_model_extrapolation_y[0] = \
            collective_assets_model.predict(collective_assets_model_extrapolation_X[0].reshape(1, -1))
        for currency_index in range(1, window_size):
            given_prices = subset_prices_nonan.values[-window_size + currency_index:].flatten()
            # Single Asset Model Predictions
            single_asset_models_previous_predicted_prices = \
                single_asset_models_extrapolation_y[:currency_index].flatten()
            single_asset_models_extrapolation_X[currency_index] = \
                np.concatenate((given_prices, single_asset_models_previous_predicted_prices))
            single_asset_models_extrapolation_y[currency_index] = \
                single_asset_models_predict(single_asset_models_extrapolation_X[currency_index].reshape(1, -1)).flatten()
            # Collective Model Predictions
            collective_assets_model_previous_predicted_prices = \
                collective_assets_model_extrapolation_y[:currency_index].flatten()
            collective_assets_model_extrapolation_X[currency_index] = \
                np.concatenate((given_prices, collective_assets_model_previous_predicted_prices))
            collective_assets_model_extrapolation_y[currency_index] = \
                collective_assets_model.predict(collective_assets_model_extrapolation_X[currency_index].reshape(1, -1)).flatten()
        # Remaining windows contain only predicted values (predicting based on previous predictions).
        for currency_index in range(window_size, num_extrapolation_dates):
            # Single Asset Model Predictions
            single_asset_models_previous_predicted_prices = \
                single_asset_models_extrapolation_y[currency_index - window_size:currency_index].flatten()
            single_asset_models_extrapolation_X[currency_index] = \
                single_asset_models_previous_predicted_prices
            single_asset_models_extrapolation_y[currency_index] = \
                single_asset_models_predict(single_asset_models_extrapolation_X[currency_index].reshape(1, -1)).flatten()
            # Collective Model Predictions
            collective_assets_model_previous_predicted_prices = \
                collective_assets_model_extrapolation_y[currency_index - window_size:currency_index].flatten()
            collective_assets_model_extrapolation_X[currency_index] = \
                collective_assets_model_previous_predicted_prices
            collective_assets_model_extrapolation_y[currency_index] = \
                collective_assets_model.predict(collective_assets_model_extrapolation_X[currency_index].reshape(1, -1)).flatten()

        # Plot predictions for the rest of 2017 and 2018.
        for currency_index in range(subset_num_currencies):
            currency_label = subset_currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[currency_index]
            label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
            # Single Asset Model Predictions
            collective_ax_current.plot(extrapolation_dates, single_asset_models_extrapolation_y[:, currency_index],
                                       color=ml_model_single_asset_color, label=ml_model_single_asset_label)
            # Collective Model Predictions
            collective_ax_current.plot(extrapolation_dates, collective_assets_model_extrapolation_y[:, currency_index],
                                       color=ml_model_collective_color, label=ml_model_collective_label)
            # Monte Carlo predictions
            collective_ax_current.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                                       color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(collective_ax_current, extrapolation_dates,
                                             MC_predicted_values_ranges, label_and_ticker, color='cyan')
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) Predicted Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            # Single Asset Model Predictions
            indiv_fig_ax.plot(extrapolation_dates, single_asset_models_extrapolation_y[:, currency_index],
                              color=ml_model_single_asset_color, label=ml_model_single_asset_label)
            # Collective Model Predictions
            indiv_fig_ax.plot(extrapolation_dates, collective_assets_model_extrapolation_y[:, currency_index],
                              color=ml_model_collective_color, label=ml_model_collective_label)
            # Monte Carlo predictions
            indiv_fig_ax.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                              color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(indiv_fig_ax, extrapolation_dates,
                                             MC_predicted_values_ranges, label_and_ticker, color='cyan')
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) Predicted Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            indiv_fig.savefig('{}/{}_predictions_{}.png'
                              .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                              dpi=figure_dpi)
        collective_fig.savefig('{}/predictions_{}.png'
                               .format(figures_dir, window_size), dpi=figure_dpi)
        collective_fig.clf()

        # Plot the predicitons for the rest of 2017 and 2018 along with the actual values for the date range used.
        for currency_index in range(subset_num_currencies):
            currency_label = subset_currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = subset_currencies_tickers[currency_index]
            label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
            # Collective plot
            collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
            # Actual values
            collective_ax_current.plot(dates[window_size:], y[:, currency_index],
                                       color=actual_values_color, label=actual_values_label)
            # Single Asset Model Predictions
            collective_ax_current.plot(extrapolation_dates, single_asset_models_extrapolation_y[:, currency_index],
                                       color=ml_model_single_asset_color, label=ml_model_single_asset_label)
            # Collective Model Predictions
            collective_ax_current.plot(extrapolation_dates, collective_assets_model_extrapolation_y[:, currency_index],
                                       color=ml_model_collective_color, label=ml_model_collective_label)
            # Monte Carlo predictions
            collective_ax_current.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                                       color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(collective_ax_current, extrapolation_dates,
                                             MC_predicted_values_ranges, label_and_ticker, color='cyan')
            collective_ax_current.set_xlabel('Date')
            collective_ax_current.set_ylabel('Close')
            collective_ax_current.set_title("{} ({}) True + Predicted Closing Value ({} day window)"
                                            .format(currency_label, currency_ticker, window_size))
            collective_ax_current.legend()
            # Individual plot
            indiv_fig = plt.figure(figsize=(12, 6))
            indiv_fig_ax = indiv_fig.add_subplot(111)
            # Actual values
            indiv_fig_ax.plot(dates[window_size:], y[:, currency_index],
                              color=actual_values_color, label=actual_values_label)
            # Single Asset Model Predictions
            indiv_fig_ax.plot(extrapolation_dates, single_asset_models_extrapolation_y[:, currency_index],
                              color=ml_model_single_asset_color, label=ml_model_single_asset_label)
            # Collective Model Predictions
            indiv_fig_ax.plot(extrapolation_dates, collective_assets_model_extrapolation_y[:, currency_index],
                              color=ml_model_collective_color, label=ml_model_collective_label)
            # Monte Carlo predictions
            indiv_fig_ax.plot(extrapolation_dates, subset_monte_carlo_predicted_values[label_and_ticker],
                              color=monte_carlo_color, label=monte_carlo_label)
            # Monte Carlo predictions (95% confidence interval)
            monte_carlo_plot_confidence_band(indiv_fig_ax, extrapolation_dates,
                                             MC_predicted_values_ranges, label_and_ticker, color='cyan')
            indiv_fig_ax.set_xlabel('Date')
            indiv_fig_ax.set_ylabel('Close')
            indiv_fig_ax.set_title("{} ({}) True + Predicted Closing Value ({} day window)"
                                   .format(currency_label, currency_ticker, window_size))
            indiv_fig_ax.legend()
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            indiv_fig.savefig('{}/{}_actual_plus_predictions_{}.png'
                              .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                              dpi=figure_dpi)
        collective_fig.savefig('{}/actual_plus_predictions_{}.png'.format(figures_dir, window_size), dpi=figure_dpi)
        plt.close('all')


if __name__ == '__main__':
    from machine_learning import keras_init
    keras_init(gpu_mem_frac=0.5)
    main()
