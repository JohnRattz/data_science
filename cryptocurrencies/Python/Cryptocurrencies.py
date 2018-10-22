# TODO: Remove memory profiling when done debugging memory consumption.
# from memory_profiler import profile
# @profile
def main():
    ### Environment and Pre-load Library Settings###
    from utilities.Python.machine_learning import keras_init
    # `gpu_mem_frac` is the fraction of GPU memory dedicated to the TensorFlow backend
    # of Keras and may need to change based on your system and settings used - notably batch size in model training.
    # For reference, I run this script with a GTX 1080 Ti, which has 11 GiB of memory.
    keras_init(gpu_mem_frac=0.5)

    ### Imports ###

    # TODO: Remove these lines when done debugging memory consumption.
    # import os
    # import psutil
    # process = psutil.Process(os.getpid())

    # Import miscellaneous global variables.
    import sys
    sys.path.insert(0, '../../globals/Python')
    from globals import utilities_dir, num_lgc_prcs

    import os
    import warnings
    import time

    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    warnings.simplefilter('ignore')
    import seaborn as sns
    from functools import partial
    from utilities.Python.plotting import KerasPlotLosses

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
    from keras import backend as K
    from machine_learning import create_keras_regressor, keras_reg_grid_search, keras_convert_optimizer_obj_to_name, \
                                 reshape_data_from_2D_to_keras

    # Custom utility functions
    # sys.path.insert(0, '../../' + utilities_dir)
    from analysis import find_Markowitz_optimal_portfolio_weights, calc_CAPM_betas, CAPM_RoR, run_monte_carlo_financial_simulation
    from type_conversions import pandas_dt_to_str
    from utilities.Python.plotting import add_value_text_to_seaborn_barplot, monte_carlo_plot_confidence_band
    from utilities.Python.conversions import returns_to_yearly

    # Optimizers
    from keras.optimizers import Adam

    ### End Imports ###

    ### Main Settings and Constants ###

    ## Data Import Settings##
    # NOTE: See `ETL.load_data()` `date_range` parameter documentation for date ranges for daily and hourly data.
    # Whether to load from CSV files or a local MySQL database ('csv' or 'sql').
    load_source = 'csv'
    analysis_data_resolution = 'daily'
    analysis_data_date_range = ('2013-05-01', '2017-10-31') # Make sure this is at least three months to avoid errors.
    # analysis_data_date_range = ('2017-11-01', '2018-03-31')
    analysis_time_unit = 'days' if analysis_data_resolution == 'daily' else 'hours'
    annualize_returns = partial(returns_to_yearly, resolution=analysis_data_resolution)
    # The resolution of data for the machine learning models.
    model_data_resolution = 'hourly'
    model_data_date_range = ('2017-07-01', '2018-03-31')
    model_last_extrapolation_time = pd.to_datetime('2018-08-31')
    model_time_unit_singular = 'day' if model_data_resolution == 'daily' else 'hour'
    model_time_unit_plural = 'days' if model_data_resolution == 'daily' else 'hours'

    ## Machine Learning Settings ##
    # TODO (model_loading_1): Create boolean variable `train_models_not_present` which specifies whether to train models missing
    # TODO (model_loading_1): on secondary storage when `load_models == True`.
    # TODO (model_loading_2): Detect if `load_models = False` will throw an error before running. If so, give a clear error message.
    load_models = True # Whether or not to load models that have already been created by this program.
    # The number of logical processors to use during grid search.
    num_lgc_prcs_grd_srch = min(3, num_lgc_prcs)
    num_cv_splits = 3 # The number of folds in cross validation.
    # TODO: Add a parameter to determine whether or not param set occurrences are pickled?
    # Whether or not to use cross validation in Keras grid search.
    keras_use_cv_grd_srch = False
    # The level of verbosity in `keras_reg_grid_search()`.
    keras_grd_srch_verbose = 1

    ## Plotting Settings ##
    # TODO: Ensure this logic works even if using no Keras models.
    # Whether or not to plot analysis plots.
    make_analysis_plots = True
    # Whether or not to plot prediction plots.
    make_prediction_plots = True
    # Whether or not to plot loss for Keras neural networks over time during training when not using cross validation.
    # This setting only matters when `keras_use_cv_grd_srch == False`.
    make_keras_loss_plots = False
    # Whether or not to make plots of parameter set occurrences for Keras neural networks.
    # These figures are used to visualize the most successful parameter sets for Keras neural networks.
    make_keras_param_set_occurrence_plots = False
    # Whether Keras figures will be made for individual assets.
    make_asset_keras_figures = make_keras_loss_plots
    # Whether figures specific to individual assets will be made for all assets.
    make_all_assets_figures = make_analysis_plots
    # Whether figures specific to individual assets will be made for assets selected for model training.
    make_selected_assets_figures = make_analysis_plots or make_prediction_plots or make_asset_keras_figures
    # Whether non-asset-specific Keras figures will be made.
    make_keras_figures = make_keras_loss_plots or make_keras_param_set_occurrence_plots
    # Whether any figures at all will be created.
    make_any_figures = (make_analysis_plots or make_prediction_plots or
                        make_keras_loss_plots or make_keras_param_set_occurrence_plots)

    ## Figure Variables and Paths ##
    figure_size = (12, 6) # The base figure size.
    figure_dpi = 300
    figures_dir = 'figures'
    if make_any_figures and not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    ## Models Variables and Paths ##
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    ## Param Sets Variables and Paths ##
    param_sets_dir = 'param_sets'
    if not os.path.exists(param_sets_dir):
        os.makedirs(param_sets_dir)

    ## Parameter Set Occurrences Variables and Paths ##
    param_set_occurrences_dir = 'param_set_occurrences_keras'
    if not os.path.exists(param_set_occurrences_dir):
        os.makedirs(param_set_occurrences_dir)

    ### End Main Settings ###

    # Data Importing
    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        load_data(resolution=analysis_data_resolution, date_range=analysis_data_date_range, source=load_source)

    # Directory Structure Creation (Non-asset-specific Figures)
    if make_all_assets_figures:
        currency_figures_subdirs = ['{}/{}'.format(figures_dir, currency_label.replace(' ', '_'))
                                    for currency_label in currencies_labels]
        for currency_figures_subdir in currency_figures_subdirs:
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
    keras_figures_dir = None # Line needed for calls to `keras_reg_grid_search()`.
    if make_keras_figures:
        keras_figures_dir = '{}/{}'.format(figures_dir, 'keras')
        if not os.path.exists(keras_figures_dir):
            os.makedirs(keras_figures_dir)

    # Value Trends
    if make_analysis_plots:
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

    # Removing Currencies with Short Histories

    # See where values are absent and keep only currencies with reasonably lengthy histories.
    # Convert the timestamps to strings with less length for y-axis labels.
    old_prices_index = prices.index
    prices.index = prices.index.map(lambda date: pandas_dt_to_str(date))
    is_null_prices = pd.isnull(prices)
    if make_analysis_plots:
        # Select dates every 120 days for tick labels.
        ax = sns.heatmap(is_null_prices, yticklabels=120)
        absent_values_fig = ax.get_figure()
        plt.tight_layout()
        absent_values_fig.savefig('{}/absent_values.png'.format(figures_dir))
        prices.index = old_prices_index

    # TODO: Determine the cryptocurrencies to *keep* first, then programtically determine the ones to remove.
    if analysis_data_resolution == 'daily':
        currencies_labels_tickers_to_remove = np.array(
            [['Bitcoin Cash', 'BCH'], ['BitConnect', 'BCC'], ['Ethereum Classic', 'ETC'],
             ['Iota', 'MIOTA'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
             ['Qtum', 'QTUM'], ['Stratis', 'STRAT'], ['Waves', 'WAVES']])
    else:
        currencies_labels_tickers_to_remove = np.array([['Omisego', 'OMG'], ['Ethereum Classic', 'ETC'],
                                                        ['Neo', 'NEO']])
    currencies_labels_to_remove = currencies_labels_tickers_to_remove[:, 0]
    currencies_tickers_to_remove = currencies_labels_tickers_to_remove[:, 1]
    currencies_labels_and_tickers_to_remove = ["{} ({})".format(currencies_label, currencies_ticker)
                                               for currencies_label, currencies_ticker in
                                               zip(currencies_labels_to_remove, currencies_tickers_to_remove)]
    print("Removing currencies from analysis data: {}".format(currencies_labels_and_tickers_to_remove))
    subset_prices = prices.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_num_currencies = len(subset_prices.columns)
    subset_prices_nonan = subset_prices.dropna()
    print("Beginning and ending {} with data for remaining currencies in analysis data: {}, {}".
          format(analysis_time_unit, subset_prices_nonan.index[0], subset_prices_nonan.index[-1]))

    num_non_nan_times = len(subset_prices_nonan)
    print("Considering {} {} of price information for analysis (as many as without NaN values)."
          .format(num_non_nan_times, analysis_time_unit))

    # Directory Structure Creation (Asset-specific Figures)
    selected_currency_figures_subdirs = ['{}/{}'.format(figures_dir, currency_label.replace(' ', '_'))
                                         for currency_label in currencies_labels
                                         if currency_label not in currencies_labels_to_remove]
    if make_selected_assets_figures:
        for currency_figures_subdir in selected_currency_figures_subdirs:
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
    keras_figures_subdirs = ['{}/{}'.format(currency_figures_subdir, 'keras')
                             for currency_figures_subdir in selected_currency_figures_subdirs]
    if make_asset_keras_figures:
        for keras_figures_subdir in keras_figures_subdirs:
            if not os.path.exists(keras_figures_subdir):
                os.makedirs(keras_figures_subdir)

    # Find the returns.
    # `pd.DataFrame` of returns over time for the subset of assets.
    subset_returns = subset_prices_nonan.pct_change()
    # `pd.Series` of average yearly returns for the subset of assets.
    subset_returns_yearly = annualize_returns(subset_returns.mean())
    # print("subset_returns_yearly:", subset_returns_yearly)
    # log_returns = np.log(prices / prices.shift(1))
    # subset_log_returns = log_returns.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)

    # Find correlations.
    # TODO: Make value text smaller to fully fit into boxes.
    if make_analysis_plots:
        clustermap_params = dict(annot=True, annot_kws=dict(fontsize=12), fmt='.2f', vmin=-1, vmax=1)
        plt.figure(figsize=figure_size, dpi=figure_dpi)
        # Price Correlations
        price_correlations = subset_prices.corr()
        price_correlations_fig = sns.clustermap(price_correlations, **clustermap_params)
        price_correlations_fig.savefig('{}/price_correlations.png'.format(figures_dir))
        plt.clf()
        # Return Correlations
        returns_correlations = subset_returns.corr()
        returns_correlations_fig = sns.clustermap(returns_correlations, **clustermap_params)
        returns_correlations_fig.savefig('{}/return_correlations.png'.format(figures_dir))
        plt.clf()

    # Volatility Examination
    if make_analysis_plots:
        # Find the standard deviations in returns for each year.
        returns_std_by_year = subset_returns.groupby(subset_returns.index.year).std()
        # Standard deviations in returns for 2017 in descending order.
        returns_std_2017 = returns_std_by_year.loc[2017]
        returns_std_2017.sort_values(ascending=False, inplace=True)
        plt.subplots(figsize=figure_size)
        plotting_data = returns_std_2017.to_frame(name='Volatility').reset_index()
        volatility_plot = sns.barplot(x='Name', y='Volatility', data=plotting_data, palette='viridis')
        # print("Volatitility plotting_data:", plotting_data)
        add_value_text_to_seaborn_barplot(volatility_plot, plotting_data['Volatility'])
        plt.title('Volatility (2017)')
        plt.savefig('{}/volatility.png'.format(figures_dir), dpi=figure_dpi)

    # Find (daily) asset betas (covariance_with_market / market_variance).
    market_index = 'Bitcoin (BTC)'
    subset_betas = calc_CAPM_betas(subset_returns, market_index)
    subset_betas.sort_values(ascending=False, inplace=True)
    # subset_betas = betas.drop(labels=currencies_labels_and_tickers_to_remove)
    # Visualize the beta values.
    if make_analysis_plots:
        fig, ax = plt.subplots(figsize=figure_size)
        plotting_data = subset_betas.to_frame(name='Beta').reset_index()
        # print("Beta plotting_data:", plotting_data)
        beta_plot = sns.barplot(ax=ax, x='Name', y='Beta', data=plotting_data, palette='viridis')
        # Show values in the figure.
        add_value_text_to_seaborn_barplot(beta_plot, plotting_data['Beta'])
        plt.title('Betas (Bitcoin (BTC) as Market Index)')
        plt.savefig('{}/CAPM_betas.png'.format(figures_dir), dpi=figure_dpi)

    # Find assets' rates of return according to CAPM.
    # The annual rate of return of a risk-free asset is set to 2.5%, since that has
    # usually been the rate of return of 10-year U.S. government bonds.
    return_risk_free = 0.025
    CAPM_expected_rates_of_return = CAPM_RoR(subset_betas, subset_returns_yearly, market_index, return_risk_free)
    CAPM_expected_rates_of_return.sort_values(ascending=False, inplace=True)
    # print("CAPM_expected_rates_of_return:", CAPM_expected_rates_of_return)
    # Visualize the rates of return required to justify investing in
    # the assets versus investing in the market index according to CAPM.
    if make_analysis_plots:
        fig, ax = plt.subplots(figsize=figure_size)
        plotting_data = CAPM_expected_rates_of_return.to_frame(name='Expected Rate of Return').reset_index()
        CAPM_plot = sns.barplot(ax=ax, x='Name', y='Expected Rate of Return', data=plotting_data, palette='viridis')
        # print("CAPM plotting_data:", plotting_data)
        # Show values in the figure.
        add_value_text_to_seaborn_barplot(CAPM_plot, plotting_data['Expected Rate of Return'], percent=True)
        plt.title('CAPM Expected Rates of Return')
        plt.savefig('{}/CAPM_expected_rates_of_return.png'.format(figures_dir), dpi=figure_dpi)

    # Plot the rates of return.
    if make_analysis_plots:
        # subset_mean_returns = subset_returns.mean()
        # Put assets in the same order as in the CAPM expected rates of return figure.
        # print("returns new index order:", CAPM_expected_rates_of_return.index)
        plotting_data = subset_returns_yearly.reindex(CAPM_expected_rates_of_return.index, copy=False)
        plotting_data = plotting_data.to_frame(name='Rate of Return').reset_index()
        # subset_mean_returns.rename('Rate of Return')
        # print("returns plotting_data:", plotting_data)
        fig, ax = plt.subplots(figsize=figure_size)
        returns_plot = sns.barplot(ax=ax, x='Name', y='Rate of Return', data=plotting_data, palette='viridis')
        # Show values in the figure.
        add_value_text_to_seaborn_barplot(returns_plot, plotting_data['Rate of Return'], percent=True)
        plt.title('Rates of Return')
        plt.savefig('{}/rates_of_return.png'.format(figures_dir), dpi=figure_dpi)

    # TODO: Calculate Sharpe ratios for individual assets ((return_asset - return_risk_free)/std_asset)?

    # Determine the optimal portfolio using Markowitz optimization.
    if make_analysis_plots:
        optimal_weights = find_Markowitz_optimal_portfolio_weights(subset_returns, return_risk_free)
        optimal_weights.sort_values(ascending=False, inplace=True)
        # Create a visualization with the weights.
        fig, ax = plt.subplots(figsize=figure_size)
        plotting_data = optimal_weights.to_frame(name='Weight').reset_index()
        portfolio_weights_plot = sns.barplot(ax=ax, x='Name', y='Weight', data=plotting_data, palette='viridis')
        # Show values in the figure.
        add_value_text_to_seaborn_barplot(portfolio_weights_plot, plotting_data['Weight'])
        plt.title('Markowitz Optimal Portfolio Weights')
        plt.savefig('{}/optimal_portfolio_weights.png'.format(figures_dir), dpi=figure_dpi)

    # Load the data for model training.
    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        load_data(resolution=model_data_resolution, date_range=model_data_date_range, source=load_source)
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
    num_non_nan_times = len(subset_prices_nonan)
    date_times = subset_prices_nonan.index.values
    print("Considering {} {} of price information for model training (as many as without NaN values)."
          .format(num_non_nan_times, model_time_unit_plural))

    model_last_data_time = prices.index[-1]
    model_offset_time = {'days': 1} if model_data_resolution == 'daily' else {'hours': 1}
    model_first_extrapolation_time = model_last_data_time + pd.DateOffset(**model_offset_time)
    model_extrapolation_times = np.array(
        pd.date_range(model_first_extrapolation_time, model_last_extrapolation_time,
                      freq='D' if model_data_resolution == 'daily' else 'H'))
    model_num_extrapolation_times = len(model_extrapolation_times)

    # Run Monte Carlo simulation to predict future values.
    MC_predicted_values_ranges, MC_predicted_values = run_monte_carlo_financial_simulation(prices, model_extrapolation_times)
    subset_monte_carlo_predicted_values_ranges = \
        MC_predicted_values_ranges.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_monte_carlo_predicted_values = \
        MC_predicted_values.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)

    # Directory Structure Creation (Machine Learning Models)
    currency_models_subdirs = ['{}/{}'.format(models_dir, currency_label.replace(' ', '_'))
                               for currency_label in subset_currencies_labels]
    for currency_models_subdir in currency_models_subdirs:
        if not os.path.exists(currency_models_subdir):
            os.makedirs(currency_models_subdir)

    # Directory Structure Creation (Param Sets)
    param_sets_subdirs = ['{}/{}'.format(param_sets_dir, currency_label.replace(' ', '_'))
                          for currency_label in subset_currencies_labels]
    for param_sets_subdir in param_sets_subdirs:
        if not os.path.exists(param_sets_subdir):
            os.makedirs(param_sets_subdir)

    # We will predict closing prices based on these numbers of days preceding the date of prediction.
    # The max `window_size` is `num_non_nan_times`, but even reasonably close values may result in poor models or even
    # training failures (errors) due to small or even empty test sets, respectively, during cross validation.
    if model_data_resolution == 'daily': # Window sizes in days - 1 week, 2 weeks, and 1 to 12 months.
        window_sizes = np.array([7, 14] + list(range(30, 361, 30)))
    else: # Window sizes in hours - 1 week, 2 weeks, 1 month, and 2 months.
        # With just 14 days and 7 cryptocurrencies, we have 14 * 24 * 7 = 2352 features.
        window_sizes = 24*7*np.array([1, 2, 4, 8, 12])
    window_sizes = window_sizes[::-1] # User larger to smaller window sizes to catch memory-related errors early.

    # Directory Structure Creation (Keras Optimization Trend Figures - "Loss Plots")
    if make_asset_keras_figures:
        for keras_figures_subdir in keras_figures_subdirs:
            for window_size in window_sizes:
                keras_figures_window_subdir = os.path.join(keras_figures_subdir, 'w_{}'.format(window_size))
                if not os.path.exists(keras_figures_window_subdir):
                    os.makedirs(keras_figures_window_subdir)

    keras_param_names = None # The names of parameters used in Keras neural network grid search. This is set later.
    param_set_occurrences_pickle_paths = ['{}/param_set_occurrences_{}.pkl'.format(param_set_occurrences_dir, window_size)
                                          for window_size in window_sizes]
    keras_window_size_scores_pickle_path = '{}/window_size_scores.pkl'.format(param_set_occurrences_dir)

    # Column name in resulting DataFrame that denotes what asset a row pertains to ('all' for collective asset models).
    asset_col_str = 'asset'
    window_size_col_str = 'window_size'
    keras_window_size_scores = {} # The highest scores of the collective assets model for each window size.

    ### Main Loop ###
    for window_size_ind, window_size in enumerate(window_sizes):
        # Data Extraction
        num_windows = len(subset_prices_nonan) - 2*window_size + 1
        num_features = subset_num_currencies * window_size
        X = np.empty((num_windows, num_features), dtype=np.float64)
        y = np.empty((num_windows, num_features), dtype=np.float64)
        for window_index in range(num_windows):
            X[window_index,:] = \
                subset_prices_nonan[window_index:window_index + window_size].values.flatten()
            y[window_index,:] = \
                subset_prices_nonan[window_index + window_size:window_index + 2*window_size].values.flatten()
        # print("X[{}], X.shape:".format(window_size), X[window_size], X.shape)
        # print("y[{}], y.shape:".format(0), y[0], y.shape)

        # Feature Scaling
        # print("X, y shapes:", X.shape, y.shape)
        # print("X, y combined shape:", np.concatenate((X,y)).shape)
        # print("num nan X:", np.isnan(X).sum())
        # print("num nan y:", np.isnan(y).sum())
        # scaler = StandardScaler()
        # scaler.fit(np.stack(X,y))
        # scaler.transform(X)
        # scaler.transform(y)
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)
        # print("X_scaled[{}], X_scaled.shape:".format(window_size), X_scaled[window_size], X_scaled.shape)
        # print("y_scaled[{}], y_scaled.shape:".format(0), y_scaled[0], y_scaled.shape)

        ### Model Specifications ##

        # Neural Network models
        # Multilayer Perceptron regressor
        # model_neural_net = MLPRegressor()
        # The number of neurons in the hidden layers will be around half the mean of the number of inputs and outputs.
        # hidden_layer_median_size = int(round((num_features + subset_num_currencies)/2))
        single_hidden_layer_size = 96
        neural_net_hidden_layer_sizes = [(single_hidden_layer_size, single_hidden_layer_size // 8)]
        # params_neural_net = {'hdn_lyr_szs_nn_collective': neural_net_hidden_layer_sizes,
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
        keras_model_nn = "Keras_NN"
        # params_neural_net = {'model__input_dim': [subset_num_currencies],
        #                      'model__hidden_layer_sizes': neural_net_hidden_layer_sizes,
        #                      'model__output_dim': [subset_num_currencies],
        #                      'model__epochs': [1000],
        #                      'model__batch_size': [100]}
        # TODO: Choose hidden layer sizes according to https://stats.stackexchange.com/a/1097/191200
        frst_hdn_lyr_sz_nn_single_asset = 512
        frst_hdn_lyr_sz_nn_collective = 1024
        # three_hidden_layers = (first_hidden_layer_size_collective, first_hidden_layer_size_collective // 2,
        #                        first_hidden_layer_size_collective // 4)
        # four_hidden_layers = three_hidden_layers + (first_hidden_layer_size_collective // 8,)
        # five_hidden_layers = four_hidden_layers + (first_hidden_layer_size_collective // 16,)
        # six_hidden_layers = five_hidden_layers + (first_hidden_layer_size_collective // 32,)
        # hdn_lyr_szs_nn_collective = [three_hidden_layers, four_hidden_layers, five_hidden_layers, six_hidden_layers]
        #
        # hdn_lyr_szs_nn_collective = [(first_hidden_layer_size_collective, first_hidden_layer_size_collective // 2,
        #                        first_hidden_layer_size_collective // 4,
        #                        first_hidden_layer_size_collective // 8)]
        # hdn_lyr_szs_nn_collective = [(first_hidden_layer_size_collective, first_hidden_layer_size_collective // 4,
        #                        first_hidden_layer_size_collective // 16,
        #                        first_hidden_layer_size_collective // 64)]

        hdn_lyr_szs_nn_single_asset = [(frst_hdn_lyr_sz_nn_single_asset, frst_hdn_lyr_sz_nn_single_asset)]#,
                                        #frst_hdn_lyr_sz_nn_single_asset // 16, frst_hdn_lyr_sz_nn_single_asset // 64)]
        hdn_lyr_szs_nn_collective = [(frst_hdn_lyr_sz_nn_collective, frst_hdn_lyr_sz_nn_collective)]#,
                                      #frst_hdn_lyr_sz_nn_collective // 16, frst_hdn_lyr_sz_nn_collective // 64)]

        keras_params_nn = {
             'batch_size': [48],  # A too large batch size results in device OOMs.
             'hidden_layer_sizes': None, # Added later to handle single asset and collective asset models separately.
             'hidden_layer_type': ['LSTM'],
             #'dropout_rate': [0.1],
             'optimizer': [Adam],
             # Parameters for Adam optimizer.
             'lr': [1e-3],
             'beta_1': [0.9],#[0.7, 0.9],
             'beta_2': [0.999]}#[0.9, 0.999]}
        keras_param_names = list(keras_params_nn.keys())
        # The number of epochs to train for in grid search. There must be at least 2 epochs for loss plotting.
        # NOTE: This will likely impact run time much more significantly than `keras_nn_epochs` by a factor of
        # the number of parameter combinations in `keras_params_nn`, further multiplied by `n_splits` used
        # for cross validation (the `cv` object below) if `keras_use_cv_grd_srch`.
        keras_nn_epochs_grd_srch = 20
        keras_nn_epochs_grd_srch = max(2, keras_nn_epochs_grd_srch) \
            if not keras_use_cv_grd_srch & make_keras_loss_plots else keras_nn_epochs_grd_srch
        # The number of epochs to train for without or after cross validation.
        keras_nn_epochs = 20

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
        model_extra_trees = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ExtraTreesRegressor())
        ])
        params_extra_trees = {'model__n_estimators': [500],
                              'model__min_samples_split': [2, 5, 10],
                              'model__max_features': ['auto', 'sqrt', 'log2']}
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
        models_to_test = [keras_model_nn]  # keras_neural_network
        param_grids = [keras_params_nn]  # keras_neural_network
        # models_to_test = [keras_model_rnn]  # keras_recurrent_neural_network
        # param_grids = [keras_params_rnn]  # keras_recurrent_neural_network
        # models_to_test = [model_svr]  # SVR
        # param_grids = [params_svr]  # SVR
        # models_to_test = [params_extra_trees] # Extra-Trees
        # param_grids = [params_extra_trees] # Extra-Trees

        # Specify the cross validation method.
        cv = TimeSeriesSplit(n_splits=num_cv_splits)
        keras_cv = cv if keras_use_cv_grd_srch else None

        # Populates `keras_param_sets_occurrences`. Tracks the number of occurrences of parameter sets for window sizes.
        keras_param_sets_occurrences_window = {}
        keras_param_set_occurrences_pickle_path = param_set_occurrences_pickle_paths[window_size_ind]

        ### Model Training ###

        # Single Asset Model (consider only windows of values for one asset each (disallows learning of correlations))
        # Store paths and types of models to load, predict with, and delete each model to save memory.
        single_asset_models_paths, single_asset_models_types, single_asset_models_param_sets = [], [], []
        single_asset_models = [None]*subset_num_currencies

        # Build a model for each cryptocurrency.
        for currency_index in range(subset_num_currencies):
            def get_X_subset(X, asset_index, num_assets):
                """
                Given a 2D feature matrix `X` of windows of asset values,
                return the subset for a given asset by index.

                Parameters
                ----------
                X: numpy.ndarray
                    The data to extract the desired subset from.
                asset_index: int
                    The asset data to retrieve.
                num_assets: int
                    The total number of assets - needed to determine the stride.
                """
                return X[:, asset_index::num_assets]

            def get_y_subset(y, asset_index, num_assets):
                """
                Given a 2D feature matrix `y` of asset values,
                return the subset for a given asset by index.

                Parameters
                ----------
                y: numpy.ndarray
                    The data to extract the desired subset from.
                asset_index: int
                    The asset data to retrieve.
                num_assets: int
                    The total number of assets - needed to determine the stride.
                """
                # return y[:, asset_index].reshape(-1, 1)
                return y[:, asset_index::num_assets]

            currency_label = subset_currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_models_subdir = '{}/{}'.format(models_dir, currency_label_no_spaces)
            currency_param_sets_subdir = "{}/{}".format(param_sets_dir, currency_label_no_spaces)
            currency_figures_subdir = \
                '{}/{}'.format(figures_dir, currency_label_no_spaces) if make_selected_assets_figures else None
            keras_figures_subdir = \
                '{}/{}'.format(currency_figures_subdir, 'keras') if make_asset_keras_figures else None
            keras_figures_window_subdir = \
                '{}/{}'.format(keras_figures_subdir, 'w_{}'.format(window_size)) if make_asset_keras_figures else None
            # Paths
            model_base_path = "{}/{}_model_{}".format(currency_models_subdir, currency_label_no_spaces, window_size)
            model_pickle_path = "{}.pkl".format(model_base_path)
            param_set_base_path = "{}/{}_param_set_{}".format(currency_param_sets_subdir, currency_label_no_spaces, window_size)
            param_set_pickle_path = "{}.pkl".format(param_set_base_path)
            model_path, model_type = (None, None)

            # Data for only this cryptocurrency.
            X_subset_scaled = get_X_subset(X_scaled, currency_index, subset_num_currencies)
            y_subset_scaled = get_y_subset(y_scaled, currency_index, subset_num_currencies)
            # print("X_subset_scaled.shape, y_subset_scaled.shape:",
            #       X_subset_scaled.shape, y_subset_scaled.shape)
            # print("X_subset_scaled[{}]:".format(window_size), X_subset_scaled[window_size], X_subset_scaled.shape)
            # print("y_subset_scaled[{}]:".format(0), y_subset_scaled[0], y_subset_scaled.shape)

            # Feature Scaling
            # X_subset_scaled = StandardScaler().fit_transform(X_subset)
            # y_subset_scaled = StandardScaler().fit_transform(y_subset)

            if not load_models:
                currency_label_and_ticker = subset_currencies_labels_and_tickers[currency_index]
                print("Currently training a model for {} with a window size of {} {}."
                    .format(currency_label_and_ticker, window_size, model_time_unit_plural))

                # Tuples of scores, the corresponding models, and the best batch sizes for Keras models.
                # score_model_batch_size_tuples = []
                best_score, best_model, best_param_set = (-np.inf, None, None)

                # Create a single asset model.
                for i in range(len(models_to_test)):
                    model_to_test = models_to_test[i]
                    param_grid = param_grids[i]
                    param_grid['hidden_layer_sizes'] = hdn_lyr_szs_nn_single_asset
                    # Keras needs to be trained differently than scikit-learn models even though Keras models can
                    # be "wrapped" to function as such. This is required (AFAIK) to save them to secondary storage.
                    if model_to_test == keras_model_nn:
                        X_subset_scaled_dict = reshape_data_from_2D_to_keras(param_grid, 1, X_subset_scaled)
                        # print("y_subset_scaled.shape[1]:", y_subset_scaled.shape[1])
                        model, score, param_set = \
                            keras_reg_grid_search(X_subset_scaled_dict, y_subset_scaled, build_fn=create_keras_regressor,
                                                  param_grid=param_grid, epochs=keras_nn_epochs,
                                                  cv_epochs=keras_nn_epochs_grd_srch, cv=keras_cv,
                                                  scoring=r2_score, verbose=keras_grd_srch_verbose,
                                                  plot_losses=make_keras_loss_plots, plotting_dir=keras_figures_window_subdir,
                                                  figure_title_prefix='{} (window size {})'.format(currency_label, window_size),
                                                  figure_kwargs={'figsize':figure_size, 'dpi':figure_dpi})
                        param_set['optimizer'] = keras_convert_optimizer_obj_to_name(param_set['optimizer'])
                        param_vals_tup = tuple(param_set.values()) + ('single', window_size)
                        num_occurrences = keras_param_sets_occurrences_window.setdefault(param_vals_tup, 0)
                        keras_param_sets_occurrences_window[param_vals_tup] = num_occurrences + 1
                    else:
                        grid_search = GridSearchCV(model_to_test, param_grid, scoring=make_scorer(r2_score),
                                                   cv=cv, n_jobs=num_lgc_prcs_grd_srch)
                        grid_search.fit(X_subset_scaled, y_subset_scaled.ravel())
                        model = grid_search.best_estimator_
                        score = grid_search.best_score_
                        param_set = grid_search.best_params_
                    best_model = model if score > best_score else best_model
                    best_param_set = param_set if score > best_score else best_param_set
                    best_score = score if score > best_score else best_score
                time.sleep(1)  # Wait 1 second for printing from GridSearchCV to complete.
                print("Best model and score for asset {} and window size {}: {}"
                      .format(currency_label_and_ticker, window_size, (best_model, best_score)))
                # Save the best model for this window size.
                if type(best_model) is keras.models.Sequential:
                    model_path = model_base_path + '.h5'
                    ks_models.save_model(best_model, model_path)
                else:
                    model_path = model_pickle_path
                    with open(model_pickle_path, "wb") as model_outfile:
                        pickle.dump(best_model, model_outfile)
                # Save the best parameter set for this window size.
                with open(param_set_pickle_path, "wb") as param_set_outfile:
                    pickle.dump(best_param_set, param_set_outfile)
            else:
                # Model Loading
                try: # Try to load the neural network model first (runs on GPUs).
                    best_model = ks_models.load_model(model_base_path + '.h5')
                    model_path = model_base_path + '.h5'
                except OSError:
                    model_path = model_pickle_path
                    with open(model_pickle_path, "rb") as model_infile:
                        best_model = pickle.load(model_infile)
                # Param Set Loading
                with open(param_set_pickle_path, "rb") as param_set_infile:
                    best_param_set = pickle.load(param_set_infile)
            single_asset_models_paths.append(model_path)
            single_asset_models_types.append(type(best_model))
            single_asset_models_param_sets.append(best_param_set)
        best_model = None
        K.clear_session()  # Deallocate models to free GPU memory.

        # Collective Model (consider windows of values for all assets)
        collective_assets_model_path, collective_assets_model_type, collective_assets_model_param_set = (None, None, None)
        collective_assets_model = None
        model_base_path = "{}/model_{}".format(models_dir, window_size)
        model_pickle_path = "{}.pkl".format(model_base_path)

        param_set_base_path = "{}/param_set_{}".format(param_sets_dir,window_size)
        param_set_pickle_path = "{}.pkl".format(param_set_base_path)

        if not load_models:
            print("Currently training a model for all assets collectively "
                  "with a window size of {} {}.".format(window_size, model_time_unit_plural))

            # Tuples of scores and the corresponding models (along with the best batch sizes for Keras models)
            # score_model_batch_size_tuples = []
            best_score, best_model, best_param_set = (-np.inf, None, None)

            # Create the models.
            for i in range(len(models_to_test)):
                model_to_test = models_to_test[i]
                param_grid = param_grids[i]
                param_grid['hidden_layer_sizes'] = hdn_lyr_szs_nn_collective
                if model_to_test == keras_model_nn:
                    # The data has different formats based on parameter values in parameter sets.
                    X_scaled_dict = reshape_data_from_2D_to_keras(param_grid, subset_num_currencies, X_scaled)
                    model, score, param_set = \
                        keras_reg_grid_search(X_scaled_dict, y_scaled, build_fn=create_keras_regressor,
                                              param_grid=param_grid,
                                              epochs=keras_nn_epochs, cv_epochs=keras_nn_epochs_grd_srch,
                                              cv=keras_cv, scoring=r2_score, verbose=keras_grd_srch_verbose,
                                              plot_losses=make_keras_loss_plots, plotting_dir=keras_figures_dir,
                                              figure_title_prefix='Collective (window size {})'.format(window_size),
                                              figure_kwargs={'figsize': figure_size, 'dpi': figure_dpi})
                    param_set['optimizer'] = keras_convert_optimizer_obj_to_name(param_set['optimizer'])
                    param_vals_tup = tuple(param_set.values()) + ('all', window_size)
                    num_occurrences = keras_param_sets_occurrences_window.setdefault(param_vals_tup, 0)
                    keras_param_sets_occurrences_window[param_vals_tup] = num_occurrences + 1
                else:
                    grid_search = GridSearchCV(model_to_test, param_grid, scoring=make_scorer(r2_score),
                                               cv=cv, n_jobs=num_lgc_prcs_grd_srch)
                    grid_search.fit(X_scaled, y_scaled.ravel())
                    model = grid_search.best_estimator_
                    score = grid_search.best_score_
                    param_set = grid_search.best_params_
                best_model = model if score > best_score else best_model
                best_param_set = param_set if score > best_score else best_param_set
                best_score = score if score > best_score else best_score
            time.sleep(1)  # Wait 1 second for printing from GridSearchCV to complete.
            # Choose the model with the best score.
            # score_model_batch_size_tuples.sort(key=lambda tup: tup[0], reverse=True)
            # score = score_model_batch_size_tuples[0][0]
            # collective_assets_model = score_model_batch_size_tuples[0][1]
            print("Best collective model and score for window size {}: {}"
                  .format(window_size, (best_model, best_score)))
            # Save the best model for this window size.
            collective_assets_model_type = type(best_model)
            if type(best_model) is keras.models.Sequential:
                collective_assets_model_path = model_base_path + '.h5'
                ks_models.save_model(best_model, model_base_path + '.h5')
            else:
                collective_assets_model_path = model_pickle_path
                with open(model_pickle_path, "wb") as model_outfile:
                    pickle.dump(best_model, model_outfile)
            # Save the best parameter set for this window size.
            with open(param_set_pickle_path, "wb") as param_set_outfile:
                pickle.dump(best_param_set, param_set_outfile)
            # Save the best score for this window size.
            keras_window_size_scores[window_size] = best_score
            with open(keras_window_size_scores_pickle_path, 'wb') as keras_window_size_scores_outfile:
                pickle.dump(keras_window_size_scores, keras_window_size_scores_outfile)
        else:
            # Model Loading
            try: # Try to load the neural network model first (runs on GPUs).
                best_model = ks_models.load_model(model_base_path + '.h5')
                collective_assets_model_path = model_base_path + '.h5'
            except OSError:
                collective_assets_model_path = model_pickle_path
                with open(model_pickle_path, "rb") as model_infile:
                    best_model = pickle.load(model_infile)
            # Param Set Loading
            with open(param_set_pickle_path, "rb") as param_set_infile:
                best_param_set = pickle.load(param_set_infile)
        collective_assets_model_type = type(best_model)
        collective_assets_model_param_set = best_param_set
        best_model = None
        K.clear_session()

        # Save the Keras parameter set occurrence counts for this window size.
        if not load_models:
            with open(keras_param_set_occurrences_pickle_path, "wb") as param_set_occurrences_outfile:
                pickle.dump(keras_param_sets_occurrences_window, param_set_occurrences_outfile)

        # Validation and Visualization (for each window size)
        if make_prediction_plots:
            print("Predicting prices with a window of {} {} of preceding currency values.".format(window_size, model_time_unit_plural))

            # Single Asset Model Predictions
            def single_asset_models_predict(X):
                """
                Returns predictions for the separate single asset models in the format of
                predictions from the collective assets model.

                **NOTE: Be sure the models are loaded first!**

                Parameters
                ----------
                X: numpy.ndarray
                    The feature matrix to predict for.
                """
                # print("Entering single_asset_models_predict()!")
                # print("X.shape:", X.shape)
                # num_time_indices = int(np.ceil(len(X)/window_size))
                # single_asset_models_pred = np.empty((len(X),subset_num_currencies), dtype=np.float64)
                single_asset_models_pred = np.empty_like(X)
                # print("single_asset_models_pred:", single_asset_models_pred.shape)
                for currency_index in range(subset_num_currencies):
                    param_set = single_asset_models_param_sets[currency_index]
                    X_subset = get_X_subset(X, currency_index, subset_num_currencies)
                    # print("num nans X_subset:", np.isnan(X_subset).sum(), X_subset.shape)
                    single_asset_model = single_asset_models[currency_index]
                    # Handle any reshaping for models that require it.
                    if type(single_asset_model) is keras.models.Sequential:
                        X_subset = reshape_data_from_2D_to_keras(param_set, 1, X_subset)
                    # print("num nans X_subset reshaped:", np.isnan(X_subset).sum(), X_subset.shape)
                    # print("X_subset:", X_subset.shape)
                    # print("num nans single_asset_model.predict(X_subset):",
                    #       np.isnan(single_asset_model.predict(X_subset)).sum(), single_asset_model.predict(X_subset).shape)
                    single_asset_models_pred[:, currency_index::subset_num_currencies] = single_asset_model.predict(X_subset)
                    # print("num nans single_asset_models_pred[:, {}::{}]:".format(currency_index, subset_num_currencies),
                    #       np.isnan(single_asset_models_pred[:, currency_index::subset_num_currencies]).sum())
                # print("X.shape:", X.shape)
                # print("single_asset_models_pred.shape:", single_asset_models_pred.shape)
                # print("Leaving single_asset_models_predict()!")
                return single_asset_models_pred

            def collective_assets_model_predict(X):
                """
                Returns predictions for the collective assets model.

                **NOTE: Be sure the model is loaded first!**

                Parameters
                ----------
                X: numpy.ndarray
                    The feature matrix to predict for.
                """
                param_set = collective_assets_model_param_set
                # Handle any reshaping for models that require it.
                if type(collective_assets_model) is keras.models.Sequential:
                    X = reshape_data_from_2D_to_keras(param_set, subset_num_currencies, X)
                collective_assets_model_pred = collective_assets_model.predict(X)
                # print("X.shape:", X.shape)
                # print("collective_assets_model_pred.shape:", collective_assets_model_pred.shape)
                return collective_assets_model_pred

            def load_model(model_type, model_path):
                """
                Loads a model.

                Parameters
                ----------
                model_type: type
                    The type of the model to load.
                model_path: str
                    The path to the model serialization file.
                """
                if model_type is keras.models.Sequential:
                    model = ks_models.load_model(model_path)
                else:
                    with open(model_path, "rb") as model_infile:
                        model = pickle.load(model_infile)
                return model

            def load_single_asset_models():
                """"
                Loads single asset models from their serialization files into `single_asset_models`.
                """
                for currency_index in range(subset_num_currencies):
                    single_asset_models[currency_index] = \
                        load_model(single_asset_models_types[currency_index], single_asset_models_paths[currency_index])

            def load_collective_assets_model():
                """"
                Loads the collective assets model from its serialization file into `collective_assets_model`.
                """
                nonlocal collective_assets_model
                collective_assets_model = load_model(collective_assets_model_type, collective_assets_model_path)

            def fmt_pred_for_vis(predictions, num_times_excess, num_times_expected, num_features):
                """
                Formats predictions for visualization. The last window overlaps the second last window,
                so we need to remove the overlapping predictions.

                Parameters
                ----------
                predictions: numpy.ndarray
                    A 2D NumPy array in which rows are windows of data that don't overlap each other,
                    except for the last and second last windows.
                num_times_excess: int
                    The number of excess entries in the predictions.
                num_times_expected: int
                    The number of entries expected in the predictions.
                num_features: int
                    The number of features in the output (i.e. number of columns).

                Returns
                -------
                predictions_fmt: numpy.ndarray
                    A 2D NumPy array of predictions - with features in rows and entries in columns.
                """
                # print(predictions.shape)
                # print(predictions[:-1].shape)
                # print(predictions[-1, num_times_excess:].shape)
                return np.concatenate(
                    [*predictions[:-1], predictions[-1, num_times_excess*num_features:]])\
                    .reshape((num_times_expected, num_features))

            # Create version of `fmt_pred_for_vis()` for predictions for available data (validation).
            # Because we are windowing, we only need some windows to get all possible predictions for available data.
            X_scaled_fmt_for_pred = X_scaled[::window_size]
            # print("num nans X_scaled_fmt_for_pred:", np.isnan(X_scaled_fmt_for_pred).sum())
            # print("y.shape, window_size:", y.shape, window_size)
            y_fmt_for_vis = y[::window_size]
            # print("num nans y_fmt_for_vis:", np.isnan(y_fmt_for_vis).sum())
            # print("Before, X_scaled_fmt_for_pred.shape, y_fmt_for_pred.shape:",
            #       X_scaled_fmt_for_pred.shape, y_fmt_for_vis.shape)
            # If `X` and `y` can't be exactly covered with non-overlapping windows,
            # we need to handle the last window specially - here, we retrieve its data.
            if X_scaled.shape[0] % window_size != 0:
                # print("X_scaled_fmt_for_pred:", X_scaled_fmt_for_pred.shape)
                # print("X_scaled[-window_size]:", X_scaled[[-window_size]].shape)
                X_scaled_fmt_for_pred = np.concatenate([X_scaled_fmt_for_pred, X_scaled[[-1]]])#-window_size]]])
                y_fmt_for_vis = np.concatenate([y_fmt_for_vis, y[[-1]]])#-window_size]]])
            # print("num nans X_scaled_fmt_for_pred finished:", np.isnan(X_scaled_fmt_for_pred).sum())
            # print("num nans y_fmt_for_vis finished:", np.isnan(y_fmt_for_vis).sum())
            # print("After, X_scaled_fmt_for_pred.shape, y_fmt_for_pred.shape:",
            #       X_scaled_fmt_for_pred.shape, y_fmt_for_vis.shape)
            # The number of points that will be in the predictions for available data.
            pred_num_times = int(np.prod(np.array(X_scaled_fmt_for_pred.shape)) / subset_num_currencies)
            # The expected number of points in the predictions for available data.
            pred_num_times_expected = num_non_nan_times - window_size
            # The excess number of points that will be in the predictions for available data.
            pred_num_times_excess = pred_num_times - pred_num_times_expected
            # print("y_pred_num_times:", y_pred_num_times)
            # print("y_pred_num_times_expected:", y_pred_num_times_expected)
            # print("y_pred_num_times_excess:", y_pred_num_times_excess)
            fmt_pred_for_vis_val = \
                partial(fmt_pred_for_vis, num_times_excess=pred_num_times_excess,
                        num_times_expected=pred_num_times_expected, num_features=subset_num_currencies)

            # Create version of `fmt_pred_for_vis()` for predictions for extrapolated data (predictions).
            # The number of windows in the extrapolation outputs before formatting for visualization.
            num_orig_extrp_rows = int(np.ceil(model_num_extrapolation_times / window_size))
            # The shape of extrapolation data before formatting for visualization.
            extrp_y_shape = (num_orig_extrp_rows, subset_num_currencies * window_size)
            # The number of points that will be in the extrapolation predictions for available data.
            extrp_pred_num_times = int(num_orig_extrp_rows * window_size)
            #np.prod(np.array(extrp_y_shape)) / subset_num_currencies)
            # The expected number of points in the extrapolation predictions for available data.
            extrp_pred_num_times_expected = model_num_extrapolation_times
            # The excess number of points that will be in the extrapolation predictions for available data.
            extrp_pred_num_times_excess = extrp_pred_num_times - extrp_pred_num_times_expected
            # print("extrp_pred_num_times:", extrp_pred_num_times)
            # print("extrp_pred_num_times_expected:", extrp_pred_num_times_expected)
            # print("extrpgy_pred_num_times_excess:", extrp_pred_num_times_excess)
            fmt_pred_for_vis_extrp = \
                partial(fmt_pred_for_vis, num_times_excess=extrp_pred_num_times_excess,
                        num_times_expected=extrp_pred_num_times_expected, num_features=subset_num_currencies)

            y_fmt_for_vis = fmt_pred_for_vis_val(y_fmt_for_vis) # Format available data for visualization.

            load_single_asset_models()
            single_asset_models_pred = \
                fmt_pred_for_vis_val(y_scaler.inverse_transform(single_asset_models_predict(X_scaled_fmt_for_pred)))
            # print("single_asset_models_pred:", single_asset_models_pred.shape)
            single_asset_models = [None]*len(single_asset_models) # Remove the models from memory.
            K.clear_session()

            # Collective Model Predictions
            load_collective_assets_model()
            collective_assets_model_pred = \
                fmt_pred_for_vis_val(y_scaler.inverse_transform(collective_assets_model_predict(X_scaled_fmt_for_pred)))
            # print("collective_assets_model_pred:", collective_assets_model_pred.shape)
            collective_assets_model = None
            K.clear_session()

            # print("date_times[window_size:]:", date_times[window_size:].shape)

            ### Plotting Predictions ###
            # The colors of lines for various things.
            actual_values_color = 'blue'
            actual_values_label = 'True'
            ml_model_single_asset_color = 'red'
            ml_model_single_asset_label = 'ML Model (Single Asset Predictors)'
            ml_model_collective_color = 'orange'
            ml_model_collective_label = 'ML Model (Collective Predictor)'
            monte_carlo_color = 'green'
            monte_carlo_label = 'Monte Carlo AVG'

            # print("Plotting validation figures!")

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
                collective_ax_current.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                           color=actual_values_color, alpha=0.5, label=actual_values_label)
                # Single Asset Model Predictions
                collective_ax_current.plot(date_times[window_size:], single_asset_models_pred[:, currency_index],
                                           color=ml_model_single_asset_color, alpha=0.5, label=ml_model_single_asset_label)
                # Collective Model Predictions
                collective_ax_current.plot(date_times[window_size:], collective_assets_model_pred[:, currency_index],
                                           color=ml_model_collective_color, alpha=0.5, label=ml_model_collective_label)
                collective_ax_current.set_xlabel('Date')
                collective_ax_current.set_ylabel('Close')
                collective_ax_current.set_title("{} ({}) Closing Value ({} {} window)"
                                                .format(currency_label, currency_ticker, window_size,
                                                        model_time_unit_singular))
                collective_ax_current.legend()
                # Individual plot
                indiv_fig = plt.figure(figsize=(12, 6))
                indiv_fig_ax = indiv_fig.add_subplot(111)
                # Actual values
                indiv_fig_ax.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                  color=actual_values_color, alpha=0.5, label=actual_values_label)
                # Single Asset Model Predictions
                indiv_fig_ax.plot(date_times[window_size:], single_asset_models_pred[:, currency_index],
                                  color=ml_model_single_asset_color, alpha=0.5, label=ml_model_single_asset_label)
                # Collective Model Predictions
                indiv_fig_ax.plot(date_times[window_size:], collective_assets_model_pred[:, currency_index],
                                  color=ml_model_collective_color, alpha=0.5, label=ml_model_collective_label)
                indiv_fig_ax.set_xlabel('Date')
                indiv_fig_ax.set_ylabel('Close')
                indiv_fig_ax.set_title("{} ({}) Closing Value ({} {} window)"
                                       .format(currency_label, currency_ticker, window_size,
                                               model_time_unit_singular))
                indiv_fig_ax.legend()
                currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
                indiv_fig.savefig('{}/{}_validation_{}.png'
                                  .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                                  dpi=figure_dpi)
            collective_fig.savefig('{}/validation_{}.png'
                                   .format(figures_dir, window_size), dpi=figure_dpi)
            collective_fig.clf()

            # print("Getting predictions!")

            # Get the models' predictions for the rest of 2017 and 2018.
            # TODO (1): Make this a function that is run for both single asset models and the collective assets model.
            # TODO (2): Inverse transform results of `single_asset_models_predict()` and `collective_assets_model_predict()`.
            # TODO (3): (Performance) Slide window by `window_size` time units - not 1 time unit - for prediction steps.

            # Single Asset Model Predictions
            load_single_asset_models()
            single_asset_models_extrapolation_y = \
                np.zeros((extrp_y_shape), dtype=np.float64)
            # print("model_num_extrapolation_times:", model_num_extrapolation_times)
            # print("single_asset_models_extrapolation_y:", single_asset_models_extrapolation_y.shape)
            # First `window_size` windows contain known values.
            # given_prices = subset_prices_nonan.values[-window_size:].flatten()
            # given_prices = X_scaled[-1]
            # print("given_prices:", given_prices.shape)
            # single_asset_models_extrapolation_X[0] = given_prices
            # print("single_asset_models_extrapolation_X[0].reshape(1,-1):",
            #       single_asset_models_extrapolation_X[0].reshape(1, -1)[:subset_num_currencies],
            #       single_asset_models_extrapolation_X[0].reshape(1, -1)[-subset_num_currencies:],)
            # Predict future values based on a window of the last given values.
            single_asset_models_extrapolation_y[0] = \
                y_scaler.inverse_transform(single_asset_models_predict(
                    X_scaled[-1].reshape(1,-1)))
            # print("num nans X_scaled:", np.isnan(X_scaled).sum())
            # print("num nans single_asset_models_predict(X_scaled[-1].reshape(1,-1)):",
            #       np.isnan(single_asset_models_predict(X_scaled[-1].reshape(1, -1))).sum())
            # print("num nans single_asset_models_extrapolation_y[0]:",
            #       np.isnan(single_asset_models_extrapolation_y[0]).sum())
            # print("single_asset_models_extrapolation_y[0]:", single_asset_models_extrapolation_y[0])
            # Predict future values based on windows of previously predicted future values.
            for window_index in range(1, len(single_asset_models_extrapolation_y)):
                # print(single_asset_models_extrapolation_y[window_index-1].reshape(1,-1))
                # print(y_scaler.transform(single_asset_models_extrapolation_y[window_index-1].reshape(1,-1)))
                # print(y_scaler.inverse_transform(single_asset_models_predict(
                #         y_scaler.transform(single_asset_models_extrapolation_y[window_index-1].reshape(1,-1)))))
                single_asset_models_extrapolation_y[window_index] = \
                    y_scaler.inverse_transform(single_asset_models_predict(
                        X_scaler.transform(single_asset_models_extrapolation_y[window_index-1].reshape(1,-1))))
            # single_asset_models_extrapolation_X[0] = given_prices
            # single_asset_models_extrapolation_y[0] = \
            #     y_scaler.inverse_transform(single_asset_models_predict(
            #         single_asset_models_extrapolation_X[0].reshape(1, -1)))
            # for currency_index in range(1, window_size):
            #     given_prices = subset_prices_nonan.values[-window_size + currency_index:].flatten()
            #     single_asset_models_previous_predicted_prices = \
            #         single_asset_models_extrapolation_y[:currency_index].flatten()
            #     single_asset_models_extrapolation_X[currency_index] = \
            #         np.concatenate((given_prices, single_asset_models_previous_predicted_prices))
            #     single_asset_models_extrapolation_y[currency_index] = \
            #         y_scaler.inverse_transform(single_asset_models_predict(
            #             single_asset_models_extrapolation_X[currency_index].reshape(1, -1)).flatten())
            # Remaining windows contain only predicted values (predicting based on previous predictions).
            # for currency_index in range(window_size, model_num_extrapolation_times):
            #     single_asset_models_previous_predicted_prices = \
            #         single_asset_models_extrapolation_y[currency_index - window_size:currency_index].flatten()
            #     single_asset_models_extrapolation_X[currency_index] = \
            #         single_asset_models_previous_predicted_prices
            #     single_asset_models_extrapolation_y[currency_index] = \
            #         y_scaler.inverse_transform(single_asset_models_predict(
            #             single_asset_models_extrapolation_X[currency_index].reshape(1, -1)).flatten())
            single_asset_models = [None] * len(single_asset_models)
            K.clear_session()
            # Format the extrapolations for visualization.
            single_asset_models_extrapolation_y = fmt_pred_for_vis_extrp(single_asset_models_extrapolation_y)

            # Collective Assets Model Predictions
            load_collective_assets_model()
            collective_assets_model_extrapolation_y = \
                np.zeros((int(np.ceil(model_num_extrapolation_times / window_size)),
                          subset_num_currencies * window_size), dtype=np.float64)
            # collective_assets_model_extrapolation_X = \
            #     np.zeros((model_num_extrapolation_times, subset_num_currencies * window_size), dtype=np.float64)
            # collective_assets_model_extrapolation_y = \
            #     np.zeros((model_num_extrapolation_times, subset_num_currencies), dtype=np.float64)
            collective_assets_model_extrapolation_y[0] = \
                y_scaler.inverse_transform(collective_assets_model_predict(
                    X_scaled[-1].reshape(1, -1)))
            # print("collective_assets_model_extrapolation_y[0]:", collective_assets_model_extrapolation_y[0])
            # Predict future values based on windows of previously predicted future values.
            for window_index in range(1, len(collective_assets_model_extrapolation_y)):
                collective_assets_model_extrapolation_y[window_index] = \
                    y_scaler.inverse_transform(collective_assets_model_predict(
                        X_scaler.transform(collective_assets_model_extrapolation_y[window_index - 1].reshape(1, -1))))
            # given_prices = subset_prices_nonan.values[-window_size:].flatten()
            # collective_assets_model_extrapolation_X[0] = given_prices
            # collective_assets_model_extrapolation_y[0] = \
            #     y_scaler.inverse_transform(collective_assets_model_predict(
            #         collective_assets_model_extrapolation_X[0].reshape(1, -1)))
            # for currency_index in range(1, window_size):
            #     given_prices = subset_prices_nonan.values[-window_size + currency_index:].flatten()
            #     collective_assets_model_previous_predicted_prices = \
            #         collective_assets_model_extrapolation_y[:currency_index].flatten()
            #     collective_assets_model_extrapolation_X[currency_index] = \
            #         np.concatenate((given_prices, collective_assets_model_previous_predicted_prices))
            #     collective_assets_model_extrapolation_y[currency_index] = \
            #         y_scaler.inverse_transform(collective_assets_model_predict(
            #             collective_assets_model_extrapolation_X[currency_index].reshape(1, -1)).flatten())
            # for currency_index in range(window_size, model_num_extrapolation_times):
            #     collective_assets_model_previous_predicted_prices = \
            #         collective_assets_model_extrapolation_y[currency_index - window_size:currency_index].flatten()
            #     collective_assets_model_extrapolation_X[currency_index] = \
            #         collective_assets_model_previous_predicted_prices
            #     collective_assets_model_extrapolation_y[currency_index] = \
            #         y_scaler.inverse_transform(collective_assets_model_predict(
            #             collective_assets_model_extrapolation_X[currency_index].reshape(1, -1)).flatten())
            collective_assets_model = None
            K.clear_session()
            # Format the extrapolations for visualization.
            collective_assets_model_extrapolation_y = fmt_pred_for_vis_extrp(collective_assets_model_extrapolation_y)

            # print("Plotting prediction figures!")

            # Plot predictions for the rest of 2017 and 2018.
            for currency_index in range(subset_num_currencies):
                currency_label = subset_currencies_labels[currency_index]
                currency_label_no_spaces = currency_label.replace(' ', '_')
                currency_ticker = subset_currencies_tickers[currency_index]
                label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
                # Collective plot
                collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
                # Single Asset Model Predictions
                collective_ax_current.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                           color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                # Collective Model Predictions
                collective_ax_current.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                           color=ml_model_collective_color, label=ml_model_collective_label)
                # Monte Carlo predictions
                collective_ax_current.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                           color=monte_carlo_color, label=monte_carlo_label)
                # Monte Carlo predictions (95% confidence interval)
                monte_carlo_plot_confidence_band(collective_ax_current, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                collective_ax_current.set_xlabel('Date')
                collective_ax_current.set_ylabel('Close')
                collective_ax_current.set_title("{} ({}) Predicted Closing Value ({} {} window)"
                                                .format(currency_label, currency_ticker, window_size,
                                                        model_time_unit_singular))
                collective_ax_current.legend()
                # Individual plot
                indiv_fig = plt.figure(figsize=(12, 6))
                indiv_fig_ax = indiv_fig.add_subplot(111)
                # Single Asset Model Predictions
                indiv_fig_ax.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                  color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                # Collective Model Predictions
                indiv_fig_ax.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                  color=ml_model_collective_color, label=ml_model_collective_label)
                # Monte Carlo predictions
                indiv_fig_ax.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                  color=monte_carlo_color, label=monte_carlo_label)
                # Monte Carlo predictions (95% confidence interval)
                monte_carlo_plot_confidence_band(indiv_fig_ax, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                indiv_fig_ax.set_xlabel('Date')
                indiv_fig_ax.set_ylabel('Close')
                indiv_fig_ax.set_title("{} ({}) Predicted Closing Value ({} {} window)"
                                       .format(currency_label, currency_ticker, window_size,
                                               model_time_unit_singular))
                indiv_fig_ax.legend()
                currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
                indiv_fig.savefig('{}/{}_predictions_{}.png'
                                  .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                                  dpi=figure_dpi)
            collective_fig.savefig('{}/predictions_{}.png'
                                   .format(figures_dir, window_size), dpi=figure_dpi)
            collective_fig.clf()

            # print("Plotting predictions and actual values together!")

            # Plot the predictions for the rest of 2017 and 2018 along with the actual values for the date range used.
            for currency_index in range(subset_num_currencies):
                currency_label = subset_currencies_labels[currency_index]
                currency_label_no_spaces = currency_label.replace(' ', '_')
                currency_ticker = subset_currencies_tickers[currency_index]
                label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
                # Collective plot
                collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
                # Actual values
                collective_ax_current.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                           color=actual_values_color, label=actual_values_label)
                # Single Asset Model Predictions
                collective_ax_current.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                           color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                # Collective Model Predictions
                collective_ax_current.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                           color=ml_model_collective_color, label=ml_model_collective_label)
                # Monte Carlo predictions
                collective_ax_current.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                           color=monte_carlo_color, label=monte_carlo_label)
                # Monte Carlo predictions (95% confidence interval)
                monte_carlo_plot_confidence_band(collective_ax_current, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                collective_ax_current.set_xlabel('Date')
                collective_ax_current.set_ylabel('Close')
                collective_ax_current.set_title("{} ({}) True + Predicted Closing Value ({} {} window)"
                                                .format(currency_label, currency_ticker, window_size,
                                                        model_time_unit_singular))
                collective_ax_current.legend()
                # Individual plot
                indiv_fig = plt.figure(figsize=(12, 6))
                indiv_fig_ax = indiv_fig.add_subplot(111)
                # Actual values
                indiv_fig_ax.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                  color=actual_values_color, label=actual_values_label)
                # Single Asset Model Predictions
                indiv_fig_ax.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                  color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                # Collective Model Predictions
                indiv_fig_ax.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                  color=ml_model_collective_color, label=ml_model_collective_label)
                # Monte Carlo predictions
                indiv_fig_ax.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                  color=monte_carlo_color, label=monte_carlo_label)
                # Monte Carlo predictions (95% confidence interval)
                monte_carlo_plot_confidence_band(indiv_fig_ax, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                indiv_fig_ax.set_xlabel('Date')
                indiv_fig_ax.set_ylabel('Close')
                indiv_fig_ax.set_title("{} ({}) True + Predicted Closing Value ({} {} window)"
                                       .format(currency_label, currency_ticker, window_size,
                                               model_time_unit_singular))
                indiv_fig_ax.legend()
                currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
                indiv_fig.savefig('{}/{}_actual_plus_predictions_{}.png'
                                  .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                                  dpi=figure_dpi)
            collective_fig.savefig('{}/actual_plus_predictions_{}.png'.format(figures_dir, window_size), dpi=figure_dpi)
        plt.close('all')

    if make_keras_param_set_occurrence_plots:
        # print("Making Keras param set occurrence plots!")

        with open(keras_window_size_scores_pickle_path, 'rb') as keras_window_size_scores_infile:
            keras_window_size_scores = pickle.load(keras_window_size_scores_infile)
        keras_best_window_size = max(keras_window_size_scores, key=keras_window_size_scores.get)
        # Used to record and analyze (via a pandas DataFrame) the number of occurrences of each parameter set.
        keras_param_sets_occurrences = {}
        for keras_param_set_occurrences_pickle_path in param_set_occurrences_pickle_paths:
            with open(keras_param_set_occurrences_pickle_path, "rb") as model_infile:
                param_set_occurrences_window = pickle.load(model_infile)
            for k,v in param_set_occurrences_window.items():
                keras_param_sets_occurrences[k] = keras_param_sets_occurrences.get(k,0) + param_set_occurrences_window[k]

        # Plot the number of occurrences for each parameter set for the window size with the best score.
        keras_unique_param_sets_tuples = list(keras_param_sets_occurrences.keys())
        col_names_to_add = [asset_col_str, window_size_col_str]  # Order matters here.
        index = pd.MultiIndex.from_tuples(keras_unique_param_sets_tuples, names=keras_param_names + col_names_to_add)
        keras_param_sets_occurrences_series = pd.Series(list(keras_param_sets_occurrences.values()), index=index)
        num_occurrences_str = 'num_occurrences'
        keras_param_sets_occurrences_frame = keras_param_sets_occurrences_series.to_frame(num_occurrences_str)
        keras_param_sets_occurrences_frame.reset_index(inplace=True)
        plotting_data = keras_param_sets_occurrences_frame.loc[
            keras_param_sets_occurrences_frame[window_size_col_str] == keras_best_window_size]
        plotting_data.drop(window_size_col_str, axis=1, inplace=True)
        plotting_index_headers = list(set(plotting_data.columns) - {num_occurrences_str, window_size_col_str, asset_col_str})
        index_col_str = ", ".join(plotting_index_headers)
        plotting_data[index_col_str] = plotting_data[plotting_index_headers].apply(tuple, axis=1)
        plotting_data.drop(plotting_index_headers, axis=1, inplace=True)
        plotting_data.sort_values(by=num_occurrences_str, ascending=False, inplace=True)
        fig, ax = plt.subplots()
        sns.barplot(x=index_col_str, y=num_occurrences_str, hue=asset_col_str, data=plotting_data, ax=ax)
        x_ticks_rotation_amt = 15
        plt.xticks(rotation=x_ticks_rotation_amt, fontsize=6)
        plt.title('Parameter set occurrences for window size {}'.format(keras_best_window_size))
        plt.tight_layout()
        plt.savefig('{}/param_set_occurrences_keras_{}_{}.png'
                    .format(keras_figures_dir, frst_hdn_lyr_sz_nn_single_asset,
                            frst_hdn_lyr_sz_nn_collective), dpi=figure_dpi)

if __name__ == '__main__':
    main()
