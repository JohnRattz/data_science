import sys
# Import miscellaneous global variables.
sys.path.insert(0,'../../globals/Python')
from globals import utilities_dir

# Custom utilities
sys.path.insert(0,'../../' + utilities_dir)
from dict_ops import get_combinations
from plotting import KerasPlotLosses

from sklearn.metrics import r2_score

import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

### Keras ###

def keras_init(gpu_mem_frac=0.3):
    """
    Configures keras at the beginning of script.

    Parameters
    ----------
    gpu_mem_frac: float
        The fraction of GPU memory allocated to the TensorFlow backend.
    """
    # Remove TensorFlow (Keras' default backend) debugging prints.
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Configure the TensorFlow backend.
    config = tf.ConfigProto()
    # Limit the GPU memory allocated to TensorFlow.
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac
    set_session(tf.Session(config=config))


def create_keras_regressor(input_dim, hidden_layer_sizes, output_dim,
                           optimizer=optimizers.Adam, optimizer_params={},
                           loss='mean_squared_error',
                           metrics=[], **kwargs):
    """
    TODO: Document this function.
    TODO: Use max norm (Dense(kernel_constraint=maxnorm(m)))?
    :return:
    """
    dropout_rate = kwargs.get('dropout_rate', None)
    # Construct the optimizer.
    optimizer = optimizer(**optimizer_params)

    # print("input_dim, hidden_layer_sizes, output_dim: ", input_dim, hidden_layer_sizes, output_dim)
    regressor = Sequential()
    # Add dropout for input layer.
    if dropout_rate is not None:
        regressor.add(Dropout(dropout_rate, input_shape=(input_dim,)))
    # Adding the input layer and the first hidden layer
    regressor.add(Dense(units=hidden_layer_sizes[0], kernel_initializer='uniform', activation='relu',
                        input_dim=input_dim))
    # Adding the remaining hidden layers.
    for layer_size in hidden_layer_sizes:
        if dropout_rate is not None:
            regressor.add(Dropout(dropout_rate))
        regressor.add(Dense(units=layer_size, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    regressor.add(Dense(units=output_dim, kernel_initializer='uniform'))
    # Compiling the ANN
    regressor.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return regressor


def extract_optimizer_params(optimizer, param_set):
    """
    Extracts the optimizer parameters (except 'optimizer') in a parameter set being trained on.
    The corresponding entries in the parameter set dictionary are not removed.

    Parameters
    ----------
    optimizer: keras.optimizers.Optimizer
        A Keras optimizer, such as the Adam optimizer.
    param_set: dict
        A dictionary of parameter names and values to be tested - one value per parameter.

    Returns
    -------
    optimizer_params: dict
        A dictionary containing any matched optimizer parameters in `param_set`.
    """
    optimizer_params = {}
    # Specify the names (keys) of parameters we are looking for in kwargs.
    optimizer_param_names = []
    if optimizer is optimizers.Adam:
        optimizer_param_names = ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay', 'amsgrad']
    # Extract the parameters.
    for optimizer_param_name in [param_name for param_name in param_set
                                 if param_name in optimizer_param_names]:
        optimizer_params[optimizer_param_name] = param_set[optimizer_param_name]
    return optimizer_params

def extract_non_optimizer_params(optimizer, param_set):
    """
    Extracts the non-optimizer parameters in a parameter set being trained on.
    The corresponding entries in the parameter set dictionary are not removed.

    Parameters
    ----------
    optimizer: keras.optimizers.Optimizer
        A Keras optimizer, such as the Adam optimizer.
    param_set: dict
        A dictionary of parameter names and values to be tested - one value per parameter.

    Returns
    -------
    non_optimizer_params: dict
        A dictionary containing any matched non-optimizer parameters in `param_set`.
    """
    non_optimizer_params = {}
    # Specify the names (keys) of parameters we are looking for in kwargs.
    optimizer_param_names = ['optimizer']
    if optimizer is optimizers.Adam:
        optimizer_param_names += ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay', 'amsgrad']
    # param_names = list(param_set.keys())
    # non_optimizer_param_names = list(set(param_names) - set(optimizer_param_names))
    # Extract the parameters.
    for non_optimizer_param_name in [param_name for param_name in param_set
                                     if param_name not in optimizer_param_names]:
        non_optimizer_params[non_optimizer_param_name] = param_set[non_optimizer_param_name]
    return non_optimizer_params

def keras_reg_grid_search(X, y, build_fn, output_dim, param_grid, epochs, cv=None, scoring=r2_score, scale=True,
                          verbose=0, plot_losses=False, plotting_dir=None, figure_title_prefix="", figure_kwargs={}):
    """
    TODO: Document this function (X, y, build_fn, cv).

    Parameters
    ----------
    output_dim: int
        The number of outputs of the neural network.
    param_grid: dict
        A dictionary of parameter names and values to be tested.
        Must contain 'batch_size' and 'hidden_layer_sizes` entries.
    epochs: int
        The number of epochs to run during training.
    scoring: function
        A scoring function, like ``sklearn.metrics.r2_score``.
    scale: bool
        Whether or not to standard scale the data before training.
    verbose: int
        0, 1, or 2. 0 = silent, 1 = updates on grid search (# param sets completed),
        2 = Keras verbosity 1 (progress bar), 3 = Keras verbosity 2 (one line per epoch).
    plot_losses: boolean
        Whether or not to plot losses for all param sets trained on in one figure.
        If ``True``, ``plotting_dir`` must not be ``None``, and ``cv`` must be ``None``.
    plotting_dir: str
        The directory in which to store loss plots.
    figure_title_prefix: str
        A string to prefix to the figure title.
    figure_kwargs: dict
        A ``dict`` of keyword arguments for construction of a matplotlib Figure for the loss plot.
    """
    # Verbosity for Keras fit().
    keras_verbose = max(0, verbose - 1)
    loss_plotters = {} if plot_losses else None # Dictionary mapping non-optimizer parameter values to loss plotters.

    def train_on_param_set(batch_size, hidden_layer_sizes, param_set):
        """
        TODO: Document this function.
        :return:
        """
        optimizer = param_set.get('optimizer')
        optimizer_params = extract_optimizer_params(optimizer, param_set)
        # print("optimizer_params:", optimizer_params)
        non_optimizer_params = extract_non_optimizer_params(optimizer, param_set)
        # print("non_optimizer_params:", non_optimizer_params)
        non_optimizer_param_vals = tuple(non_optimizer_params.values())
        # print("non_optimizer_param_vals:", non_optimizer_param_vals)
        # Select the appropriate loss_plotter or create if needed.
        loss_plotter = loss_plotters.get(non_optimizer_param_vals, None) if plot_losses else None
        if plot_losses and (loss_plotter is None):
            loss_plotter = loss_plotters[non_optimizer_param_vals] = \
                KerasPlotLosses(epochs, plotting_dir, figure_title_prefix, **figure_kwargs) if plot_losses else None
        # print("id(loss_plotter):", id(loss_plotter))
        # Build the model.
        model_building_param_set = param_set.copy()
        model_building_param_set.pop('batch_size', None)
        model_building_param_set.pop('hidden_layer_sizes', None)
        model = build_fn(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes,
                         output_dim=output_dim, optimizer_params=optimizer_params, **model_building_param_set)
        score = 0
        n_splits = 0
        if cv is not None:
            # Train and test on the cross validation folds.
            for train_indices, test_indices in cv.split(X):
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                if scale:
                    X_scaler = StandardScaler()
                    y_scaler = StandardScaler()
                    X_scaler.fit(X_train)
                    X_train = X_scaler.transform(X_train)
                    X_test = X_scaler.transform(X_test)
                    y_scaler.fit(y_train)
                    y_train = y_scaler.transform(y_train)
                    y_test = y_scaler.transform(y_test)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=keras_verbose)
                y_pred = model.predict(X_test)
                score += scoring(y_test, y_pred)
                n_splits += 1
            score = score / n_splits
        else: # Train without cross validation.
            if plot_losses:
                loss_plotter.set_optimizer(optimizer)
                loss_plotter.set_optimizer_params(optimizer_params)
                loss_plotter.set_non_optimizer_params(non_optimizer_params)
            X_scaled, y_scaled = [None]*2
            if scale:
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
                X_scaler.fit(X)
                X_scaled = X_scaler.transform(X)
                y_scaler.fit(y)
                y_scaled = y_scaler.transform(y)
            model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=batch_size, verbose=keras_verbose,
                      callbacks=[loss_plotter])
            y_pred = model.predict(X)
            score = scoring(y,y_pred)
        return model, score

    input_dim = X.shape[1]

    # print("Keras model input shape: ", X.shape, y.shape)
    # Test a model for each parameter set.
    param_sets = get_combinations(param_grid)
    best_model = None
    best_score = -float('inf')
    best_batch_size = None
    # Fraction of parameter sets trained with
    frac_param_sets_cmplt = 0.
    # Fraction of parameter sets trained with as of the last print
    frac_param_sets_cmplt_lst_prt = 0.
    num_chars_prg_bar = 10  # The number of characters within the progress bar.
    # Minimum difference in the fraction of parameter sets completed to print
    frac_param_sets_cmplt_min_diff_prt = 1.0/num_chars_prg_bar
    for param_set in param_sets:
        batch_size = param_set.get('batch_size')
        hidden_layer_sizes = param_set.get('hidden_layer_sizes')
        model, score = train_on_param_set(batch_size, hidden_layer_sizes, param_set)
        if score > best_score:
            best_model = model
            best_score = score
            best_batch_size = batch_size
        if verbose >= 1:
            frac_param_sets_cmplt += 1.0 / len(param_sets)
            if frac_param_sets_cmplt >= frac_param_sets_cmplt_lst_prt + frac_param_sets_cmplt_min_diff_prt:
                frac_param_sets_cmplt_lst_prt = frac_param_sets_cmplt
                prg_bar_inner_str = ('>'*int(round(num_chars_prg_bar*frac_param_sets_cmplt))).ljust(num_chars_prg_bar)
                # TODO: Print over the same line in a multi-platform and reasonably terminal-agnostic way in Python3.
                print('[{}] Completed {:%} of param sets'
                      .format(prg_bar_inner_str, frac_param_sets_cmplt))
    # TODO: Train best model on full dataset after cross validation (if `cv is None`, this is unnecessary).
    if plot_losses:
        for loss_plotter in loss_plotters.values():
            loss_plotter.save_figure()
    return best_model, best_score, best_batch_size

### End Keras ###