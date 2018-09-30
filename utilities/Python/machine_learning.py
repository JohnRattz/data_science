import sys
# Import miscellaneous global variables.
sys.path.insert(0,'../../globals/Python')
from globals import utilities_dir

# Custom utilities
sys.path.insert(0,'../../' + utilities_dir)
from dict_ops import get_dict_combinations
from plotting import KerasPlotLosses
from scale import scale_data_with_train

from sklearn.metrics import r2_score

import tensorflow as tf

import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras import optimizers
from sklearn.preprocessing import StandardScaler
import numpy as np

### Keras ###

def keras_init(tf_min_log_level='3', gpu_mem_frac=0.3):
    """
    Configures keras and its TensorFlow backend. Usually placed at the beginning of a script.

    Parameters
    ----------
    tf_min_log_level: str
        A string representation of an integer. One of ['1', '2', '3'].
        Higher numbers for this parameter allow fewer prints from the TensorFlow backend.
    gpu_mem_frac: float
        The fraction of GPU on-board memory allocated to the TensorFlow backend.
        Note that models_keras_rnn_512_1024_20e_20e that do not use TensorFlow components that allow memory swapping between GPU and system memory,
        such as those used by Keras layers like `Sequential`, may cause OOM errors preventing models_keras_rnn_512_1024_20e_20e from being trained
        and potentially crashing the Python interpreter with an uncaught and otherwise difficult-to-handle exception.
    """
    # Remove TensorFlow (Keras' default backend) debugging prints.
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_min_log_level

    # Configure the TensorFlow backend.
    config = tf.ConfigProto()
    # Limit the GPU memory allocated to TensorFlow.
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac
    set_session(tf.Session(config=config))


def create_keras_regressor(input_shape, hidden_layer_sizes, output_dim, optimizer=optimizers.Adam,
                           optimizer_params={}, hidden_layer_type='Dense', loss='mean_squared_error',
                           metrics=[], **kwargs):
    """
    TODO: Document this function.
    TODO: Generalize this to classification.
    TODO: Use max norm (Dense(kernel_constraint=maxnorm(m)))?
    """
    # print("hidden_layer_type:", hidden_layer_type)
    dropout_rate = kwargs.get('dropout_rate', None)
    # Construct the optimizer.
    optimizer = optimizer(**optimizer_params)

    # TODO: Instead of `input_dim`, determine based on shape of `X_train` and `layer_type`.

    regressor = Sequential()
    # Add the hidden layers.
    for layer_num, layer_size in enumerate(hidden_layer_sizes):
        # These are the parameters for this hidden layer. The parameters here are common to all hidden layers.
        params = dict(units=layer_size)
        if layer_num == 0: # Specify the input shape for the first hidden layer.
            params['input_shape'] = input_shape
        # Create this layer.
        if hidden_layer_type == "Dense":
            params = {**params, **dict(kernel_initializer='uniform', activation='relu')}
            regressor.add(Dense(**params))
        elif hidden_layer_type == "LSTM":
            if layer_num < len(hidden_layer_sizes) - 1:  # If this is not the last hidden layer...
                params['return_sequences'] = True
            regressor.add(LSTM(**params))
        # Add a dropout layer.
        if dropout_rate is not None:
            regressor.add(Dropout(dropout_rate))
    # Adding the output layer and compiling the network.
    regressor.add(Dense(units=output_dim, kernel_initializer='uniform'))
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
    # Extract the parameters.
    for non_optimizer_param_name in [param_name for param_name in param_set
                                     if param_name not in optimizer_param_names]:
        non_optimizer_params[non_optimizer_param_name] = param_set[non_optimizer_param_name]
    return non_optimizer_params

def keras_reg_grid_search(X_dict, y, build_fn, param_grid, epochs, cv_epochs=None, cv=None,
                          scoring=r2_score, verbose=0, plot_losses=False,
                          plotting_dir=None, figure_title_prefix="", figure_kwargs={}, plotting_kwargs={}):
    """
    TODO: Why use `build_fn` when there is only one sensible build function to call?
    TODO: Document this function (X_dict, y, build_fn).
    TODO: Try to sleep the calling thread on Keras model `fit()` calls (low priority)?
    TODO: Plot losses for each param set or the best param set (add parameter to specify).

    Parameters
    ----------
    param_grid: dict
        A dictionary of parameter names and values to be tested.
        Must contain 'batch_size' and 'hidden_layer_sizes` entries.
    epochs: int
        The number of epochs to run during grid search without cross validation (i.e. ``cv is None``) or on the best
        parameter set found by grid search with cross validation.
    cv_epochs: int
        The number of epochs to run for each cross validation fold during grid search if cross validation is specified
        (i.e. ``cv is not None``). Defaults to ``epochs``.
    cv: cross-validation generator (e.g. `sklearn.model_selection.KFold`)
        An object to be used as a cross-validation generator.
    scoring: function
        A scoring function, like ``sklearn.metrics.r2_score``.
    verbose: int
        Can be 0, 1, 2, or 3. 0 = silent, 1 = updates on grid search (# param sets completed),
        2 = Keras verbosity 1 (progress bar), 3 = Keras verbosity 2 (one line per epoch).
    plot_losses: boolean
        Whether or not to plot losses for all param sets trained on in one figure.
        If ``True``, ``plotting_dir`` must not be ``None``, and ``cv`` must be ``None``.
    plotting_dir: str
        The directory in which to store loss plots.
    figure_title_prefix: str
        A string to prefix the loss plot figure title.
    figure_kwargs: dict
        A ``dict`` of keyword arguments for construction of a matplotlib Figure for the loss plot.
    plotting_kwargs: dict
        A ``dict`` of keyword arguments for the plotting of the loss plot (`matplotlib.pyplot.plot()`).
    """

    def create_model_from_param_set(param_set, optimizer_params, build_fn):
        """
        TODO: Document this function.
        """
        model_building_param_set = param_set.copy()
        model_building_param_set.pop('batch_size', None)
        hidden_layer_type = model_building_param_set['hidden_layer_type']
        input_shape = None
        if hidden_layer_type == 'Dense':
            input_shape = X_dict[hidden_layer_type].shape
        elif hidden_layer_type == 'LSTM':
            # For Keras RNNs, the shape is (batch_size, timesteps, input_dim).
            # See https://keras.io/layers/recurrent/#rnn for more information.
            input_shape = X_dict[hidden_layer_type].shape[1:]
        output_dim = y.shape[1]
        hidden_layer_sizes = model_building_param_set.pop('hidden_layer_sizes', None)
        model = build_fn(input_shape=input_shape, hidden_layer_sizes=hidden_layer_sizes,
                         output_dim=output_dim, optimizer_params=optimizer_params, **model_building_param_set)
        return model

    def train_on_param_set(param_set):
        """
        TODO: Document this function.
        """
        # Extract the appropriately formatted data.
        hidden_layer_type = param_set['hidden_layer_type']
        X = X_dict[hidden_layer_type]
        nonlocal y

        # Extract some elements of the parameter set.
        batch_size = param_set.get('batch_size')
        optimizer = param_set.get('optimizer')
        optimizer_params = extract_optimizer_params(optimizer, param_set)

        model = create_model_from_param_set(param_set, optimizer_params, build_fn)
        non_optimizer_params = extract_non_optimizer_params(optimizer, param_set)
        non_optimizer_param_vals = tuple(non_optimizer_params.values())
        # Select the appropriate loss_plotter or create one if needed.
        loss_plotter = loss_plotters.get(non_optimizer_param_vals, None) if plot_losses else None
        if plot_losses and (loss_plotter is None):
            loss_plotter = loss_plotters[non_optimizer_param_vals] = \
                KerasPlotLosses(epochs, plotting_dir, figure_title_prefix, figure_kwargs,
                                plotting_kwargs) if plot_losses else None
        score = 0
        n_splits = 0
        if cv is not None:
            # Train and test on the cross validation folds.
            for train_indices, test_indices in cv.split(X):
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                # if scale:
                ## CV Scaling ##
                # Data shape can vary based on parameters like 'hidden_layer_type',
                # so the data must be reshaped for `StandardScaler`, then restored to its original shape.
                X_train_reshaped, X_test_reshaped = reshape_data_from_keras_to_2D(X_train, X_test)
                X_train_reshaped, X_test_reshaped, X_scaler = scale_data_with_train(X_train_reshaped, X_test_reshaped)
                X_train, X_test = X_train_reshaped.reshape(X_train.shape), X_test_reshaped.reshape(X_test.shape)
                y_train_reshaped, y_test_reshaped = reshape_data_from_keras_to_2D(y_train, y_test)
                y_train_reshaped, y_test_reshaped, y_scaler = scale_data_with_train(y_train_reshaped, y_test_reshaped)
                y_train, y_test = y_train_reshaped.reshape(y_train.shape), y_test_reshaped.reshape(y_test.shape)
                ## End CV Scaling ##
                model.fit(X_train, y_train, epochs=cv_epochs, batch_size=batch_size, verbose=keras_verbose)
                y_pred = model.predict(X_test)
                # TODO: Find out why the following error is thrown for large batch sizes (specifically with RNNs?):
                # TODO: ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
                score += scoring(y_test, y_pred)
                n_splits += 1
            score = score / n_splits
        else: # Train without cross validation.
            if plot_losses:
                loss_plotter.set_optimizer(optimizer)
                loss_plotter.set_optimizer_params(optimizer_params)
                loss_plotter.set_non_optimizer_params(non_optimizer_params)
            # if scale:
            #     X_reshaped = reshape_data_from_keras_to_2D(X)[0]
            #     X_reshaped, X_scaler = scale_data_with_train(X_reshaped)
            #     X = X_reshaped.reshape(X.shape)
            #     y_reshaped = reshape_data_from_keras_to_2D(y)[0]
            #     y_reshaped, y_scaler = scale_data_with_train(y_reshaped)
            #     y = y_reshaped.reshape(y.shape)
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=keras_verbose,
                      callbacks=[loss_plotter] if plot_losses else None)
            y_pred = model.predict(X)
            score = scoring(y,y_pred)
        del model
        K.clear_session()  # Deallocate models_keras_rnn_512_1024_20e_20e to free GPU memory.
        return score, batch_size, optimizer_params

    cv_epochs = epochs if cv_epochs is None else cv_epochs

    # Verbosity for Keras fit().
    keras_verbose = max(0, verbose - 1)
    loss_plotters = {} if plot_losses else None  # Dictionary mapping non-optimizer parameter values to loss plotters.

    # Test a model for each parameter set.
    param_sets = get_dict_combinations(param_grid)
    best_score = -float('inf')
    best_param_set = None
    best_batch_size = None
    best_optimizer_params = None
    # Fraction of parameter sets trained with
    frac_param_sets_cmplt = 0.
    # Fraction of parameter sets trained with as of the last print
    frac_param_sets_cmplt_lst_prt = 0.
    num_chars_prg_bar = 10  # The number of characters within the progress bar
    # Minimum difference in the fraction of parameter sets completed to print
    frac_param_sets_cmplt_min_diff_prt = 1.0/num_chars_prg_bar

    for param_set in param_sets:
        score, batch_size, optimizer_params = train_on_param_set(param_set)
        if score > best_score:
            # best_model = model
            best_score = score
            best_param_set = param_set
            best_batch_size = batch_size
            best_optimizer_params = optimizer_params
        if verbose >= 1:
            frac_param_sets_cmplt += 1.0 / len(param_sets)
            if frac_param_sets_cmplt >= frac_param_sets_cmplt_lst_prt + frac_param_sets_cmplt_min_diff_prt:
                frac_param_sets_cmplt_lst_prt = frac_param_sets_cmplt
                prg_bar_inner_str = ('>'*int(round(num_chars_prg_bar*frac_param_sets_cmplt))).ljust(num_chars_prg_bar)
                # TODO: Print over the same line in a multi-platform and reasonably terminal-agnostic way in Python3.
                print('[{}] Completed {:%} of param sets'
                      .format(prg_bar_inner_str, frac_param_sets_cmplt))
    hidden_layer_type = best_param_set['hidden_layer_type']
    X = X_dict[hidden_layer_type]
    # Train the best model on the full dataset.
    best_model = create_model_from_param_set(best_param_set, best_optimizer_params, build_fn)
    # if scale:
    #     X_reshaped = reshape_data_from_keras_to_2D(X)[0]
    #     X_reshaped, X_scaler = scale_data_with_train(X_reshaped)
    #     X = X_reshaped.reshape(X.shape)
    #     y_reshaped = reshape_data_from_keras_to_2D(y)[0]
    #     y_reshaped, y_scaler = scale_data_with_train(y_reshaped)
    #     y = y_reshaped.reshape(y.shape)
    best_model.fit(X, y, epochs=epochs, batch_size=best_batch_size, verbose=keras_verbose)
    if plot_losses:
        for loss_plotter in loss_plotters.values():
            loss_plotter.save_figure()
    return best_model, best_score, best_param_set

def keras_convert_optimizer_obj_to_name(optimizer):
    """
    Returns a simple string name for a Keras optimizer object.
    """
    if optimizer is optimizers.Adam:
        return 'Adam'

### End Keras ###

def reshape_data_from_2D_to_keras(params, output_dim, *args):
    """
    Reshapes data to a 2D representation based on values in a parameter set or grid, such as 'hidden_layer_type'.
    Notably, sklearn scalers such as `StandardScaler` require their data to be in 2D representation.
    This should work for reshaping data of both `X` (feature matrix) and `y` ("label" matrix).

    Parameters
    ----------
    params: dict
        Either a parameter set or a parameter grid.
    output_dim: int
        The number of outputs of the neural network.
    *args:
        The data to reshape, which could be the full data, training data, testing data, validation data,
        or some combination of those.

    Returns
    -------
    reshaped_data: list or dict
        If `params` is a parameter set, a list of the same data as in `args`, but reshaped.
        If `params`is a parameter grid, a dictionary mapping every hidden layer type to its reshaped data.
    """
    def get_data(hidden_layer_type):
        """
        Given a hidden layer type, return the properly formatted data.
        """
        if hidden_layer_type == 'Dense':
            return args[0] if len(args) == 1 else args
        if hidden_layer_type == 'LSTM':
            data_reshaped = list(map(lambda data: data.reshape(data.shape[0], int(data.shape[1]/output_dim), output_dim), args))
            return data_reshaped[0] if len(data_reshaped) == 1 else data_reshaped

    # If `params` is a parameter dictionary.
    if isinstance(list(params.values())[0], list):
        output_dict = {}
        # TODO: Does the shape alteration code depend on whether `X` or `y` is used for a given entry in `args`?
        if 'Dense' in params['hidden_layer_type']:
            output_dict['Dense-'] = get_data('Dense')
        if 'LSTM' in params['hidden_layer_type']:
            output_dict['LSTM'] = get_data('LSTM')
        return output_dict
    else: # If `params` is a parameter set.
        return get_data(params['hidden_layer_type'])

def reshape_data_from_keras_to_2D(*args, param_set=None):
    """
    Reshapes data to a 2D representation based on values in the parameter set, such as 'hidden_layer_type'.
    Notably, sklearn scalers such as `StandardScaler` require their data to be in 2D representation.
    Currently, the same code works for reshaping data for both `X` (feature matrix) and `y` ("label" matrix).

    Parameters
    ----------
    *args: list
        The data to reshape, which could be the full data, training data, testing data, validation data,
        or some combination of those.
    param_set: dict
        A dictionary of parameter names and values.

    Returns
    -------
    reshaped_data: list
        The reshaped data - same length as `args`.
    """
    # So far, `param_set` has not been needed to determine how to reshape.
    return list(map(lambda data: data.reshape(data.shape[0], -1), args))