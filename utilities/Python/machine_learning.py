# Import miscellaneous global variables.
import sys
sys.path.insert(0,'../../globals/Python')
from globals import utilities_dir

# Custom utility functions
sys.path.insert(0,'../../' + utilities_dir)
from dict_ops import get_combinations

from sklearn.metrics import r2_score

import keras
from keras.models import Sequential
from keras.layers import Dense # TODO: Add Dropout, Convolution, and Max Pooling?
from sklearn.preprocessing import StandardScaler

import numpy as np

### Keras ###

def create_keras_regressor(input_dim, hidden_layer_sizes, output_dim, optimizer='adam', loss='mean_squared_error',
                           metrics=['accuracy'], **kwargs):
    """
    TODO: Document this function.
    :return:
    """
    # print("input_dim, hidden_layer_sizes, output_dim: ", input_dim, hidden_layer_sizes, output_dim)
    regressor = Sequential()
    # Adding the input layer and the first hidden layer
    regressor.add(Dense(units=hidden_layer_sizes[0], kernel_initializer='uniform', activation='relu',
                        input_dim=input_dim))
    # Adding the remaining hidden layers.
    for layer_size in hidden_layer_sizes:
        regressor.add(Dense(units=layer_size, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    regressor.add(Dense(units=output_dim, kernel_initializer='uniform'))
    # Compiling the ANN
    regressor.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return regressor


def keras_reg_grid_search(X, y, build_fn, input_dim, output_dim, param_grid,
                          epochs, scoring, cv, scale=True, verbose=0):
    """
    TODO: Document this function (`param_grid` must have a `batch_size` entry).
    :return:
    """
    def train_on_param_set(batch_size, param_set=dict()):
        """
        TODO: Document this function.
        :return:
        """
        model = None
        if len(param_set) == 0:
            model = build_fn(input_dim=input_dim, output_dim=output_dim)
        else:
            model = build_fn(input_dim=input_dim, output_dim=output_dim, **param_set)
        score = 0
        n_splits = 0
        # Train and test on the cross validation folds.
        for train_indices, test_indices in cv.split(X):
            # print("train_indices.shape, test_indices.shape: ",
            #       train_indices.shape, test_indices.shape)
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            if scale:
                X_scaler.fit(X_train)
                X_train = X_scaler.transform(X_train)
                X_test = X_scaler.transform(X_test)
                y_scaler.fit(y_train)
                y_train = y_scaler.transform(y_train)
                y_test = y_scaler.transform(y_test)
            # print("X_train[:5], y_train[:5]: ", X_train[:5], y_train[:5])
            # print("X_train.shape, y_train.shape: ", X_train.shape, y_train.shape)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            y_pred = model.predict(X_test)
            # y_pred = y_scaler.inverse_transform(y_pred)
            # print("y_test.shape, y_pred.shape: ", y_test.shape, y_pred.shape)
            # print("y_test, y_pred: ", np.stack([y_test, y_pred], axis=1))
            score += scoring(y_test, y_pred)
            n_splits += 1
        score = score / n_splits
        # print("score: ", score)
        return model, score

    # print("Keras model input shape: ", X.shape, y.shape)
    # Test a model for each parameter set.
    param_sets = get_combinations(param_grid)
    best_model = None
    best_score = -float('inf')
    best_batch_size = None
    for param_set in param_sets:
        batch_size = param_set['batch_size']
        # print("param_set: ", param_set)
        model, score = train_on_param_set(batch_size, param_set)
        if score > best_score:
            best_model = model
            best_score = score
            best_batch_size = batch_size
    print("best_score, best_batch_size: ", best_score, best_batch_size)
    return best_model, best_score, best_batch_size


### End Keras ###