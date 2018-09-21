from sklearn.preprocessing import StandardScaler

def scale_data_with_train(*args):
    """
    Standard scales data with a scaler trained only on the training data.

    Parameters
    ----------
    *args: list
        The data to scale, which could be the full data, training data, testing data, validation data,
        or some combination of those. The first argument must be the training data.

    Returns
    -------
    transformed_args_and_scaler: list
        The scaled versions of `*args`. The last member of this list
        is the scaler (an `sklearn.preprocessing.StandardScaler`).
    """
    scaler = StandardScaler().fit(args[0])
    return list(map(lambda data: scaler.transform(data), args)) + [scaler]