from sklearn.preprocessing import StandardScaler

def scale_data_with_train(*args):
    """
    Standard scales data with a scaler trained only on the training data.

    Returns
    -------
    *args: list
        The data to scale, which could be the full data, training data, testing data, validation data,
        or some combination of those. The first argument must be the training data.
    scaler: sklearn.preprocessing.StandardScaler
        The sklearn scaler used to scale the training and testing data.
    """
    scaler = StandardScaler().fit(args[0])
    return list(map(lambda data: scaler.transform(data), args)) + [scaler]