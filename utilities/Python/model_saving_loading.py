import tempfile

import keras.models
from keras.models import load_model, model_from_json
from sklearn.externals import joblib


### Keras ###

# def make_keras_picklable():
#     """
#     Makes Keras models (specifically, `keras.models.Model`) picklable (not Keras wrappers).
#     Credit to http://zachmoshe.com/2017/04/03/pickling-keras-models.html
#     """
#     def __getstate__(self):
#         model_str = ""
#         with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
#             keras.models.save_model(self, fd.name, overwrite=True)
#             model_str = fd.read()
#         d = { 'model_str': model_str }
#         return d
#
#     def __setstate__(self, state):
#         with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
#             fd.write(state['model_str'])
#             fd.flush()
#             model = keras.models.load_model(fd.name)
#         self.__dict__ = model.__dict__
#
#     cls = keras.models.Model
#     cls.__getstate__ = __getstate__
#     cls.__setstate__ = __setstate__

# def save_keras_model(model, model_filepath, weights_filepath):
#     """
#     # TODO: Document this method.
#     Save a Keras model to the file system.
#     Credit goes to `this article. <https://machinelearningmastery.com/save-load-keras-deep-learning-models>`_
#     Parameters
#     ----------
#     model:
#     model_filepath: str
#         The base path to the model JSON file. Omit the '.json' extension.
#     weights_filepath: str
#         The base path to the weights HDF5 file. Omit the '.h5' extension.
#     """
#     keras.models
#     # Serialize model to JSON
#     model_json = model.to_json()
#     with open(model_filepath + ".json", "w") as json_file:
#         json_file.write(model_json)
#     # Serialize weights to HDF5
#     model.save_weights(weights_filepath + ".h5")
#
# def load_keras_model(model_filepath, weights_filepath):
#     """
#     Load a Keras model from the file system.
#     Credit goes to `this article. <https://machinelearningmastery.com/save-load-keras-deep-learning-models>`_
#
#     Parameters
#     ----------
#     model_filepath: str
#         The base path to the model JSON file. Omit the '.json' extension.
#     weights_filepath: str
#         The base path to the weights HDF5 file. Omit the '.h5' extension.
#     """
#     # load json and create model
#     json_file = open(model_filepath + '.json', 'r')
#     model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(model_json)
#     # load weights into new model
#     model.load_weights(weights_filepath + '.h5')
#     return model


### End Keras ###