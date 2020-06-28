import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open("../data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vector(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vector(y) for y in va_d[1]]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vector(y) for y in te_d[1]]
    test_data = list(zip(test_inputs, test_results))

    return training_data, validation_data, test_data

def vector(y):
    v = np.zeros((10, 1))
    v[y] = 1.0
    return v