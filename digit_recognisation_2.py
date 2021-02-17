from scipy.io import loadmat
mnist = loadmat('mnist-original.mat')

data = mnist['data'].T
label = mnist['label'].T
label = label.reshape(len(label),)

import matplotlib.pyplot as plt
import imageio
import numpy as np

X_train, X_test, y_train, y_test = data[:60000], data[60000:], label[:60000], label[60000:]
shuffle_index_train = np.random.permutation(60000)
shuffle_index_test = np.random.permutation(10000)

X_train, y_train = X_train[shuffle_index_train], y_train[shuffle_index_train]
X_test, y_test = X_test[shuffle_index_test], y_test[shuffle_index_test]

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

def neuralnetwork():
    classifier = Sequential()
    
    classifier.add(Dense(256, activation = 'relu', input_shape = (784,), kernel_initializer = 'uniform'))
    classifier.add(Dense(256, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(10, activation = 'softmax', kernel_initializer = 'uniform'))
    
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

network = KerasClassifier(build_fn = neuralnetwork, epochs = 10, batch_size = 32)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = network, X = X_train, y = y_train, cv = 3)