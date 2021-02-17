from scipy.io import loadmat
mnist = loadmat('D:\machine learning\projects\mnist digits\mnist-original.mat')

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
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

classifier = Sequential()

classifier.add(Dense(256, activation = 'relu', input_shape = (784,), kernel_initializer = 'uniform'))
classifier.add(Dense(256, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(10, activation = 'softmax', kernel_initializer = 'uniform'))

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, epochs = 25)

classifier.evaluate(X_test, y_test)