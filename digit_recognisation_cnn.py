import numpy as np
import pandas as pd
import matplotlib.pyplot

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(64, (3,3), activation = 'relu', input_shape = (28, 28, 1)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory(directory = 'trainingSet/trainingSet',
                                                 target_size = (28,28),
                                                 color_mode = 'grayscale',
                                                 class_mode = 'categorical',
                                                 batch_size = 32)

classifier.fit_generator(generator = training_set, epochs = 30, steps_per_epoch = 10)
import imageio
newimage = np.array(imageio.imread('testSet//testSet//img_453.jpg')).reshape(1,28,28,1)
y_pred = classifier.predict(newimage)
category = np.argmax(y_pred)