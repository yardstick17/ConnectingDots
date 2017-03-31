# -*- coding: utf-8 -*-
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.models import Sequential

CUSTOM_OUTPUT_CATEGORIES = 4
IMAGE_RESCALE_SIZE = 227


def DeepNet():
    # uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
    model = Sequential()

    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(3, IMAGE_RESCALE_SIZE, IMAGE_RESCALE_SIZE),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))

    model.add(Dense(CUSTOM_OUTPUT_CATEGORIES, name='dense_3'))
    model.add(Activation('softmax'))

    return model
