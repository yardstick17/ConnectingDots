#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.models import Sequential

from neural_networks.convolutional_neural_network.networks.spp_pooling.spatial_pyramid_pooling import \
    SpatialPyramidPooling

CUSTOM_OUTPUT_CATEGORIES = 2


def Spp():
    # uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
    model = Sequential()

    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(3, None, None), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(CUSTOM_OUTPUT_CATEGORIES, name='dense_3'))
    model.add(Activation('softmax'))
    return model
