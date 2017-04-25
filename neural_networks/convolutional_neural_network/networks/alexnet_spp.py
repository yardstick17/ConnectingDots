# -*- coding: utf-8 -*-
from keras.engine import Input
from keras.engine import merge
from keras.engine import Model
from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Sequential

from neural_networks.convolutional_neural_network.networks.spp_pooling.spatial_pyramid_pooling import \
    SpatialPyramidPooling

CUSTOM_OUTPUT_CATEGORIES = 2


def AlextNetSpp():
    inputs = Input(shape=(3, None, None), name='Input Layer')

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2), name='Max-Pool-1')(conv_1)
    conv_2 = cross_channel_normalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2), name='zero-padding-1')(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2), name='max-pooling')(conv_2)
    conv_3 = cross_channel_normalization()(conv_3)

    conv_3 = ZeroPadding2D((1, 1), name='zero-padding-2')(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1), name='zero-padding-3')(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1), name='zero-padding-4')(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    # dense_1 = MaxPooling2D(kernal_size, strides=kernal_size, name='max-pool-convpool_5')(conv_5)
    dense_1 = SpatialPyramidPooling([1, 2, 4])(conv_5)

    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(CUSTOM_OUTPUT_CATEGORIES, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction)

    return model


def Spp():
    # uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
    model = Sequential()

    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(3, None, None), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(CUSTOM_OUTPUT_CATEGORIES, name='dense_3'))
    model.add(Activation('softmax'))
    return model
