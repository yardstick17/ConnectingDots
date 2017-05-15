# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from neural_networks.convolutional_neural_network.convnet_keras.convnetskeras.convnets import convnet
from neural_networks.convolutional_neural_network.networks.image_quality import Spp
from neural_networks.convolutional_neural_network.pre_processing.image_preprocessing_for_spp import \
    preprocess_image_batch
#from sklearn.model_selection import train_test_split

PREFIX_DIR_LOCATION = os.path.expanduser('~/training_data/eyeq/')
IMAGE_MODEL_CHECKPOINT = os.path.join(os.path.expanduser('~/ConnectingDots'),
                                      'eyeq_model.model.checkpoint.h5')

IMAGE_MODEL_WEIGHTS = os.path.join(os.path.expanduser('~/ConnectingDots'), 'eyeq_model.model.weights.h5')
BEST_IMAGE_MODEL = os.path.expanduser('~/ConnectingDots/eyeq_model_best_accuracy.h5')

DATA_SRC_DIRECTORY = os.path.join(PREFIX_DIR_LOCATION,
                                  'zomato_datasets')
CLASS_NAME_MAPPING = {}
# IMAGE_RESCALE_SIZE = DeepNet.IMAGE_RESCALE_SIZE
CSV_LOG_FILENAME = os.path.join(os.path.expanduser('~/ConnectingDots'), 'eyeq_model_train_log.csv')
PER_CLASS_MAX_IMAGES = 50000
NB_EPOCH = 3
NETWORK_MODEL = 'alexnet'


def get_model(NETWORK_MODEL, nb_classes):
    return convnet(NETWORK_MODEL, nb_classes)


def get_data():
    data_set_input_images_files, data_set_input_images_true_label = get_class_wise_images_and_true_label()
    processed_input_images_dict, image_lable_dict = preprocess_image_batch(data_set_input_images_files,
                                                                           data_set_input_images_true_label)
    return processed_input_images_dict, image_lable_dict


def get_class_wise_images_and_true_label():
    data_sources = glob(PREFIX_DIR_LOCATION + '/*')
    data_set_input_images = []
    data_set_input_images_true_label = []
    global CLASS_NAME_MAPPING

    for directory in data_sources:
        directory = glob(directory + '/*')
        print('directory;', directory)
        index = 0
        for sub_directory in directory:
            if os.path.isdir(sub_directory):
                class_dir_name = sub_directory.split('/')[-1]
                CLASS_NAME_MAPPING[index] = class_dir_name
                image_class_files = glob(sub_directory + '/*.jpg')[:PER_CLASS_MAX_IMAGES]
                data_set_input_images.extend(image_class_files)
                data_set_input_images_true_label.extend([[index]] * len(image_class_files))
                index += 1
    return data_set_input_images, data_set_input_images_true_label


def get_callbacks():
    callbacks = []
    import keras
    callbacks.append(keras.callbacks.CSVLogger(CSV_LOG_FILENAME,append=True))
    callbacks.append(keras.callbacks.ModelCheckpoint(
        IMAGE_MODEL_CHECKPOINT,
        monitor='val_loss',
        # verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto')
    )
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=BEST_IMAGE_MODEL,
        verbose=1,
        save_best_only=True,
    ))
    return callbacks


def train_network():
    global CLASS_NAME_MAPPING
    nb_classes = len(CLASS_NAME_MAPPING.keys())
    print('processing data for training')
    # X_train, Y_train, X_test, Y_test, nb_classes = get_data()
    processed_input_images_dict, image_lable_dict = get_data()
    print('getting model')
    model = Spp()

    sgd = SGD(lr=0.0001)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.summary()
    callbacks = get_callbacks()

    print('fitting model for the dataset')

    iteration = 0
    try:
        while (iteration < 200):
            iteration += 1
            print('Iteration :: ', iteration)
            for output_shape in processed_input_images_dict:
                print('Epoch for shape :', output_shape)
                X = processed_input_images_dict[output_shape]
                Y = image_lable_dict[output_shape]
                # X_train = np.concatenate(X)
                X_train = X
                y = np.concatenate(Y)
                Y_train = np_utils.to_categorical(y, nb_classes=nb_classes)  # to get sofmax shape of (None, nb_classes)
                X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=3)
                model.fit(X_train, y_train, batch_size=8, nb_epoch=NB_EPOCH, validation_data=(X_test, y_test),
                          shuffle='batch', callbacks=callbacks)

    except KeyboardInterrupt as ke:
        model.save('eyeq_model.h5')

    model.save('eyeq_model.h5')


if __name__ == '__main__':
        train_network()
