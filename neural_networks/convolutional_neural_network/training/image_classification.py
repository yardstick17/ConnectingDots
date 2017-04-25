import os
from glob import glob

import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils

from neural_networks.convolutional_neural_network.convnet_keras.convnetskeras.convnets import convnet
from neural_networks.convolutional_neural_network.pre_processing.image_batch_preprocessing import \
    preprocess_image_batch

PREFIX_DIR_LOCATION = os.path.expanduser('~/ConectingDots')
IMAGE_MODEL_CHECKPOINT = os.path.join(PREFIX_DIR_LOCATION,
                                      'trained_models',
                                      'image_clazzification_final.model.checkpoint.h5')

IMAGE_MODEL_WEIGHTS = os.path.join(PREFIX_DIR_LOCATION,
                                   'trained_models',
                                   'image_clazzification.model.weights.h5')
BEST_IMAGE_MODEL = os.path.join(PREFIX_DIR_LOCATION,
                                'trained_models',
                                'image_classification_best_accuracy.h5')

DATA_SRC_DIRECTORY = os.path.join(PREFIX_DIR_LOCATION,
                                  'dataset')
CLASS_NAME_MAPPING = {}
# IMAGE_RESCALE_SIZE = DeepNet.IMAGE_RESCALE_SIZE
CSV_LOG_FILENAME = os.path.join(PREFIX_DIR_LOCATION,
                                'logs',
                                'model_train_log.csv')
PER_CLASS_MAX_IMAGES = 10
NB_EPOCH = 2
NETWORK_MODEL = 'alexnet'


def get_model(NETWORK_MODEL, nb_classes):
    return convnet(NETWORK_MODEL, nb_classes)


def get_data():
    data_set_input_images_files, data_set_input_images_true_label = get_class_wise_images_and_true_label()
    processed_input_images = [preprocess_image_batch([image], NETWORK_MODEL)
                               for image in data_set_input_images_files]

    global CLASS_NAME_MAPPING
    nb_classes = len(CLASS_NAME_MAPPING.keys())
    print('Number of classes for Classification found : ', nb_classes, '\n', CLASS_NAME_MAPPING)
    X_train = np.concatenate(processed_input_images)

    y_out = np.concatenate(data_set_input_images_true_label)
    y_out = np_utils.to_categorical(y_out, nb_classes=nb_classes)  # to get sofmax shape of (None, nb_classes)
    Y_train = y_out

    print(X_train.shape)
    print(Y_train.shape)

    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train)
    split_number = int(len(X_train) * 0.2)
    X_test = X_train[:split_number]
    Y_test = Y_train[:split_number]
    return X_train[split_number:], Y_train[split_number:], X_test, Y_test, nb_classes


def get_class_wise_images_and_true_label():
    directory = glob(DATA_SRC_DIRECTORY + '/*')
    data_set_input_images = []
    data_set_input_images_true_label = []
    global CLASS_NAME_MAPPING
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
    callbacks.append(keras.callbacks.CSVLogger(CSV_LOG_FILENAME))
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
    print('processing data for training')
    X_train, Y_train, X_test, Y_test, nb_classes = get_data()
    print('getting model')
    model = get_model(NETWORK_MODEL, nb_classes)

    sgd = SGD()
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.summary()
    callbacks = get_callbacks()
    print('fitting model for the dataset')
    model.fit(X_train, Y_train, nb_epoch=NB_EPOCH, validation_data=(X_test, Y_test),
              shuffle='batch', callbacks=callbacks)


if __name__ == '__main__':
    train_network()
