import os
from glob import glob

import numpy as np
from keras.optimizers import SGD

from neural_networks.convolutional_neural_network.networks import DeepNet
from neural_networks.convolutional_neural_network.networks.DeepNet import DeepNet as DeepNetwork
from neural_networks.convolutional_neural_network.pre_processing.image_batch_preprocessing import \
    preprocess_image_batch

IMAGE_MODEL_CHECKPOINT = os.path.expanduser('~/image_clazzification_final.model.checkpoint.h5')
IMAGE_MODEL_WEIGHTS = os.path.expanduser('~/image_clazzification.model.weights.h5')
IMAGE_MODEL = os.path.expanduser('~/image_clazzification.model.h5')

DATA_SRC_DIRECTORY = os.path.expanduser('~/ConectingDots/data/zomato_photos/*')
class_name_mapping = {}
IMAGE_RESCALE_SIZE = DeepNet.IMAGE_RESCALE_SIZE


def get_model():
    return DeepNetwork()


def get_data():
    global DATA_SRC_DIRECTORY
    directory = glob(DATA_SRC_DIRECTORY)

    data_set_input = []
    data_set_true_label = []
    global class_name_mapping
    index = 0
    for sub_directory in directory:

        if os.path.isdir(sub_directory):
            class_dir_name = sub_directory.split('/')[-1]
            image_class_files = glob(sub_directory + '/*.jpg')[:10]

            class_image_inputs = [preprocess_image_batch([image], color_mode='rgb',
                                                         img_size=(IMAGE_RESCALE_SIZE, IMAGE_RESCALE_SIZE),
                                                         crop_size=(IMAGE_RESCALE_SIZE, IMAGE_RESCALE_SIZE))
                                  for image in image_class_files]
            data_set_input.extend(class_image_inputs)
            data_set_true_label.extend([ [index] ] * len(class_image_inputs))
            class_name_mapping[index] = class_dir_name
            index += 1
    print('class_name_mapping: ', class_name_mapping)

    X_train = np.concatenate(data_set_input)
    Y_train = np.array(data_set_true_label)
    print(X_train.shape)
    print(Y_train.shape)

    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train)
    split_number = int(len(X_train) * 0.2)
    X_test = X_train[:split_number]
    Y_test = Y_train[:split_number]
    return X_train[split_number:], Y_train[split_number:], X_test, Y_test


def get_callbacks():
    callbacks = []
    import keras
    callbacks.append(keras.callbacks.ModelCheckpoint(
        # 'image_clazzification_final.model.{epoch:02d}-{val_loss:.2f}.hdf5'
        IMAGE_MODEL_CHECKPOINT,
        monitor='val_loss',
        # verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto')
    )
    return callbacks


def train_network():
    X_train, Y_train, X_test, Y_test = get_data()
    model = get_model()
    NB_EPOCH = 100

    sgd = SGD()
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.summary()
    callbacks = get_callbacks()
    model.fit(X_train, Y_train, nb_epoch=NB_EPOCH, validation_data=(X_test, Y_test),
              shuffle='batch', callbacks=callbacks)


if __name__ == '__main__':
    train_network()
