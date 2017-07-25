#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
import time

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

BIAS = 1


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + math.exp(-x))


def delta_weight(eta, true_label, predicted_label, x):
    """

    :param eta: Learning rate
    :param true_label:
    :param predicted_label:
    :param x:
    """
    lambda_param = -1
    delta_w = lambda_param * eta * (predicted_label - true_label) * x
    return delta_w


def training_perceptron(eta, X, Y, number_of_epoch=5000):
    """

    :param eta: learning rate of perceptron
    :param X: the feature set for training
    :param Y: the target value against feature set
    """

    logging.info('Training Config:\nNumber_of_epoch: {} Eta: {}'.format(number_of_epoch, eta))
    W = np.random.rand(1, X.shape[1] + 1)
    loss_log = []
    X = np.insert(X, 2, values=1, axis=1)
    for epoch in range(number_of_epoch):
        X, Y = shuffle(X, Y)
        loss = 0.0

        for index, (feature_row, true_label) in enumerate(zip(X, Y)):
            theta = np.dot(np.array(feature_row), W.T)
            # predicted_output = 1 if theta > 0 else 0
            predicted_output = float(theta)

            loss += (true_label - predicted_output) ** 2
            delta_W = [delta_weight(eta, true_label, predicted_output, x) for x in feature_row]
            logging.debug([feature_row, true_label, np.around(W, decimals=1), predicted_output, theta, delta_W])

            W = np.add(W, delta_W)
        if epoch % 10 == 0:
            loss_log.append([epoch, loss])
        logging.info('Epoch Summary : Epoch: {} Loss: {}'.format(epoch, loss))
        if loss < 0.001:
            break

        time.sleep(0.001)
    df = pd.DataFrame(loss_log, columns=['Epoch', 'Loss'])
    logging.info(df)
    df.to_csv('training_log.csv')
    return number_of_epoch


def print_training_details(tracking_param_list):
    import pandas as pd
    df = pd.DataFrame(tracking_param_list,
                      columns=['feature_row', 'true_label', 'weight vector', 'predicted_output', 'theta', 'delta_W'])
    print(df)


if __name__ == '__main__':
    logging.basicConfig(format='[%(name)s] [%(asctime)s] %(levelname)s : %(message)s', level=logging.INFO)
    pd.set_option('display.width', 200)

    X = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [0, 1, 1]])
    Y = np.array([0, 1, 1, 0])


    eta = 0.01 # learning Rate


    print('Training Data : \n', pd.DataFrame([[x, y] for x, y in zip(X, Y)], columns=['X', 'Y']))

    W = training_perceptron(eta, X, Y)
