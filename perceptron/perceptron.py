#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
from sklearn.utils import shuffle


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + math.exp(-x))


def delta_weight(beta, true_label, predicted_label, x):
    """

    :param beta:
    :param true_label:
    :param predicted_label:
    :param x:
    """
    delta_w = -1 * beta * (predicted_label - true_label) * x
    return delta_w


def training_perceptron(beta, X, Y, bias=1, number_of_epoch=5000):
    """

    :param beta: learning rate of perceptron
    :param X: the feature set for training
    :param Y: the target value against feature set
    """

    W = np.random.rand(1, X.shape[1])
    tracking_param_list = []  # not used in training perceptron
    while (number_of_epoch):
        track_for_all_case_true = True
        X, Y = shuffle(X, Y)
        for feature_row, true_label in zip(X, Y):
            theta = np.dot(np.array(feature_row), W.T) + bias
            predicted_output = 1 if theta > 0 else 0
            delta_W = [delta_weight(beta, true_label, predicted_output, x) for x in feature_row]

            tracking_param_list.append(
                [[feature_row], true_label, np.around(W, decimals=1), predicted_output, theta,
                 delta_W])  # not used in tuning params

            W = np.add(W, delta_W)
            track_for_all_case_true = track_for_all_case_true and (int(predicted_output) == int(true_label))

        if track_for_all_case_true:
            print('perceptron for given dataset trained...')
            break
        number_of_epoch -= 1
    import pandas as pd
    df = pd.DataFrame(tracking_param_list,
                      columns=['feature_row', 'true_label', 'weight vector', 'predicted_output', 'theta', 'delta_W'])
    pd.set_option('display.width', 200)
    print('number_of_epoch : ', number_of_epoch)
    print(df)
    return number_of_epoch


if __name__ == '__main__':
    X = [[0, 1, 0, 1, 1], [0, 1, 1, 1, 0], [1, 0, 1, 0, 1], [0, 1, 1, 0, 0]]
    Y = [1, 1, 1, 0]
    beta = 0.1
    W = training_perceptron(beta, np.array(X), np.array(Y))